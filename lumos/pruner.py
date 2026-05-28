import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
import math
from torch.autograd import Variable
from typing import Dict, Tuple, Optional
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
limit_a, limit_b, epsilon = -0.1, 1.1, 1e-6


class _FlattenSelect(nn.Module):
    """Flatten input and select only the surviving (non-pruned) feature indices.

    Replaces ``nn.Flatten`` after structural input-feature pruning of the first
    Linear layer.  ``keep_indices`` is a 1-D buffer of the flat feature positions
    that were *not* gated off by L0.
    """

    def __init__(self, keep_indices):
        super().__init__()
        self.register_buffer(
            'keep_indices',
            torch.tensor(sorted(keep_indices), dtype=torch.long),
        )

    def forward(self, x):
        return x.flatten(1)[:, self.keep_indices]


class L0Pruner:
    """
    L0 regularization pruner for neural networks.
    
    This class applies L0 regularization to PyTorch neural networks by adding
    learnable gates to nn.Linear, nn.Conv2d, and nn.Conv3d layers. The gates
    are sampled from a hard-concrete distribution during training and used
    to prune channels/units during inference.
    
    Attributes:
        model: The neural network model to be pruned
        droprate_init: Initial dropout rate for L0 gates
        temperature: Temperature for the concrete distribution
        weight_decay: Strength of the L2 penalty
        lamba: Strength of the L0 penalty
        local_rep: Whether to use separate gate samples per minibatch element
        device: Device for computations
        layer_gates: Dictionary mapping layer names to their L0 gates (qz_loga)
        ignore_layers: List of layers to ignore
    """
    
    def __init__(
        self,
        model: nn.Module,
        droprate_init: float = 0.5,
        temperature: float = 2.0 / 3.0,
        weight_decay: float = 1.0,
        lamba: float = 1.0,
        lamda: float = 50000,
        local_rep: bool = False,
        device: Optional[str] = None,
        ignore_layers: Optional[list] = None,
        sparse_training: bool = False,
        writer_dir: Optional[str] = None,
    ):
        """
        Initialize the L0Pruner.
        
        Args:
            model: PyTorch neural network model to be pruned
            droprate_init: Initial dropout rate for L0 gates (default: 0.5)
            temperature: Temperature for the concrete distribution (default: 2.0/3.0)
            weight_decay: Strength of the L2 penalty (default: 1.0)
            lamba: Strength of the L0 penalty compared to L2 penalty (default: 1.0)
            lamda: Strength of the L0 penalty to the loss (default: 50000)
            local_rep: Whether to use separate gate samples per minibatch element (default: False)
            device: Device for computations (default: auto-detect)
            ignore_layers: List of layers to ignore (default: None)
            sparse_training: Whether to initialize gates and modify forward passes immediately (default: False)
            writer_dir: Directory for tensorboard writer (default: None)
        """
        self.model = model
        self.droprate_init = droprate_init if droprate_init != 0.0 else 0.5
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.lamba = lamba
        self.lamda = lamda
        self.local_rep = local_rep
        self.device = device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.ignore_layers = ignore_layers if ignore_layers is not None else []
        
        # Storage for L0 gates and masks
        self.layer_gates = {}
        self.layer_masks = {}
        self.layer_gates_params_par = {}
        
        if sparse_training:
            # Initialize L0 gates for supported layers
            self._initialize_l0_gates()
            
            # Modify layer forward methods
            self._modify_layer_forwards()
            
        # Use lazy import to avoid TensorBoard compatibility issues
        if writer_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=writer_dir)
            except Exception as e:
                print(f"Warning: Could not initialize TensorBoard writer: {e}")
                self.writer = None
        else:
            self.writer = None
        
        # for regularize_v1
        self.regularize_status = False
        self.regularize_step = 0
        
    def _initialize_l0_gates(self):
        """Initialize L0 gates (qz_loga) for supported layers."""
        for name, module in self.model.named_modules():
            if name in self.ignore_layers or type(module) in self.ignore_layers:
                continue
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                # Determine the dimension for the gate
                if isinstance(module, nn.Linear):
                    dim_z = module.in_features  # Gate input features for Linear
                else:  # Conv2d or Conv3d
                    dim_z = module.out_channels  # Gate output channels for Conv
                
                # Create learnable parameter qz_loga
                qz_loga = nn.Parameter(
                    torch.Tensor(dim_z).to(self.device)
                )
                
                # Initialize qz_loga
                qz_loga.data.normal_(
                    math.log(1 - self.droprate_init) - math.log(self.droprate_init), 
                    1e-2
                )
                
                # Store the gate
                self.layer_gates[name] = qz_loga
                self.layer_gates_params_par[name] = (qz_loga.numel(), module.weight.numel())
                
                # Register parameter with the model
                setattr(module, "qz_loga", qz_loga)
                module.register_parameter("qz_loga", qz_loga)
                
                # store the original length of the qz_loga
                module.qz_loga_length = dim_z
    
    def _modify_layer_forwards(self):
        """Modify forward methods of supported layers to incorporate L0 gates."""
        for name, module in self.model.named_modules():
            if name in self.ignore_layers or type(module) in self.ignore_layers:
                continue
            if name in self.layer_gates:
                # Store original forward method as an attribute of the module
                module._original_forward = module.forward
                
                # Create new forward method with L0 gates
                if isinstance(module, nn.Linear):
                    module.forward = self._create_linear_forward(name, module)
                elif isinstance(module, nn.Conv2d):
                    module.forward = self._create_conv2d_forward(name, module)
                elif isinstance(module, nn.Conv3d):
                    module.forward = self._create_conv3d_forward(name, module)
    
    def _create_linear_forward(self, name: str, module: nn.Linear):
        """Create modified forward method for nn.Linear layers."""
        def forward(input_):
            if self.local_rep or not self.model.training:
                # Sample gates for each batch element or during inference
                m = self._sample_z_linear(name, input_.size(0), sample=self.model.training)
                xin = input_.mul(m)
                output = F.linear(xin, module.weight, module.bias)
            else:
                # Sample weights directly
                weights = self._sample_weights_linear(name, module)
                output = F.linear(input_, weights, module.bias)
            
            return output
        
        return forward
    
    def _create_conv2d_forward(self, name: str, module: nn.Conv2d):
        """Create modified forward method for nn.Conv2d layers."""
        def forward(input_):
            if self.local_rep or not self.model.training:
                # Apply convolution first, then gate
                output = F.conv2d(
                    input_, module.weight, module.bias, module.stride,
                    module.padding, module.dilation, module.groups
                )
                z = self._sample_z_conv2d(name, output.size(0), sample=self.model.training)
                return output.mul(z)
            else:
                # Sample weights directly
                weights = self._sample_weights_conv2d(name, module)
                return F.conv2d(
                    input_, weights, None, module.stride,
                    module.padding, module.dilation, module.groups
                )
        
        return forward
    
    def _create_conv3d_forward(self, name: str, module: nn.Conv3d):
        """Create modified forward method for nn.Conv3d layers."""
        def forward(input_):
            if self.local_rep or not self.model.training:
                # Apply convolution first, then gate
                output = F.conv3d(
                    input_, module.weight, module.bias, module.stride,
                    module.padding, module.dilation, module.groups
                )
                z = self._sample_z_conv3d(name, output.size(0), sample=self.model.training)
                return output.mul(z)
            else:
                # Sample weights directly
                weights = self._sample_weights_conv3d(name, module)
                return F.conv3d(
                    input_, weights, None, module.stride,
                    module.padding, module.dilation, module.groups
                )
        
        return forward
    
    def cdf_qz(self, x: torch.Tensor, qz_loga: torch.Tensor) -> torch.Tensor:
        """Compute the CDF of the 'stretched' concrete distribution."""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - qz_loga).clamp(
            min=epsilon, max=1 - epsilon
        )
    
    def quantile_concrete(self, x: torch.Tensor, qz_loga: torch.Tensor) -> torch.Tensor:
        """Compute the quantile (inverse CDF) of the 'stretched' concrete distribution."""
        x = x.to(qz_loga.device)
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + qz_loga) / self.temperature
        )
        return y * (limit_b - limit_a) + limit_a
    
    def get_eps(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Generate uniform random numbers for the concrete distribution."""
        eps = torch.empty(size, device=self.device).uniform_(epsilon, 1 - epsilon)
        return Variable(eps)
    
    def _sample_z_linear(self, name: str, batch_size: int, sample: bool = True) -> torch.Tensor:
        """Sample hard-concrete gates for nn.Linear layers."""
        qz_loga = self.layer_gates[name]
        dim_z = qz_loga.size(0)
        
        if sample:
            eps = self.get_eps((batch_size, dim_z))
            z = self.quantile_concrete(eps, qz_loga)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.sigmoid(qz_loga).view(1, dim_z)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)
    
    def _sample_z_conv2d(self, name: str, batch_size: int, sample: bool = True) -> torch.Tensor:
        """Sample hard-concrete gates for nn.Conv2d layers."""
        qz_loga = self.layer_gates[name]
        dim_z = qz_loga.size(0)
        
        if sample:
            eps = self.get_eps((batch_size, dim_z))
            z = self.quantile_concrete(eps, qz_loga).view(batch_size, dim_z, 1, 1)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.sigmoid(qz_loga).view(1, dim_z, 1, 1)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)
    
    def _sample_z_conv3d(self, name: str, batch_size: int, sample: bool = True) -> torch.Tensor:
        """Sample hard-concrete gates for nn.Conv3d layers."""
        qz_loga = self.layer_gates[name]
        dim_z = qz_loga.size(0)
        
        if sample:
            eps = self.get_eps((batch_size, dim_z))
            z = self.quantile_concrete(eps, qz_loga).view(batch_size, dim_z, 1, 1, 1)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.sigmoid(qz_loga).view(1, dim_z, 1, 1, 1)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)
    
    def _sample_weights_linear(self, name: str, module: nn.Linear) -> torch.Tensor:
        """Sample weights for nn.Linear layers."""
        qz_loga = self.layer_gates[name]
        z = self.quantile_concrete(self.get_eps(qz_loga.size()), qz_loga)
        m = F.hardtanh(z, min_val=0, max_val=1)
        return module.weight * m.view(1, -1)
    
    def _sample_weights_conv2d(self, name: str, module: nn.Conv2d) -> torch.Tensor:
        """Sample weights for nn.Conv2d layers."""
        qz_loga = self.layer_gates[name]
        z = self.quantile_concrete(self.get_eps(qz_loga.size()), qz_loga)
        m = F.hardtanh(z, min_val=0, max_val=1)
        return module.weight * m.view(-1, 1, 1, 1)
    
    def _sample_weights_conv3d(self, name: str, module: nn.Conv3d) -> torch.Tensor:
        """Sample weights for nn.Conv3d layers."""
        qz_loga = self.layer_gates[name]
        z = self.quantile_concrete(self.get_eps(qz_loga.size()), qz_loga)
        m = F.hardtanh(z, min_val=0, max_val=1)
        return module.weight * m.view(-1, 1, 1, 1, 1)
    
    def constrain_parameters(self):
        """Constrain qz_loga parameters for numerical stability."""
        for qz_loga in self.layer_gates.values():
            qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))
            
    def regularize(self) -> torch.Tensor:
        """
        Compute the L0 regularization loss for all modified layers.
        
        Returns:
            Total regularization loss as a scalar tensor
        """
        regularization = 0.0
        
        for name, module in self.model.named_modules():
            if name in self.ignore_layers or type(module) in self.ignore_layers:
                continue
            if name in self.layer_gates:
                qz_loga = self.layer_gates[name]
                q0 = 1 - self.cdf_qz(torch.tensor(0.0).to(qz_loga.device), qz_loga)
                
                if isinstance(module, nn.Linear):
                    # Matches reference L0Dense._reg_w: lamba is inside the sum over out_features.
                    # Each input gate controls out_features weights, so L0 cost = lamba * out_features.
                    logpw_col = torch.sum(
                        -(0.5 * self.weight_decay * module.weight.pow(2)) - self.lamba, 0
                    )
                    logpw = torch.sum(q0 * logpw_col)
                    logpb = 0
                    if module.bias is not None:
                        logpb = -torch.sum(0.5 * self.weight_decay * module.bias.pow(2))
                    regularization += logpw + logpb
                
                elif isinstance(module, nn.Conv2d):
                    # Matches reference L0Conv2d._reg_w: lamba inside sum over (in_ch, kH, kW).
                    # Each output-channel gate controls in_ch*kH*kW weights, so
                    # L0 cost per gate = lamba * in_ch * kH * kW.
                    logpw_col = (
                        torch.sum(-(0.5 * self.weight_decay * module.weight.pow(2)) - self.lamba, 3)
                        .sum(2).sum(1)
                    )
                    logpw = torch.sum(q0 * logpw_col)
                    logpb = 0
                    if module.bias is not None:
                        logpb = -torch.sum(
                            q0 * (0.5 * self.weight_decay * module.bias.pow(2) - self.lamba)
                        )
                    regularization += logpw + logpb
                
                elif isinstance(module, nn.Conv3d):
                    # Same pattern for Conv3d: lamba inside sum over (in_ch, kD, kH, kW).
                    logpw_col = (
                        torch.sum(-(0.5 * self.weight_decay * module.weight.pow(2)) - self.lamba, 4)
                        .sum(3).sum(2).sum(1)
                    )
                    logpw = torch.sum(q0 * logpw_col)
                    logpb = 0
                    if module.bias is not None:
                        logpb = -torch.sum(
                            q0 * (0.5 * self.weight_decay * module.bias.pow(2) - self.lamba)
                        )
                    regularization += logpw + logpb
        
        reg_loss = -regularization / self.lamda
        return reg_loss
    
    def get_prune_ratio(self) -> float:
        """Get the prune ratio for all layers."""
        masks = self.get_masks()
        total_params = 0
        pruned_params = 0
        for name, (qz_loga_numel, weight_numel) in self.layer_gates_params_par.items():
            mask = masks[name]
            pruned_params += torch.sum(mask.flatten() < 1e-6) * weight_numel / len(mask.flatten())
            total_params += weight_numel
            
        return pruned_params / total_params
            
    def get_masks(self) -> Dict[str, torch.Tensor]:
        """
        Get current masks for all layers.
        
        Returns:
            Dictionary mapping layer names to their masks
        """
        masks = {}
        for name, qz_loga in self.layer_gates.items():
            if name in self.ignore_layers or type(self.model.get_submodule(name)) in self.ignore_layers:
                continue
            masks[name] = self.get_mask(name)
            
        return masks

    def get_mask(self, name: str):
        """Get current masks for a specific layer."""
        qz_loga = self.layer_gates[name]
        if isinstance(self.model.get_submodule(name), nn.Linear):
            pi = torch.sigmoid(qz_loga).view(1, -1)
            mask = F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)
            return mask.view(-1, 1)
        else:  # Conv2d or Conv3d
            pi = torch.sigmoid(qz_loga).view(1, -1, 1, 1)
            if isinstance(self.model.get_submodule(name), nn.Conv3d):
                pi = pi.unsqueeze(-1)
            mask = F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)
            return mask
        
    def get_mask_from_layer_qz_loga(self, module: nn.Module) -> torch.Tensor:
        """Get the mask from the qz_loga of a specific layer."""
        qz_loga = getattr(module, "qz_loga")
        if isinstance(module, nn.Linear):
            pi = torch.sigmoid(qz_loga).view(1, -1)
            mask = F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)
            return mask.view(-1, 1)
        else:  # Conv2d or Conv3d
            pi = torch.sigmoid(qz_loga).view(1, -1, 1, 1)
            if isinstance(module, nn.Conv3d):
                pi = pi.unsqueeze(-1)
            mask = F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)
            return mask
    
    @staticmethod
    def viz_masks(masks: Dict[str, torch.Tensor], dir: str = "run/viz/masks", step: int = 0):
        """Visualize the masks for all layers."""
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            shutil.rmtree(dir)
            os.makedirs(dir)    
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=dir)
            for name, mask in masks.items():
                writer.add_histogram(f"mask/{name}", mask.flatten(), global_step=step)
            writer.close()
        except Exception as e:
            print(f"Warning: Could not use TensorBoard for mask visualization: {e}")
    
    @staticmethod
    def viz_weights(model: nn.Module, dir: str = "run/viz/weights", step: int = 0):
        """Visualize the weights for all layers."""
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            shutil.rmtree(dir)
            os.makedirs(dir)    
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=dir)
        except Exception as e:
            print(f"Warning: Could not use TensorBoard for weight visualization: {e}")
            return
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                # Basic histogram for all layers
                writer.add_histogram(f"weight/{name}", module.weight.flatten(), global_step=step)
                
                if isinstance(module, nn.Linear):
                    weight_2d = module.weight.detach().cpu().numpy()
                    writer.add_image(
                        f"weight_heatmap/{name}", 
                        torch.from_numpy(weight_2d).unsqueeze(0).unsqueeze(0), 
                        global_step=step,
                        dataformats='NCHW'
                    )
                
                elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
                    weight = module.weight.detach().cpu()
                    kernels = weight.view(weight.size(0), -1)
                    fig, ax = plt.subplots(figsize=(15, 8))
                    sns.boxplot(data=kernels.t().numpy(), ax=ax)
                    ax.set_xlabel("Kernel Index (Output Channel)")
                    ax.set_ylabel("Weight Value")
                    ax.set_title(f"Weight Distribution for Kernels in {name}")
                    writer.add_figure(f"weight_boxplot/{name}", fig, global_step=step)
                    plt.close(fig)
        
        try:
            writer.close()
        except:
            pass
    
    def merge_mask(self) -> nn.Module:
        """Merge masks into model weights and restore original forward methods.

        Uses a **binary** (0/1) gate: a channel is either fully active (mask=1,
        weight unchanged) or pruned (mask=0, weight zeroed).  Using the raw soft
        gate would scale active-channel weights by ~0.5 (sigmoid(0)*1.2-0.1),
        which corrupts the calibration of any subsequent fine-tuning pass and
        makes the weights incorrect when loading a fine-tuned checkpoint.
        """
        merged_model = self.model
        
        for name, module in merged_model.named_modules():
            if name in self.layer_gates:
                # Binary mask: 0 for pruned channels, 1 for active channels.
                # A channel is pruned iff its soft gate value is ~0 (qz_loga < -2.4).
                soft = self.get_mask(name)
                mask = (soft > 0.0).float()
                
                if isinstance(module, nn.Linear):
                    mask_flat = mask.flatten()
                    module.weight.data = module.weight.data * mask_flat.view(1, -1)
                
                elif isinstance(module, nn.Conv2d):
                    mask_flat = mask.flatten()
                    module.weight.data = module.weight.data * mask_flat.view(-1, 1, 1, 1)
                    if module.bias is not None:
                        module.bias.data = module.bias.data * mask_flat
                
                elif isinstance(module, nn.Conv3d):
                    mask_flat = mask.flatten()
                    module.weight.data = module.weight.data * mask_flat.view(-1, 1, 1, 1, 1)
                    if module.bias is not None:
                        module.bias.data = module.bias.data * mask_flat
                
                if hasattr(module, '_original_forward'):
                    module.forward = module._original_forward
        
        return merged_model

    @staticmethod
    def _get_zero_out_indices(module, epsilon=1e-6):
        """Indices of output channels whose weights AND bias are all zero.
        
        After BN fusion, a conv channel with zero weight but non-zero fused
        bias still produces a constant non-zero output.  Such channels are NOT
        safe to structurally remove.
        """
        w = module.weight.data
        per_ch = w.view(w.size(0), -1).abs().max(dim=1)[0]
        zero_weight = per_ch < epsilon
        
        if module.bias is not None:
            zero_bias = module.bias.data.abs() < epsilon
            zero_both = zero_weight & zero_bias
        else:
            zero_both = zero_weight
        
        return torch.where(zero_both)[0].cpu().tolist()

    @staticmethod
    def _get_zero_in_indices(module, epsilon=1e-6):
        """Indices of input features whose weights are all zero."""
        w = module.weight.data
        per_feat = w.abs().max(dim=0)[0]
        return torch.where(per_feat < epsilon)[0].cpu().tolist()

    @staticmethod
    def _safe_prune_indices(group, candidate_idxs, epsilon=1e-6):
        """Filter candidate prune indices to keep only those that are safe.
        
        In residual networks, pruning one layer's output channels propagates
        through the dependency graph to coupled layers.  A candidate index is
        unsafe if a *coupled output-channel layer* (reached via the residual
        add) has non-zero weights at that index.
        
        Key insight: when we prune output channels of layer A, the dependency
        graph also removes the corresponding *input* channels of layer B
        (the next layer).  Those input channels may have non-zero weights, but
        they receive zero input (from A's zeroed output), so removing them is
        lossless.  We therefore only check output-channel dependencies, not
        input-channel ones.
        
        We also skip BatchNorm layers since they are purely dependent — they
        just scale/shift and are removed together with the channel.
        
        Args:
            group: A torch_pruning Group (iterable of (Dep, idxs) pairs).
            candidate_idxs: The initially proposed prune indices.
            epsilon: Threshold below which a value is considered zero.
        
        Returns:
            List of indices that are safe to prune.
        """
        candidate_set = set(candidate_idxs)
        
        for dep, dep_idxs in group:
            mod = dep.target.module if hasattr(dep.target, 'module') else None
            if mod is None or not hasattr(mod, 'weight'):
                continue
            
            handler_name = dep.handler.__name__ if hasattr(dep.handler, '__name__') else ''
            
            # Skip BatchNorm — purely dependent, removed with the channel
            if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d)):
                continue
            
            # Skip input-channel pruning — safe because the upstream output
            # channel is zero, so these input channels receive zero regardless
            # of their weight values.
            if 'in' in handler_name:
                continue
            
            # Only check output-channel dependencies (these are the coupled
            # layers reached through residual additions).
            if 'out' in handler_name:
                w = mod.weight.data
                if isinstance(mod, (nn.Conv2d, nn.Conv3d)):
                    per_ch = w.view(w.size(0), -1).abs().max(dim=1)[0]
                elif isinstance(mod, nn.Linear):
                    per_ch = w.abs().max(dim=1)[0]
                else:
                    continue
                
                # Also check bias (after BN fusion, zero-weight channels may
                # still have non-zero fused bias)
                has_bias = hasattr(mod, 'bias') and mod.bias is not None
                    
                for i, orig_idx in enumerate(dep_idxs):
                    if i < len(candidate_idxs) and orig_idx < len(per_ch):
                        weight_nonzero = per_ch[orig_idx].item() > epsilon
                        bias_nonzero = (has_bias and orig_idx < len(mod.bias.data)
                                        and mod.bias.data[orig_idx].abs().item() > epsilon)
                        if weight_nonzero or bias_nonzero:
                            candidate_set.discard(candidate_idxs[i])
        
        return sorted(candidate_set)

    @staticmethod
    def _fuse_bn_into_conv(model):
        """Fuse BatchNorm layers into their preceding Conv layers.
        
        Converts Conv→BN pairs into a single Conv with adjusted weights and bias.
        After fusion, zero-weight conv channels truly produce zero output with
        no BN running_mean/beta offset.  This is essential for lossless
        structural pruning.
        
        The BN modules are replaced with nn.Identity().
        """
        module_map = {n: m for n, m in model.named_modules()}
        
        for name, module in list(model.named_modules()):
            if not isinstance(module, (nn.Conv2d, nn.Conv3d)):
                continue
            
            # Try to find the corresponding BN layer
            bn_candidates = [name.replace('conv', 'bn')]
            parts = name.rsplit('.', 1)
            if len(parts) == 2 and parts[1].isdigit():
                bn_candidates.append(f"{parts[0]}.{int(parts[1]) + 1}")
            
            for bn_name in bn_candidates:
                if bn_name not in module_map:
                    continue
                bn = module_map[bn_name]
                if not isinstance(bn, (nn.BatchNorm2d, nn.BatchNorm3d)):
                    continue
                
                # Fuse: new_weight = (gamma / sqrt(var + eps)) * weight
                #        new_bias = gamma * (- mean) / sqrt(var + eps) + beta
                gamma = bn.weight.data
                beta = bn.bias.data
                mean = bn.running_mean
                var = bn.running_var
                eps = bn.eps
                
                std = torch.sqrt(var + eps)
                scale = gamma / std
                
                # Fuse into conv weight
                if isinstance(module, nn.Conv2d):
                    module.weight.data = module.weight.data * scale.view(-1, 1, 1, 1)
                else:
                    module.weight.data = module.weight.data * scale.view(-1, 1, 1, 1, 1)
                
                # Create bias if conv didn't have one
                if module.bias is None:
                    module.bias = nn.Parameter(torch.zeros(module.out_channels, device=module.weight.device))
                module.bias.data = scale * (module.bias.data - mean) + beta
                
                # Replace BN with Identity
                bn_parts = bn_name.rsplit('.', 1)
                if len(bn_parts) == 2:
                    parent = model.get_submodule(bn_parts[0])
                    setattr(parent, bn_parts[1], nn.Identity())
                else:
                    setattr(model, bn_name, nn.Identity())
                
                break  # Found and fused, move to next conv
        
        return model

    def prune_model(self, example_inputs=None):
        """Apply permanent pruning based on learned masks using torch_pruning.
        
        Pipeline:
        1. merge_mask() — zeros conv weights at pruned channels
        2. BN fusion — folds BN into Conv (zero-weight channels get fused bias)
        3. Zero negative-bias channels — for zero-weight channels where the
           fused bias is negative, relu kills it, so bias can be zeroed too
        4. Structural pruning via torch_pruning with safety checks — only
           prunes channels where BOTH weight and bias are zero
        
        Args:
            example_inputs: Example inputs for building the dependency graph.
        """
        # First, merge masks into the model weights and restore original forward methods
        merged_model = self.merge_mask()
        
        # Fuse BatchNorm into Conv layers
        merged_model = self._fuse_bn_into_conv(merged_model)
        
        # For zero-weight channels with negative fused bias in layers that are
        # DIRECTLY followed by ReLU (not by residual-add-then-relu), the relu
        # kills the constant output, so the bias can safely be zeroed.
        # In BasicBlock: conv1 is followed by relu; conv2 is followed by add+relu.
        # In Bottleneck: conv1, conv2 are followed by relu; conv3 by add+relu.
        # We identify "relu-next" layers by name convention: convN where N is
        # not the last conv in the block.
        epsilon = 1e-6
        for name, module in merged_model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Conv3d)) or module.bias is None:
                continue
            
            # Determine if this conv is directly followed by ReLU.
            # conv1 in BasicBlock/Bottleneck: yes (relu right after bn1)
            # conv2 in Bottleneck: yes (relu right after bn2)
            # conv2 in BasicBlock: NO (add + relu)
            # conv3 in Bottleneck: NO (add + relu)
            # shortcut.0: NO (add + relu, or just passes through)
            # Standalone convs (stem conv1): followed by bn+relu, yes
            parts = name.rsplit('.', 1)
            is_relu_next = False
            if len(parts) == 2:
                last = parts[1]
                if last == 'conv1':
                    # conv1 in any block: always followed by bn1 + relu
                    is_relu_next = True
                elif last == 'conv2':
                    # conv2 in Bottleneck: followed by bn2 + relu
                    # conv2 in BasicBlock: followed by bn2 + add + relu (NOT safe)
                    # Check if parent has conv3 (Bottleneck) or not (BasicBlock)
                    try:
                        parent = merged_model.get_submodule(parts[0])
                        if hasattr(parent, 'conv3'):
                            is_relu_next = True  # Bottleneck
                    except Exception:
                        pass
            elif name == 'conv1':
                # Stem conv1: followed by bn1 + relu
                is_relu_next = True
            
            if not is_relu_next:
                continue
            
            w = module.weight.data
            per_ch = w.view(w.size(0), -1).abs().max(dim=1)[0]
            zero_weight = per_ch < epsilon
            negative_bias = module.bias.data < -epsilon
            killable = zero_weight & negative_bias
            if killable.any():
                module.bias.data[killable] = 0.0
        
        # For MLP-style linear chains: gates are placed on in_features, so
        # merge_mask zeros input *columns* of each Linear layer.  Structural
        # pruning via prune_linear_in_channels requires the predecessor's
        # *output rows* to also be zero (they appear as a coupled
        # prune_linear_out_channels dependency in the DG group).  Without
        # this propagation, _safe_prune_indices rejects almost every candidate
        # because the predecessor's output rows still have non-zero weights.
        #
        # Propagation rule: for each consecutive Linear→Linear pair where the
        # second layer is gated (in self.layer_gates), zero the output rows of
        # the first layer that correspond to zero input columns of the second.
        # We reset on any Conv layer to avoid incorrect propagation across
        # Conv→Flatten→Linear boundaries.
        _prev_linear_mod = None
        for _lname, _lmod in merged_model.named_modules():
            if isinstance(_lmod, (nn.Conv2d, nn.Conv3d)):
                _prev_linear_mod = None  # break linear chain at conv
            elif isinstance(_lmod, nn.Linear):
                if _prev_linear_mod is not None and _lname in self.layer_gates:
                    _zero_in = self._get_zero_in_indices(_lmod)
                    if _zero_in:
                        _idx = torch.tensor(_zero_in, dtype=torch.long,
                                            device=_prev_linear_mod.weight.device)
                        _prev_linear_mod.weight.data[_idx, :] = 0.0
                        if _prev_linear_mod.bias is not None:
                            _prev_linear_mod.bias.data[_idx] = 0.0
                _prev_linear_mod = _lmod

        # Build a map: gated-Linear-name → preceding-Flatten-name.
        # Used in the post-loop to replace Flatten with _FlattenSelect after all
        # DG-based pruning has run (DG may shrink the layer's output dimension
        # first, so we must defer input-feature pruning until after that).
        _flatten_preceded = {}          # { linear_name: flatten_name }
        _last_flat_name = None
        _seen_param_since_flat = False
        for _n, _m in merged_model.named_modules():
            if isinstance(_m, nn.Flatten):
                _last_flat_name = _n
                _seen_param_since_flat = False
            elif isinstance(_m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                if (isinstance(_m, nn.Linear)
                        and _n in self.layer_gates
                        and _last_flat_name is not None
                        and not _seen_param_since_flat):
                    _flatten_preceded[_n] = _last_flat_name
                _seen_param_since_flat = True

        # Identify unwrapped parameters that should be ignored during pruning
        unwrapped_params = []
        for name, param in merged_model.named_parameters():
            if name in ['pos_embedding', 'cls_token'] or 'pos_embedding' in name or 'cls_token' in name:
                unwrapped_params.append((param, -1))
        
        # Build the dependency graph
        DG = tp.DependencyGraph()
        DG.build_dependency(
            merged_model, 
            example_inputs=example_inputs.to(self.device),
            unwrapped_parameters=unwrapped_params,
            ignored_params=[]
        )
        
        # Add unwrapped parameters to ignore list
        for name, param in merged_model.named_parameters():
            if name in ['pos_embedding', 'cls_token'] or 'pos_embedding' in name or 'cls_token' in name:
                self.ignore_layers.append(name)
        
        # Track which layers have been handled to avoid double-pruning
        pruned_layers = set()
        fully_pruned_layers = set()
        
        for name in list(self.layer_gates.keys()):
            if name in self.ignore_layers or 'mlp.net.0' in name or name in pruned_layers:
                continue
            
            try:
                module = merged_model.get_submodule(name)
            except AttributeError:
                continue
            
            if isinstance(module, nn.Linear):
                candidate_idx = self._get_zero_in_indices(module)
                if not candidate_idx:
                    continue
                
                total = module.in_features
                # Build the group first to check safety
                group = DG.get_pruning_group(module, tp.prune_linear_in_channels, idxs=candidate_idx)

                # If the group has no coupled parametric OUTPUT-channel dependency,
                # the inputs come from a non-parametric source (e.g. nn.Flatten).
                # Pruning would shrink the weight matrix but leave the actual data
                # shape unchanged → runtime shape mismatch.  Skip structural removal;
                # the zeroed columns already provide sparsity at no accuracy cost.
                has_parametric_predecessor = any(
                    ('out' in (dep.handler.__name__ if hasattr(dep.handler, '__name__') else ''))
                    and hasattr(dep.target.module if hasattr(dep.target, 'module') else None, 'weight')
                    for dep, _ in group
                )
                if not has_parametric_predecessor:
                    # Inputs come from a non-parametric source (e.g. Flatten).
                    # Deferred to the post-loop below so DG-based pruning of
                    # this layer's output channels runs first.
                    continue

                safe_idx = self._safe_prune_indices(group, candidate_idx)
                
                if not safe_idx:
                    skipped = len(candidate_idx) - len(safe_idx)
                    if skipped > 0:
                        print(f"  {name}: {skipped} indices skipped (non-zero in coupled layers)")
                    continue
                
                if len(safe_idx) >= total:
                    print(f"Fully-pruned layer (skip): {name} ({len(safe_idx)}/{total} features)")
                    fully_pruned_layers.add(name)
                    continue
                
                # Re-get group with safe indices only
                group = DG.get_pruning_group(module, tp.prune_linear_in_channels, idxs=safe_idx)
                print(f"Pruning layer: {name}, type: Linear, {len(safe_idx)}/{total} features "
                      f"(filtered from {len(candidate_idx)} candidates)")
                group.prune()
                pruned_layers.add(name)
                
            elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
                candidate_idx = self._get_zero_out_indices(module)
                if not candidate_idx:
                    continue
                
                total = module.out_channels
                # Build the group first to check safety
                group = DG.get_pruning_group(module, tp.prune_conv_out_channels, idxs=candidate_idx)
                safe_idx = self._safe_prune_indices(group, candidate_idx)
                
                if not safe_idx:
                    skipped = len(candidate_idx) - len(safe_idx)
                    if skipped > 0:
                        print(f"  {name}: {skipped} indices skipped (non-zero in coupled layers)")
                    continue
                
                if len(safe_idx) >= total:
                    print(f"Fully-pruned layer (skip): {name} ({len(safe_idx)}/{total} channels)")
                    fully_pruned_layers.add(name)
                    continue
                
                # Re-get group with safe indices only
                group = DG.get_pruning_group(module, tp.prune_conv_out_channels, idxs=safe_idx)
                print(f"Pruning layer: {name}, type: Conv, {len(safe_idx)}/{total} channels "
                      f"(filtered from {len(candidate_idx)} candidates)")
                group.prune()
                pruned_layers.add(name)

        # ── Post-loop: structural pruning of input features for Flatten-preceded ──
        # Linear layers (e.g. Flatten → layers.0 in an MLP).
        # We re-fetch each module after DG-based pruning has potentially reduced its
        # output channels, recompute zero input columns on the post-DG weight, then
        # directly slice the weight matrix and replace the preceding nn.Flatten with
        # a _FlattenSelect that discards the same feature positions at runtime.
        for _fl_lin_name, _fl_flat_name in _flatten_preceded.items():
            if _fl_lin_name in pruned_layers:
                continue
            try:
                _fl_mod = merged_model.get_submodule(_fl_lin_name)
            except AttributeError:
                continue

            _fl_cands = self._get_zero_in_indices(_fl_mod)
            if not _fl_cands:
                continue

            _fl_n_total = _fl_mod.in_features
            _fl_keep = sorted(set(range(_fl_n_total)) - set(_fl_cands))
            if not _fl_keep:
                continue   # degenerate: every input feature gated off

            # Rebuild the Linear keeping only surviving input columns
            _fl_new = nn.Linear(len(_fl_keep), _fl_mod.out_features,
                                bias=_fl_mod.bias is not None).to(self.device)
            _fl_keep_t = torch.tensor(_fl_keep, dtype=torch.long)
            _fl_new.weight.data = _fl_mod.weight.data[:, _fl_keep_t]
            if _fl_mod.bias is not None:
                _fl_new.bias.data = _fl_mod.bias.data.clone()

            # Replace the Linear in the model
            _fl_parts = _fl_lin_name.rsplit('.', 1)
            if len(_fl_parts) == 2:
                setattr(merged_model.get_submodule(_fl_parts[0]), _fl_parts[1], _fl_new)
            else:
                setattr(merged_model, _fl_lin_name, _fl_new)

            # Replace the preceding Flatten with _FlattenSelect
            _fl_selector = _FlattenSelect(_fl_keep).to(self.device)
            _fl_flat_parts = _fl_flat_name.rsplit('.', 1)
            if len(_fl_flat_parts) == 2:
                setattr(merged_model.get_submodule(_fl_flat_parts[0]),
                        _fl_flat_parts[1], _fl_selector)
            else:
                setattr(merged_model, _fl_flat_name, _fl_selector)

            pruned_layers.add(_fl_lin_name)

            # Print indices summary (up to 20 shown, remainder collapsed)
            _fl_n_pruned = len(_fl_cands)
            _fl_n_show   = min(_fl_n_pruned, 20)
            _fl_more     = (f" ... +{_fl_n_pruned - _fl_n_show} more"
                            if _fl_n_pruned > _fl_n_show else "")
            print(f"Pruning layer: {_fl_lin_name}, type: Linear (input features), "
                  f"{_fl_n_pruned}/{_fl_n_total} features removed, {len(_fl_keep)} kept")
            print(f"  Pruned input feature indices [{_fl_n_pruned} total]: "
                  f"{_fl_cands[:_fl_n_show]}{_fl_more}")

        # Post-prune cleanup: replace fully-pruned layers with 0-channel
        # placeholder modules so forward methods can detect them.
        for name in fully_pruned_layers:
            module = merged_model.get_submodule(name)
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                cls = nn.Conv2d if isinstance(module, nn.Conv2d) else nn.Conv3d
                new_conv = cls(
                    in_channels=module.in_channels,
                    out_channels=0,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=1,
                    bias=module.bias is not None,
                ).to(self.device)
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent = merged_model.get_submodule(parts[0])
                    setattr(parent, parts[1], new_conv)
                else:
                    setattr(merged_model, name, new_conv)
                    
            elif isinstance(module, nn.Linear):
                new_linear = nn.Linear(
                    in_features=0,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                ).to(self.device)
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent = merged_model.get_submodule(parts[0])
                    setattr(parent, parts[1], new_linear)
                else:
                    setattr(merged_model, name, new_linear)
            
            # Also replace the corresponding BatchNorm layer (if any)
            bn_candidates = []
            bn_candidates.append(name.replace('conv', 'bn'))
            parts = name.rsplit('.', 1)
            if len(parts) == 2 and parts[1].isdigit():
                bn_candidates.append(f"{parts[0]}.{int(parts[1]) + 1}")
            seen = set()
            bn_candidates = [c for c in bn_candidates if c != name and c not in seen and not seen.add(c)]
            
            for bn_name in bn_candidates:
                try:
                    bn_module = merged_model.get_submodule(bn_name)
                    if isinstance(bn_module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                        new_bn = type(bn_module)(0).to(self.device)
                        bn_parts = bn_name.rsplit('.', 1)
                        if len(bn_parts) == 2:
                            parent = merged_model.get_submodule(bn_parts[0])
                            setattr(parent, bn_parts[1], new_bn)
                        else:
                            setattr(merged_model, bn_name, new_bn)
                        break
                except Exception:
                    continue
        
        if fully_pruned_layers:
            print(f"\nFully pruned {len(fully_pruned_layers)} layer(s): {sorted(fully_pruned_layers)}")
            print("These layers are bypassed via residual/shortcut connections at inference time.\n")
        
        return merged_model
    
    @staticmethod
    def log_qz_loga(pruner, epoch, batch_idx=None, writer=None, external_nums={}):
        step = epoch if batch_idx is None else epoch * 10000 + batch_idx
        
        for i, (name, qz_loga) in enumerate(pruner.layer_gates.items()):
            tag = f"qz_loga/{i}-{name}"
            arr = qz_loga.detach().cpu()
            writer.add_histogram(tag, arr, step)
        
        # log the prune ratio
        masks = pruner.get_masks()
        for i, (name, mask) in enumerate(masks.items()):
            tag = f"active_gates/{i}-{name}"
            active_gates = torch.sum(mask.flatten() > 1e-6)
            writer.add_scalar(tag, active_gates, step)
        
        # log the external numbers
        for key, value in external_nums.items():
            writer.add_scalar(f"external/{key}", value, step)
    
    def update_temperature(self, temperature: float):
        """Update the temperature parameter for the concrete distribution."""
        self.temperature = temperature
    
    def get_pruned_model(self, example_inputs=None) -> nn.Module:
        """
        Get a copy of the model with pruning applied.
        
        Returns:
            Pruned model
        """
        pruned_model = self.prune_model(example_inputs)
        return pruned_model
