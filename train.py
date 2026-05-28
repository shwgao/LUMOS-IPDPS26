import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import sys

from lumos.pruner import L0Pruner
from lumos.models import get_model
from lumos.data import get_dataset
from lumos.config import L0Config, TrainConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="LUMOS: L0 Regularization Pruning Training")
    
    # Dataset and Model
    parser.add_argument(
        "--dataset",
        type=str,
        default="fluid",
        choices=["fluid", "cosmoflow", "molhiv", "medical", "cifar10", "mnist"],
        help="Dataset to use",
    )
    parser.add_argument("--data-root", type=str, default="./data", help="Root directory for dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=[
            "auto",
            "fluid",
            "cosmoflow",
            "molhiv",
            "medical",
            "medical_vit",
            "medical_vit_small",
            "medical_vit_base",
            "medical_vit_large",
            "resnet18",
            "resnet18_2x",
            "mnist_mlp",
        ],
        help="Model architecture",
    )
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (optional override)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=200, help="Total epochs for training")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (default: 0.1 for SGD, use 0.001 for Adam)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="Optimizer (default: sgd, matching reference)")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    # L0 Pruner parameters
    parser.add_argument("--droprate-init", type=float, default=0.5, help="Initial drop rate for gates")
    parser.add_argument("--temperature", type=float, default=2.0/3.0, help="Temperature for concrete distribution")
    parser.add_argument("--lamba", type=float, default=0.001, help="L0 penalty per weight (default: 0.001)")
    parser.add_argument("--lamda", type=float, default=None, help="Regularization divisor (default: N = training set size)")
    parser.add_argument("--gate-lr-scale", type=float, default=0.1,
                        help="LR multiplier for qz_loga gate parameters relative to base lr. "
                             "Keeps gates from saturating immediately with large lr. (default: 0.1)")
    parser.add_argument("--gate-lr-warmup", type=int, default=0,
                        help="Number of warmup epochs during which gate_lr is near-zero. "
                             "Allows the model to learn good features before L0 pruning starts. "
                             "Useful for large/wide models that collapse under immediate L0 pressure. (default: 0)")
    parser.add_argument("--finetune-epochs", type=int, default=0, help="Additional fine-tuning epochs after L0 training (gates merged into weights)")
    parser.add_argument("--target-prune-ratio", type=float, default=0.0,
                        help="Stop L0 training early when prune_ratio >= this value, then immediately fine-tune. "
                             "0.0 means disabled (train for full --epochs). (default: 0.0)")
    
    # Resume/Pretrain
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model")

    return parser.parse_args()

def train(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get Dataset
    num_classes, train_dst, val_dst, input_size = get_dataset(args.dataset, data_root=args.data_root, batch_size=args.batch_size)
    if args.num_classes is not None:
        num_classes = args.num_classes
    args.num_classes = num_classes 
    
    # DataLoader handling for different dataset types
    if args.dataset in ['molhiv']:
        from torch_geometric.loader import DataLoader as PyGDataLoader
        trainloader = PyGDataLoader(train_dst, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testloader = PyGDataLoader(val_dst, batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        trainloader = torch.utils.data.DataLoader(train_dst, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(val_dst, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Get Model
    # Map dataset to default model if auto is used
    if args.model == 'auto':
        if args.dataset == 'medical':
            args.model = 'medical_vit'
        elif args.dataset == 'fluid':
            args.model = 'fluid'
        elif args.dataset == 'cosmoflow':
            args.model = 'cosmoflow'
        elif args.dataset == 'molhiv':
            args.model = 'molhiv'
        elif args.dataset == 'cifar10':
            args.model = 'resnet18_2x'
        elif args.dataset == 'mnist':
            args.model = 'mnist_mlp'

    model = get_model(args.model, num_classes=args.num_classes)
    
    if args.pretrained:
        logger.info(f"Loading pretrained model from {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(args.device)
    
    # Compute lamda default: use N (dataset size), matching the reference paper's -(1/N) scaling.
    if args.lamda is None:
        args.lamda = float(len(train_dst))
        logger.info(f"Using lamda = N = {args.lamda:.0f} (training set size)")

    # Initialize Pruner
    # CRITICAL: Reference scales weight_decay by N (dataset size) inside the model:
    #   self.weight_decay = N * weight_decay  (see L0WideResNet.__init__)
    # This makes the L2 term in logpw_col = -(0.5 * N*wd * w^2 + lamba) dominant.
    # With w^2 ~ 0.01 and N*wd=25: L2 term = 0.125 >> lamba=0.001.
    # Without this scaling (wd=5e-4), L2 term is 2.5e-6, essentially zero,
    # so the L0 gradient through q0 never gets large enough to push gates closed.
    N = float(len(train_dst))
    pruner_weight_decay = N * args.weight_decay   # e.g. 50000 * 5e-4 = 25.0
    logger.info(f"Pruner weight_decay = N * wd = {N:.0f} * {args.weight_decay} = {pruner_weight_decay}")
    logger.info("Initializing L0 Pruner...")
    pruner = L0Pruner(
        model,
        droprate_init=args.droprate_init,
        temperature=args.temperature,
        weight_decay=pruner_weight_decay,
        lamba=args.lamba,
        lamda=args.lamda,
        device=args.device,
        sparse_training=True, # Enable L0 gates
        writer_dir=os.path.join(args.output_dir, "tensorboard")
    )
    
    # Optimizer and scheduler — default SGD+momentum matches reference (train_wide_resnet.py)
    # Gate parameters (qz_loga) use a lower lr to prevent immediate saturation to ±4.6.
    # With SGD lr=0.1, task-loss gradients push qz_loga to the upper bound in just a few epochs,
    # after which hardtanh and sigmoid derivatives vanish and the L0 gradient can no longer
    # move gates. A 10x lower lr for gates allows the L0 signal to compete.
    gate_params  = [p for n, p in model.named_parameters() if 'qz_loga' in n]
    weight_params = [p for n, p in model.named_parameters() if 'qz_loga' not in n]
    gate_lr = args.lr * args.gate_lr_scale

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            [{'params': weight_params, 'lr': args.lr},
             {'params': gate_params,   'lr': gate_lr}],
            momentum=args.momentum, weight_decay=0, nesterov=True
        )
        # Drop LR at 60%, 80%, 90% of total epochs, ×0.2 each (reference uses fixed [60,120,160])
        epoch_drop = [int(0.6 * args.epochs), int(0.8 * args.epochs), int(0.9 * args.epochs)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=epoch_drop, gamma=0.2)
        logger.info(f"Optimizer: SGD (weight lr={args.lr}, gate lr={gate_lr:.4f}, "
                    f"momentum={args.momentum}), MultiStepLR milestones={epoch_drop}")
    else:
        optimizer = optim.Adam(
            [{'params': weight_params, 'lr': args.lr},
             {'params': gate_params,   'lr': gate_lr}]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        logger.info(f"Optimizer: Adam (weight lr={args.lr}, gate lr={gate_lr:.4f}), CosineAnnealingLR")
    
    # Criterion selection
    if args.dataset in ['fluid', 'cosmoflow']:
        criterion = nn.MSELoss()
        is_regression = True
    elif args.dataset == 'molhiv':
        # MolHIV is binary classification but uses BCEWithLogits usually with ROC-AUC eval
        # However, model outputs [batch, 1] or [batch, 2]? 
        # If model outputs 2 classes, CrossEntropy. If 1, BCE.
        # GNN model defaults to Linear(emb_dim, num_class). If num_class=2, output is [B, 2].
        # PyG MolHIV target is usually [B, 1].
        # We need to check data.y shape in loop.
        criterion = nn.CrossEntropyLoss() # Default, adjust if needed
        is_regression = False
    else:
        criterion = nn.CrossEntropyLoss()
        is_regression = False
    
    best_acc = -float('inf') if not is_regression else float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(trainloader):
            # Handle PyG batch vs Tuple batch
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch
                inputs, targets = inputs.to(args.device), targets.to(args.device)
            else:
                # PyG Batch object
                inputs = batch.to(args.device)
                targets = batch.y.to(args.device)
                
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Adjust shapes for loss
            if args.dataset == 'molhiv':
                # Target might be [B, 1], output [B, 2]
                if targets.dim() > 1 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)
                
            loss = criterion(outputs, targets)
            
            # L0 Regularization applied every epoch (no warmup, matching reference)
            l0_loss = pruner.regularize()
            total_train_loss = loss + l0_loss
            
            total_train_loss.backward()
            optimizer.step()
            pruner.constrain_parameters()  # clamp qz_loga to prevent extreme values
            
            total_loss += loss.item()
            
            if not is_regression:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            else:
                total += targets.size(0)
            
            if batch_idx % 100 == 0:
                acc_str = f"| Acc: {100.*correct/total:.2f}%" if not is_regression else ""
                logger.info(f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f} | L0: {l0_loss.item():.4f} {acc_str}")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for batch in testloader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch
                    inputs, targets = inputs.to(args.device), targets.to(args.device)
                else:
                    inputs = batch.to(args.device)
                    targets = batch.y.to(args.device)
                    
                outputs = model(inputs)
                if args.dataset == 'molhiv' and targets.dim() > 1 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)
                    
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                if not is_regression:
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                else:
                    val_total += 1 # Count batches or samples? Loss is mean.
        
        prune_ratio = pruner.get_prune_ratio()
        
        if not is_regression:
            val_acc = 100. * val_correct / val_total
            logger.info(f"Epoch {epoch} Completed. Val Acc: {val_acc:.2f}%. Prune Ratio: {prune_ratio:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_sparse.pth"))
        else:
            avg_val_loss = val_loss / len(testloader)
            logger.info(f"Epoch {epoch} Completed. Val Loss: {avg_val_loss:.4f}. Prune Ratio: {prune_ratio:.4f}")
            
            if avg_val_loss < best_acc: # best_acc is best_loss here
                best_acc = avg_val_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_sparse.pth"))
        
        scheduler.step()
        # Gate LR management: pin gate_lr constant regardless of scheduler decay.
        # Both SGD+MultiStepLR and Adam+CosineAnnealingLR decay param_groups[1]['lr']; we
        # override it so the L0 pruning pressure is sustained throughout training.
        # During warmup (SGD only), keep gates near-frozen so the model first learns
        # good representations before pruning begins.
        if epoch < args.gate_lr_warmup:
            optimizer.param_groups[1]['lr'] = gate_lr * 0.001  # near-frozen (warmup)
        else:
            optimizer.param_groups[1]['lr'] = gate_lr  # constant target (no decay)

        # Early-stop L0 training when target prune ratio is reached
        if args.target_prune_ratio > 0.0 and prune_ratio >= args.target_prune_ratio:
            logger.info(f"Target prune ratio {args.target_prune_ratio:.4f} reached at epoch {epoch} "
                        f"(prune_ratio={prune_ratio:.4f}). Stopping L0 training early.")
            break

    logger.info(f"Training finished.")
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model_sparse.pth"))

    # Fine-tuning phase: merge gates into weights, then retrain with clean forward (no gating)
    if args.finetune_epochs > 0:
        logger.info(f"Merging L0 mask into weights before fine-tuning...")
        # merge_mask() scales weights by deterministic gate values and restores original
        # forward methods, so both model.train() and model.eval() use the same computation.
        # This eliminates the stochastic-train / deterministic-eval mismatch that causes
        # val accuracy oscillation.
        pruner.merge_mask()

        # Record which output channels are zeroed so we can re-zero them after each
        # optimizer step (prevents pruned channels from regrowing during fine-tuning).
        keep_masks = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                per_ch = module.weight.data.view(module.weight.size(0), -1).abs().max(dim=1)[0]
                keep_masks[name] = (per_ch >= 1e-6)
            elif isinstance(module, nn.Linear):
                per_feat = module.weight.data.abs().max(dim=0)[0]
                keep_masks[name] = (per_feat >= 1e-6)

        ft_params = [p for n, p in model.named_parameters() if 'qz_loga' not in n]
        if args.optimizer == 'sgd':
            ft_optimizer = optim.SGD(ft_params, lr=args.lr * 0.1,
                                     momentum=args.momentum, weight_decay=0)
            ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(ft_optimizer, T_max=args.finetune_epochs)
        else:
            ft_optimizer = optim.Adam(ft_params, lr=args.lr * 0.1)
            ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(ft_optimizer, T_max=args.finetune_epochs)
        best_ft_acc = -float('inf') if not is_regression else float('inf')

        for ft_epoch in range(args.finetune_epochs):
            model.train()
            ft_correct = 0
            ft_total = 0
            for batch in trainloader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch
                    inputs, targets = inputs.to(args.device), targets.to(args.device)
                else:
                    inputs = batch.to(args.device)
                    targets = batch.y.to(args.device)
                ft_optimizer.zero_grad()
                outputs = model(inputs)
                if args.dataset == 'molhiv' and targets.dim() > 1 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)
                loss = criterion(outputs, targets)
                loss.backward()
                ft_optimizer.step()
                # Re-zero pruned channels to prevent them from regrowing
                with torch.no_grad():
                    for name, module in model.named_modules():
                        if name in keep_masks:
                            if isinstance(module, nn.Conv2d):
                                module.weight.data[~keep_masks[name]] = 0.0
                            elif isinstance(module, nn.Linear):
                                module.weight.data[:, ~keep_masks[name]] = 0.0
                if not is_regression:
                    _, predicted = outputs.max(1)
                    ft_total += targets.size(0)
                    ft_correct += predicted.eq(targets).sum().item()

            ft_scheduler.step()

            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            with torch.no_grad():
                for batch in testloader:
                    if isinstance(batch, (tuple, list)):
                        inputs, targets = batch
                        inputs, targets = inputs.to(args.device), targets.to(args.device)
                    else:
                        inputs = batch.to(args.device)
                        targets = batch.y.to(args.device)
                    outputs = model(inputs)
                    if args.dataset == 'molhiv' and targets.dim() > 1 and targets.shape[1] == 1:
                        targets = targets.squeeze(1)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    if not is_regression:
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()

            prune_ratio = pruner.get_prune_ratio()
            if not is_regression:
                val_acc = 100. * val_correct / val_total
                train_acc_str = f" | Train Acc: {100.*ft_correct/ft_total:.2f}%" if ft_total > 0 else ""
                logger.info(f"Finetune Epoch {ft_epoch} | Val Acc: {val_acc:.2f}%{train_acc_str} | Prune Ratio: {prune_ratio:.4f}")
                if val_acc > best_ft_acc:
                    best_ft_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_finetuned.pth"))
            else:
                avg_val_loss = val_loss / len(testloader)
                logger.info(f"Finetune Epoch {ft_epoch} | Val Loss: {avg_val_loss:.4f} | Prune Ratio: {prune_ratio:.4f}")
                if avg_val_loss < best_ft_acc:
                    best_ft_acc = avg_val_loss
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_finetuned.pth"))

        torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model_finetuned.pth"))
        logger.info(f"Fine-tuning finished. Best val {'acc' if not is_regression else 'loss'}: {best_ft_acc:.4f}")

if __name__ == "__main__":
    args = get_args()
    train(args)
