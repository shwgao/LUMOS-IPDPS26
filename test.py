import argparse
import os
import torch
import torch.nn as nn
import logging
import torch_pruning as tp
from lumos.pruner import L0Pruner
from lumos.models import get_model
from lumos.data import get_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="LUMOS: Test and Prune Model")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="fluid",
        choices=["fluid", "cosmoflow", "molhiv", "medical", "cifar10", "mnist"],
        help="Dataset",
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
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to sparse model checkpoint (state_dict)")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save pruned model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    # Pruner parameters needed for reconstruction
    parser.add_argument("--droprate-init", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0/3.0)
    
    return parser.parse_args()

def test(model, testloader, device, dataset_name):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.MSELoss() if dataset_name in ['fluid', 'cosmoflow'] else nn.CrossEntropyLoss()
    is_regression = dataset_name in ['fluid', 'cosmoflow']
    
    with torch.no_grad():
        for batch in testloader:
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
            else:
                inputs = batch.to(device)
                targets = batch.y.to(device)
                
            outputs = model(inputs)
            if dataset_name == 'molhiv' and targets.dim() > 1 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
            
            if is_regression:
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                total += 1
            else:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
    if is_regression:
        return total_loss / total # Avg Loss
    else:
        return 100. * correct / total # Accuracy

def main(args):
    # Get Dataset
    num_classes, train_dst, val_dst, input_size = get_dataset(args.dataset, data_root=args.data_root, batch_size=args.batch_size)
    if args.num_classes is not None:
        num_classes = args.num_classes
    args.num_classes = num_classes
    
    if args.dataset in ['molhiv']:
        from torch_geometric.loader import DataLoader as PyGDataLoader
        testloader = PyGDataLoader(val_dst, batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        testloader = torch.utils.data.DataLoader(val_dst, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
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

    # Get Model
    model = get_model(args.model, num_classes=args.num_classes)
    model = model.to(args.device)
    
    logger.info("Initializing L0 Pruner with sparse training enabled to load gates...")
    pruner = L0Pruner(
        model,
        droprate_init=args.droprate_init,
        temperature=args.temperature,
        device=args.device,
        sparse_training=True  # Important: Must initialize gates to load them
    )
    
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)

    # Restore normal (non-gated) forwards WITHOUT touching the weights.
    # The fine-tuned checkpoint has weights calibrated for scale=1.0 (no gating).
    # merge_mask() would scale active channel weights by their gate value (~0.5 for
    # qz_loga≈0), halving activations and collapsing accuracy.  We only need to
    # remove the custom L0 forward hooks; the zeroed weight channels are already the
    # structural pruning signal used by get_pruned_model().
    for _, module in model.named_modules():
        if hasattr(module, '_original_forward'):
            module.forward = module._original_forward

    # Test sparse model
    score = test(model, testloader, args.device, args.dataset)
    metric = "Loss" if args.dataset in ['fluid', 'cosmoflow'] else "Accuracy"
    logger.info(f"{metric} of sparse model (with gates): {score:.4f}")
    
    # Perform structural pruning
    logger.info("Performing structural pruning...")
    
    # Example inputs
    if args.dataset in ['molhiv']:
        example_inputs = next(iter(testloader))
        example_inputs = example_inputs.to(args.device)
    else:
        if input_size is None:
             # Fallback for some datasets if input_size not defined
             example_inputs = torch.randn(1, 3, 224, 224).to(args.device) 
        else:
             example_inputs = torch.randn(input_size).to(args.device)
    
    # Get stats before pruning
    # Note: torch_pruning might not support GNN/PyG inputs directly in count_ops_and_params easily
    # We try, catch if fails
    try:
        ori_ops, ori_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    except Exception as e:
        logger.warning(f"Could not count ops/params: {e}")
        ori_ops, ori_params = 0, 0
    
    # Prune
    # Pruning graph models might be tricky with dependency graph if PyG modules are custom
    # But GINConv usually uses standard Linear layers which we annotated.
    try:
        pruned_model = pruner.get_pruned_model(example_inputs=example_inputs)
    except Exception as e:
        logger.error(f"Pruning failed: {e}")
        return

    # Get stats after pruning
    try:
        pruned_ops, pruned_params = tp.utils.count_ops_and_params(pruned_model, example_inputs=example_inputs)
        logger.info(f"Pruning Results:")
        logger.info(f"Params: {ori_params/1e6:.2f}M -> {pruned_params/1e6:.2f}M")
        logger.info(f"FLOPs: {ori_ops/1e6:.2f}M -> {pruned_ops/1e6:.2f}M")
    except:
        pass
    
    # Test pruned model
    score_pruned = test(pruned_model, testloader, args.device, args.dataset)
    logger.info(f"{metric} of pruned model: {score_pruned:.4f}")
    
    # Save pruned model
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "pruned_model.pth")
    torch.save(pruned_model, save_path)
    logger.info(f"Saved pruned model to {save_path}")

if __name__ == "__main__":
    args = get_args()
    main(args)
