# LUMOS

LUMOS is a lightweight and effective L0 regularization-based pruning framework for neural networks.

## Features

- **L0 Regularization**: Automatically learns which weights to keep and which to prune. As well as can be used for the input features.
- **Structural Pruning**: Supports structural pruning for Linear, Conv2d, and Conv3d layers.
- **Sparsity Learning**: Trains the model to be sparse from the start or fine-tunes a pretrained model.
- **Modular Design**: Separated models, data loaders, and pruner logic.
- **Early-Stop by Target Prune Ratio**: Stops L0 training as soon as a target sparsity is reached, then fine-tunes — preventing accuracy collapse caused by over-pruning before recovery.

## Example Models

- **Fluid**: MLP for fluid simulation data.
- **CosmoFlow**: 3D CNN for cosmological parameter estimation.
- **MolHIV**: GNN for molecular property prediction (OGB).
- **Medical**: Vision Transformer (MedicalViT) for medical image classification (Brain Tumor, COVID19, Skin Cancer).
- **MNIST**: MLP (784→512→256→128→10) for handwritten digit classification.
- **CIFAR10**: ResNet18 / 2× width ResNet18 (`--model auto`).

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install torch torchvision torch-pruning matplotlib seaborn torch-geometric ogb tfrecord-lite
   ```

## Usage

### 1. Training (Sparsity Learning)

To train a model with L0 regularization:

```bash
# Fluid (Regression)
python train.py --dataset fluid --model auto --lamba 1.0 --epochs 50

# CosmoFlow (Regression)
python train.py --dataset cosmoflow --model auto --lamba 1.0 --epochs 20

# MolHIV (Graph Classification)
python train.py --dataset molhiv --model auto --lamba 1.0 --epochs 50

# Medical (Image Classification)
python train.py --dataset medical --model auto --lamba 1.0 --epochs 50

# CIFAR10 (Image Classification)
python train.py --dataset cifar10 --model auto --lamba 1.0 --epochs 50
```

Key arguments:
- `--lamba`: Controls the strength of L0 regularization (higher = more sparsity).
- `--epochs`: Number of L0 training epochs (upper bound; may terminate early via `--target-prune-ratio`).
- `--finetune-epochs`: Number of fine-tuning epochs after L0 training completes.
- `--target-prune-ratio`: *(Recommended)* Stop L0 training early when the hard-gate prune ratio reaches this value, then immediately fine-tune. Prevents accuracy collapse. Set to `0.0` to disable (default).
- `--data-root`: Path to dataset directory (defaults to `./data` or cluster paths).

#### Demo MNIST recipe (≥40% sparsity, ≥95% accuracy)

The L0 pruner gates **input features** of each Linear layer, so the first layer
gates the raw 784-pixel inputs — meaning pruned pixels are never used by the
network regardless of their value.

```bash
python train.py \
  --dataset mnist --data-root ./data --model auto \
  --epochs 100 --batch-size 256 --lr 0.01 --optimizer adam \
  --lamba 0.002 --gate-lr-scale 1.0 \
  --droprate-init 0.5 --finetune-epochs 30 \
  --target-prune-ratio 0.40 \
  --output-dir ./output/mnist_run
```

**Result:**

| Metric | Target | Achieved |
|--------|--------|----------|
| Overall prune ratio | ≥ 0.40 | **0.4033** |
| Val Accuracy | ≥ 95% | **98.25%** |
| Pixel inputs pruned | — | **394 / 784 (50.3%)** |

- L0 training early-stopped at epoch 11 (prune_ratio = 0.4033).
- Fine-tuning (30 epochs): val acc stable at ~98.25%.
- `best_model_finetuned.pth` checkpoint: 394 of 784 input pixel columns are zeroed in `layers.0.weight` (first hidden layer). The network never reads those pixels.

#### Demo CIFAR-10 recipe (≥50% sparsity, ≥85% accuracy)

```bash
python train.py \
  --dataset cifar10 --data-root ./data --model auto \
  --epochs 200 --batch-size 128 --lr 0.1 --momentum 0.9 --optimizer sgd \
  --weight-decay 5e-4 --lamba 0.002 --gate-lr-scale 1.0 \
  --droprate-init 0.5 --finetune-epochs 100 \
  --target-prune-ratio 0.50 \
  --output-dir ./output/run_cifar10
```

**Result:**

| Metric | Target | Achieved |
|--------|--------|----------|
| Prune Ratio | ≥ 0.50 | **0.5049** |
| Val Accuracy | ≥ 85% | **95.28%** |

- L0 training early-stopped at epoch 79 (prune_ratio = 0.5049).
- Fine-tuning (100 epochs, cosine LR): val acc rose from 91.48% (epoch 0) → **95.28%** (best).

### 2. Pruning and Testing

After training, you can prune the model structurally and evaluate it:

```bash
# CIFAR-10
python test.py --dataset cifar10 --model auto \
  --checkpoint ./output/run_cifar10/best_model_finetuned.pth \
  --output-dir ./output/run_cifar10_pruned

# MNIST
python test.py --dataset mnist --model auto \
  --checkpoint ./output/mnist_run/best_model_finetuned.pth \
  --output-dir ./output/mnist_pruned
```

This will:
1. Load the trained sparse model (with gates).
2. Permanently prune the layers based on the learned gates.
3. Report FLOPs/Params reduction and performance (Accuracy or Loss).
4. Save the structurally pruned model.

## Project Structure

- `lumos/`
  - `pruner.py`: Core L0Pruner class.
  - `models/`: Model definitions (Fluid MLP, CosmoFlow, GNN, MedicalViT, ResNet).
  - `data/`: Dataset loading and preprocessing.
  - `config.py`: Configuration classes.
- `train.py`: Training script (supports `--target-prune-ratio` early-stop).
- `test.py`: Pruning and testing script.
