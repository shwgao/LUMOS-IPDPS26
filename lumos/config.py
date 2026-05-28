from dataclasses import dataclass
from typing import Optional

@dataclass
class L0Config:
    """Configuration for L0 Pruner"""
    droprate_init: float = 0.5
    temperature: float = 2.0 / 3.0
    weight_decay: float = 5e-4
    lamba: float = 1.0  # L0 penalty strength
    lamda: float = 1.0  # Loss weight
    local_rep: bool = False
    reg_warmup: int = 0  # Epochs to wait before applying regularization
    
    # Pruning parameters
    iterative_steps: int = 400
    pruning_ratio: float = 0.5
    
    # Other
    seed: int = 42

@dataclass
class TrainConfig:
    """Configuration for training"""
    dataset: str = "resnet"
    data_root: str = "./data"
    model: str = "resnet18"
    batch_size: int = 64
    epochs: int = 100
    lr: float = 0.01
    lr_decay_milestones: list = (60, 80)
    lr_decay_gamma: float = 0.1
    device: str = "cuda"
    output_dir: str = "./run"
    pretrained: Optional[str] = None
    resume: Optional[str] = None

