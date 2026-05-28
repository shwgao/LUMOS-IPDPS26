from .medical_vit import medical_vit, medical_vit_base, medical_vit_small, medical_vit_large
from .fluid_mlp import fluid_mlp
from .cosmoflow import cosmoflow_model
from .gnn import gnn_model
from .resnet_cifar import resnet18, resnet18_2x
from .mnist_mlp import mnist_mlp

MODEL_DICT = {
    # Medical
    'medical_vit': medical_vit,
    'medical_vit_small': medical_vit_small,
    'medical_vit_base': medical_vit_base,
    'medical_vit_large': medical_vit_large,
    'medical': medical_vit, # alias

    # Fluid
    'fluid': fluid_mlp,

    # CosmoFlow
    'cosmoflow': cosmoflow_model,

    # GNN (Molhiv)
    'molhiv': gnn_model,

    # CIFAR
    'resnet18': resnet18,
    'resnet18_2x': resnet18_2x,

    # MNIST
    'mnist_mlp': mnist_mlp,
}

def get_model(name: str, num_classes=None, **kwargs):
    if name not in MODEL_DICT:
        raise ValueError(f"Model {name} not found. Available models: {list(MODEL_DICT.keys())}")
    
    model_fn = MODEL_DICT[name]
    
    # Filter kwargs to only pass what the model expects/needs if necessary
    # For now, we pass all kwargs, but some models might not accept them.
    # We can be more selective if needed.
    
    return model_fn(num_classes=num_classes, **kwargs)
