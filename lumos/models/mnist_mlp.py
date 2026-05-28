import torch.nn as nn


class MnistMLP(nn.Module):
    """Simple MLP for MNIST.

    Input: (N, 1, 28, 28)  → flattened to (N, 784) inside forward().
    Architecture: 784 → 512 → 256 → 128 → num_classes
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


def mnist_mlp(num_classes: int = 10, **kwargs):
    return MnistMLP(num_classes=num_classes)
