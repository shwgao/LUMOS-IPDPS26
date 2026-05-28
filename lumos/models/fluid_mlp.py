import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_classes=3):
        super(MLP, self).__init__()
        
        layer_dims = [784, 300, 100]
        input_dim = 15
        
        layers = []
        for i, dimh in enumerate(layer_dims):
            inp_dim = input_dim if i == 0 else layer_dims[i - 1]
            layers += [nn.Linear(inp_dim, dimh)]
            layers += [nn.ReLU()]
        layers.append(nn.Linear(layer_dims[-1], num_classes))
        self.output = nn.Sequential(*layers)

    def forward(self, x):
        return self.output(x)

def fluid_mlp(num_classes=3, **kwargs):
    return MLP(num_classes=num_classes)
