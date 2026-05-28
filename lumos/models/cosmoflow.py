import torch
import torch.nn as nn
import torch.nn.functional as nnf

# Settings from engine/models/cosmoflow/settings.json
COSMOFLOW_SETTINGS = {
    "n_conv_layers": 4,
    "n_conv_filters": 32,
    "conv_kernel": 2,
    "dropout_rate": 0.5,
    "device": "cuda",
    "beta_ema": 0.999,
    "N": 1,
    "flat_shape": [1, 128, 128, 128]
}

class Conv3DActMP(nn.Module):
    def __init__(
            self,
            conv_kernel: int,
            conv_channel_in: int,
            conv_channel_out: int,
            isbn: bool = False,
    ):
        super().__init__()

        self.conv = nn.Conv3d(conv_channel_in, conv_channel_out, kernel_size=conv_kernel, stride=1, padding=1, bias=True)
        self.isbn = isbn
        self.bn = nn.BatchNorm3d(conv_channel_out)
        self.act = nn.LeakyReLU(negative_slope=0.3)
        self.mp = nn.MaxPool3d(kernel_size=2, stride=2)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.isbn:
            return self.mp(self.act(self.bn(self.conv(x))))
        else:
            return self.mp(self.act(self.conv(x)))


class CosmoFlow(nn.Module):
    def __init__(self, inference=False, using_reg=False):
        super().__init__()
        settings = COSMOFLOW_SETTINGS

        n_conv_layers = settings["n_conv_layers"]
        n_conv_filters = settings["n_conv_filters"]
        conv_kernel = settings["conv_kernel"]
        dropout_rate = settings["dropout_rate"]
        
        self.conv_seq = nn.ModuleList()
        input_channel_size = 4

        for i in range(n_conv_layers):
            output_channel_size = n_conv_filters * (1 << i)
            self.conv_seq.append(Conv3DActMP(conv_kernel, input_channel_size, output_channel_size))
            input_channel_size = output_channel_size

        # Calculate flattened size: 128 -> 64 -> 32 -> 16 -> 8
        # 128 // (2^4) = 8
        flatten_inputs = 128 // (2 ** n_conv_layers)
        flatten_inputs = (flatten_inputs ** 3) * input_channel_size
        
        self.dense1 = nn.Linear(flatten_inputs, 128, bias=True)
        self.dense2 = nn.Linear(128, 64, bias=True)
        self.output = nn.Linear(64, 4, bias=True)

        self.dropout_rate = dropout_rate
        if self.dropout_rate is not None:
            self.dr1 = nn.Dropout(p=self.dropout_rate)
            self.dr2 = nn.Dropout(p=self.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, 4, 128, 128, 128]
        for i, conv_layer in enumerate(self.conv_seq):
            x = conv_layer(x)

        # Flatten
        # Original code: x = x.permute(0, 2, 3, 4, 1).flatten(1)
        # This permutation puts channels last before flattening.
        # Check dense1 input size. 
        # If input is [B, C, D, H, W], permute to [B, D, H, W, C]
        # Then flatten to [B, D*H*W*C]
        x = x.permute(0, 2, 3, 4, 1).flatten(1)

        x = nnf.leaky_relu(self.dense1(x), negative_slope=0.3)
        if self.dropout_rate is not None:
            x = self.dr1(x)

        x = nnf.leaky_relu(self.dense2(x), negative_slope=0.3)
        if self.dropout_rate is not None:
            x = self.dr2(x)

        # Output scaling
        x = torch.sigmoid(self.output(x)) * 1.2
        return x

def cosmoflow_model(**kwargs):
    valid_args = ['inference', 'using_reg']
    cf_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
    return CosmoFlow(**cf_kwargs)
