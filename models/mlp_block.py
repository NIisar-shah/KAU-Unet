"""MLP token block — parameter-comparable ablation baseline against the
Tokenized KAN Block (Ablation B of the manuscript).

Mirrors the structure of TokenizedKANBlock exactly (LayerNorm in, stacked
inner transformations, LayerNorm-with-residual, depth-wise convolution
residual) so that the only architectural difference is the inner
transformation: KANLinear layers in the KAN block vs. two-layer MLPs in
this block.

Parameter accounting at the manuscript's default configuration
(channels=256, num_layers=2, grid_size=5, spline_order=3):

    TokenizedKANBlock (two KANLinear layers):
        2 * (256*256 + 256*256*8 + 256*256) = 1,310,720 parameters
        (base_weight + spline_weight + spline_scaler per KAN layer)

    MLPTokenBlock (two MLP layers, hidden_mult=5):
        2 * (256*1280 + 1280 + 1280*256 + 256) ~= 1,313,792 parameters

The two blocks match to within 0.3% on parameter count, so any
performance difference between KAN-UNet and the KAN-replaced-by-MLP
variant cannot plausibly be attributed to added parameter capacity.
"""
import torch
import torch.nn as nn


class MLPTokenBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int = 2,
        hidden_mult: int = 5,
        mlp_ratio: float = 1.0,
        activation: str = "gelu",
    ):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        ffn_hidden = channels * hidden_mult

        act_layer = {"gelu": nn.GELU, "relu": nn.ReLU,
                     "silu": nn.SiLU}[activation]

        self.norm_in = nn.LayerNorm(channels)
        self.mlp_layers = nn.ModuleList()
        in_dim = channels
        for i in range(num_layers):
            out_dim = hidden if i < num_layers - 1 else channels
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(in_dim, ffn_hidden),
                act_layer(),
                nn.Linear(ffn_hidden, out_dim),
            ))
            in_dim = out_dim

        self.norm_out = nn.LayerNorm(channels)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3,
                                padding=1, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        tokens = x.flatten(2).transpose(1, 2)
        residual = tokens
        tokens = self.norm_in(tokens)

        for layer in self.mlp_layers:
            tokens = layer(tokens)

        tokens = self.norm_out(tokens + residual)

        x_out = tokens.transpose(1, 2).reshape(B, C, H, W)
        x_out = x_out + self.dwconv(x_out)
        return x_out