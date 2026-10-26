"""Tokenized KAN block for bottleneck integration.

Pipeline:
    feature map [B, C, H, W]
        -> flatten to tokens [B, HW, C]
        -> LayerNorm
        -> stacked KAN layers with residual
        -> LayerNorm
        -> reshape back to [B, C, H, W]
        -> depthwise 3x3 convolution residual for local context

The KAN layers provide global, non-linear token mixing through learnable
spline activations, while the trailing depthwise convolution restores the
short-range spatial coherence required for boundary-accurate echocardiographic
segmentation.

The `use_tanh` flag (default True) is forwarded to each KANLinear and
controls the tanh-bounded input transformation described in Section 3.2.3
of the manuscript. Setting use_tanh=False reverts to the unmodified
B-spline KANLinear formulation (Ablation A).
"""
import torch
import torch.nn as nn

from .kan_layer import KANLinear


class TokenizedKANBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_kan_layers: int = 2,
        grid_size: int = 5,
        spline_order: int = 3,
        mlp_ratio: float = 1.0,
        use_tanh: bool = True,     # NEW: forwarded to every KANLinear
    ):
        super().__init__()
        hidden = int(channels * mlp_ratio)

        self.norm_in = nn.LayerNorm(channels)
        self.kan_layers = nn.ModuleList()
        in_dim = channels
        for i in range(num_kan_layers):
            out_dim = hidden if i < num_kan_layers - 1 else channels
            self.kan_layers.append(
                KANLinear(
                    in_features=in_dim,
                    out_features=out_dim,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    use_tanh=use_tanh,     # NEW: forward
                )
            )
            in_dim = out_dim

        self.norm_out = nn.LayerNorm(channels)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        tokens = x.flatten(2).transpose(1, 2)
        residual = tokens
        tokens = self.norm_in(tokens)

        for layer in self.kan_layers:
            tokens = layer(tokens)

        tokens = self.norm_out(tokens + residual)

        x_out = tokens.transpose(1, 2).reshape(B, C, H, W)
        x_out = x_out + self.dwconv(x_out)
        return x_out