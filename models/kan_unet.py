"""KAN-augmented U-Net for echocardiographic segmentation.

The encoder and decoder follow a conventional convolutional U-shape, which
preserves the strong locality prior required for speckle-rich ultrasound
images. The deepest semantic representation is refined by a Tokenized KAN
Bottleneck (TKB), enabling expressive non-linear modelling of global
anatomical structure with parameter-efficient B-spline activations.

I/O contract is identical to the baseline:
    forward([B, in_channels, H, W]) -> [B, out_channels, H, W]

Three model variants are exported:
    KAN_UNet           -- the proposed architecture (default)
    KAN_UNet_NoTanh    -- Ablation A: KAN bottleneck without tanh bounding
    KAN_UNet_MLP       -- Ablation B: KAN bottleneck replaced with
                          a parameter-comparable MLP block
"""
import torch
import torch.nn as nn

from .baseline_unet import DoubleConv, Down, Up
from .kan_blocks import TokenizedKANBlock
from .mlp_block import MLPTokenBlock


class KAN_UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        kan_num_layers: int = 2,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        use_tanh: bool = True,
        bottleneck_type: str = "kan",
    ):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)

        # Conditional bottleneck construction.
        # The attribute name `bottleneck_kan` is kept for both variants
        # so that no changes to `forward()` are needed.
        if bottleneck_type == "kan":
            self.bottleneck_kan = TokenizedKANBlock(
                channels=c4,
                num_kan_layers=kan_num_layers,
                grid_size=kan_grid_size,
                spline_order=kan_spline_order,
                use_tanh=use_tanh,
            )
        elif bottleneck_type == "mlp":
            self.bottleneck_kan = MLPTokenBlock(
                channels=c4,
                num_layers=kan_num_layers,
                hidden_mult=5,
            )
        else:
            raise ValueError(
                f"Unknown bottleneck_type: {bottleneck_type!r}. "
                f"Expected one of ['kan', 'mlp'].")

        self.up1 = Up(c4, c3, c3)
        self.up2 = Up(c3, c2, c2)
        self.up3 = Up(c2, c1, c1)

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4 = self.bottleneck_kan(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.out_conv(x)


# ==================================================================
# Ablation variants (Section 4.X of the manuscript).
# Each is a thin subclass of KAN_UNet that hardcodes the ablation
# flag while declaring its own __init__ signature. The explicit
# signature is required so that models/__init__.py's _filter_params
# (which uses inspect.signature) correctly filters cfg.model.params.
# ==================================================================


class KAN_UNet_NoTanh(KAN_UNet):
    """Ablation A: KAN-UNet with the tanh-bounded input transformation
    disabled on the KANLinear spline path. Reverts to the unmodified
    B-spline formulation used in U-KAN (Li et al., AAAI 2025)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        kan_num_layers: int = 2,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            kan_num_layers=kan_num_layers,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order,
            use_tanh=False,
            bottleneck_type="kan",
        )


class KAN_UNet_MLP(KAN_UNet):
    """Ablation B: KAN bottleneck replaced with a parameter-comparable
    MLP block (Linear -> GELU -> Linear, two layers, hidden_mult=5).
    Tests whether the empirical improvement over the convolutional
    baseline is KAN-specific or merely capacity-related."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        kan_num_layers: int = 2,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            kan_num_layers=kan_num_layers,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order,
            bottleneck_type="mlp",
        )