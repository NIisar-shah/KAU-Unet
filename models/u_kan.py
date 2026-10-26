"""
u_kan.py
========

Reference implementation of U-KAN (Li et al., AAAI 2025) for controlled
comparison against our KAN-bottleneck variants on CAMUS.

Architectural specification, following the U-KAN paper:
    * U-shaped encoder-decoder backbone.
    * Convolutional stages at higher spatial resolutions.
    * Tokenized KAN (TokKAN) blocks at deeper stages: flatten spatial
      dimensions to tokens, apply LayerNorm + KAN token-channel mixing +
      depth-wise convolution for local inductive bias, add residual.
    * Skip connections between matching encoder-decoder stages.
    * 1x1 conv output head to num_classes logits.

This is a reference implementation reproducing the published architectural
specification, not a port of the official code. It reuses our pipeline's
existing KANLinear so that optimizer, loss, augmentation, and training
protocol are identical to the other variants under comparison. The
``use_tanh`` flag (default True) threads our tanh-support correction into
every KAN layer; set it False to recover the unmodified B-spline formulation
of the original U-KAN. This flag is what the tanh-generality experiment
toggles.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# The pipeline's KANLinear lives in either kan_layer.py or kan_blocks.py.
try:
    from .kan_layer import KANLinear
except ImportError:
    from .kan_blocks import KANLinear


# =============================================================================
# CORE BLOCKS
# =============================================================================

class ConvBlock(nn.Module):
    """Standard double-conv block: Conv-BN-ReLU x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TokKANBlock(nn.Module):
    """Tokenized KAN block: LayerNorm -> KAN token-channel mixing -> residual;
    followed by depth-wise 3x3 conv + residual for local spatial inductive bias.

    Input and output tensors are (B, C, H, W); tokens are (B, H*W, C).
    """

    def __init__(self, dim: int, kan_grid_size: int = 5,
                 kan_spline_order: int = 3, use_tanh: bool = True):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.kan = KANLinear(
            dim, dim,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            use_tanh=use_tanh,          # <-- thread the tanh-support flag
        )
        self.dw_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten spatial dims to tokens (B, HW, C)
        tokens = x.flatten(2).transpose(1, 2)

        # KAN token-channel mixing with LayerNorm and residual
        h = self.norm(tokens)                      # (B, HW, C)
        h = h.reshape(-1, C)                       # (B*HW, C) for KANLinear
        h = self.kan(h)                            # (B*HW, C)
        h = h.reshape(B, H * W, C)                 # (B, HW, C)
        tokens = tokens + h                        # residual on tokens

        # Reshape back to spatial (B, C, H, W)
        x = tokens.transpose(1, 2).reshape(B, C, H, W)

        # Depth-wise convolution for local inductive bias, with residual
        x = x + self.dw_conv(x)
        return x


# =============================================================================
# U-KAN MODEL
# =============================================================================

class UKAN(nn.Module):
    """U-KAN: convolutional encoder stages at higher spatial resolutions
    plus tokenized KAN stages at deeper spatial resolutions, with a U-shaped
    decoder that mirrors this structure and uses skip connections.

    Architecture at 256x256 input:
        Encoder:  Conv(c1)@256 -> Conv(c2)@128 -> Conv(c3)@64
                  -> TokKAN(c4)@32 -> TokKAN(c5)@16
        Decoder:  TokKAN(c4)@32 (skip=s4)
                  -> Conv(c3)@64 (skip=s3, concat)
                  -> Conv(c2)@128 (skip=s2, concat)
                  -> Conv(c1)@256 (skip=s1, concat)
        Output:   1x1 conv -> num_classes logits

    use_tanh (default True): applies the tanh-support correction to every
    KAN layer. False recovers the unmodified B-spline U-KAN.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        encoder_dims=(32, 64, 128, 160, 256),
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        use_tanh: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert len(encoder_dims) == 5, "encoder_dims must have five stages (3 conv + 2 KAN)"
        c1, c2, c3, c4, c5 = encoder_dims

        # ---- Encoder ----
        self.enc1 = ConvBlock(in_channels, c1)          # skip at 256
        self.enc2 = ConvBlock(c1, c2)                    # skip at 128
        self.enc3 = ConvBlock(c2, c3)                    # skip at 64

        self.proj4 = nn.Conv2d(c3, c4, 1)                # channel projection into KAN dim
        self.enc4 = TokKANBlock(c4, kan_grid_size, kan_spline_order, use_tanh)  # skip at 32

        self.proj5 = nn.Conv2d(c4, c5, 1)                # channel projection into KAN dim
        self.enc5 = TokKANBlock(c5, kan_grid_size, kan_spline_order, use_tanh)  # bottleneck at 16

        self.pool = nn.MaxPool2d(2)

        # ---- Decoder ----
        # KAN decoder stage (mirrors enc4)
        self.up5 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec5 = TokKANBlock(c4, kan_grid_size, kan_spline_order, use_tanh)

        # Conv decoder stages (mirror enc3, enc2, enc1)
        self.up4 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec4 = ConvBlock(c3 * 2, c3)

        self.up3 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec3 = ConvBlock(c2 * 2, c2)

        self.up2 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec2 = ConvBlock(c1 * 2, c1)

        # ---- Output head ----
        self.out_conv = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x):
        # ---- Encoder ----
        s1 = self.enc1(x)                                # (B, c1, 256, 256)
        s2 = self.enc2(self.pool(s1))                    # (B, c2, 128, 128)
        s3 = self.enc3(self.pool(s2))                    # (B, c3,  64,  64)

        s4_in = self.proj4(self.pool(s3))                # (B, c4,  32,  32)
        s4 = self.enc4(s4_in)                            # (B, c4,  32,  32)

        b_in = self.proj5(self.pool(s4))                 # (B, c5,  16,  16)
        b = self.enc5(b_in)                              # (B, c5,  16,  16)

        # ---- Decoder ----
        d5 = self.up5(b)                                 # (B, c4,  32,  32)
        d5 = self.dec5(d5 + s4)                          # additive skip; KAN block

        d4 = self.up4(d5)                                # (B, c3,  64,  64)
        d4 = self.dec4(torch.cat([d4, s3], dim=1))       # concat skip; conv block

        d3 = self.up3(d4)                                # (B, c2, 128, 128)
        d3 = self.dec3(torch.cat([d3, s2], dim=1))

        d2 = self.up2(d3)                                # (B, c1, 256, 256)
        d2 = self.dec2(torch.cat([d2, s1], dim=1))

        return self.out_conv(d2)                         # (B, num_classes, 256, 256)