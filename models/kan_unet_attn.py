"""Attention-gated KAN-UNet.

Adds attention gates on the three encoder->decoder skip connections,
following the additive attention formulation of Oktay et al. (MIDL 2018,
"Attention U-Net: Learning Where to Look for the Pancreas"). Each skip
connection is filtered by an attention map computed from the
corresponding decoder upsampled features, allowing the network to
selectively suppress irrelevant skip context (e.g. acoustic shadow
regions in echocardiograms, off-anatomy speckle).

This is a complementary architectural axis to the bottleneck-design
investigation: where the KAN bottleneck and auxiliary supervision
strategies control WHAT the deepest representation captures, the
attention gates control WHICH skip information the decoder receives
back at each upsampling stage.

Attention gate formulation (per skip):
    Given skip x_l (B, C_skip, H, W) and gating signal g (B, C_gate, H_g, W_g):
    1. Upsample g bilinearly to (H, W) if needed.
    2. alpha = sigmoid(psi(ReLU(W_x(x_l) + W_g(g))))     in [0, 1]
    3. filtered_skip = alpha * x_l                        (broadcast over channels)

The alpha map is a single-channel attention mask shared across all
skip channels, broadcast multiplicatively. This is the most parameter-
efficient form of additive attention gating (as in the original paper).

Parameter overhead vs KAN_UNet:
    ag1 (skip=c3, gate=c4): ~25K
    ag2 (skip=c2, gate=c3): ~6.3K
    ag3 (skip=c1, gate=c2): ~1.6K
    Total: ~33K (negligible vs 3.24 M backbone)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan_unet import KAN_UNet


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al. 2018) for skip-connection
    filtering.

    Inter-channel dimension defaults to max(skip_channels // 2, 16) for
    a sensible parameter trade-off.
    """

    def __init__(self, skip_channels, gate_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 16)
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        """
        Args:
            x: skip feature (B, skip_channels, H, W)
            g: gating signal (B, gate_channels, H_g, W_g) with
               H_g <= H and W_g <= W.
        Returns:
            filtered skip (B, skip_channels, H, W).
        """
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:],
                              mode='bilinear', align_corners=False)
        x_p = self.W_x(x)
        g_p = self.W_g(g)
        att = self.psi(self.relu(x_p + g_p))    # (B, 1, H, W)
        return x * att


class AttKAN_UNet(KAN_UNet):
    """KAN-UNet with attention gates on all three encoder->decoder skips.

    Bottleneck, encoder, and decoder structure are inherited unchanged
    from KAN_UNet (tanh-bounded KAN bottleneck, no boundary aux head).
    The only architectural difference is that the three skip features
    are filtered by attention gates conditioned on the corresponding
    decoder feature one level deeper.

    Forward returns a single tensor in both train and eval mode (no
    auxiliary heads), so the existing Trainer single-tensor path
    handles training without modification.
    """

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
            use_tanh=True,
            bottleneck_type="kan",
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # ag1: gate skip x3 (c3 channels, H/4) with bottleneck x4 (c4, H/8 -> upsampled to H/4)
        # ag2: gate skip x2 (c2,        H/2) with decoder  u1 (c3, H/4 -> H/2)
        # ag3: gate skip x1 (c1,        H)   with decoder  u2 (c2, H/2 -> H)
        self.ag1 = AttentionGate(skip_channels=c3, gate_channels=c4)
        self.ag2 = AttentionGate(skip_channels=c2, gate_channels=c3)
        self.ag3 = AttentionGate(skip_channels=c1, gate_channels=c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.bottleneck_kan(x4)

        # Attention-gated skips: filter each skip by the deeper decoder
        # signal before passing it into the Up module.
        x3_gated = self.ag1(x3, x4)
        u1 = self.up1(x4, x3_gated)

        x2_gated = self.ag2(x2, u1)
        u2 = self.up2(u1, x2_gated)

        x1_gated = self.ag3(x1, u2)
        u3 = self.up3(u2, x1_gated)

        return self.out_conv(u3)