"""KAN-UNet+ family: KAN bottleneck augmented with various auxiliary
supervision strategies. All variants subclass the base KAN_UNet_Plus
which exposes three independent toggles via __init__ kwargs:

  - boundary_loss_weight (float):
        Scales the boundary BCE term in the total loss. The Trainer
        reads model.boundary_loss_weight at runtime, so this can be
        varied across subclasses without retraining the architecture.
        Phase 1 sweep: 0.5 (default), 0.2, 0.1.

  - use_deep_supervision (bool):
        Adds three auxiliary segmentation heads at decoder stages
        1, 2, 3 (channel counts c3, c2, c1), each producing a full-
        resolution prediction. Trained jointly with auxiliary Dice
        losses, weights [0.4, 0.2, 0.1] from deepest to shallowest.

  - use_multiscale_boundary (bool):
        Adds boundary heads at decoder stages 1 and 2 in addition to
        the final-resolution boundary head. Each auxiliary boundary
        head is supervised by a downsampled Sobel target. Phase 2 only.

  - use_sdf (bool):
        Replaces the binary boundary head with a multi-channel signed-
        distance regression head (one channel per foreground class).
        Supervised by smooth-L1 against the Sobel-based SDF target.
        Phase 2 only.

Forward-pass return contract:
  - Eval mode: always a single tensor (seg_logits) for back-compat
    with evaluate.py.
  - Train mode: a tuple, structure depends on which auxiliary heads
    are active. The Trainer dispatches on tuple length.

Tuple structures in train mode:
  - boundary only:                (seg, boundary)
  - boundary + DS:                (seg, boundary, ds1, ds2, ds3)
  - multi-scale boundary:         (seg, boundary, boundary_s1, boundary_s2)
  - SDF (no binary boundary):     (seg, sdf)
  - SDF + DS:                     (seg, sdf, ds1, ds2, ds3)

The Trainer inspects the model's `aux_signature` attribute (a tuple of
strings like ("boundary",) or ("boundary", "ds1", "ds2", "ds3")) to
know how to unpack and weight each component.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan_unet import KAN_UNet
from .baseline_unet import DoubleConv


# =============================================================================
# Base class: configurable auxiliary supervision
# =============================================================================
class KAN_UNet_Plus(KAN_UNet):
    """KAN-UNet with optional boundary, deep-supervision, multi-scale
    boundary, and SDF auxiliary supervision heads.

    The default configuration matches the original KAN-UNet+ (binary
    boundary head only, weight 0.5). Subclasses change the toggles.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        kan_num_layers: int = 2,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        boundary_loss_weight: float = 0.5,
        boundary_pos_weight: float = 5.0,
        boundary_thickness: int = 2,
        use_deep_supervision: bool = False,
        ds_weights=(0.4, 0.2, 0.1),
        use_multiscale_boundary: bool = False,
        msb_weights=(1.0, 0.5, 0.25),
        use_sdf: bool = False,
        sdf_weight: float = 0.5,
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

        # Channel widths matching the parent KAN_UNet construction.
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        # Store all toggles as instance attributes; the Trainer reads
        # these at runtime to know how to compute and weight losses.
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.boundary_pos_weight = float(boundary_pos_weight)
        self.boundary_thickness = int(boundary_thickness)
        self.use_deep_supervision = bool(use_deep_supervision)
        self.ds_weights = tuple(ds_weights)
        self.use_multiscale_boundary = bool(use_multiscale_boundary)
        self.msb_weights = tuple(msb_weights)
        self.use_sdf = bool(use_sdf)
        self.sdf_weight = float(sdf_weight)

        # Number of foreground classes used for SDF (set externally by
        # Trainer if needed; default to out_channels - 1).
        self._n_fg_classes = max(out_channels - 1, 1)

        # ----- Always-present primary boundary head ---------------------
        # Replaced by SDF head when use_sdf=True.
        if self.use_sdf:
            self.sdf_head = nn.Conv2d(c1, self._n_fg_classes, kernel_size=1)
            self.boundary_head = None
        else:
            self.boundary_head = nn.Conv2d(c1, 1, kernel_size=1)
            self.sdf_head = None

        # ----- Deep supervision heads (3 stages, all upsampled to H, W) -
        if self.use_deep_supervision:
            # Decoder stage outputs are: up1 -> c3, up2 -> c2, up3 -> c1.
            # We attach heads at the OUTPUT of each up-block. Up1 output
            # has c3 channels (deepest), up3 has c1 (final, already
            # supervised by out_conv). For DS we use up1 (c3) and up2 (c2).
            # The third "head" is at the final resolution and operates on
            # the same features as out_conv; for simplicity we wire three
            # heads at the up1/up2/up3 outputs.
            self.ds_head_1 = nn.Conv2d(c3, out_channels, kernel_size=1)
            self.ds_head_2 = nn.Conv2d(c2, out_channels, kernel_size=1)
            self.ds_head_3 = nn.Conv2d(c1, out_channels, kernel_size=1)
        else:
            self.ds_head_1 = None
            self.ds_head_2 = None
            self.ds_head_3 = None

        # ----- Multi-scale boundary heads (Phase 2) ---------------------
        if self.use_multiscale_boundary:
            self.boundary_head_s1 = nn.Conv2d(c3, 1, kernel_size=1)
            self.boundary_head_s2 = nn.Conv2d(c2, 1, kernel_size=1)
        else:
            self.boundary_head_s1 = None
            self.boundary_head_s2 = None

        # ----- Auxiliary signature (tuple of strings) -------------------
        # Tells the Trainer which auxiliary heads are active and in what
        # order they appear in the forward-pass tuple.
        self.aux_signature = self._build_aux_signature()

    def _build_aux_signature(self):
        sig = []
        if self.use_sdf:
            sig.append("sdf")
        else:
            sig.append("boundary")
        if self.use_multiscale_boundary:
            sig.extend(["boundary_s1", "boundary_s2"])
        if self.use_deep_supervision:
            sig.extend(["ds1", "ds2", "ds3"])
        return tuple(sig)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x):
        # Re-run the parent encoder/bottleneck/decoder path, capturing
        # the intermediate up-block outputs so the auxiliary heads can
        # tap them off.
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.bottleneck_kan(x4)

        u1 = self.up1(x4, x3)   # (B, c3, H/4, W/4)
        u2 = self.up2(u1, x2)   # (B, c2, H/2, W/2)
        u3 = self.up3(u2, x1)   # (B, c1, H,    W)

        seg_logits = self.out_conv(u3)

        if not self.training:
            return seg_logits

        # ----- Build training-time tuple per aux_signature --------------
        outputs = [seg_logits]

        H, W = seg_logits.shape[-2:]

        # Primary auxiliary head: boundary (1 channel) OR SDF (K_fg channels).
        if self.use_sdf:
            outputs.append(self.sdf_head(u3))
        else:
            outputs.append(self.boundary_head(u3))

        # Multi-scale boundary heads (Phase 2): upsample to full res so
        # the Trainer can use the same Sobel target.
        if self.use_multiscale_boundary:
            b_s1 = self.boundary_head_s1(u1)
            b_s1 = F.interpolate(b_s1, size=(H, W),
                                 mode="bilinear", align_corners=False)
            b_s2 = self.boundary_head_s2(u2)
            b_s2 = F.interpolate(b_s2, size=(H, W),
                                 mode="bilinear", align_corners=False)
            outputs.extend([b_s1, b_s2])

        # Deep-supervision heads.
        if self.use_deep_supervision:
            ds1 = self.ds_head_1(u1)
            ds1 = F.interpolate(ds1, size=(H, W),
                                mode="bilinear", align_corners=False)
            ds2 = self.ds_head_2(u2)
            ds2 = F.interpolate(ds2, size=(H, W),
                                mode="bilinear", align_corners=False)
            ds3 = self.ds_head_3(u3)   # already at full res
            outputs.extend([ds1, ds2, ds3])

        return tuple(outputs)


# =============================================================================
# Phase 1 variants: boundary-weight sweep and deep supervision
# =============================================================================
class KAN_UNet_Plus_Lambda02(KAN_UNet_Plus):
    """Phase 1A: KAN-UNet+ with boundary loss weight lambda=0.2.
    Hypothesis: lambda=0.5 is over-regularizing on the dense multi-class
    seg loss; lowering it should free seg loss to dominate and let
    boundary serve as a gentle regularizer."""

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
            boundary_loss_weight=0.2,
        )


class KAN_UNet_Plus_Lambda01(KAN_UNet_Plus):
    """Phase 1A (backup): KAN-UNet+ with boundary loss weight lambda=0.1.
    Use only if lambda=0.2 still appears to over-regularize."""

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
            boundary_loss_weight=0.1,
        )


class KAN_UNet_Plus_DS(KAN_UNet_Plus):
    """Phase 1B: KAN-UNet+ with deep supervision in addition to the
    default boundary loss (lambda=0.5). Three auxiliary Dice losses at
    decoder stages, weights (0.4, 0.2, 0.1)."""

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
            use_deep_supervision=True,
        )


class KAN_UNet_Plus_Combo(KAN_UNet_Plus):
    """Phase 1 follow-up (only train if neither lambda02 nor DS alone
    gives a clean win): combines lambda=0.2 with deep supervision."""

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
            boundary_loss_weight=0.2,
            use_deep_supervision=True,
        )


# =============================================================================
# Phase 2 variants: multi-scale boundary supervision and SDF regression
# =============================================================================
class KAN_UNet_Plus_MSB(KAN_UNet_Plus):
    """Phase 2A: Multi-scale boundary supervision. Adds boundary heads at
    decoder stages 1, 2 in addition to the final-resolution head.
    Hypothesis: forcing boundary-relevant features at multiple resolutions
    may improve HD95 across class scales (LV myo is small, LA is large)."""

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
            boundary_loss_weight=0.2,           # use the (presumed-best)
                                                # lambda from Phase 1
            use_multiscale_boundary=True,
        )


class KAN_UNet_Plus_SDF(KAN_UNet_Plus):
    """Phase 2B: Signed-distance-function regression instead of binary
    boundary BCE. Mathematically tied to HD95 because the SDF *is* a
    pixel-distance function, so regressing it directly supervises the
    boundary-distance metric we report.

    Replaces the boundary BCE head with a multi-channel (one per
    foreground class) SDF regression head trained with smooth-L1."""

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
            use_sdf=True,
            sdf_weight=0.1,
        )
