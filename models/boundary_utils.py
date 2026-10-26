"""Boundary-aware auxiliary loss utilities.

Provides three families of utilities used by KAN_UNet_Plus variants:

  - compute_boundary_target(gt_mask, ...) : binary boundary target via Sobel
        edge detection on each foreground-class mask, unioned. Used by the
        Phase 1 / KAN_UNet_Plus default boundary head.

  - boundary_loss(boundary_logits, boundary_target, pos_weight) : BCE-with-
        logits between predicted boundary logits and the binary target.

  - compute_sdf_target(gt_mask, ...) : signed-distance-function target.
        Inside each foreground region the value is the negative distance to
        the boundary; outside it is the positive distance. Used by the
        Phase 2 kan_unet_plus_sdf variant. Mathematically tied to HD95
        because the SDF *is* a distance, so regressing it directly
        supervises the metric we care about.

  - sdf_loss(sdf_logits, sdf_target) : L1 loss in the SDF domain plus an
        optional region term (Boundary Loss, Kervadec et al. MICCAI 2019,
        equation 3) implemented via product with predicted foreground mask.

Default boundary thickness is 2 pixels (dilated from the 1-pixel Sobel
edge by a 5x5 max-pool). This thickness is a deliberately small lobe
around the true class boundary -- thick enough to provide a learnable
target gradient, thin enough that the loss focuses the model's
attention on the actual transition region.
"""
import torch
import torch.nn.functional as F


# Sobel kernels stored as module-level constants and moved to the input
# device on each call (negligible overhead).
_SOBEL_X = torch.tensor(
    [[-1., 0., 1.],
     [-2., 0., 2.],
     [-1., 0., 1.]],
).view(1, 1, 3, 3)

_SOBEL_Y = torch.tensor(
    [[-1., -2., -1.],
     [ 0.,  0.,  0.],
     [ 1.,  2.,  1.]],
).view(1, 1, 3, 3)


# =============================================================================
# Phase 1 / default boundary head: binary boundary BCE
# =============================================================================
@torch.no_grad()
def compute_boundary_target(
    gt_mask: torch.Tensor,
    num_classes: int,
    foreground_classes=None,
    thickness: int = 2,
) -> torch.Tensor:
    """Compute binary boundary target from a per-pixel class label map.

    Args:
        gt_mask: (B, H, W) integer label map in {0, ..., num_classes-1}.
        num_classes: total number of classes including background.
        foreground_classes: list of class indices treated as foreground.
            Default: all non-zero classes ([1, 2, ..., num_classes-1]).
        thickness: optional dilation half-width in pixels around each
            class boundary. thickness=0 gives a strict 1-pixel Sobel
            edge. thickness=2 (default) gives a 5-pixel-wide boundary
            band, which is easier to learn and matches the HD95
            tolerance of a few pixels.

    Returns:
        boundary: (B, 1, H, W) float tensor in {0, 1} marking the union
            of class-boundary pixels in the foreground region.
    """
    if foreground_classes is None:
        foreground_classes = list(range(1, num_classes))

    device = gt_mask.device
    B, H, W = gt_mask.shape

    sobel_x = _SOBEL_X.to(device)
    sobel_y = _SOBEL_Y.to(device)

    boundary = torch.zeros(B, 1, H, W, device=device)
    for c in foreground_classes:
        cls_mask = (gt_mask == c).float().unsqueeze(1)
        gx = F.conv2d(cls_mask, sobel_x, padding=1)
        gy = F.conv2d(cls_mask, sobel_y, padding=1)
        edges = torch.sqrt(gx * gx + gy * gy + 1e-12)
        boundary = torch.maximum(boundary, (edges > 0.5).float())

    if thickness > 0:
        k = 2 * thickness + 1
        boundary = F.max_pool2d(boundary, kernel_size=k,
                                stride=1, padding=thickness)

    return boundary


def boundary_loss(
    boundary_logits: torch.Tensor,
    boundary_target: torch.Tensor,
    pos_weight: float = 5.0,
) -> torch.Tensor:
    """BCE-with-logits boundary loss with positive-class up-weighting.

    Boundary pixels are sparse (~5-10% of image area after dilation),
    so pos_weight > 1 prevents the loss from collapsing to the trivial
    all-zero solution.

    Args:
        boundary_logits: (B, 1, H, W) raw logits from boundary_head.
        boundary_target: (B, 1, H, W) binary target from
            compute_boundary_target.
        pos_weight: scalar multiplier on positive-class loss.
            Default 5.0 corresponds roughly to inverse class frequency.

    Returns:
        Scalar loss tensor.
    """
    pos_weight_tensor = torch.tensor(
        pos_weight, device=boundary_logits.device,
        dtype=boundary_logits.dtype,
    )
    return F.binary_cross_entropy_with_logits(
        boundary_logits, boundary_target,
        pos_weight=pos_weight_tensor,
    )


# =============================================================================
# Phase 2 variant: signed-distance regression (Kervadec et al. MICCAI 2019)
# =============================================================================
@torch.no_grad()
def compute_sdf_target(
    gt_mask: torch.Tensor,
    num_classes: int,
    foreground_classes=None,
    clip: float = 20.0,
) -> torch.Tensor:
    """Compute signed-distance-function target from a per-pixel class
    label map.

    For each foreground class, compute the distance transform of the
    binary mask AND of its complement; the SDF is the complement-distance
    minus the foreground-distance (positive outside, negative inside).
    The final target is the SUM of per-class SDFs (one channel per
    foreground class).

    This approximates the SDF using a 4-iteration approximate erosion
    + chamfer expansion that runs on GPU without requiring scipy. The
    approximation is sufficient for a learning target (we don't need
    exact distances, only smoothly-varying signals away from the
    boundary).

    Args:
        gt_mask: (B, H, W) integer label map.
        num_classes: total number of classes including background.
        foreground_classes: which class indices contribute. Defaults to
            all non-zero classes.
        clip: clamp absolute SDF values at +/- clip pixels to prevent
            far-away pixels from dominating the L1 loss.

    Returns:
        sdf: (B, K_fg, H, W) float tensor, where K_fg is the number of
            foreground classes. Negative inside the class region,
            positive outside, in pixel units.
    """
    if foreground_classes is None:
        foreground_classes = list(range(1, num_classes))

    device = gt_mask.device
    dtype = torch.float32
    B, H, W = gt_mask.shape
    K_fg = len(foreground_classes)
    sdf = torch.zeros(B, K_fg, H, W, device=device, dtype=dtype)

    for k, c in enumerate(foreground_classes):
        m = (gt_mask == c).to(dtype).unsqueeze(1)  # (B, 1, H, W)
        # Approximate distance transform: iteratively grow a chamfer
        # distance. _approx_distance_transform(mask) is zero INSIDE
        # the mask and grows outward, so:
        #   dist_from_fg = distance from foreground (zero inside fg)
        #   dist_from_bg = distance from background (zero outside fg)
        dist_from_fg = _approx_distance_transform(m, max_iters=int(clip),
                                                  device=device)
        dist_from_bg = _approx_distance_transform(1.0 - m, max_iters=int(clip),
                                                  device=device)
        # Standard SDF convention: positive outside the region, negative
        # inside. Outside fg: dist_from_fg > 0, dist_from_bg = 0
        #   -> sdf = dist_from_fg (positive) [outside]
        # Inside fg:  dist_from_fg = 0, dist_from_bg > 0
        #   -> sdf = -dist_from_bg (negative) [inside]
        sdf_k = (dist_from_fg - dist_from_bg).squeeze(1)
        sdf[:, k] = sdf_k.clamp(-clip, clip)

    return sdf


@torch.no_grad()
def _approx_distance_transform(
    mask: torch.Tensor, max_iters: int = 20, device=None,
) -> torch.Tensor:
    """Approximate Euclidean distance transform on GPU via iterative
    dilation. For each background pixel, the result is the L_inf distance
    (in pixels) to the nearest foreground pixel, capped at max_iters.

    Args:
        mask: (B, 1, H, W) binary tensor.
        max_iters: cap on the distance value.

    Returns:
        dist: (B, 1, H, W) float tensor of distances. Distance is 0
            inside the mask and grows outward.
    """
    if device is None:
        device = mask.device
    dist = torch.zeros_like(mask)
    # The "frontier" is the set of pixels not yet reached. We dilate the
    # mask one pixel per iteration; pixels newly reached at iteration k
    # get distance k.
    cur = mask.clone()
    for k in range(1, max_iters + 1):
        # 3x3 max-pool dilates by 1 pixel.
        dilated = F.max_pool2d(cur, kernel_size=3, stride=1, padding=1)
        newly = (dilated - cur).clamp(min=0)  # pixels reached at step k
        dist = dist + newly * float(k)
        cur = dilated
        if cur.min() > 0.5:
            # all pixels reached, no further iterations needed
            break
    return dist


def sdf_loss(
    sdf_logits: torch.Tensor,
    sdf_target: torch.Tensor,
    smooth_l1_beta: float = 1.0,
) -> torch.Tensor:
    """Smooth-L1 (Huber) loss between predicted and target SDF maps.

    Smooth-L1 is less sensitive to occasional large errors near the image
    border than plain L1, and easier to optimize than L2 because gradients
    do not blow up for large prediction errors at initialization.

    Args:
        sdf_logits: (B, K_fg, H, W) predicted SDF values from the model.
        sdf_target: (B, K_fg, H, W) target SDF from compute_sdf_target.
        smooth_l1_beta: transition point between L2 and L1 regimes.

    Returns:
        Scalar loss tensor.
    """
    return F.smooth_l1_loss(sdf_logits, sdf_target, beta=smooth_l1_beta)
