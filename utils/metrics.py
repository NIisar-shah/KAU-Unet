"""Evaluation metrics for binary and multi-class echocardiographic segmentation.

All hard-decision metrics operate on integer label maps of shape ``[B, H, W]``
(values in ``{0..K-1}``). Use :func:`predict_labels` to convert raw model
logits into a label map; this hides the difference between the binary
single-channel sigmoid output and the multi-class softmax output.

For soft (probability-based) metrics, see ``engine/losses.py`` — those live
with the loss because that's where their gradients matter.
"""
from typing import Dict

import numpy as np
import torch


# -----------------------------------------------------------------------------
# logits -> label map
# -----------------------------------------------------------------------------
def predict_labels(logits: torch.Tensor) -> torch.Tensor:
    """Convert raw model output to a hard label map ``[B, H, W]``.

    Binary head (``out_channels == 1``): sigmoid + 0.5 threshold.
    Multi-class head (``out_channels  > 1``): argmax across the class dim.
    """
    if logits.size(1) == 1:
        return (torch.sigmoid(logits[:, 0]) > 0.5).long()
    return logits.argmax(dim=1).long()


# -----------------------------------------------------------------------------
# per-class hard metrics (torch, batched)
# -----------------------------------------------------------------------------
def dice_per_class(
    pred_labels: torch.Tensor,
    target_labels: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-class hard Dice, mean over batch. Returns ``Tensor[num_classes]``.

    For each class ``c``: ``2|P n T| / (|P|+|T|)`` averaged over batch.
    Classes absent from both prediction and target in every batch element
    return Dice = 1.0 (vacuous match), via the ``eps`` smoothing.
    """
    out = torch.zeros(num_classes, device=pred_labels.device)
    for c in range(num_classes):
        p = (pred_labels == c).float().view(pred_labels.size(0), -1)
        t = (target_labels == c).float().view(target_labels.size(0), -1)
        inter = (p * t).sum(dim=1)
        denom = p.sum(dim=1) + t.sum(dim=1)
        out[c] = ((2.0 * inter + eps) / (denom + eps)).mean()
    return out


def iou_per_class(
    pred_labels: torch.Tensor,
    target_labels: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-class IoU, mean over batch. Returns ``Tensor[num_classes]``."""
    out = torch.zeros(num_classes, device=pred_labels.device)
    for c in range(num_classes):
        p = (pred_labels == c).float().view(pred_labels.size(0), -1)
        t = (target_labels == c).float().view(target_labels.size(0), -1)
        inter = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1) - inter
        out[c] = ((inter + eps) / (union + eps)).mean()
    return out


# -----------------------------------------------------------------------------
# legacy binary helpers (kept so older code paths still compile)
# -----------------------------------------------------------------------------
def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor,
                            eps: float = 1e-6) -> float:
    """Scalar foreground Dice. Accepts either binary or multi-class logits.

    targets may be ``[B, H, W]`` long or ``[B, 1, H, W]`` float -- both work.
    Reports mean Dice across all foreground classes (1..K-1) for multi-class,
    or class-1 Dice for binary.
    """
    pred = predict_labels(logits)
    if targets.dim() == 4:
        targets = targets.squeeze(1)
    targets = targets.long()
    num_classes = max(int(pred.max().item()), int(targets.max().item()), 1) + 1
    if logits.size(1) > 1:
        num_classes = max(num_classes, logits.size(1))
    d = dice_per_class(pred, targets, num_classes, eps)
    if num_classes == 2:
        return float(d[1].item())
    return float(d[1:].mean().item())


def iou_from_logits(logits: torch.Tensor, targets: torch.Tensor,
                    eps: float = 1e-6) -> float:
    """Scalar foreground IoU -- same conventions as ``dice_score_from_logits``."""
    pred = predict_labels(logits)
    if targets.dim() == 4:
        targets = targets.squeeze(1)
    targets = targets.long()
    num_classes = max(int(pred.max().item()), int(targets.max().item()), 1) + 1
    if logits.size(1) > 1:
        num_classes = max(num_classes, logits.size(1))
    i = iou_per_class(pred, targets, num_classes, eps)
    if num_classes == 2:
        return float(i[1].item())
    return float(i[1:].mean().item())


# -----------------------------------------------------------------------------
# HD95: per-class boundary metric
# -----------------------------------------------------------------------------
def hausdorff_95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance, in pixels.

    Both inputs are 2-D binary numpy arrays of the same shape (one foreground
    class at a time). Returns ``nan`` if either mask is empty so empty-mask
    cases can be filtered out of averages instead of contaminating them.
    """
    from scipy.ndimage import distance_transform_edt

    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float("nan")

    dt_pred = distance_transform_edt(1 - pred_mask)
    dt_gt = distance_transform_edt(1 - gt_mask)

    d_gt_to_pred = dt_pred[gt_mask.astype(bool)]
    d_pred_to_gt = dt_gt[pred_mask.astype(bool)]

    return float(np.percentile(
        np.concatenate([d_gt_to_pred, d_pred_to_gt]), 95
    ))


def hausdorff_95_per_class(
    pred_labels: np.ndarray,
    target_labels: np.ndarray,
    foreground_classes,
) -> Dict[int, float]:
    """Per-class HD95 on a single 2-D label map. Returns ``{c: hd95_or_nan}``.

    ``pred_labels`` and ``target_labels`` are 2-D ``np.ndarray`` of label
    indices (e.g. as produced by :func:`predict_labels` per sample, moved to
    CPU/numpy). One foreground class at a time is binarised and passed to
    :func:`hausdorff_95`.
    """
    out: Dict[int, float] = {}
    for c in foreground_classes:
        p = (pred_labels == c).astype(np.uint8)
        g = (target_labels == c).astype(np.uint8)
        out[c] = hausdorff_95(p, g)
    return out