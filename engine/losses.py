"""Segmentation losses for binary and multi-class echocardiographic masks.

Single loss class ``SegLoss`` for both heads: it inspects ``num_classes`` and
internally dispatches to ``BCEWithLogits + soft Dice`` (binary,
``out_channels == 1``) or ``CrossEntropy + soft Dice`` (multi-class,
``out_channels == num_classes >= 2``). The soft Dice term is averaged over
foreground classes only -- including background pumps the loss towards a
trivial all-background prediction at the start of training.

Targets are always integer label maps of shape ``[B, H, W]``. The loss does
any reshaping internally so the trainer does not need to know which head it
is feeding.
"""
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_dice_per_class(probs: torch.Tensor, target_onehot: torch.Tensor,
                         eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice per class, mean over batch. ``probs`` and ``target_onehot``
    are both ``[B, K, H, W]``. Returns ``Tensor[K]``."""
    B, K = probs.shape[:2]
    p = probs.view(B, K, -1)
    t = target_onehot.view(B, K, -1)
    inter = (p * t).sum(-1)
    denom = p.sum(-1) + t.sum(-1)
    dice_b = (2.0 * inter + eps) / (denom + eps)
    return dice_b.mean(dim=0)


class SegLoss(nn.Module):
    """CE/BCE + soft Dice, automatic binary / multi-class dispatch.

    Args:
        num_classes: total number of classes including background. Use 2 for
            binary (the model still emits a single sigmoid channel, but
            losses and metrics reason in two-class label space).
        foreground_classes: which class indices contribute to the soft Dice
            term. Defaults to ``[1, ..., num_classes-1]`` (everything except
            background).
        ce_weight: weight on the cross-entropy / BCE term.
        dice_weight: weight on the soft Dice term.
        class_weights: optional per-class weights for the CE term (unused for
            binary). Must have length ``num_classes`` if provided.
    """

    def __init__(
        self,
        num_classes: int,
        foreground_classes: Optional[Sequence[int]] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[Iterable[float]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 (binary uses 2: bg + fg).")
        self.num_classes = num_classes
        self.foreground_classes = (
            list(foreground_classes)
            if foreground_classes is not None
            else list(range(1, num_classes))
        )
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.eps = eps

        if num_classes == 2:
            self.cls_loss = nn.BCEWithLogitsLoss()
        else:
            w = None
            if class_weights is not None:
                w = torch.tensor(list(class_weights), dtype=torch.float32)
            self.cls_loss = nn.CrossEntropyLoss(weight=w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, 1, H, W]      (binary)   or  [B, K, H, W] (multi-class)
        targets: [B, H, W] long with values in {0..num_classes-1}
        """
        if targets.dim() == 4:
            # Tolerate legacy [B, 1, H, W] float targets from older trainers.
            targets = targets.squeeze(1)
        targets = targets.long()

        if self.num_classes == 2:
            # ----- BCE path -----
            cls = self.cls_loss(logits, targets.float().unsqueeze(1))
            # Build a 2-channel softmax-like distribution from a single sigmoid
            # logit so the Dice machinery is the same as multi-class.
            p_fg = torch.sigmoid(logits)                  # [B, 1, H, W]
            probs = torch.cat([1.0 - p_fg, p_fg], dim=1)  # [B, 2, H, W]
        else:
            # ----- CE path -----
            cls = self.cls_loss(logits, targets)
            probs = F.softmax(logits, dim=1)              # [B, K, H, W]

        target_onehot = F.one_hot(targets, num_classes=self.num_classes)\
                          .permute(0, 3, 1, 2).float()

        dice_b = _soft_dice_per_class(probs, target_onehot, self.eps)
        fg_dice = dice_b[self.foreground_classes].mean()
        dice_term = 1.0 - fg_dice

        return self.ce_weight * cls + self.dice_weight * dice_term


# -----------------------------------------------------------------------------
# Backward-compat shim
# -----------------------------------------------------------------------------
class BCEDiceLoss(SegLoss):
    """Legacy alias used by older configs. Binary segmentation (num_classes=2)."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__(num_classes=2, ce_weight=bce_weight, dice_weight=dice_weight)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    """Stand-alone soft Dice over foreground classes. Kept for compatibility
    with any external code that imported this name directly."""
    if targets.dim() == 4:
        targets = targets.squeeze(1)
    targets = targets.long()
    num_classes = max(int(targets.max().item()) + 1, 2)
    if logits.size(1) > 1:
        num_classes = max(num_classes, logits.size(1))

    if logits.size(1) == 1:
        p_fg = torch.sigmoid(logits)
        probs = torch.cat([1.0 - p_fg, p_fg], dim=1)
    else:
        probs = F.softmax(logits, dim=1)

    target_onehot = F.one_hot(targets, num_classes=num_classes)\
                      .permute(0, 3, 1, 2).float()
    dice_b = _soft_dice_per_class(probs, target_onehot, eps)
    return 1.0 - dice_b[1:].mean()


def build_loss(num_classes: int, **kwargs) -> SegLoss:
    """Factory used by the trainer. Pass ``num_classes`` from the config."""
    return SegLoss(num_classes=num_classes, **kwargs)