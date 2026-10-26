"""Training loop for binary or multi-class echocardiographic segmentation.

The same ``Trainer`` drives all configurations. It dispatches the loss
computation on whether the model's forward returns a single tensor or a
tuple, and -- for tuple-returning models -- on the model's
``aux_signature`` attribute which lists the auxiliary heads in their
forward-pass order.

Supported auxiliary heads (in any combination):
  - "boundary"     : binary Sobel-derived boundary (BCE-with-logits)
  - "sdf"          : signed-distance regression (smooth-L1)
  - "boundary_s1"  : multi-scale boundary at decoder stage 1
  - "boundary_s2"  : multi-scale boundary at decoder stage 2
  - "ds1"/"ds2"/"ds3" : deep-supervision Dice/CE heads at decoder stages

For models whose forward returns a single tensor (Baseline, KAN-UNet,
KAN-UNet-NoTanh, KAN-UNet-MLP), the tuple branch is never taken, so
their training behaviour is unchanged from before the Phase 1 / Phase 2
work.
"""
import os
import csv
from typing import Dict, Optional, Sequence, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .losses import build_loss
from utils.metrics import (
    predict_labels,
    dice_per_class,
    iou_per_class,
    hausdorff_95,
)
from models.boundary_utils import (
    compute_boundary_target,
    boundary_loss,
    compute_sdf_target,
    sdf_loss,
)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        device,
        save_dir: str,
        num_classes: int,
        foreground_classes: Optional[Sequence[int]] = None,
        class_names: Optional[Sequence[str]] = None,
        num_epochs: int = 100,
        validate_every: int = 1,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        # Auxiliary-loss config (Trainer-level fallback defaults; the
        # MODEL is the authority -- if model.X exists, it overrides
        # these). Kept for backward compatibility with older train.py.
        boundary_loss_weight: float = 0.5,
        boundary_pos_weight: float = 5.0,
        boundary_thickness: int = 2,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.num_epochs = num_epochs
        # How often (in epochs) to run validation + checkpointing. 1 = every
        # epoch (original behaviour). Larger values are used by the data-
        # efficiency sweep, where epochs are scaled up by 1/fraction so the
        # validation cadence stays constant in gradient-STEP space. The final
        # epoch is always validated regardless of this value.
        self.validate_every = max(1, int(validate_every))
        self.scheduler = scheduler

        self.num_classes = num_classes
        self.foreground_classes = (
            list(foreground_classes)
            if foreground_classes is not None
            else list(range(1, num_classes))
        )
        if class_names is None:
            class_names = [f"class{c}" for c in range(num_classes)]
        if len(class_names) != num_classes:
            raise ValueError(
                f"class_names length ({len(class_names)}) "
                f"must match num_classes ({num_classes})"
            )
        self.class_names = list(class_names)

        # Auxiliary-loss defaults. Model attributes (if present) take
        # precedence over these Trainer-level defaults.
        self.default_boundary_loss_weight = float(boundary_loss_weight)
        self.default_boundary_pos_weight = float(boundary_pos_weight)
        self.default_boundary_thickness = int(boundary_thickness)

        os.makedirs(self.save_dir, exist_ok=True)
        self.criterion = build_loss(
            num_classes=num_classes,
            foreground_classes=self.foreground_classes,
        )
        self.best_val_dice = -1.0
        self.log_path = os.path.join(self.save_dir, "log.csv")

        if not os.path.exists(self.log_path):
            header = ["epoch", "train_loss", "train_dice",
                      "val_loss", "val_dice_mean_fg",
                      "val_iou_mean_fg", "val_hd95_mean_fg"]
            for c in self.foreground_classes:
                header.append(f"val_dice_{self.class_names[c]}")
            header.append("lr")
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    # -------------------------------------------------------------------------
    # batch -> (images, targets)
    # -------------------------------------------------------------------------
    def _prepare_batch(self, batch: Dict[str, torch.Tensor]):
        ed = batch["ed_frame"].to(self.device)
        es = batch["es_frame"].to(self.device)
        ed_m = batch["ed_mask"].to(self.device).long()
        es_m = batch["es_mask"].to(self.device).long()
        images = torch.cat([ed, es], dim=0)
        targets = torch.cat([ed_m, es_m], dim=0)
        return images, targets

    # -------------------------------------------------------------------------
    # Auxiliary loss helpers
    # -------------------------------------------------------------------------
    def _get(self, attr: str, fallback):
        """Read an aux-loss config attribute from the model if it exists,
        otherwise fall back to the Trainer's default."""
        return getattr(self.model, attr, fallback)

    def _compute_aux_losses(self, aux_tensors, aux_signature, targets):
        """Given the auxiliary output tensors (in the model's declared
        aux_signature order) and the integer GT, compute each component
        loss and return:
            total_aux_loss (scalar Tensor),
            components_dict (str -> float).
        """
        components = {}
        total_aux = torch.zeros((), device=targets.device, dtype=torch.float32)

        # Pull per-model config (with Trainer defaults as fallback).
        boundary_weight = self._get("boundary_loss_weight",
                                    self.default_boundary_loss_weight)
        boundary_pos = self._get("boundary_pos_weight",
                                 self.default_boundary_pos_weight)
        boundary_thick = self._get("boundary_thickness",
                                   self.default_boundary_thickness)
        ds_weights = self._get("ds_weights", (0.4, 0.2, 0.1))
        msb_weights = self._get("msb_weights", (1.0, 0.5, 0.25))
        sdf_weight = self._get("sdf_weight", 0.5)

        # Pre-compute targets that may be reused across heads.
        boundary_target = None
        sdf_target = None

        for name, tensor in zip(aux_signature, aux_tensors):
            if name == "boundary":
                if boundary_target is None:
                    with torch.no_grad():
                        boundary_target = compute_boundary_target(
                            targets,
                            num_classes=self.num_classes,
                            foreground_classes=self.foreground_classes,
                            thickness=boundary_thick,
                        )
                comp = boundary_loss(tensor, boundary_target,
                                     pos_weight=boundary_pos)
                total_aux = total_aux + boundary_weight * comp
                components["boundary"] = comp.item()

            elif name in ("boundary_s1", "boundary_s2"):
                if boundary_target is None:
                    with torch.no_grad():
                        boundary_target = compute_boundary_target(
                            targets,
                            num_classes=self.num_classes,
                            foreground_classes=self.foreground_classes,
                            thickness=boundary_thick,
                        )
                # MSB weights: msb_weights[0] is for the primary "boundary"
                # head, msb_weights[1] for s1, msb_weights[2] for s2.
                w_idx = 1 if name == "boundary_s1" else 2
                w = boundary_weight * msb_weights[w_idx]
                comp = boundary_loss(tensor, boundary_target,
                                     pos_weight=boundary_pos)
                total_aux = total_aux + w * comp
                components[name] = comp.item()

            elif name == "sdf":
                if sdf_target is None:
                    with torch.no_grad():
                        sdf_target = compute_sdf_target(
                            targets,
                            num_classes=self.num_classes,
                            foreground_classes=self.foreground_classes,
                        )
                comp = sdf_loss(tensor, sdf_target)
                total_aux = total_aux + sdf_weight * comp
                components["sdf"] = comp.item()

            elif name in ("ds1", "ds2", "ds3"):
                # Use the same SegLoss criterion as the primary head.
                idx = int(name[-1]) - 1   # ds1 -> 0, ds2 -> 1, ds3 -> 2
                comp = self.criterion(tensor, targets)
                total_aux = total_aux + ds_weights[idx] * comp
                components[name] = comp.item()

            else:
                raise ValueError(f"Unknown aux signature entry: {name!r}")

        return total_aux, components

    # -------------------------------------------------------------------------
    # epoch loops
    # -------------------------------------------------------------------------
    def train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_seg_loss = 0.0
        running_aux_loss = 0.0
        aux_active = False
        aux_component_running: Dict[str, float] = {}

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}")
        for batch in pbar:
            images, targets = self._prepare_batch(batch)

            outputs = self.model(images)

            if isinstance(outputs, tuple):
                aux_active = True
                seg_logits = outputs[0]
                aux_tensors = outputs[1:]
                aux_signature = getattr(
                    self.model, "aux_signature",
                    ("boundary",) if len(aux_tensors) == 1 else None,
                )
                if aux_signature is None:
                    raise RuntimeError(
                        "Model returned a tuple but has no aux_signature "
                        "attribute. Cannot dispatch aux losses."
                    )

                seg_loss = self.criterion(seg_logits, targets)

                aux_total, components = self._compute_aux_losses(
                    aux_tensors, aux_signature, targets,
                )
                loss = seg_loss + aux_total

                running_seg_loss += seg_loss.item()
                running_aux_loss += aux_total.item()
                for k, v in components.items():
                    aux_component_running[k] = \
                        aux_component_running.get(k, 0.0) + v
            else:
                seg_logits = outputs
                loss = self.criterion(seg_logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                preds = predict_labels(seg_logits.detach())
                d = dice_per_class(preds, targets, self.num_classes)
                batch_dice = float(d[self.foreground_classes].mean().item())

            running_loss += loss.item()
            running_dice += batch_dice

            if aux_active:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    seg=f"{seg_loss.item():.4f}",
                    aux=f"{aux_total.item():.4f}",
                    dice=f"{batch_dice:.4f}",
                )
            else:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    dice=f"{batch_dice:.4f}",
                )

        n = len(self.train_loader)
        if aux_active:
            comp_str = "  ".join(
                f"{k}={v/n:.4f}" for k, v in aux_component_running.items()
            )
            print(f"  [aux] seg={running_seg_loss/n:.4f}  "
                  f"aux={running_aux_loss/n:.4f}  components: {comp_str}")

        return running_loss / n, running_dice / n

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        running_loss = 0.0
        dice_sum = torch.zeros(self.num_classes, device=self.device)
        iou_sum = torch.zeros(self.num_classes, device=self.device)
        hd95_per_class: Dict[int, List[float]] = {c: [] for c in self.foreground_classes}
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Val   {epoch}")
        for batch in pbar:
            images, targets = self._prepare_batch(batch)
            logits = self.model(images)
            # Defensive unwrap in case a future model returns a tuple
            # even in eval mode.
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = self.criterion(logits, targets)

            preds = predict_labels(logits)
            dice_sum += dice_per_class(preds, targets, self.num_classes)
            iou_sum += iou_per_class(preds, targets, self.num_classes)
            running_loss += loss.item()
            n_batches += 1

            preds_np = preds.cpu().numpy()
            gts_np = targets.cpu().numpy()
            for b in range(preds_np.shape[0]):
                for c in self.foreground_classes:
                    p_c = (preds_np[b] == c).astype(np.uint8)
                    g_c = (gts_np[b] == c).astype(np.uint8)
                    h = hausdorff_95(p_c, g_c)
                    if not np.isnan(h):
                        hd95_per_class[c].append(h)

        mean_dice = (dice_sum / n_batches).cpu()
        mean_iou = (iou_sum / n_batches).cpu()
        mean_loss = running_loss / n_batches
        hd95_mean = {
            c: (float(np.mean(v)) if v else float("nan"))
            for c, v in hd95_per_class.items()
        }
        return mean_loss, mean_dice, mean_iou, hd95_mean

    # -------------------------------------------------------------------------
    # logging / checkpointing
    # -------------------------------------------------------------------------
    def _save_csv_log(self, epoch, tl, td, vl, vd, vi, vh):
        fg = self.foreground_classes
        mean_fg_dice = float(vd[fg].mean().item())
        mean_fg_iou = float(vi[fg].mean().item())
        hd_vals = [vh[c] for c in fg]
        valid = [h for h in hd_vals if not np.isnan(h)]
        mean_fg_hd95 = float(np.mean(valid)) if valid else float("nan")

        row = [
            epoch, f"{tl:.6f}", f"{td:.6f}",
            f"{vl:.6f}", f"{mean_fg_dice:.6f}",
            f"{mean_fg_iou:.6f}",
            f"{mean_fg_hd95:.4f}" if not np.isnan(mean_fg_hd95) else "nan",
        ]
        for c in fg:
            row.append(f"{float(vd[c].item()):.6f}")
        row.append(f"{self.optimizer.param_groups[0]['lr']:.8f}")

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        return mean_fg_dice, mean_fg_iou, mean_fg_hd95

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float,
                        mean_fg_dice: float):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "foreground_classes": self.foreground_classes,
        }
        torch.save(ckpt, os.path.join(self.save_dir, "last.pt"))
        if mean_fg_dice > self.best_val_dice:
            self.best_val_dice = mean_fg_dice
            ckpt["best_val_dice"] = self.best_val_dice
            torch.save(ckpt, os.path.join(self.save_dir, "best.pt"))

    # -------------------------------------------------------------------------
    # driver
    # -------------------------------------------------------------------------
    def fit(self):
        for epoch in range(self.num_epochs):
            tl, td = self.train_one_epoch(epoch)

            # Validate (and checkpoint) every `validate_every` epochs, and
            # always on the final epoch. The scheduler still steps every
            # epoch so the cosine schedule is unaffected by the cadence.
            is_last = (epoch == self.num_epochs - 1)
            if ((epoch + 1) % self.validate_every == 0) or is_last:
                vl, vd, vi, vh = self.validate(epoch)
                mean_fg_dice, mean_fg_iou, mean_fg_hd95 = \
                    self._save_csv_log(epoch, tl, td, vl, vd, vi, vh)
                self.save_checkpoint(epoch, tl, vl, mean_fg_dice)

                per_class_str = "  ".join(
                    f"{self.class_names[c]}={float(vd[c].item()):.3f}"
                    for c in self.foreground_classes
                )
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_loss={tl:.4f} train_dice={td:.4f} | "
                    f"val_loss={vl:.4f} val_dice={mean_fg_dice:.4f} "
                    f"val_iou={mean_fg_iou:.4f} "
                    f"val_hd95={mean_fg_hd95:.2f}  ({per_class_str})"
                )
            else:
                print(f"[Epoch {epoch:03d}] train_loss={tl:.4f} "
                      f"train_dice={td:.4f}  (val skipped)")

            if self.scheduler is not None:
                self.scheduler.step()