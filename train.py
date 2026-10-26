"""Main entry point for fully supervised echocardiographic segmentation.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/kan_tanh.yaml --train_fraction 0.10 --seed 42 \
        --save_dir runs/deff/kan_tanh_frac0.10_seed42
"""
import argparse
import math
import random
import yaml

import torch
from torch.utils.data import DataLoader

from data import CamusDataset, list_patient_ids, split_patient_ids
from models import build_model
from engine import Trainer
from utils import set_seed


def build_optimizer(model, cfg_opt):
    name = cfg_opt["name"].lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg_opt["lr"],
            weight_decay=cfg_opt["weight_decay"],
        )
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg_opt["lr"])
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg_sched):
    if cfg_sched is None:
        return None
    name = cfg_sched["name"].lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg_sched["T_max"],
            eta_min=cfg_sched["eta_min"],
        )
    raise ValueError(f"Unknown scheduler: {name}")


def main(cfg_path: str, train_fraction=None, seed=None, save_dir=None):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- CLI overrides -----------------------------------------------------
    # Let a single config drive the whole data-efficiency sweep: a bash loop
    # varies --train_fraction / --seed / --save_dir, the config stays put.
    # NOTE: this overrides experiment.seed (model init + dataloader order +
    # subsample draw). It does NOT touch data.split_seed, so the val/test
    # partition is identical across every run.
    if seed is not None:
        cfg["experiment"]["seed"] = int(seed)
    if train_fraction is not None:
        cfg["data"]["train_fraction"] = float(train_fraction)
    if save_dir is not None:
        cfg["experiment"]["save_dir"] = save_dir

    set_seed(cfg["experiment"]["seed"])

    # --- task / class bookkeeping ------------------------------------------
    task = cfg.get("task", "multiclass")
    num_classes = int(cfg.get("num_classes", 4))
    class_names = cfg.get("class_names")
    if class_names is None:
        class_names = [f"class{c}" for c in range(num_classes)]
    foreground_classes = cfg.get("foreground_classes",
                                 list(range(1, num_classes)))

    # The model's output-channel count is fully determined by the task:
    # 1 channel for binary (BCE-style head), num_classes for multi-class
    # (softmax head). Setting it here guarantees no drift between the
    # config's task field and the model that gets built.
    cfg["model"]["params"]["out_channels"] = 1 if task == "binary" else num_classes

    # ----- data ------------------------------------------------------------
    all_ids = list_patient_ids(cfg["data"]["root"])
    # Patient split is held fixed across seeds (data.split_seed); only the
    # model init / dataloader order vary with experiment.seed.
    split_seed = cfg["data"].get("split_seed", cfg["experiment"]["seed"])
    splits = split_patient_ids(
        all_ids,
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        seed=split_seed,
    )
    print(f"Patients - train: {len(splits['train'])}, "
          f"val: {len(splits['val'])}, test: {len(splits['test'])}")

    # ----- data-efficiency subsampling (TRAIN ONLY; val/test untouched) ----
    # Nested & seed-tied: for a given seed, a smaller fraction is a STRICT
    # subset of every larger one, because each fraction takes a prefix of the
    # same seed-shuffled ordering of the full training set. This folds "which
    # patients were drawn" variance into the seed band you already report, and
    # leaves val/test exactly as split_patient_ids produced them.
    train_fraction = float(cfg["data"].get("train_fraction", 1.0))
    n_full_train = len(splits["train"])
    if train_fraction < 1.0:
        subsample_seed = cfg["data"].get("subsample_seed", cfg["experiment"]["seed"])
        order = list(splits["train"])
        random.Random(subsample_seed).shuffle(order)
        n_keep = max(1, round(train_fraction * n_full_train))
        splits["train"] = sorted(order[:n_keep])
        print(f"Data-efficiency: train_fraction={train_fraction} "
              f"(subsample_seed={subsample_seed}) -> "
              f"{len(splits['train'])} / {n_full_train} train patients")

    # ----- equalize optimization budget across fractions -------------------
    # Fixed EPOCHS would give low-fraction runs proportionally fewer gradient
    # steps (steps/epoch = floor(n_patients * n_views / batch) under
    # drop_last=True), confounding "less data" with "less training" -- the
    # first thing a reviewer attacks on a data-efficiency curve. We hold the
    # TOTAL gradient-step count ~constant by scaling epochs so a fraction-run
    # matches the full-data step budget, and stretch the cosine schedule
    # (T_max) over the same horizon. Validation then runs every `val_every`
    # epochs, so the ~base_epochs validations land at the same STEP cadence as
    # the full run instead of thousands of times at low fractions. Model
    # selection stays "best validation Dice".
    base_epochs = int(cfg["train"]["num_epochs"])
    n_views = len(cfg["data"]["views"])
    batch = int(cfg["train"]["batch_size"])
    steps_per_epoch_full = max(1, (n_full_train * n_views) // batch)
    steps_per_epoch_frac = max(1, (len(splits["train"]) * n_views) // batch)
    target_total_steps = base_epochs * steps_per_epoch_full
    effective_epochs = max(base_epochs,
                           math.ceil(target_total_steps / steps_per_epoch_frac))
    val_every = max(1, effective_epochs // base_epochs)
    if effective_epochs != base_epochs:
        cfg["train"]["num_epochs"] = effective_epochs
        if cfg["train"].get("scheduler") and "T_max" in cfg["train"]["scheduler"]:
            cfg["train"]["scheduler"]["T_max"] = effective_epochs
        print(f"Budget: base {base_epochs} epochs (~{target_total_steps} steps) -> "
              f"{effective_epochs} epochs at fraction {train_fraction}; "
              f"validate every {val_every} epochs")

    train_ds = CamusDataset(
        root=cfg["data"]["root"],
        patient_ids=splits["train"],
        views=tuple(cfg["data"]["views"]),
        image_size=cfg["data"]["image_size"],
        target_label=cfg["data"].get("target_label", 1),
        augment=cfg["train"]["augment"],
        task=task,
        file_ext=cfg["data"].get("file_ext", ".nii.gz"),
    )
    val_ds = CamusDataset(
        root=cfg["data"]["root"],
        patient_ids=splits["val"],
        views=tuple(cfg["data"]["views"]),
        image_size=cfg["data"]["image_size"],
        target_label=cfg["data"].get("target_label", 1),
        augment=False,
        task=task,
        file_ext=cfg["data"].get("file_ext", ".nii.gz"),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    # ----- model -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['model']['name']}  |  task: {task}  |  "
          f"num_classes: {num_classes}  |  "
          f"out_channels: {cfg['model']['params']['out_channels']}  |  "
          f"trainable params: {n_params/1e6:.3f} M")

    optimizer = build_optimizer(model, cfg["train"]["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["train"].get("scheduler"))

    # ----- boundary-aware aux loss config (only used by KAN_UNet_Plus) -----
    # The Trainer reads these via cfg["train"].get("boundary_*", default).
    # For non-boundary-aware models, the Trainer detects the model's
    # forward-pass return type (single tensor vs. tuple) and falls back
    # to the original single-loss path, so these settings are no-ops.
    boundary_loss_weight = cfg["train"].get("boundary_loss_weight", 0.5)
    boundary_pos_weight = cfg["train"].get("boundary_pos_weight", 5.0)
    boundary_thickness = cfg["train"].get("boundary_thickness", 2)

    # ----- train -----------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir=cfg["experiment"]["save_dir"],
        num_classes=2 if task == "binary" else num_classes,
        foreground_classes=foreground_classes if task != "binary" else [1],
        class_names=class_names if task != "binary"
                    else ["background", class_names[1] if len(class_names) > 1 else "foreground"],
        num_epochs=cfg["train"]["num_epochs"],
        validate_every=val_every,
        scheduler=scheduler,
        # Boundary-aware auxiliary loss (Trainer ignores these for models
        # whose forward() returns a single tensor).
        boundary_loss_weight=boundary_loss_weight,
        boundary_pos_weight=boundary_pos_weight,
        boundary_thickness=boundary_thickness,
    )
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--train_fraction", type=float, default=None,
                        help="Fraction of TRAIN patients to keep (val/test fixed). "
                             "Omit or 1.0 for full training set.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override experiment.seed (init + subsample draw). "
                             "Does NOT change data.split_seed.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override experiment.save_dir (one dir per sweep run).")
    args = parser.parse_args()
    main(args.config, train_fraction=args.train_fraction,
         seed=args.seed, save_dir=args.save_dir)