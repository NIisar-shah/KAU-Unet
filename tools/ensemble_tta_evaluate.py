"""Ensemble + TTA evaluation across all 5 seeds of each variant.

For each variant with 5 trained seeds in <runs_root>/<model>_seed{N}/:
  - Load all 5 best.pt checkpoints into 5 model instances.
  - For each held-out test (patient, view, phase) tuple:
      a. Get softmax probabilities from each of the 5 seed models on
         the original input.
      b. Get softmax probabilities from each on horizontally flipped
         input, then flip the prediction back.
      c. Average across all 10 probability maps (5 seeds * 2 TTA).
      d. Take argmax to produce the ensemble label map.
  - Compute Dice / IoU / HD95 against GT.
  - Write per-sample and summary CSVs into
      <output_dir>/<model>/test_per_sample.csv
      <output_dir>/<model>/test_summary.csv
  - At the end, write an aggregate CSV across all evaluated variants:
      <output_dir>/aggregate.csv

Usage:
    cd /workspace/kan_fully_sup
    python tools/ensemble_tta_evaluate.py \
        --runs_root ./runs \
        --config configs/default.yaml \
        --output_dir ./runs/ensemble_results

Wall time (Quadro GV100): ~30-45 minutes for all 8 variants, with 75
test patients x 2 phases x 5 seeds x 2 TTA = 1500 forward passes per
variant.
"""
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import CamusDataset, list_patient_ids, split_patient_ids
from models import build_model
from utils import set_seed, hausdorff_95


RUN_DIR_RE = re.compile(r"^(?P<model>.+)_seed(?P<seed>-?\d+)$")


# -----------------------------------------------------------------------------
# Per-sample ensemble + TTA inference
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_with_tta(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
    """Return softmax probabilities averaged over original + horizontal
    flip TTA. ``img`` is (1, C, H, W)."""
    logits = model(img)
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = torch.softmax(logits, dim=1)

    img_flip = torch.flip(img, dims=[3])
    logits_flip = model(img_flip)
    if isinstance(logits_flip, tuple):
        logits_flip = logits_flip[0]
    probs_flip = torch.softmax(logits_flip, dim=1)
    probs_flip = torch.flip(probs_flip, dims=[3])

    return (probs + probs_flip) / 2.0


@torch.no_grad()
def ensemble_predict(models: List[torch.nn.Module],
                     img: torch.Tensor) -> torch.Tensor:
    """Mean softmax probabilities across an ensemble of models (each
    with TTA). Returns (1, K, H, W)."""
    probs_sum = None
    for m in models:
        p = predict_with_tta(m, img)
        probs_sum = p if probs_sum is None else (probs_sum + p)
    return probs_sum / len(models)


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------
def find_complete_variants(runs_root: Path, min_seeds: int = 5
                           ) -> Dict[str, Dict[int, Path]]:
    """Scan runs_root for {model}_seed{N}/ directories with best.pt;
    return only variants having at least min_seeds complete seeds."""
    variants: Dict[str, Dict[int, Path]] = {}
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        m = RUN_DIR_RE.match(d.name)
        if not m:
            continue
        ckpt = d / "best.pt"
        if not ckpt.exists():
            continue
        model_name = m.group("model")
        seed = int(m.group("seed"))
        variants.setdefault(model_name, {})[seed] = ckpt
    return {k: v for k, v in variants.items() if len(v) >= min_seeds}


# -----------------------------------------------------------------------------
# Per-variant evaluation
# -----------------------------------------------------------------------------
def evaluate_variant(
    model_name: str,
    ckpt_paths: Dict[int, Path],
    cfg: dict,
    device: torch.device,
    test_ds: CamusDataset,
    foreground_classes: Sequence[int],
    class_names: Sequence[str],
) -> List[Dict]:
    cfg_local = {**cfg}
    cfg_local["model"] = {**cfg["model"], "name": model_name}

    models: List[torch.nn.Module] = []
    for seed in sorted(ckpt_paths.keys()):
        m = build_model(cfg_local).to(device)
        ckpt = torch.load(ckpt_paths[seed], map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)

    records: List[Dict] = []
    for i in range(len(test_ds)):
        item = test_ds[i]
        patient = item["patient"]
        view = item["view"]
        for phase in ("ED", "ES"):
            img = item[f"{phase.lower()}_frame"].unsqueeze(0).to(device)
            gt = item[f"{phase.lower()}_mask"].unsqueeze(0).to(device).long()

            probs = ensemble_predict(models, img)
            preds = probs.argmax(dim=1)         # (1, H, W)

            preds_np = preds[0].cpu().numpy()
            gt_np = gt[0].cpu().numpy()

            for c in foreground_classes:
                p = (preds_np == c).astype(np.uint8)
                g = (gt_np == c).astype(np.uint8)
                inter = float((p & g).sum())
                p_sum = float(p.sum())
                g_sum = float(g.sum())
                dice = (2.0 * inter + 1e-6) / (p_sum + g_sum + 1e-6)
                iou = (inter + 1e-6) / (p_sum + g_sum - inter + 1e-6)
                hd95 = hausdorff_95(p, g)
                records.append({
                    "patient": patient,
                    "view": view,
                    "phase": phase,
                    "class_id": c,
                    "class_name": class_names[c],
                    "dice": float(dice),
                    "iou": float(iou),
                    "hd95": float(hd95),
                })

    # Free GPU memory before next variant.
    for m in models:
        del m
    torch.cuda.empty_cache()

    return records


# -----------------------------------------------------------------------------
# Summary aggregation
# -----------------------------------------------------------------------------
def summarize(records: List[Dict], model_name: str,
              foreground_classes: Sequence[int],
              class_names: Sequence[str]) -> List[Dict]:
    def _row(group: str, recs: List[Dict]) -> Dict:
        d = [r["dice"] for r in recs]
        i = [r["iou"] for r in recs]
        h = [r["hd95"] for r in recs if not np.isnan(r["hd95"])]
        return {
            "model": model_name,
            "group": group,
            "n_samples": len(recs),
            "dice_mean": float(np.mean(d)) if d else float("nan"),
            "dice_std": (float(np.std(d, ddof=1)) if len(d) > 1 else 0.0),
            "iou_mean": float(np.mean(i)) if i else float("nan"),
            "iou_std": (float(np.std(i, ddof=1)) if len(i) > 1 else 0.0),
            "hd95_mean": float(np.mean(h)) if h else float("nan"),
            "hd95_std": (float(np.std(h, ddof=1)) if len(h) > 1 else 0.0),
        }

    rows = [_row("overall", records)]
    for c in foreground_classes:
        cname = class_names[c]
        rows.append(_row(cname, [r for r in records if r["class_id"] == c]))
    for v in sorted({r["view"] for r in records}):
        rows.append(_row(v, [r for r in records if r["view"] == v]))
    for ph in ("ED", "ES"):
        rows.append(_row(ph, [r for r in records if r["phase"] == ph]))
    return rows


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default="./runs")
    parser.add_argument("--config", type=str,
                        default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str,
                        default="./runs/ensemble_results")
    parser.add_argument("--variants", type=str, nargs="*", default=None,
                        help="Restrict to specific variant names. "
                             "Default: all variants with 5+ seeds.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["experiment"]["seed"])

    task = cfg.get("task", "multiclass")
    num_classes = int(cfg.get("num_classes", 4))
    class_names = cfg.get("class_names") or [
        f"class{c}" for c in range(num_classes)]
    foreground_classes = cfg.get("foreground_classes",
                                 list(range(1, num_classes)))
    cfg["model"]["params"]["out_channels"] = \
        1 if task == "binary" else num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the test set once -- it's the same across all variants
    # because data.split_seed is fixed.
    all_ids = list_patient_ids(cfg["data"]["root"])
    splits = split_patient_ids(
        all_ids,
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        seed=cfg["data"].get("split_seed", 42),
    )
    test_ds = CamusDataset(
        root=cfg["data"]["root"],
        patient_ids=splits["test"],
        views=tuple(cfg["data"]["views"]),
        image_size=cfg["data"]["image_size"],
        target_label=cfg["data"].get("target_label", 1),
        augment=False,
        task=task,
        file_ext=cfg["data"].get("file_ext", ".nii.gz"),
    )

    variants = find_complete_variants(runs_root)
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants}

    print(f"Evaluating ensemble+TTA on {len(variants)} variants:")
    for name, seeds in variants.items():
        print(f"  {name}: seeds {sorted(seeds.keys())}")
    print()

    aggregate_rows: List[Dict] = []
    for name, seed_paths in sorted(variants.items()):
        print(f"=== {name} ===")
        try:
            records = evaluate_variant(
                name, seed_paths, cfg, device, test_ds,
                foreground_classes, class_names)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue

        out_d = output_dir / name
        out_d.mkdir(parents=True, exist_ok=True)
        if records:
            with open(out_d / "test_per_sample.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                w.writeheader()
                for r in records:
                    w.writerow(r)

        summary = summarize(records, name, foreground_classes, class_names)
        with open(out_d / "test_summary.csv", "w", newline="") as f:
            if summary:
                w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
                w.writeheader()
                for r in summary:
                    w.writerow(r)
        aggregate_rows.extend(summary)

        overall = summary[0]
        print(f"  Overall  Dice={overall['dice_mean']:.4f} +/- "
              f"{overall['dice_std']:.4f}   "
              f"HD95={overall['hd95_mean']:.2f} +/- "
              f"{overall['hd95_std']:.2f}")
        for r in summary[1:]:
            if r["group"] in class_names[1:]:
                print(f"  {r['group']:18s} Dice={r['dice_mean']:.4f}  "
                      f"HD95={r['hd95_mean']:.2f}")
        print()

    if aggregate_rows:
        with open(output_dir / "aggregate.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
            w.writeheader()
            for r in aggregate_rows:
                w.writerow(r)
        print(f"Wrote {output_dir / 'aggregate.csv'}")
        print(f"     {output_dir / '<variant>' / 'test_per_sample.csv'}")
        print(f"     {output_dir / '<variant>' / 'test_summary.csv'}")


if __name__ == "__main__":
    main()