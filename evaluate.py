"""Held-out test-set evaluation for a trained checkpoint.

Loads the saved checkpoint and computes per-sample, per-class Dice / IoU /
HD95 on the test split that was set aside at training time. Reproduces the
exact same patient-level split by re-using ``config.experiment.seed``.

Two CSVs are written next to the checkpoint:

* ``test_per_sample.csv``   one row per (patient, view, phase, class) sample
* ``test_summary.csv``      aggregations -- overall, by-class, by-view,
                            by-phase, by-(view, phase), by-(class, view,
                            phase) -- with mean and std for each metric

The summary is also printed to stdout as a markdown table so you can paste
it straight into the manuscript.

Usage:
    python evaluate.py \
        --config configs/default.yaml \
        --checkpoint runs/kan_unet_camus_fullsup/best.pt
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from data import CamusDataset, list_patient_ids, split_patient_ids
from models import build_model
from utils import set_seed, hausdorff_95, predict_labels


# -----------------------------------------------------------------------------
# inference
# -----------------------------------------------------------------------------
def per_sample_metrics(
    model: torch.nn.Module,
    dataset: CamusDataset,
    device: torch.device,
    foreground_classes: Sequence[int],
    class_names: Sequence[str],
) -> List[Dict]:
    """Inference one (patient, view, phase) at a time so the metadata of each
    prediction is preserved for downstream stratified aggregation. Returns a
    long-format list with one record per (patient, view, phase, class)."""
    model.eval()
    records: List[Dict] = []
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            patient = item["patient"]
            view = item["view"]
            for phase in ("ED", "ES"):
                img = item[f"{phase.lower()}_frame"].unsqueeze(0).to(device)
                gt = item[f"{phase.lower()}_mask"].unsqueeze(0).to(device).long()

                outputs = model(img)
                # KAN_UNet_Plus returns a single tensor in eval mode by design,
                # but defensively unwrap a tuple in case some future variant
                # returns (seg_logits, ...) under model.eval() as well.
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                preds = predict_labels(logits)            # [1, H, W] long

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
                        "hd95": float(hd95),  # may be nan
                    })
    return records


# -----------------------------------------------------------------------------
# aggregation
# -----------------------------------------------------------------------------
def aggregate(
    records: List[Dict],
    by: Optional[Sequence[str]] = None,
    label: str = "overall",
) -> List[Dict]:
    """Group records by metadata fields, compute mean / std / valid-count for
    each metric. ``by=None`` collapses everything into a single row."""
    if not by:
        groups: List[Tuple[str, List[Dict]]] = [(label, records)]
    else:
        bucket: Dict[Tuple, List[Dict]] = {}
        for r in records:
            key = tuple(r[b] for b in by)
            bucket.setdefault(key, []).append(r)
        groups = [("+".join(str(k) for k in key), recs)
                  for key, recs in sorted(bucket.items())]

    rows: List[Dict] = []
    for group_label, recs in groups:
        row: Dict = {"group": group_label, "n": len(recs)}
        for metric in ("dice", "iou", "hd95"):
            vals = [r[metric] for r in recs
                    if not (isinstance(r[metric], float) and np.isnan(r[metric]))]
            if vals:
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                row[f"{metric}_n_valid"] = len(vals)
            else:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_n_valid"] = 0
        rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def format_table(rows: List[Dict], title: str = "") -> str:
    lines: List[str] = []
    if title:
        lines.append(f"### {title}")
    lines.append("| group | n | Dice (mean +/- std) | IoU (mean +/- std) | HD95 px (mean +/- std) |")
    lines.append("|---|---:|---|---|---|")
    for r in rows:
        d = f"{r['dice_mean']:.4f} +/- {r['dice_std']:.4f}"
        i = f"{r['iou_mean']:.4f} +/- {r['iou_std']:.4f}"
        h = f"{r['hd95_mean']:.2f} +/- {r['hd95_std']:.2f}"
        lines.append(f"| {r['group']} | {r['n']} | {d} | {i} | {h} |")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# entry point
# -----------------------------------------------------------------------------
def main(cfg_path: str, ckpt_path: str, output_dir: Optional[str] = None) -> None:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"]["seed"])

    task = cfg.get("task", "multiclass")
    num_classes = int(cfg.get("num_classes", 4))
    class_names = cfg.get("class_names")
    if class_names is None:
        class_names = [f"class{c}" for c in range(num_classes)]
    foreground_classes = cfg.get("foreground_classes",
                                 list(range(1, num_classes)))

    # Mirror train.py: the model's output-channel count is fully determined
    # by the task and num_classes. Setting it here guarantees the rebuilt
    # model architecture matches the saved checkpoint regardless of what
    # used_config.yaml literally contains under model.params.
    cfg["model"]["params"]["out_channels"] = 1 if task == "binary" else num_classes

    # Re-derive the exact same test split that training held out.
    # data.split_seed (fixed across seeds) decides patient assignment.
    all_ids = list_patient_ids(cfg["data"]["root"])
    split_seed = cfg["data"].get("split_seed", cfg["experiment"]["seed"])
    splits = split_patient_ids(
        all_ids,
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        seed=split_seed,
    )
    test_ids = splits["test"]
    print(f"Evaluating on {len(test_ids)} held-out test patients "
          f"(views={cfg['data']['views']}, task={task}, classes={class_names})")

    test_ds = CamusDataset(
        root=cfg["data"]["root"],
        patient_ids=test_ids,
        views=tuple(cfg["data"]["views"]),
        image_size=cfg["data"]["image_size"],
        target_label=cfg["data"].get("target_label", 1),
        augment=False,
        task=task,
        file_ext=cfg["data"].get("file_ext", ".nii.gz"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {ckpt_path}  "
          f"(epoch={ckpt.get('epoch', '?')}, "
          f"best_val_dice={ckpt.get('best_val_dice', float('nan')):.4f})")

    records = per_sample_metrics(
        model, test_ds, device, foreground_classes, class_names,
    )

    overall = aggregate(records, by=None, label="overall")
    by_class = aggregate(records, by=["class_name"])
    by_view = aggregate(records, by=["view"])
    by_phase = aggregate(records, by=["phase"])
    by_view_phase = aggregate(records, by=["view", "phase"])
    by_class_view_phase = aggregate(records, by=["class_name", "view", "phase"])

    out_dir = Path(output_dir) if output_dir else Path(ckpt_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(records, out_dir / "test_per_sample.csv")
    write_csv(
        overall + by_class + by_view + by_phase + by_view_phase + by_class_view_phase,
        out_dir / "test_summary.csv",
    )

    print()
    print(format_table(overall, "Overall (all foreground classes)"))
    print()
    print(format_table(by_class, "By class"))
    print()
    print(format_table(by_view, "By view"))
    print()
    print(format_table(by_phase, "By phase"))
    print()
    print(format_table(by_view_phase, "By view x phase"))
    print()
    print(format_table(by_class_view_phase, "By class x view x phase"))
    print()
    print(f"Wrote: {out_dir / 'test_per_sample.csv'}")
    print(f"Wrote: {out_dir / 'test_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to the same YAML used for training.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best.pt (or any model_state_dict checkpoint).")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write CSVs (default: same dir as checkpoint).")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output_dir)