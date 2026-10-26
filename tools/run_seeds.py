"""Multi-seed runner for the KAN-UNet vs Baseline-UNet ablation.

Trains each (model, seed) combination using the same base config -- only
``experiment.seed``, ``experiment.name``, ``experiment.save_dir`` and
``model.name`` vary. ``data.split_seed`` is held fixed so all seeds see the
same patient train/val/test split, isolating architecture-init variance
from data-split variance.

After each training run finishes, the script invokes ``evaluate.py`` on the
freshly written ``best.pt`` and writes the per-sample / summary CSVs in the
run directory.

Crash-safe: skips any (model, seed) pair whose ``best.pt`` already exists.
Pass ``--force`` to retrain or ``--force_eval`` to re-evaluate.

Usage:
    python tools/run_seeds.py \
        --base_config configs/default.yaml \
        --models kan_unet baseline_unet \
        --seeds 42 0 1 2 3 \
        --runs_root ./runs
"""
import argparse
import copy
import sys
from pathlib import Path

import yaml

# Make project root importable when invoked from the repo root or anywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import main as train_main
from evaluate import main as evaluate_main


def make_per_run_config(base_cfg, model_name, seed, runs_root):
    """Build a per-run config that differs from the base only in experiment
    bookkeeping and the model name. Data split is intentionally untouched."""
    cfg = copy.deepcopy(base_cfg)
    run_name = f"{model_name}_seed{seed}"
    cfg["experiment"]["name"] = run_name
    cfg["experiment"]["save_dir"] = str(Path(runs_root) / run_name)
    cfg["experiment"]["seed"] = int(seed)
    cfg["model"]["name"] = model_name
    return cfg


def write_resolved_config(cfg, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "used_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def main(args):
    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    total = len(args.models) * len(args.seeds)
    done = 0
    failures = []
    for model_name in args.models:
        for seed in args.seeds:
            done += 1
            run_name = f"{model_name}_seed{seed}"
            run_dir = runs_root / run_name
            best_pt = run_dir / "best.pt"
            cfg_path = run_dir / "used_config.yaml"

            print(f"\n{'='*72}")
            print(f"[{done}/{total}] {run_name}")
            print(f"{'='*72}")

            # ---- train ------------------------------------------------------
            try:
                if best_pt.exists() and not args.force:
                    print(f"  train skip: {best_pt} exists. Pass --force to retrain.")
                else:
                    cfg = make_per_run_config(base_cfg, model_name, seed, runs_root)
                    write_resolved_config(cfg, str(run_dir))
                    train_main(str(cfg_path))
            except Exception as e:
                print(f"  TRAIN FAILED for {run_name}: {e}")
                failures.append((run_name, "train", str(e)))
                continue

            # ---- evaluate ---------------------------------------------------
            try:
                if not best_pt.exists():
                    print(f"  eval skip: no best.pt at {best_pt}")
                    continue
                summary = run_dir / "test_summary.csv"
                if summary.exists() and not args.force_eval:
                    print(f"  eval skip: {summary} exists. Pass --force_eval to redo.")
                else:
                    evaluate_main(str(cfg_path), str(best_pt))
            except Exception as e:
                print(f"  EVAL FAILED for {run_name}: {e}")
                failures.append((run_name, "evaluate", str(e)))

    print(f"\n{'='*72}")
    if failures:
        print(f"Done with {len(failures)} failures:")
        for run_name, stage, msg in failures:
            print(f"  - {run_name} [{stage}]: {msg}")
    else:
        print("All runs completed successfully.")
    print(f"\nNext: python tools/aggregate_seeds.py --runs_root {runs_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_config", required=True)
    p.add_argument("--models", nargs="+", required=True,
                   help="e.g. kan_unet baseline_unet")
    p.add_argument("--seeds", nargs="+", type=int, required=True,
                   help="e.g. 42 0 1 2 3")
    p.add_argument("--runs_root", default="./runs")
    p.add_argument("--force", action="store_true",
                   help="Retrain even if best.pt exists.")
    p.add_argument("--force_eval", action="store_true",
                   help="Re-evaluate even if test_summary.csv exists.")
    args = p.parse_args()
    main(args)