"""Aggregate multi-seed runs into mean +/- std across seeds and emit a
publication-ready two-panel LaTeX table (Dice + HD95) with paired Wilcoxon
significance markers.

Run after ``tools/run_seeds.py`` has populated
``<runs_root>/{model}_seed{N}/`` with ``test_summary.csv`` files. Discovers
runs by directory name and parses ``{model_name}_seed{seed}`` to recover
(model, seed) tuples.

Outputs into ``<runs_root>/aggregate/``:

  * ``aggregate.csv``       mean +/- std across seeds for every (model,
                            group, metric) combination
  * ``paper_table.tex``     Two-panel Table-1: Dice on top, HD95 below.
                            Both panels carry paired-Wilcoxon significance
                            markers vs. the baseline model.
  * ``paired_tests.csv``    per-(class, metric) p-values (two-sided AND
                            one-sided in the favourable direction), deltas,
                            and n_wins_for_proposed across seeds.

Usage:
    python tools/aggregate_seeds.py --runs_root ./runs
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


RUN_DIR_RE = re.compile(r"^(?P<model>.+)_seed(?P<seed>-?\d+)$")

# Which direction is "better" for each metric. Used both for one-sided
# Wilcoxon and for the n_wins count.
METRIC_DIRECTION = {
    "dice_mean": "higher_better",
    "iou_mean":  "higher_better",
    "hd95_mean": "lower_better",
}


# -----------------------------------------------------------------------------
# discovery / loading
# -----------------------------------------------------------------------------
def parse_run_dirs(runs_root: Path) -> List[Tuple[str, int, Path]]:
    found = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        m = RUN_DIR_RE.match(d.name)
        if not m:
            continue
        if not (d / "test_summary.csv").exists():
            print(f"  skip: {d.name} has no test_summary.csv")
            continue
        found.append((m.group("model"), int(m.group("seed")), d))
    return found


def load_summary(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            metrics = {}
            for k, v in row.items():
                if k == "group":
                    continue
                if v in ("", "nan"):
                    metrics[k] = float("nan")
                else:
                    try:
                        metrics[k] = float(v)
                    except ValueError:
                        metrics[k] = float("nan")
            out[row["group"]] = metrics
    return out


def load_class_metadata(runs: List[Tuple[str, int, Path]]) -> Tuple[List[str], List[int]]:
    if not runs:
        return [], []
    cfg_path = runs[0][2] / "used_config.yaml"
    if not cfg_path.exists():
        return [], []
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    class_names = cfg.get("class_names", []) or []
    foreground_classes = cfg.get("foreground_classes",
                                 list(range(1, len(class_names))))
    return class_names, foreground_classes


# -----------------------------------------------------------------------------
# aggregation
# -----------------------------------------------------------------------------
def aggregate_across_seeds(runs: List[Tuple[str, int, Path]]):
    per_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for model, seed, run_dir in runs:
        summary = load_summary(run_dir / "test_summary.csv")
        for group, metrics in summary.items():
            for metric, val in metrics.items():
                if metric.endswith("_mean"):
                    per_seed[model][group][metric].append(val)

    aggregate: Dict = defaultdict(lambda: defaultdict(dict))
    for model, group_dict in per_seed.items():
        for group, metric_dict in group_dict.items():
            for metric, vals in metric_dict.items():
                vals_clean = [v for v in vals if not np.isnan(v)]
                if vals_clean:
                    aggregate[model][group][metric] = {
                        "mean": float(np.mean(vals_clean)),
                        "std": (float(np.std(vals_clean, ddof=1))
                                if len(vals_clean) > 1 else 0.0),
                        "n": len(vals_clean),
                    }
                else:
                    aggregate[model][group][metric] = {
                        "mean": float("nan"), "std": float("nan"), "n": 0,
                    }
    return per_seed, aggregate


# -----------------------------------------------------------------------------
# paired statistical tests
# -----------------------------------------------------------------------------
def paired_wilcoxon(b_vals, a_vals, alternative: str = "two-sided") -> float:
    """Wilcoxon signed-rank on diffs = b - a (proposed - baseline).

    alternative:
        'two-sided': tests median(diff) != 0
        'greater':   tests median(diff) > 0   (use for higher-is-better metrics)
        'less':      tests median(diff) < 0   (use for lower-is-better metrics)
    """
    pairs = [(aa, bb) for aa, bb in zip(a_vals, b_vals)
             if not (np.isnan(aa) or np.isnan(bb))]
    if len(pairs) < 3:
        return float("nan")
    diffs = np.array([bb - aa for aa, bb in pairs])
    if np.all(diffs == 0):
        return 1.0
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(diffs, alternative=alternative)
        return float(res.pvalue)
    except Exception as e:
        print(f"  Wilcoxon failed ({alternative}): {e}")
        return float("nan")


def count_wins_for_proposed(b_vals, a_vals, direction: str) -> int:
    """Count seeds where the proposed model is favorably different.
    direction: 'higher_better' or 'lower_better'."""
    pairs = [(aa, bb) for aa, bb in zip(a_vals, b_vals)
             if not (np.isnan(aa) or np.isnan(bb))]
    diffs = [bb - aa for aa, bb in pairs]
    if direction == "higher_better":
        return sum(1 for d in diffs if d > 0)
    else:
        return sum(1 for d in diffs if d < 0)


def compute_all_pvalues(per_seed, class_names, foreground_classes,
                        baseline_model, proposed_model,
                        metrics=("dice_mean", "hd95_mean")):
    """Returns {metric: {group: {'p_two': ..., 'p_one': ..., 'n_wins': ...}}}."""
    cols = [class_names[i] for i in foreground_classes] + ["overall"]
    out = {m: {} for m in metrics}
    for metric in metrics:
        direction = METRIC_DIRECTION.get(metric, "higher_better")
        one_side_alt = "greater" if direction == "higher_better" else "less"
        for c in cols:
            a = per_seed[baseline_model].get(c, {}).get(metric, [])
            b = per_seed[proposed_model].get(c, {}).get(metric, [])
            out[metric][c] = {
                "p_two": paired_wilcoxon(b, a, alternative="two-sided"),
                "p_one": paired_wilcoxon(b, a, alternative=one_side_alt),
                "n_wins": count_wins_for_proposed(b, a, direction),
                "n_pairs": sum(1 for aa, bb in zip(a, b)
                               if not (np.isnan(aa) or np.isnan(bb))),
            }
    return out


# -----------------------------------------------------------------------------
# output writers
# -----------------------------------------------------------------------------
def write_aggregate_csv(aggregate, out_path: Path) -> None:
    rows = []
    for model, group_dict in aggregate.items():
        for group, metric_dict in group_dict.items():
            row = {"model": model, "group": group}
            for metric, stats in metric_dict.items():
                row[f"{metric}_seed_mean"] = stats["mean"]
                row[f"{metric}_seed_std"] = stats["std"]
                row[f"{metric}_n_seeds"] = stats["n"]
            rows.append(row)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    keys = ["model", "group"] + [k for k in keys if k not in ("model", "group")]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_paired_tests_csv(per_seed, aggregate, pvalues, class_names,
                            foreground_classes, baseline_model, proposed_model,
                            out_path: Path) -> None:
    rows = []
    cols = [class_names[i] for i in foreground_classes] + ["overall"]
    for c in cols:
        for metric in ("dice_mean", "hd95_mean"):
            a = per_seed[baseline_model].get(c, {}).get(metric, [])
            b = per_seed[proposed_model].get(c, {}).get(metric, [])
            diffs = [bb - aa for aa, bb in zip(a, b)
                     if not (np.isnan(aa) or np.isnan(bb))]
            stats = pvalues[metric].get(c, {})
            rows.append({
                "group": c,
                "metric": metric,
                "direction": METRIC_DIRECTION[metric],
                "baseline_mean": float(np.mean(a)) if a else float("nan"),
                "proposed_mean": float(np.mean(b)) if b else float("nan"),
                "delta_mean": float(np.mean(diffs)) if diffs else float("nan"),
                "p_two_sided": stats.get("p_two", float("nan")),
                "p_one_sided": stats.get("p_one", float("nan")),
                "n_wins_for_proposed": stats.get("n_wins", 0),
                "n_pairs": stats.get("n_pairs", 0),
            })
    if not rows:
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_latex_table(aggregate, pvalues, class_names, foreground_classes,
                      out_path: Path, baseline_model: str, proposed_model: str,
                      n_seeds: int,
                      dice_decimals: int = 3,
                      hd95_decimals: int = 2) -> None:
    """Two-panel table: Dice on top, HD95 below. Plain rows, no multirow."""
    fg_names = [class_names[c] for c in foreground_classes]
    cols = fg_names + ["overall"]
    header_cells = [c.replace("_", r"\_") for c in fg_names] + ["Mean"]

    def cell(stats, marker: str = "", decimals: int = 3) -> str:
        return f"{stats['mean']:.{decimals}f} $\\pm$ {stats['std']:.{decimals}f}{marker}"

    def marker_for(p_two: float) -> str:
        if np.isnan(p_two):
            return ""
        if p_two < 0.01:
            return "$^{**}$"
        if p_two < 0.05:
            return "$^{*}$"
        return ""

    lines = [
        "% Auto-generated by tools/aggregate_seeds.py",
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Test-set Dice ($\uparrow$, higher is better) and HD95 in "
         r"pixels ($\downarrow$, lower is better) on CAMUS "
         f"(mean $\\pm$ std over {n_seeds} seeds with a fixed patient split). "
         r"Significance markers $^{*}p<0.05$, $^{**}p<0.01$ (paired Wilcoxon "
         r"signed-rank, two-sided) vs.\ the baseline.}"),
        r"\label{tab:per_class_metrics}",
        r"\begin{tabular}{ll" + "c" * len(cols) + "}",
        r"\toprule",
        "Metric & Model & " + " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]

    # ---- Dice panel ---------------------------------------------------------
    base_dice = [cell(aggregate[baseline_model][c]["dice_mean"],
                      decimals=dice_decimals)
                 for c in cols]
    prop_dice = [cell(aggregate[proposed_model][c]["dice_mean"],
                      marker_for(pvalues["dice_mean"][c]["p_two"]),
                      decimals=dice_decimals)
                 for c in cols]
    lines.append(r"Dice $\uparrow$       & Baseline U-Net  & " + " & ".join(base_dice) + r" \\")
    lines.append(r"                      & KAN-UNet (ours) & " + " & ".join(prop_dice) + r" \\")

    lines.append(r"\midrule")

    # ---- HD95 panel ---------------------------------------------------------
    base_hd95 = [cell(aggregate[baseline_model][c]["hd95_mean"],
                      decimals=hd95_decimals)
                 for c in cols]
    prop_hd95 = [cell(aggregate[proposed_model][c]["hd95_mean"],
                      marker_for(pvalues["hd95_mean"][c]["p_two"]),
                      decimals=hd95_decimals)
                 for c in cols]
    lines.append(r"HD95 (px) $\downarrow$ & Baseline U-Net  & " + " & ".join(base_hd95) + r" \\")
    lines.append(r"                       & KAN-UNet (ours) & " + " & ".join(prop_hd95) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    out_path.write_text("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# stdout
# -----------------------------------------------------------------------------
def print_summary(pvalues, aggregate, class_names, foreground_classes,
                  baseline_model: str, proposed_model: str) -> None:
    cols = [class_names[i] for i in foreground_classes] + ["overall"]
    print()
    print("Paired-Wilcoxon test summary (proposed vs baseline):")
    print(f"  {'class':18s} {'metric':10s} {'baseline':>10s} {'proposed':>10s} "
          f"{'delta':>10s} {'wins':>7s} {'p_two':>8s} {'p_one':>8s}")
    print("  " + "-" * 84)
    for cls in cols:
        for metric in ("dice_mean", "hd95_mean"):
            b = aggregate[baseline_model].get(cls, {}).get(metric, {}).get("mean", float("nan"))
            p = aggregate[proposed_model].get(cls, {}).get(metric, {}).get("mean", float("nan"))
            delta = p - b
            stats = pvalues[metric].get(cls, {})
            wins = stats.get("n_wins", 0)
            n = stats.get("n_pairs", 0)
            p_two = stats.get("p_two", float("nan"))
            p_one = stats.get("p_one", float("nan"))
            mark = ""
            if not np.isnan(p_two) and p_two < 0.05:
                mark = " *"
            elif not np.isnan(p_one) and p_one < 0.05:
                mark = " (*)"  # one-sided only
            print(f"  {cls:18s} {metric:10s} {b:>10.4f} {p:>10.4f} "
                  f"{delta:>+10.4f} {wins:>3d}/{n:<3d} "
                  f"{p_two:>8.4f} {p_one:>8.4f}{mark}")
    print()
    print("  Legend:  *   = two-sided p < 0.05")
    print("           (*) = one-sided p < 0.05 only "
          "(N=5 ceiling for two-sided = 0.0625)")


# -----------------------------------------------------------------------------
# entry point
# -----------------------------------------------------------------------------
def main(args):
    runs_root = Path(args.runs_root)
    runs = parse_run_dirs(runs_root)
    if not runs:
        print(f"No completed runs found in {runs_root}")
        return

    print(f"Found {len(runs)} completed runs:")
    for model, seed, run_dir in runs:
        print(f"  {model:20s}  seed={seed:3d}  -> {run_dir}")

    class_names, foreground_classes = load_class_metadata(runs)
    if not class_names:
        print("Could not read class_names from any used_config.yaml; "
              "pass --classes manually.")
        if args.classes:
            class_names = ["bg"] + list(args.classes)
            foreground_classes = list(range(1, len(class_names)))

    per_seed, aggregate = aggregate_across_seeds(runs)

    out_dir = runs_root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_aggregate_csv(aggregate, out_dir / "aggregate.csv")
    print(f"\nWrote: {out_dir / 'aggregate.csv'}")

    models_present = set(aggregate.keys())
    if args.baseline_model in models_present and args.proposed_model in models_present:
        pvalues = compute_all_pvalues(
            per_seed, class_names, foreground_classes,
            args.baseline_model, args.proposed_model,
        )

        n_seeds = max(
            len(per_seed[args.baseline_model].get(class_names[foreground_classes[0]], {}).get("dice_mean", [])),
            len(per_seed[args.proposed_model].get(class_names[foreground_classes[0]], {}).get("dice_mean", [])),
        )

        write_latex_table(
            aggregate, pvalues, class_names, foreground_classes,
            out_dir / "paper_table.tex",
            args.baseline_model, args.proposed_model, n_seeds,
        )
        write_paired_tests_csv(
            per_seed, aggregate, pvalues, class_names, foreground_classes,
            args.baseline_model, args.proposed_model,
            out_dir / "paired_tests.csv",
        )
        print(f"Wrote: {out_dir / 'paper_table.tex'}")
        print(f"Wrote: {out_dir / 'paired_tests.csv'}")
        print_summary(pvalues, aggregate, class_names, foreground_classes,
                      args.baseline_model, args.proposed_model)
    else:
        print(f"\n  Skipping LaTeX/paired tests: need both {args.baseline_model} "
              f"and {args.proposed_model} in {runs_root}. "
              f"Found: {sorted(models_present)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", default="./runs")
    p.add_argument("--baseline_model", default="baseline_unet")
    p.add_argument("--proposed_model", default="kan_unet")
    p.add_argument("--classes", nargs="*", default=None,
                   help="Foreground class names (only used if used_config.yaml "
                        "lookup fails).")
    args = p.parse_args()
    main(args)