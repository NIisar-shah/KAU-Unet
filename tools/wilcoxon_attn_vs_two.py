"""Per-sample paired Wilcoxon: attn vs four key comparators.

  - attn vs baseline_unet      : "improves over baseline" at per-sample power
  - attn vs kan_unet           : "attention gates beat vanilla KAN-UNet"
  - attn vs kan_unet_mlp       : "KAN+attention beats MLP bottleneck"
  - attn vs kan_unet_plus      : "boundary BCE negative result" at per-sample power

To add more comparators, just append to COMPARATORS list below.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RUNS = Path("./runs")
SEEDS = [0, 1, 2, 3, 42]
FOREGROUND = [1, 2, 3]
CLASS_NAMES = {1: "LV_endocardium", 2: "LV_myocardium", 3: "left_atrium"}

PROPOSED = "kan_unet_attn"
COMPARATORS = [
    "baseline_unet",
    "kan_unet",
    "kan_unet_mlp",
    "kan_unet_plus",
    # add more here if needed:
    # "kan_unet_no_tanh",
    # "kan_unet_plus_lambda02",
    # "kan_unet_plus_ds",
]


def load_per_sample_seed_avg(model: str) -> pd.DataFrame:
    frames = []
    for s in SEEDS:
        path = RUNS / f"{model}_seed{s}" / "test_per_sample.csv"
        frames.append(pd.read_csv(path))
    combined = pd.concat(frames, ignore_index=True)
    keys = ["patient", "view", "phase", "class_id"]
    avg = combined.groupby(keys, as_index=False)[["dice", "hd95"]].mean()
    return avg.sort_values(keys).reset_index(drop=True)


def paired_test(a, b, lower_is_better):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 5:
        return dict(n=n, delta=np.nan, wins=0, p_two=np.nan, p_one=np.nan)
    delta = (a - b).mean()
    if lower_is_better:
        wins = int((a < b).sum()); alt = "less"
    else:
        wins = int((a > b).sum()); alt = "greater"
    try:
        _, p_two = wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        p_two = np.nan
    try:
        _, p_one = wilcoxon(a, b, alternative=alt, zero_method="wilcox")
    except ValueError:
        p_one = np.nan
    return dict(n=n, delta=delta, wins=wins, p_two=p_two, p_one=p_one)


def compare_pair(name_a, name_b):
    A = load_per_sample_seed_avg(name_a)
    B = load_per_sample_seed_avg(name_b)
    keys = ["patient", "view", "phase", "class_id"]
    if not A[keys].equals(B[keys]):
        print(f"  KEY MISMATCH between {name_a} and {name_b}; aborting.")
        return
    print(f"\n=== {name_a} vs {name_b} ===")
    print(f"  {'group':18s} {'metric':6s} {'mean_A':>8s} {'mean_B':>8s} "
          f"{'delta':>9s} {'wins':>10s} {'p_two':>10s} {'p_one':>10s}")
    fg_A = A[A["class_id"].isin(FOREGROUND)]
    fg_B = B[B["class_id"].isin(FOREGROUND)]
    for metric in ["dice", "hd95"]:
        lower = (metric == "hd95")
        r = paired_test(fg_A[metric].values, fg_B[metric].values, lower)
        m_a, m_b = fg_A[metric].mean(), fg_B[metric].mean()
        stars = (" **" if r["p_two"] < 0.01 else
                 " *"  if r["p_two"] < 0.05 else
                 " †"  if r["p_one"] < 0.05 else "")
        print(f"  {'overall':18s} {metric:6s} {m_a:8.4f} {m_b:8.4f} "
              f"{r['delta']:+9.4f}  {r['wins']}/{r['n']:<5d}  "
              f"{r['p_two']:10.5f} {r['p_one']:10.5f}{stars}")
    for cls_id in FOREGROUND:
        cname = CLASS_NAMES[cls_id]
        cA = A[A["class_id"] == cls_id]
        cB = B[B["class_id"] == cls_id]
        for metric in ["dice", "hd95"]:
            lower = (metric == "hd95")
            r = paired_test(cA[metric].values, cB[metric].values, lower)
            m_a, m_b = cA[metric].mean(), cB[metric].mean()
            stars = (" **" if r["p_two"] < 0.01 else
                     " *"  if r["p_two"] < 0.05 else
                     " †"  if r["p_one"] < 0.05 else "")
            print(f"  {cname:18s} {metric:6s} {m_a:8.4f} {m_b:8.4f} "
                  f"{r['delta']:+9.4f}  {r['wins']}/{r['n']:<5d}  "
                  f"{r['p_two']:10.5f} {r['p_one']:10.5f}{stars}")


def main():
    for comparator in COMPARATORS:
        compare_pair(PROPOSED, comparator)
    print()
    print("Legend:  ** = two-sided p<0.01;  * = two-sided p<0.05;  "
          "† = one-sided p<0.05 only")
    print("N=900 = 3 foreground classes × 75 patients × 2 views × 2 phases")
    print("N=300 = 75 × 2 × 2  (per class)")


if __name__ == "__main__":
    main()