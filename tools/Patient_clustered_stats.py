#!/usr/bin/env python3
"""
patient_clustered_stats.py
==========================

Re-analyses the per-sample paired comparisons in Table 4 of the KAN-UNet-Attn
manuscript using statistical methods that respect the nested data structure
flagged by reviewers:

    1. Per-record metrics are averaged across the five training seeds.
    2. Patient-level cluster bootstrap (B = 5000) gives p-values that respect
       the fact that each of the 75 test patients contributes up to 12 records
       (2 views x 2 phases x 3 foreground classes). The unclustered Wilcoxon
       on N = 900 records overstates significance.
    3. Benjamini-Hochberg (BH) FDR correction at q = 0.05 is applied across
       all 32 cells of the table.

Inputs
------
    runs/<variant>_seed<s>/test_per_sample.csv
    with columns: patient, view, phase, class_id, dice, hd95
    and class_id in {1, 2, 3} for the three foreground classes.

Outputs
-------
    runs/aggregate/wilcoxon_corrected.csv   # full table for inspection
    runs/aggregate/wilcoxon_corrected.tex   # paste-ready LaTeX table

Usage
-----
    cd /workspace/kan_fully_sup
    python tools/patient_clustered_stats.py

Adapt-if-needed
---------------
    Adjust REFERENCE, COMPARATORS, and the run-directory naming convention
    below if your runs use different names.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

WORKSPACE = Path(__file__).resolve().parent.parent
RUNS_DIR  = WORKSPACE / "runs"
OUT_CSV   = RUNS_DIR / "aggregate" / "wilcoxon_corrected.csv"
OUT_TEX   = RUNS_DIR / "aggregate" / "wilcoxon_corrected.tex"

# The proposed composite is the reference; everything is reported as
# (reference - comparator).
REFERENCE   = "kan_unet_attn"
COMPARATORS = ["baseline_unet", "kan_unet", "kan_unet_mlp", "kan_unet_plus"]

SEEDS = [42, 0, 1, 2, 3]

# Bootstrap configuration
N_BOOTSTRAP = 5000
RNG_SEED    = 12345
ALPHA_FDR   = 0.05

# Foreground class mapping (matches test_per_sample.csv class_id column)
CLASS_NAMES        = {1: "LV_endo", 2: "LV_myo", 3: "LA"}
FOREGROUND_CLASSES = [1, 2, 3]

# Display strings for the LaTeX table
COMP_LATEX = {
    "baseline_unet": r"vs.\ Baseline",
    "kan_unet":      r"vs.\ KAN-UNet",
    "kan_unet_mlp":  r"vs.\ MLP",
    "kan_unet_plus": r"vs.\ KAN-UNet$^{+}$",
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_per_sample(variant: str) -> pd.DataFrame:
    """Load the 5-seed per-sample CSVs for `variant` and average each metric
    across seeds, returning one row per (patient, view, phase, class_id)."""
    dfs = []
    for seed in SEEDS:
        path = RUNS_DIR / f"{variant}_seed{seed}" / "test_per_sample.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing per-sample CSV: {path}")
        df = pd.read_csv(path)
        for col in ("patient", "view", "phase", "class_id", "dice", "hd95"):
            if col not in df.columns:
                raise KeyError(f"Column '{col}' missing in {path}")
        df["seed"] = seed
        dfs.append(df)

    all_seeds = pd.concat(dfs, ignore_index=True)
    keys = ["patient", "view", "phase", "class_id"]
    agg = (all_seeds.groupby(keys, as_index=False)[["dice", "hd95"]]
                    .mean(numeric_only=True))
    return agg


# =============================================================================
# STATISTICS
# =============================================================================

def cluster_bootstrap(diffs: np.ndarray,
                      patients: np.ndarray,
                      n_boot: int = N_BOOTSTRAP,
                      rng: np.random.Generator | None = None
                      ) -> tuple[float, float, float, float]:
    """Patient-level cluster bootstrap of the mean of `diffs`.

    Returns (mean_delta, ci_low, ci_high, p_two_sided).

    `p_two_sided` is the percentile two-sided bootstrap p-value:
        p = 2 * min(Pr[boot mean >= 0], Pr[boot mean <= 0])
    capped at 1.0.

    NaN values in `diffs` (e.g., undefined HD95) are dropped together with
    their patient labels.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    mask = ~np.isnan(diffs)
    diffs = np.asarray(diffs)[mask]
    patients = np.asarray(patients)[mask]

    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan"), 1.0

    unique_patients = np.unique(patients)
    n_pat = len(unique_patients)

    # Pre-index diffs by patient for fast bootstrap concatenation
    patient_to_diffs = {p: diffs[patients == p] for p in unique_patients}

    observed_mean = float(diffs.mean())
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(unique_patients, size=n_pat, replace=True)
        boot_means[b] = np.concatenate([patient_to_diffs[p] for p in sampled]).mean()

    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    p_above = float((boot_means >= 0).mean())
    p_below = float((boot_means <= 0).mean())
    p_two = float(min(1.0, 2 * min(p_above, p_below)))

    return observed_mean, float(ci_lo), float(ci_hi), p_two


def bh_correct(p_values: np.ndarray, alpha: float = ALPHA_FDR
               ) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR adjustment (step-up).

    Returns (adjusted_p_values, significant_mask) in the original ordering.
    `significant_mask[i] = True` iff adjusted_p_values[i] <= alpha.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked_p = p[order]

    adjusted_ranked = np.empty(m)
    cum_min = 1.0
    for k in range(m - 1, -1, -1):
        cum_min = min(cum_min, ranked_p[k] * m / (k + 1))
        adjusted_ranked[k] = cum_min

    adjusted = np.empty(m)
    adjusted[order] = adjusted_ranked
    return adjusted, adjusted <= alpha


# =============================================================================
# DIFF COMPUTATION FOR ONE CELL
# =============================================================================

def compute_paired_diffs(ref_df: pd.DataFrame,
                         comp_df: pd.DataFrame,
                         metric: str,
                         class_filter: list[int]
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Return per-record paired differences (ref - comp) and patient labels."""
    keys = ["patient", "view", "phase", "class_id"]
    ref  = ref_df [ref_df ["class_id"].isin(class_filter)]
    comp = comp_df[comp_df["class_id"].isin(class_filter)]
    merged = ref.merge(comp, on=keys, suffixes=("_ref", "_comp"))
    diffs = (merged[f"{metric}_ref"] - merged[f"{metric}_comp"]).to_numpy()
    return diffs, merged["patient"].to_numpy()


# =============================================================================
# LATEX FORMATTING
# =============================================================================

def fmt_delta_cell(mean_delta: float, metric: str,
                   p_raw: float, p_corr: float, alpha: float = ALPHA_FDR) -> str:
    """Format a single cell of the corrected Table 4 as LaTeX."""
    if metric == "dice":
        s = f"{mean_delta:+.4f}"
    else:  # hd95
        s = f"{mean_delta:+.2f}"

    if p_corr < 0.01:
        return s + r"$^{\ast\ast}$"
    if p_corr < alpha:
        return s + r"$^{\ast}$"
    if p_raw < alpha:
        return s + r"$^{(\circ)}$"
    return s


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("[info] Loading per-sample CSVs for variants:")
    for v in [REFERENCE] + COMPARATORS:
        print(f"        - {v}")
    data = {v: load_per_sample(v) for v in [REFERENCE] + COMPARATORS}
    n_pat = data[REFERENCE]["patient"].nunique()
    n_rec = len(data[REFERENCE])
    print(f"[info] {n_rec} records / {n_pat} test patients per variant")

    rng = np.random.default_rng(RNG_SEED)

    class_subsets = [
        ("Overall", FOREGROUND_CLASSES),
        ("LV_endo", [1]),
        ("LV_myo",  [2]),
        ("LA",      [3]),
    ]
    metrics = ["dice", "hd95"]

    rows = []
    for comp_variant in COMPARATORS:
        for class_label, class_ids in class_subsets:
            for metric in metrics:
                diffs, patients = compute_paired_diffs(
                    data[REFERENCE], data[comp_variant], metric, class_ids
                )
                mean_d, ci_lo, ci_hi, p_raw = cluster_bootstrap(
                    diffs, patients, n_boot=N_BOOTSTRAP, rng=rng
                )
                rows.append({
                    "comparator": comp_variant,
                    "class":      class_label,
                    "metric":     metric,
                    "n_records":  int(np.sum(~np.isnan(diffs))),
                    "n_patients": int(len(np.unique(patients))),
                    "mean_delta": mean_d,
                    "ci_lo":      ci_lo,
                    "ci_hi":      ci_hi,
                    "p_raw":      p_raw,
                })
        print(f"[info] Finished comparisons vs {comp_variant}")

    results = pd.DataFrame(rows)
    p_corr, sig_bh = bh_correct(results["p_raw"].to_numpy(), alpha=ALPHA_FDR)
    results["p_bh"]     = p_corr
    results["sig_bh"]   = sig_bh
    results["sig_raw"]  = results["p_raw"] < ALPHA_FDR

    # --- Save CSV ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\n[info] Wrote {OUT_CSV}")

    # --- Stdout summary ---
    print("\n" + "=" * 100)
    print(f"Patient-clustered bootstrap (B={N_BOOTSTRAP}, seed={RNG_SEED})  "
          f"+  Benjamini-Hochberg FDR at q={ALPHA_FDR}")
    print("=" * 100)
    for comp in COMPARATORS:
        sub = results[results["comparator"] == comp]
        print(f"\n--- KAN-UNet-Attn vs {comp} ---")
        print(f"{'Class':<10} {'Metric':<6} {'Δ':>11} {'95% CI':>26} "
              f"{'p_raw':>9} {'p_BH':>9} {'sig':>6}")
        for _, r in sub.iterrows():
            ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
            tag = ""
            if r["sig_bh"]:                tag = "  BH"
            elif r["sig_raw"]:             tag = "  raw"
            print(f"{r['class']:<10} {r['metric']:<6} {r['mean_delta']:+11.4f} "
                  f"{ci:>26} {r['p_raw']:>9.4f} {r['p_bh']:>9.4f} {tag:>6}")

    n_total      = len(results)
    n_sig_raw    = int(results["sig_raw"].sum())
    n_sig_bh     = int(results["sig_bh"].sum())
    n_lost       = int((results["sig_raw"] & ~results["sig_bh"]).sum())

    print(f"\n{'-' * 60}")
    print(f"Total cells:                                {n_total}")
    print(f"Significant before correction (p_raw<{ALPHA_FDR}):     {n_sig_raw}")
    print(f"Significant after BH-FDR (q={ALPHA_FDR}):           {n_sig_bh}")
    print(f"Cells lost to BH correction:                {n_lost}")

    # --- Build LaTeX ---
    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Per-sample paired comparisons against KAN-UNet-Attn under "
        r"a patient-level cluster bootstrap (B$=5000$ resamples of the $N=75$ "
        r"test patients) with Benjamini--Hochberg FDR correction at $q=0.05$ "
        r"applied across all 32 cells. Cells show the mean per-record "
        r"difference $\Delta=$ KAN-UNet-Attn $-$ reference. "
        r"$^{\ast\ast}$ $q<0.01$; $^{\ast}$ $q<0.05$; "
        r"$^{(\circ)}$ unadjusted $p<0.05$ but does not survive FDR correction. "
        r"Negative $\Delta$ on Dice or positive $\Delta$ on HD95 indicate "
        r"KAN-UNet-Attn underperforms the reference and are reported "
        r"transparently.}"
    )
    lines.append(r"\label{tab:wilcoxon_corrected}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    header = " & ".join(["Class", "Metric"] + [COMP_LATEX[c] for c in COMPARATORS]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for class_label, _ in class_subsets:
        for metric in metrics:
            arrow = r"$\uparrow$" if metric == "dice" else r"$\downarrow$"
            metric_label = "Dice " + arrow if metric == "dice" else "HD95 " + arrow
            cells = [class_label, metric_label]
            for comp in COMPARATORS:
                r = results[(results["comparator"] == comp)
                            & (results["class"] == class_label)
                            & (results["metric"] == metric)].iloc[0]
                cells.append(fmt_delta_cell(
                    r["mean_delta"], metric, r["p_raw"], r["p_bh"]
                ))
            lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    OUT_TEX.write_text("\n".join(lines))
    print(f"\n[info] Wrote {OUT_TEX}")

    # --- Suggested narrative ---
    print("\n" + "=" * 100)
    print("Suggested narrative updates based on what survived BH correction:")
    print("=" * 100)
    for comp in COMPARATORS:
        survivors = results[(results["comparator"] == comp) & results["sig_bh"]]
        if len(survivors) == 0:
            print(f"  vs {comp}: NO comparisons survive BH at q={ALPHA_FDR}.")
            continue
        cells = [f"{r['class']} {r['metric']}" for _, r in survivors.iterrows()]
        print(f"  vs {comp}: SURVIVES BH on {len(survivors)} cell(s):")
        for c in cells:
            print(f"      - {c}")


if __name__ == "__main__":
    main()