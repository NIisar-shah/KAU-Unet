# Manuscript numerical outputs

This directory is reserved for the non-identifiable numerical outputs that
support the manuscript tables and figures. CAMUS images and masks must not be
placed here.

Before journal submission, deposit the final outputs using this structure:

```text
results/
├── five_seed/
│   ├── aggregate.csv
│   ├── paired_tests.csv
│   └── per_seed_summary.csv
├── patient_clustered/
│   ├── wilcoxon_corrected.csv
│   └── wilcoxon_corrected.tex
├── ensemble_tta/
│   └── aggregate.csv
├── efficiency/
│   ├── measured_latency.csv
│   └── model_cost.tex
└── figures/
    └── source_values.csv
```

Each deposited file should record the generating script, Git commit, model
names, seeds, split seed, and creation date. Values must be copied from actual
run outputs; do not manually reconstruct or round source CSV files from the
manuscript tables.

The repository currently contains code and configurations but no claim is made
that every manuscript number has been independently regenerated from a fresh
environment. Update this statement when the verification run is complete.
