# Manuscript numerical outputs

This directory contains non-identifiable numerical outputs supporting the
manuscript tables, statistical analyses, and selected figures. CAMUS images,
masks, and trained checkpoints are not redistributed.

## Files

- `five_seed/aggregate.csv`: mean and standard deviation across seeds
  `42, 0, 1, 2, 3` for the revised five-model ablation.
- `five_seed/per_seed_test_summary.csv`: evaluation summaries for every
  model/seed pair in the revised five-model ablation.
- `patient_clustered/wilcoxon_corrected.csv`: KAU-Net paired comparisons
  against U-Net, KAN-UNet without tanh, MLP-bottleneck U-Net, and KAN-UNet.
  Results use a patient-clustered bootstrap with 5,000 resamples, bootstrap
  seed 12345, and Benjamini-Hochberg FDR correction across 32 comparisons.
- `ensemble_tta/aggregate.csv`: five-seed ensemble with horizontal-flip TTA
  for the four architectures reported in the ensemble table.
- `efficiency/measured_efficiency.csv`: parameter count, FLOPs, latency,
  memory, and FPS derived from measured mean latency on the reported GPU.
- `figures/data_efficiency_source_data.csv`: inferential source values for
  the reduced-training-data analysis.
- `figures/stratified_kau_improvement_source_data.csv`: source values for the
  stratified KAU-Net improvement map.

## Provenance

The five-seed and ensemble files were collected from the original run
directories. The clustered-statistics file was regenerated from the original
per-sample CSV outputs after aligning the comparator set with the revised
five-model manuscript. The efficiency values correspond to a single
`256 x 256 x 3` input with batch size 1, 10 warmup passes, and 100 measured
passes on an NVIDIA Quadro GV100 GPU.

Model registry keys used in these files are:

| Registry key | Manuscript name |
|---|---|
| `baseline_unet` | U-Net |
| `kan_unet_no_tanh` | KAN-UNet (no tanh) |
| `kan_unet_mlp` | MLP-bottleneck U-Net |
| `kan_unet` | KAN-UNet |
| `kan_unet_attn` | KAU-Net |

Raw CAMUS data remain subject to the original provider's access and usage
conditions.
