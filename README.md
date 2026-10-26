# KAU-Net — KAN-UNet-Attn for cardiac ultrasound segmentation

KAU-Net is a U-Net encoder–decoder for four-class 2D echocardiographic
segmentation: background, left-ventricular endocardium, myocardium, and left
atrium. Its bottleneck uses a tokenized Kolmogorov–Arnold Network (KAN), with
`tanh` bounding applied only to the B-spline path. Three additive attention
gates filter encoder skip features using decoder-derived signals.

This repository provides the reference research implementation and
reproducibility workflow associated with the KAU-Net manuscript. It is not a
clinical device.

## Repository contents

- `models/kan_unet_attn.py`: proposed KAU-Net architecture
  (`kan_unet_attn` registry key).
- `models/baseline_unet.py`, `models/kan_unet.py`, and
  `models/mlp_block.py`: controlled comparison architectures.
- `train.py`, `evaluate.py`, `engine/`, `data/`, and `utils/`:
  common training and held-out evaluation pipeline.
- `configs/`: fixed experimental configurations using patient split seed 42.
- `tools/run_seeds.py`: paired five-seed experiment runner.
- `tools/aggregate_seeds.py`: across-seed aggregation and paired comparisons.
- `tools/patient_clustered_stats.py`: patient-clustered bootstrap with
  Benjamini-Hochberg FDR correction.
- `tools/ensemble_tta_evaluate.py`: five-seed ensemble with horizontal-flip TTA.
- `tools/model_cost.py`: parameter, FLOP, memory, and latency measurements.
- `tests/`: CPU smoke tests for model construction and deterministic splitting.

## Environment

The manuscript experiments used Python 3.10-compatible code and PyTorch 2.x.
Install a PyTorch build appropriate for the local CPU/CUDA environment first,
then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The minimum compatible package versions are recorded in `requirements.txt`.
Before the archival release, the authors should additionally record the exact
Python, PyTorch, CUDA, GPU, operating-system, and package versions used for the
reported experiments in `ENVIRONMENT.md`.

## CAMUS data

Download CAMUS from its original provider and comply with its terms:

https://www.creatis.insa-lyon.fr/Challenge/camus/

CAMUS images and annotations are not redistributed here. Set `data.root` to
the directory that directly contains the patient folders. The expected layout is:

```text
<CAMUS_ROOT>/
├── patient0001/
│   ├── patient0001_2CH_ED.nii.gz
│   ├── patient0001_2CH_ED_gt.nii.gz
│   ├── patient0001_2CH_ES.nii.gz
│   ├── patient0001_2CH_ES_gt.nii.gz
│   ├── patient0001_4CH_ED.nii.gz
│   ├── patient0001_4CH_ED_gt.nii.gz
│   ├── patient0001_4CH_ES.nii.gz
│   └── patient0001_4CH_ES_gt.nii.gz
├── patient0002/
└── ...
```

If the downloaded archive contains a parent directory such as
`database_nifti/`, point `data.root` to that directory, not to its parent.
The loader also supports CAMUS MetaImage files: set `data.file_ext: .mhd` and
use the same patient-folder naming convention. No format conversion is
performed by this repository.

The label mapping is fixed as 0=background, 1=LV endocardium,
2=LV myocardium, and 3=left atrium.

## Smoke test without CAMUS

```bash
python -m pytest -q tests
```

Or run the model-only check directly:

```bash
python -c "import torch; from models import build_model; m=build_model({'model':{'name':'kan_unet_attn','params':{'in_channels':3,'out_channels':4,'base_channels':32}}}); m.eval(); print(m(torch.randn(1,3,64,64)).shape)"
```

Expected output:

```text
torch.Size([1, 4, 64, 64])
```

## Train and evaluate KAU-Net

Edit `data.root` in `configs/kau_net.yaml`, then run:

```bash
python train.py --config configs/kau_net.yaml
python evaluate.py --config configs/kau_net.yaml \
  --checkpoint runs/kau_net_seed42/best.pt
```

Evaluation writes `test_per_sample.csv` and `test_summary.csv` beside the
checkpoint.

## Reproduce the revised five-model ablation

The revised comparison contains U-Net, KAN-UNet without tanh,
MLP-bottleneck U-Net, KAN-UNet, and KAU-Net:

```bash
python tools/run_seeds.py \
  --base_config configs/default.yaml \
  --models baseline_unet kan_unet_no_tanh kan_unet_mlp kan_unet kan_unet_attn \
  --seeds 42 0 1 2 3 \
  --runs_root runs
```

The initialization seeds are `42 0 1 2 3`. The patient assignment remains
fixed through `data.split_seed: 42`.

Aggregate the runs with KAU-Net as the proposed architecture:

```bash
python tools/aggregate_seeds.py \
  --runs_root runs \
  --baseline_model baseline_unet \
  --proposed_model kan_unet_attn
```

Run the patient-clustered analysis:

```bash
python tools/patient_clustered_stats.py
```

The statistical script compares KAU-Net against the other four architectures
in the revised ablation. It expects all five seeds and the per-sample CSV files
created by `evaluate.py`.

## Ensemble and efficiency evaluation

```bash
python tools/ensemble_tta_evaluate.py \
  --runs_root runs \
  --config configs/default.yaml \
  --variants baseline_unet kan_unet_no_tanh kan_unet_mlp kan_unet kan_unet_attn

python tools/model_cost.py \
  --config configs/default.yaml \
  --models baseline_unet kan_unet kan_unet_attn \
  --batch_size 1 \
  --warmup 10 \
  --iters 100 \
  --out runs/aggregate/model_cost.tex
```

Latency and FPS are hardware- and software-dependent. Report them only together
with the GPU, CUDA, PyTorch, batch size, warmup count, and measurement count.
FLOPs from `fvcore` are a lower bound for custom B-spline operations.

## Numerical outputs and checkpoints

The source repository does not redistribute CAMUS data. Before journal
submission, the authors should attach or archive the numerical outputs
underlying the manuscript tables and figures, including per-seed summaries,
patient-clustered bootstrap results, ensemble summaries, and measured
efficiency results. A recommended layout is documented in `results/README.md`.

If trained checkpoints are promised in the manuscript, distribute them through
a versioned GitHub Release or an archival repository such as Zenodo. Otherwise,
the manuscript availability statement should not promise checkpoints.

## Reproducibility notes

- The fixed patient split is independent of model initialization seeds.
- Images are resized bilinearly and masks with nearest-neighbour interpolation.
- Grayscale frames are replicated to three input channels.
- The B-spline support interval depends on the extended knot vector, not only
  the nominal `[-1, 1]` grid.
- Repository code, configurations, and numerical outputs should be cited with a
  versioned release tag. For the manuscript release, use `v1.0.0`.

## Data and code availability

The CAMUS dataset must be obtained from its original custodians. This
repository contains the KAU-Net source code and the associated training,
inference, ablation, evaluation, efficiency, ensemble, and statistical-analysis
scripts. The source code is licensed under the Apache License, Version 2.0.

## Citation

Until the article DOI is available, use the provisional software citation:

> Sherazi, S. N. Y., Hu, Y., and Abbas, M. (2026).
> *KAN-UNet-Attn (KAU-Net): support-aware KAN bottleneck with gated skip
> connections for multi-structure echocardiographic segmentation*
> [Computer software]. GitHub.
> https://github.com/NIisar-shah/KAU-Unet

Machine-readable citation metadata is provided in `CITATION.cff`. Update it
with the final paper citation and DOI when available.
