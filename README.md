# KAU-Net — KAN-UNet-Attn for cardiac ultrasound segmentation

**Research implementation: code prepared for review; not yet a verified release.**

KAU-Net is a U-Net encoder–decoder for four-class 2D echocardiographic segmentation
(background, left-ventricular endocardium, myocardium and left atrium).
Its bottleneck uses a tokenized Kolmogorov–Arnold Network (KAN) with
`tanh` bounding on **only** the B-spline path. Three additive attention
gates filter encoder skip features using decoder-derived signals.

> **Paper status:** Insert the final paper title, DOI, and journal details after
> acceptance/publication. Do not describe this repository as a peer-reviewed
> implementation before the paper is published.

## What's included

- `models/kan_layer.py`: KANLinear with independent SiLU base and spline paths.
- `models/kan_blocks.py`: tokenized KAN bottleneck with LayerNorm and spatial refinement.
- `models/kan_unet_attn.py`: complete KAU-Net architecture (`kan_unet_attn` registry key).
- `models/kan_unet.py`, `models/mlp_block.py`, `models/baseline_unet.py`:
  relevant architectural controls.
- `train.py`, `evaluate.py`, `engine/`, `data/`, `utils/`:
  common CAMUS data, training, and evaluation pipeline.
- `configs/`: example configurations (split seed 42).
- `tools/run_seeds.py`, `tools/aggregate_seeds.py`,
  `tools/ensemble_tta_evaluate.py`, `tools/Patient_clustered_stats.py`:
  multi-seed and statistical evaluation scripts.

## Reproducibility setup

1. Use Python 3.10+ with a compatible PyTorch install (CPU or CUDA).
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Download CAMUS **from its original provider**, subject to its usage terms:
   https://www.creatis.insa-lyon.fr/Challenge/camus/
4. **Data format:** configurations expect the CAMUS NIfTI conversion (`.nii.gz`),
   structured as patient directories. Update `data.root` in YAML to match your
   files. The input loader supports additional formats, but do not assume the
   provider's raw files are identical to these converted inputs.
5. Confirm image and mask label mappings (0=background, 1=LV endocardium,
   2=LV myocardium, 3=left atrium). Check the loaded dataset before training.

## Minimal model smoke test (no CAMUS data required)

```bash
python -c "import torch; from models import build_model; m=build_model({'model':{'name':'kan_unet_attn','params':{'in_channels':3,'out_channels':4,'base_channels':32}}}); m.eval(); x=torch.randn(1,3,64,64); print(m(x).shape)"
```

Expected shape: `torch.Size([1, 4, 64, 64])`.

## Train and evaluate the proposed KAU-Net

```bash
python train.py --config configs/kau_net.yaml
python evaluate.py --config configs/kau_net.yaml \
  --checkpoint runs/kau_net_seed42/best.pt
```

The above paths are **examples**; `configs/kau_net.yaml` uses the local placeholder
`./datasets/CAMUS_NIFTI`. Set it to the actual location before training.

## Paired five-seed protocol

```bash
python tools/run_seeds.py --base_config configs/default.yaml \
    --models baseline_unet kan_unet_no_tanh kan_unet kan_unet_mlp kan_unet_attn \
    --seeds 42 0 1 2 3 --runs_root runs
python tools/aggregate_seeds.py --runs_root runs
python tools/Patient_clustered_stats.py
```

The patient split is determined by `data.split_seed` in the base YAML, independent
of the initialization seed. The cluster-bootstrap script expects per-sample
CSV files from these runs and includes comparisons to `kan_unet_plus`; consult
its `COMPARATORS` constant if you run only the five architectures above.

## Design and scope

- The B-spline support interval depends on the **extended knot vector**, not
  only the nominal `[-1, 1]` grid. Out-of-grid inputs are not automatically
  all-zero B-spline responses.
- The measured results reported in the manuscript are **not independently
  reproduced from this ZIP**; raw CAMUS images, trained checkpoints and per-seed
  CSV outputs are not included in the uploaded folder.
- The repository records a research implementation, **not a clinical device**.
  Timing on one GPU is not evidence of end-to-end clinical real-time performance.

## Data and code availability

The **Cardiac Acquisitions for Multi-structure Ultrasound Segmentation (CAMUS)**
dataset is provided by its original custodians and is **not redistributed**
in this repository. Researchers should obtain it from the
[official CAMUS dataset page](https://www.creatis.insa-lyon.fr/Challenge/camus/)
and comply with its access and usage conditions. The training and evaluation
scripts expect a local CAMUS installation with the image and mask labels
described in [Reproducibility setup](#reproducibility-setup). Set `data.root`
in the YAML configuration to the local dataset directory.

This repository contains the KAU-Net source code and the associated
training, inference, ablation, evaluation, and statistical-analysis
scripts. In particular, `configs/kau_net.yaml` specifies the proposed
four-class model and training configuration; `train.py` and `evaluate.py`
provide the training and held-out evaluation entry points;
`tools/run_seeds.py`, `tools/aggregate_seeds.py`, and
`tools/Patient_clustered_stats.py` support the multi-seed and
patient-clustered comparison procedures. Installation requirements,
data organization, model smoke testing, and example commands are
documented above.

The source code is provided under the **Apache License, Version 2.0**
(see [LICENSE](LICENSE)). The repository is **currently private** while
the manuscript and release materials are being finalized; it is intended
to be made publicly available at
[github.com/NIisar-shah/KAU-Unet](https://github.com/NIisar-shah/KAU-Unet)
when the authors authorize the release. The repository does not include
CAMUS images, pretrained checkpoints, or original per-seed experiment
outputs. Accordingly, reproducing the reported numerical results requires
obtaining the dataset and rerunning the documented experiments; inclusion
of executable code should not be interpreted as an independently
verified reproduction of every result.

## Citation

If you use, modify, or build upon KAU-Net in academic research, please
acknowledge the software and cite the associated research article once its
bibliographic details are available. Until an article DOI is assigned, the
following is a **provisional software citation**, not a claim of journal
publication:

> Sherazi, S. N. Y., Hu, Y., and Abbas, M. (2026).
> *KAN-UNet-Attn (KAU-Net): support-aware KAN bottleneck with gated skip
> connections for multi-structure echocardiographic segmentation*
> [Computer software]. GitHub.
> https://github.com/NIisar-shah/KAU-Unet

Machine-readable citation metadata is provided in
[CITATION.cff](CITATION.cff). After the associated article is published,
the authors will update the citation metadata and this section with the
journal reference and DOI. Citing the paper does not replace compliance
with the software license or any third-party licenses.
