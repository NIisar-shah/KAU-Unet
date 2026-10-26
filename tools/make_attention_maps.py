#!/usr/bin/env python3
"""
Generate attention_maps.pdf — attention-gate figure for the KAN-UNet-Attn paper.

Updated typography for journal submission: serif (Times New Roman / Liberation
Serif fallback) family, STIX math fontset so the alpha subscripts render in
serif, larger and bolder column titles, larger row labels.

Shows attention-gate activations for three representative held-out test
patients in a 3-row x 6-column grid.
"""

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import yaml
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =============================================================================
# TYPOGRAPHY -- set BEFORE any figure is created
# =============================================================================
#
# Font-family resolution order:
#   1. Times New Roman      -- Microsoft, via ttf-mscorefonts-installer.
#   2. Liberation Serif     -- Linux metric-equivalent, free license.
#   3. DejaVu Serif         -- always available with matplotlib.
# Any of these will give a Times-family look; matplotlib falls through
# transparently if the earlier ones are missing.

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset":  "stix",   # serif math; renders alpha-subscripts cleanly
    "mathtext.rm":       "serif",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "pdf.fonttype":      42,       # embed as TrueType (editable in Illustrator)
    "ps.fonttype":       42,
})


# =============================================================================
# CONFIGURATION
# =============================================================================

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_ROOT = WORKSPACE / "datasets" / "CAMUS_NIFTI"
CKPT_PATH = WORKSPACE / "runs" / "kan_unet_attn_seed42" / "best.pt"
OUT_PATH  = WORKSPACE / "runs" / "figures" / "attention_maps.pdf"

CASES = [
    dict(patient="patient0224", view="4CH", phase="ES",
         row_label="Clean acquisition\npatient 0224\n4CH, ES"),
    dict(patient="patient0176", view="2CH", phase="ED",
         row_label="LA rescue\npatient 0176\n2CH, ED"),
    dict(patient="patient0334", view="4CH", phase="ES",
         row_label="Strong LA rescue\npatient 0334\n4CH, ES"),
]

IMG_SIZE = 256

CLASS_COLORS = np.array([
    [0,   0,   0,   0  ],
    [220, 38,  38,  180],
    [34,  197, 94,  180],
    [59,  130, 246, 180],
], dtype=np.float32) / 255.0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_camus_case(patient, view, phase, size=IMG_SIZE):
    pdir = DATA_ROOT / patient
    img_path = pdir / f"{patient}_{view}_{phase}.nii.gz"
    gt_path  = pdir / f"{patient}_{view}_{phase}_gt.nii.gz"

    if not img_path.exists():
        raise FileNotFoundError(f"Missing image: {img_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing label: {gt_path}")

    img_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path))).astype(np.float32)
    if img_arr.ndim == 3:
        img_arr = img_arr[0]

    gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))).astype(np.int64)
    if gt_arr.ndim == 3:
        gt_arr = gt_arr[0]

    img_t = torch.from_numpy(img_arr)[None, None].float()
    img_t = F.interpolate(img_t, size=(size, size), mode="bilinear", align_corners=False)
    gt_t  = torch.from_numpy(gt_arr.astype(np.float32))[None, None]
    gt_t  = F.interpolate(gt_t, size=(size, size), mode="nearest")

    raw_img = img_t[0, 0].numpy()
    raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
    gt_out  = gt_t[0, 0].long().numpy()

    img_tensor = torch.from_numpy(raw_img)[None, None].repeat(1, 3, 1, 1).float()
    return img_tensor, gt_out, raw_img


# =============================================================================
# MODEL LOADING + ATTENTION-MASK CAPTURE
# =============================================================================

def load_model(ckpt_path, device):
    sys.path.insert(0, str(WORKSPACE))
    from models import build_model

    cfg_path = ckpt_path.parent / "used_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["params"]["out_channels"] = cfg.get("num_classes", 4)
    model = build_model(cfg)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        state = (ckpt.get("model")
                 or ckpt.get("state_dict")
                 or ckpt.get("model_state_dict")
                 or ckpt)
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys:", missing[:5], "...")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys:", unexpected[:5], "...")
    return model.eval().to(device)


def register_attention_hooks(model, gate_class_name="AttentionGate"):
    captured = {}
    handles = []

    for name, mod in model.named_modules():
        if mod.__class__.__name__ != gate_class_name:
            continue
        if not hasattr(mod, "psi"):
            print(f"[warn] {name} has no `psi` attribute, skipping")
            continue

        def make_hook(gate_name):
            def hook(_module, _inp, out):
                captured[gate_name] = out.detach().cpu()
            return hook

        handles.append(mod.psi.register_forward_hook(make_hook(name)))
    return captured, handles


# =============================================================================
# VISUALIZATION
# =============================================================================

def overlay_segmentation(img_gray, seg_classes):
    rgb     = np.stack([img_gray, img_gray, img_gray], axis=-1)
    overlay = CLASS_COLORS[seg_classes]
    alpha   = overlay[..., 3:4]
    rgb     = rgb * (1 - alpha) + overlay[..., :3] * alpha
    return np.clip(rgb, 0, 1)


def upsample_to(arr, size=IMG_SIZE):
    t = torch.as_tensor(np.asarray(arr), dtype=torch.float32)
    while t.ndim < 4:
        t = t.unsqueeze(0)
    if t.shape[-2:] != (size, size):
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] Loading model from {CKPT_PATH}")
    model = load_model(CKPT_PATH, device)
    captured, handles = register_attention_hooks(model)
    print(f"[info] Hooked {len(handles)} attention gates.")
    if len(handles) != 3:
        print(f"[warn] Expected 3 gates, got {len(handles)}.")

    rows = []
    for case in CASES:
        print(f"[info] Processing {case['patient']} {case['view']} {case['phase']}")
        img, gt, raw = load_camus_case(case["patient"], case["view"], case["phase"])
        captured.clear()

        with torch.no_grad():
            logits = model(img.to(device))
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = logits.argmax(dim=1)[0].cpu().numpy()

        for k, v in sorted(captured.items(), key=lambda kv: kv[1].shape[-1]):
            print(f"  {k}: shape={tuple(v.shape)}, "
                  f"min={v.min():.3f}, mean={v.mean():.3f}, "
                  f"max={v.max():.3f}, std={v.std():.3f}")

        masks = sorted(captured.items(), key=lambda kv: kv[1].shape[-1])
        alpha_3 = upsample_to(masks[0][1]) if len(masks) > 0 else np.zeros((IMG_SIZE, IMG_SIZE))
        alpha_2 = upsample_to(masks[1][1]) if len(masks) > 1 else np.zeros((IMG_SIZE, IMG_SIZE))
        alpha_1 = upsample_to(masks[2][1]) if len(masks) > 2 else np.zeros((IMG_SIZE, IMG_SIZE))

        rows.append(dict(case=case, raw=raw, gt=gt, pred=pred,
                         a3=alpha_3, a2=alpha_2, a1=alpha_1))

    # ----- Figure layout -----
    n_rows = len(rows)
    n_cols = 6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols, 2.0 * n_rows),
        gridspec_kw=dict(wspace=0.04, hspace=0.10),
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Input",
        "Ground truth",
        "Prediction",
        r"$\alpha_{3}$ (deep)",
        r"$\alpha_{2}$ (mid)",
        r"$\alpha_{1}$ (shallow)",
    ]

    for r, row in enumerate(rows):
        ax_row = axes[r]

        ax_row[0].imshow(row["raw"], cmap="gray", vmin=0, vmax=1)
        ax_row[1].imshow(overlay_segmentation(row["raw"], row["gt"]))
        ax_row[2].imshow(overlay_segmentation(row["raw"], row["pred"]))

        for j, key in enumerate(["a3", "a2", "a1"]):
            mask = row[key]
            lo, hi = np.percentile(mask, [2, 98])
            mask_vis = np.clip((mask - lo) / (hi - lo + 1e-8), 0, 1)
            ax_row[3 + j].imshow(mask_vis, cmap="viridis", vmin=0, vmax=1)

        for c in range(n_cols):
            ax_row[c].set_xticks([])
            ax_row[c].set_yticks([])
            for spine in ax_row[c].spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("0.3")

        # ROW LABELS -- larger, bold, more pad so multi-line text breathes.
        ax_row[0].set_ylabel(
            row["case"]["row_label"],
            rotation=0, ha="right", va="center",
            fontsize=11, labelpad=70, fontweight="bold",
        )

        # COLUMN TITLES -- only on top row; larger, bold, more pad.
        if r == 0:
            for c, t in enumerate(col_titles):
                ax_row[c].set_title(t, fontsize=13, fontweight="bold", pad=8)

    # Class legend at the bottom.
    legend_handles = [
        Patch(facecolor=CLASS_COLORS[1, :3], alpha=0.7, label="LV endocardium"),
        Patch(facecolor=CLASS_COLORS[2, :3], alpha=0.7, label="LV myocardium"),
        Patch(facecolor=CLASS_COLORS[3, :3], alpha=0.7, label="Left atrium"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.02), fontsize=11, frameon=False,
    )

    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
    print(f"[done] Saved {OUT_PATH}")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()