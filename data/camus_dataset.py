"""CAMUS dataset for fully supervised LV / myocardium / LA segmentation.

Expected directory layout (the standard CAMUS NIfTI distribution from
CREATIS, also produced by ``database_nifti/``):

    root/
      patient0001/
        patient0001_2CH_ED.nii.gz
        patient0001_2CH_ED_gt.nii.gz
        patient0001_2CH_ES.nii.gz
        patient0001_2CH_ES_gt.nii.gz
        patient0001_4CH_ED.nii.gz
        patient0001_4CH_ED_gt.nii.gz
        patient0001_4CH_ES.nii.gz
        patient0001_4CH_ES_gt.nii.gz
        Info_2CH.cfg              # metadata (not read)
        Info_4CH.cfg
      patient0002/
      ...

The MetaImage ``.mhd`` layout is also supported -- pass ``file_ext='.mhd'``.

CAMUS ground-truth labels:
    0 = background
    1 = LV endocardium
    2 = LV myocardium
    3 = left atrium

This loader returns **integer label maps** (long tensors). The behaviour is
selected by ``task``:

    task="binary":      mask values in {0, 1}, where 1 = ``target_label``.
    task="multiclass":  mask values in {0..3}, the full CAMUS label set.

The dataset yields:
    {
        "ed_frame": [3, H, W] float tensor in [0, 1],
        "es_frame": [3, H, W] float tensor in [0, 1],
        "ed_mask":  [H, W]    long tensor of class indices,
        "es_mask":  [H, W]    long tensor of class indices,
        "patient":  str,
        "view":     str ("2CH" or "4CH"),
    }
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import SimpleITK as sitk
except ImportError as e:
    raise ImportError(
        "SimpleITK is required for CAMUS .mhd loading. "
        "Install with: pip install SimpleITK"
    ) from e


def _read_image(path: Path) -> np.ndarray:
    """Read any ITK-supported image format (NIfTI ``.nii.gz`` / ``.nii``,
    MetaImage ``.mhd``, MHA, NRRD, ...). Squeezes a leading singleton
    dimension (common when CAMUS 2-D frames are stored as 3-D 1-slice
    NIfTI volumes)."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _to_label_map(mask: np.ndarray, task: str, target_label: int = 1) -> np.ndarray:
    """Convert raw CAMUS GT to an integer label map.

    Binary mode produces {0, 1} with foreground = ``target_label``.
    Multi-class mode keeps the original CAMUS labelling {0, 1, 2, 3}.
    """
    if task == "binary":
        return (mask == target_label).astype(np.int64)
    return mask.astype(np.int64)


def _resize(arr: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    """Resize 2-D array to (size, size) using nearest for masks, bilinear for images."""
    import cv2
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(arr, (size, size), interpolation=interp)


def _normalise(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


class CamusDataset(Dataset):
    """Patient/view-paired CAMUS dataset.

    Each item corresponds to one (patient, view) tuple and bundles both ED and
    ES frames + masks, mirroring the structure your trainer expects.
    """

    def __init__(
        self,
        root: str,
        patient_ids: List[str],
        views: Tuple[str, ...] = ("2CH", "4CH"),
        image_size: int = 256,
        target_label: int = 1,
        augment: bool = False,
        task: str = "multiclass",
        file_ext: str = ".nii.gz",
    ):
        if task not in ("binary", "multiclass"):
            raise ValueError(f"task must be 'binary' or 'multiclass', got {task!r}")
        self.root = Path(root)
        self.image_size = image_size
        self.target_label = target_label
        self.augment = augment
        self.task = task
        self.file_ext = file_ext

        self.items: List[Tuple[str, str]] = []
        for pid in patient_ids:
            for v in views:
                ed = self.root / pid / f"{pid}_{v}_ED{file_ext}"
                es = self.root / pid / f"{pid}_{v}_ES{file_ext}"
                if ed.exists() and es.exists():
                    self.items.append((pid, v))

        if len(self.items) == 0:
            example_pid = patient_ids[0] if patient_ids else "patientXXXX"
            expected = (
                self.root / example_pid
                / f"{example_pid}_{views[0]}_ED{file_ext}"
            )
            raise RuntimeError(
                f"No CAMUS samples found under {self.root}.\n"
                f"  Expected files like:  {expected}\n"
                f"  patient_ids ({len(patient_ids)}): {patient_ids[:3]}...\n"
                f"  views: {views}\n"
                f"  file_ext: {file_ext!r}\n"
                "Verify `data.root`, that patient folders sit directly under it, "
                "and that `data.file_ext` matches your actual files."
            )

    def __len__(self) -> int:
        return len(self.items)

    def _load_pair(self, pid: str, view: str):
        d = self.root / pid
        ext = self.file_ext
        ed = _read_image(d / f"{pid}_{view}_ED{ext}")
        ed_gt = _read_image(d / f"{pid}_{view}_ED_gt{ext}")
        es = _read_image(d / f"{pid}_{view}_ES{ext}")
        es_gt = _read_image(d / f"{pid}_{view}_ES_gt{ext}")
        return ed, ed_gt, es, es_gt

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        # Horizontal flip
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        # Mild rotation in [-10, 10] degrees
        if random.random() < 0.5:
            import cv2
            angle = random.uniform(-10.0, 10.0)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0.0)
        return img, mask

    def _to_3ch_tensor(self, img: np.ndarray) -> torch.Tensor:
        # Replicate grayscale to 3 channels for compatibility with in_channels=3 networks.
        t = torch.from_numpy(img).float().unsqueeze(0)
        return t.repeat(3, 1, 1)

    def __getitem__(self, idx: int):
        pid, view = self.items[idx]
        ed, ed_gt, es, es_gt = self._load_pair(pid, view)

        ed = _resize(_normalise(ed), self.image_size, is_mask=False)
        es = _resize(_normalise(es), self.image_size, is_mask=False)
        ed_m = _resize(
            _to_label_map(ed_gt, self.task, self.target_label),
            self.image_size, is_mask=True,
        )
        es_m = _resize(
            _to_label_map(es_gt, self.task, self.target_label),
            self.image_size, is_mask=True,
        )

        if self.augment:
            ed, ed_m = self._augment(ed, ed_m)
            es, es_m = self._augment(es, es_m)

        return {
            "ed_frame": self._to_3ch_tensor(ed),
            "es_frame": self._to_3ch_tensor(es),
            "ed_mask": torch.from_numpy(ed_m.astype(np.int64)).long(),
            "es_mask": torch.from_numpy(es_m.astype(np.int64)).long(),
            "patient": pid,
            "view": view,
        }


def list_patient_ids(root: str) -> List[str]:
    """Return sorted list of patient folder names under `root`."""
    root_p = Path(root)
    return sorted([p.name for p in root_p.iterdir() if p.is_dir() and p.name.startswith("patient")])


def split_patient_ids(
    patient_ids: List[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Patient-level split (mandatory to avoid information leakage in echo)."""
    rng = random.Random(seed)
    ids = patient_ids.copy()
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]
    return {"train": train_ids, "val": val_ids, "test": test_ids}