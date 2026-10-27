"""Fast CPU checks for repository integrity."""
import torch

from data.camus_dataset import split_patient_ids
from models import build_model


def test_kau_net_forward_shape():
    cfg = {
        "model": {
            "name": "kan_unet_attn",
            "params": {
                "in_channels": 3,
                "out_channels": 4,
                "base_channels": 8,
                "kan_num_layers": 2,
                "kan_grid_size": 5,
                "kan_spline_order": 3,
            },
        }
    }
    model = build_model(cfg).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 4, 64, 64)


def test_patient_split_is_deterministic_and_disjoint():
    patient_ids = [f"patient{i:04d}" for i in range(1, 501)]
    a = split_patient_ids(patient_ids, val_ratio=0.15, test_ratio=0.15, seed=42)
    b = split_patient_ids(patient_ids, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert a == b
    assert len(a["train"]) == 350
    assert len(a["val"]) == 75
    assert len(a["test"]) == 75
    assert set(a["train"]).isdisjoint(a["val"])
    assert set(a["train"]).isdisjoint(a["test"])
    assert set(a["val"]).isdisjoint(a["test"])
    assert set(a["train"] + a["val"] + a["test"]) == set(patient_ids)
