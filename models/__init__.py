"""Model factory.

Single entry point used by ``train.py`` and the ensemble evaluation
script. ``params`` from the YAML is filtered through each model's
``__init__`` signature so the same ``params`` block can be shared
across all variants without any of them tripping over keys they
don't accept.
"""
import inspect

from .baseline_unet import BaselineUNet
from .kan_unet import KAN_UNet, KAN_UNet_NoTanh, KAN_UNet_MLP
from .kan_unet_plus import (
    KAN_UNet_Plus,
    KAN_UNet_Plus_Lambda02,
    KAN_UNet_Plus_Lambda01,
    KAN_UNet_Plus_DS,
    KAN_UNet_Plus_Combo,
    KAN_UNet_Plus_MSB,
    KAN_UNet_Plus_SDF,
)
from .kan_unet_attn import AttKAN_UNet
from .u_kan import UKAN


_REGISTRY = {
    # ----- baselines / completed runs (do not re-train) ----------------
    "baseline_unet":    BaselineUNet,
    "kan_unet":         KAN_UNet,
    "kan_unet_no_tanh": KAN_UNet_NoTanh,           # Ablation A
    "kan_unet_mlp":     KAN_UNet_MLP,              # Ablation B
    "kan_unet_plus":    KAN_UNet_Plus,             # Initial proposed (lambda=0.5)

    # ----- Phase 1 (completed, did not deliver) ------------------------
    "kan_unet_plus_lambda02": KAN_UNet_Plus_Lambda02,
    "kan_unet_plus_lambda01": KAN_UNet_Plus_Lambda01,
    "kan_unet_plus_ds":       KAN_UNet_Plus_DS,

    # ----- Phase 1 follow-up (probably not needed given Phase 1 result) -
    "kan_unet_plus_combo":    KAN_UNet_Plus_Combo,

    # ----- Phase 2 (still available as fallback) -----------------------
    "kan_unet_plus_msb":      KAN_UNet_Plus_MSB,
    "kan_unet_plus_sdf":      KAN_UNet_Plus_SDF,

    # ----- Final architectural variant ---------------------------------
    "kan_unet_attn":          AttKAN_UNet,

    # ----- Reference comparison model (external baseline) --------------
    "u_kan":                  UKAN,
}


def _filter_params(cls, params: dict) -> dict:
    allowed = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    return {k: v for k, v in params.items() if k in allowed}


def build_model(cfg: dict):
    name = cfg["model"]["name"].lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model name: {name}. "
                         f"Known: {sorted(_REGISTRY)}")
    cls = _REGISTRY[name]
    params = cfg["model"].get("params", {})
    return cls(**_filter_params(cls, params))