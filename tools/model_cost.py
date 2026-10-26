"""Measure and report model computational cost: parameters, FLOPs, and
inference latency per image. Produces a publication-ready LaTeX table.

FLOPs are reported as multiply-accumulate operations counted by fvcore
(``FlopCountAnalysis``). fvcore may miss custom-op contributions inside the
B-spline basis evaluation, so the KAN FLOPs are a *lower bound*. We
explicitly note this in the caption so the table is defensible under
review.

Latency is wall-clock time per forward pass on a single 256x256x3 input,
measured after warmup with CUDA synchronisation to avoid measuring kernel
launch overhead.

Usage:
    # Auto-detects GPU; falls back to CPU
    python tools/model_cost.py --config configs/default.yaml

    # Or compare with explicit models and write the LaTeX next to the others:
    python tools/model_cost.py \
        --config configs/default.yaml \
        --out runs/aggregate/model_cost.tex
"""
import argparse
import copy
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import build_model


# -----------------------------------------------------------------------------
# measurements
# -----------------------------------------------------------------------------
def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model: torch.nn.Module, input_shape, device) -> float:
    """Multiply-accumulate count via fvcore. Returns 0.0 on failure."""
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        x = torch.randn(*input_shape, device=device)
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fca = FlopCountAnalysis(model, x)
                fca.unsupported_ops_warnings(False)
                fca.uncalled_modules_warnings(False)
                return float(fca.total())
    except Exception as e:
        print(f"  FLOP counting failed: {e}")
        return 0.0


def benchmark_inference(model: torch.nn.Module, input_shape, device,
                        n_warmup: int = 10, n_iters: int = 100):
    """Mean and std forward-pass time (ms) per input, with CUDA sync."""
    model.eval()
    x = torch.randn(*input_shape, device=device)
    is_cuda = device.type == "cuda"

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    if is_cuda:
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)), float(np.std(times, ddof=1))


def peak_gpu_memory_mb(model: torch.nn.Module, input_shape, device) -> float:
    """Peak GPU memory (MB) during a single forward pass. Returns 0 on CPU."""
    if device.type != "cuda":
        return 0.0
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# -----------------------------------------------------------------------------
# per-model evaluation
# -----------------------------------------------------------------------------
def measure(model_name: str, cfg: dict, image_size: int, batch_size: int,
            n_warmup: int, n_iters: int, device):
    """Build the model from cfg, measure cost, free memory."""
    cfg_m = copy.deepcopy(cfg)
    cfg_m["model"]["name"] = model_name

    model = build_model(cfg_m).to(device)
    input_shape = (batch_size, cfg["model"]["params"]["in_channels"],
                   image_size, image_size)

    params = count_params(model)
    flops = compute_flops(model, input_shape, device)
    lat_mean, lat_std = benchmark_inference(
        model, input_shape, device, n_warmup=n_warmup, n_iters=n_iters,
    )
    mem_mb = peak_gpu_memory_mb(model, input_shape, device)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "params": params,
        "flops": flops,
        "latency_mean_ms": lat_mean,
        "latency_std_ms": lat_std,
        "peak_mem_mb": mem_mb,
    }


# -----------------------------------------------------------------------------
# LaTeX
# -----------------------------------------------------------------------------
def write_latex_table(results, batch_size: int, image_size: int, device,
                      out_path: Path,
                      baseline: str = "baseline_unet",
                      proposed: str = "kan_unet_attn") -> None:
    if baseline not in results or proposed not in results:
        print(f"  Skipping LaTeX -- need both {baseline} and {proposed} measurements.")
        return

    b = results[baseline]
    k = results[proposed]
    p_pct = 100.0 * (k["params"] - b["params"]) / b["params"]
    f_pct = (100.0 * (k["flops"] - b["flops"]) / b["flops"]
             if b["flops"] > 0 else float("nan"))
    l_pct = 100.0 * (k["latency_mean_ms"] - b["latency_mean_ms"]) / b["latency_mean_ms"]

    dev_str = "GPU" if device.type == "cuda" else "CPU"
    caption = (
        r"Computational cost of the proposed KAU-Net vs U-Net "
        f"on a single ${image_size}\\times{image_size}\\times 3$ input "
        f"(batch size {batch_size}). Latency reported as mean $\\pm$ std over "
        "100 forward passes after 10 warmup iterations on " + dev_str + ". "
        "FLOPs measured by fvcore; for the KAN spline-basis path this is a "
        "conservative lower bound."
    )

    flops_b = f"{b['flops']/1e9:.2f}" if b["flops"] > 0 else "n/a"
    flops_k_cell = (
        f"{k['flops']/1e9:.2f} ({f_pct:+.0f}\\%)"
        if not np.isnan(f_pct) else "n/a"
    )

    lines = [
        "% Auto-generated by tools/model_cost.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{tab:model_cost}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Params (M) & FLOPs (G) & Latency (ms/image) \\",
        r"\midrule",
        (f"Baseline U-Net  & {b['params']/1e6:.2f} "
         f"& {flops_b} "
         f"& {b['latency_mean_ms']:.2f} $\\pm$ {b['latency_std_ms']:.2f} \\\\"),
        (f"KAU-Net & {k['params']/1e6:.2f} ({p_pct:+.0f}\\%) "
         f"& {flops_k_cell} "
         f"& {k['latency_mean_ms']:.2f} $\\pm$ {k['latency_std_ms']:.2f} "
         f"({l_pct:+.0f}\\%) \\\\"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote: {out_path}")


# -----------------------------------------------------------------------------
# entry point
# -----------------------------------------------------------------------------
def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Mirror train.py's out_channels derivation so the model we measure has
    # the same architecture as the one being trained.
    task = cfg.get("task", "multiclass")
    num_classes = int(cfg.get("num_classes", 4))
    cfg["model"]["params"]["out_channels"] = 1 if task == "binary" else num_classes

    image_size = int(cfg["data"]["image_size"])
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu)
                          else "cpu")
    print(f"Device: {device}")
    print(f"Input: {args.batch_size} x 3 x {image_size} x {image_size}")
    print(f"Warmup: {args.warmup}, iters: {args.iters}")
    print()

    results = {}
    for model_name in args.models:
        print(f"Measuring {model_name}...")
        results[model_name] = measure(
            model_name, cfg,
            image_size=image_size,
            batch_size=args.batch_size,
            n_warmup=args.warmup,
            n_iters=args.iters,
            device=device,
        )

    # stdout
    print()
    header = (f"{'Model':18s} {'Params (M)':>11s} {'FLOPs (G)':>11s} "
              f"{'Latency (ms)':>16s} {'Peak mem (MB)':>15s}")
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        flops_str = (f"{r['flops']/1e9:>11.3f}" if r["flops"] > 0
                     else f"{'n/a':>11s}")
        mem_str = (f"{r['peak_mem_mb']:>15.1f}" if device.type == "cuda"
                   else f"{'n/a':>15s}")
        print(f"{name:18s} "
              f"{r['params']/1e6:>11.3f} "
              f"{flops_str} "
              f"{r['latency_mean_ms']:>9.2f} +/- {r['latency_std_ms']:>4.2f} "
              f"{mem_str}")

    # Relative overhead
    if "baseline_unet" in results and "kan_unet_attn" in results:
        b = results["baseline_unet"]
        k = results["kan_unet_attn"]
        print()
        print("Overhead (KAU-Net vs U-Net):")
        print(f"  Params:  +{(k['params']-b['params'])/1e6:.3f} M  "
              f"({100*(k['params']-b['params'])/b['params']:+.1f} %)")
        if b["flops"] > 0:
            print(f"  FLOPs:   +{(k['flops']-b['flops'])/1e9:.3f} G  "
                  f"({100*(k['flops']-b['flops'])/b['flops']:+.1f} %)")
        print(f"  Latency: +{k['latency_mean_ms']-b['latency_mean_ms']:.2f} ms  "
              f"({100*(k['latency_mean_ms']-b['latency_mean_ms'])/b['latency_mean_ms']:+.1f} %)")

    # LaTeX
    if args.out:
        out_path = Path(args.out)
        write_latex_table(results, args.batch_size, image_size, device, out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="+",
                   default=["baseline_unet", "kan_unet", "kan_unet_attn"])
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size for the latency measurement.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available.")
    p.add_argument("--out", default="runs/aggregate/model_cost.tex",
                   help="Output LaTeX table path.")
    args = p.parse_args()
    main(args)