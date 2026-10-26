# Manuscript experiment environment

This record contains the environment information preserved with the manuscript
and measured efficiency results. Fields that were not captured during the
original runs are identified explicitly rather than inferred.

## Hardware

- GPU model: NVIDIA Quadro GV100
- GPU memory: 32 GB HBM2
- CPU: not recorded
- System RAM: not recorded

## Software

- Operating system: Linux-based experiment workspace; distribution/version not recorded
- Python: exact patch version not recorded
- PyTorch: 2.4.1
- CUDA runtime: 12.1
- cuDNN: exact version not recorded
- SimpleITK: exact version not recorded
- OpenCV: exact version not recorded
- NumPy: exact version not recorded
- SciPy: exact version not recorded

## Efficiency protocol

- Input tensor: `1 x 3 x 256 x 256`
- Batch size: `1`
- Warmup passes: `10`
- Timed passes: `100`
- Device: NVIDIA Quadro GV100
- Synchronization: CUDA synchronization before and after each timed pass
- Latency definition: mean wall-clock forward-pass time per image
- FPS definition: `1000 / mean_latency_ms`
- Ensemble/TTA: excluded from the reported single-model latency

Measured values are deposited in
`results/efficiency/measured_efficiency.csv`.

## Future reruns

For future archival reruns, preserve the complete environment with:

```bash
python --version
python -m pip freeze
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The missing original patch-level versions above should not be retroactively
filled using a different machine.
