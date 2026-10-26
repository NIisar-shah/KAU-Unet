# Manuscript experiment environment

Complete this record from the machine used for the final manuscript results
before creating release `v1.0.0`. Do not estimate or infer missing versions.

## Hardware

- GPU model: **TO BE CONFIRMED**
- GPU memory: **TO BE CONFIRMED**
- CPU: **TO BE CONFIRMED**
- RAM: **TO BE CONFIRMED**

## Software

- Operating system: **TO BE CONFIRMED**
- Python: **TO BE CONFIRMED**
- PyTorch: **TO BE CONFIRMED**
- CUDA runtime: **TO BE CONFIRMED**
- cuDNN: **TO BE CONFIRMED**
- SimpleITK: **TO BE CONFIRMED**
- OpenCV: **TO BE CONFIRMED**
- NumPy: **TO BE CONFIRMED**
- SciPy: **TO BE CONFIRMED**

## Efficiency protocol

- Input tensor: `1 x 3 x 256 x 256`
- Batch size: `1`
- Warmup passes: `10`
- Timed passes: `100`
- Synchronization: CUDA synchronization before and after each timed pass
- Latency definition: mean wall-clock forward-pass time per image
- FPS definition: `1000 / mean_latency_ms`

## Capture commands

Record the outputs of the following commands from the experiment environment:

```bash
python --version
python -m pip freeze
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
