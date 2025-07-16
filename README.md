# Depth Boosted Splatting
An experiment testing if injecting a monocular depth prior into the 3D Gaussian Splatting (3DGS) loss function improves reconstruction quality.
The research prototype aims to demonstrate a measurable gain in PSNR and visual sharpness by adding a single-frame depth supervision term.

WORK IN PROGRESS!!

### Credits
Depth Boosted Splatting [Original Repository](https://github.com/DepthAnything/Depth-Anything-V2) 

Gaussian Splatting [Original Repository](https://github.com/graphdeco-inria/gaussian-splatting)

Tanks and Temples [Open Source Image Dataset](https://www.tanksandtemples.org/download/) - used for training and evaluation

## System Requirements

### OS
- Linux (Ubuntu 22.04+ recommended)  
- macOS (Apple Silicon supported via MPS fallback)

### Hardware
- **Recommended**: NVIDIA GPU with CUDA 11.8 support  
- **Supported**: Apple M1/M2 (via PyTorch MPS backend)  
- **Fallback**: CPU-only (functional but significantly slower)

### Python
- Python 3.10 or later  
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) recommended for environment management

### Memory & Storage
- At least **16 GB RAM**  
- At least **10 GB free disk space** (for checkpoints, outputs, and cache)

### Dependencies
Installable via `environment.yml` or `requirements.txt`:
- `torch` ≥ 2.0 (CUDA 11.8 or MPS/CPU version)
- `torchvision`
- `numpy`
- `tqdm`
- `opencv-python` (with OpenEXR support)
- `OpenEXR`, `Imath` (for saving `.exr` depth maps)
- `huggingface_hub` (for automatic model checkpoint downloads)

### Optional (for 3D Gaussian Splatting)
- `diff_gaussian_rasterization` (`3dgs_accel`) module for GPU-accelerated rendering



### Notes
- environment-full is the dependencies for my VM
