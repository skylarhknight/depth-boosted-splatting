#!/usr/bin/env python3
# run_depth.py

import os
# Enable CPU fallback for unsupported MPS ops (must come before torch import)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
from pathlib import Path
import cv2 
import numpy as np
import torch

# 1) Add the Depth-Anything-V2 code to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Depth-Anything-V2"))

# 2) Import the correct model class
from depth_anything_v2.dpt import DepthAnythingV2

# 3) Select the best device: CUDA → MPS → CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# 4) Locate your frames folder (or fallback to project root)
base = Path("frames")
if not base.exists():
    base = Path(".")

# 5) Gather all PNG frames
frame_files = sorted(base.glob("frame_*.png"))
if not frame_files:
    raise FileNotFoundError(f"No frame_*.png files found in {base.resolve()}")

# 6) Pick the first frame
img_path = frame_files[0]
print(f"Using input frame: {img_path}")

# 7) Load the image (BGR format)
raw = cv2.imread(str(img_path))
if raw is None:
    raise FileNotFoundError(f"Couldn’t load image at {img_path}")

# 8) Initialize the model on the chosen device
model = DepthAnythingV2(encoder="vitb").to(device).eval()

# 9) Run inference (infer_image does preprocessing internally)
depth_map = model.infer_image(raw, input_size=518)

# 10) Prepare output path
out_dir = Path("frames") if Path("frames").exists() else Path(".")
depth_name = img_path.stem  # e.g. "frame_001"
out_path = out_dir / f"{depth_name}.exr"

# 11) Save the depth map using PyOpenEXR
import OpenEXR, Imath
h, w = depth_map.shape
pt = Imath.PixelType(Imath.PixelType.FLOAT)
hdr = OpenEXR.Header(w, h)
hdr['channels'] = {'R': Imath.Channel(pt)}

exr = OpenEXR.OutputFile(str(out_path), hdr)
exr.writePixels({'R': depth_map.tobytes()})
exr.close()

print(f"Saved depth map to {out_path}")
