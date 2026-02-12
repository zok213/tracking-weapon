import torch
from ultralytics import YOLO
import cv2
import sys
from pathlib import Path
import os

# Force CPU or GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Paths
ckpt_path = Path("/home/student/Toan/checkpoints/vtmot_person_only_rgbt/epoch1.pt")
data_root = Path("/home/student/Toan/data/VT-MOT_Person_Only")
img_path = data_root / "images/train/photo-0306-01_000001.jpg"
ir_path  = data_root / "images_ir/train/photo-0306-01_000001.jpg"

print(f"Checking checkpoint: {ckpt_path}")
if not ckpt_path.exists():
    print("Checkpoint not found yet.")
    sys.exit(1)

# Load Model
try:
    # We need to construct the model first because checking state dict is safer
    # But usually YOLO(ckpt) works if it's a full checkpoint. 
    # Our script saves `state_dict`. So we need structure.
    # Load structure
    init_path = "/home/student/Toan/models/yolo26x_rgbt_init.pt"
    model = YOLO(init_path) 
    
    # SURGERY: We MUST replicate the training surgery (nc=1)
    from ultralytics.nn.modules import Detect
    head = model.model.model[-1]
    if isinstance(head, Detect) and head.nc != 1:
        print(f"SURGERY: Replacing Head (nc={head.nc} -> nc=1)")
        ch = [m[0].conv.in_channels for m in head.cv2]
        new_head = Detect(nc=1, ch=ch)
        new_head.i = head.i
        new_head.f = head.f
        new_head.type = head.type
        model.model.model[-1] = new_head
        model.model.nc = 1

    # Load state dict
    sd = torch.load(ckpt_path, map_location=device)
    model.model.load_state_dict(sd)
    model.model.to(device)
    print("Model loaded successfully.")
    
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Prepare Input
import numpy as np
rgb = cv2.imread(str(img_path))
ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))

# Preprocess
rgb = rgb.astype(np.float32) / 255.0
rgb = rgb[..., ::-1] # BGR to RGB
ir = ir.astype(np.float32) / 255.0
ir = ir[..., None]
blob = np.concatenate([rgb, ir], axis=-1) # H,W,4
blob = blob.transpose(2, 0, 1) # 4,H,W
blob = torch.tensor(blob).unsqueeze(0).to(device)

# Inference
model.model.eval()
with torch.no_grad():
    preds = model.model(blob)

# Decode
# YOLO head output: list or tuple. Usually [preds]
if isinstance(preds, (list, tuple)):
    preds = preds[0]

# Preds shape: [Batch, 4+Conf, Anchors]
print(f"Raw Output Shape: {preds.shape}")
print(f"Max Value: {preds.max()}")
print(f"Min Value: {preds.min()}")
print(f"Mean Value: {preds.mean()}")

# Post-Process (Simple manual check)
# cls scores are usually after the first 4.
# For 1 class: [Batch, 5, Anchors] (x,y,w,h, conf)
# Check max confidence
conf = preds[0, 4, :]
print(f"Max Confidence: {conf.max().item()}")
print(f"Num Detections > 0.5: {(conf > 0.5).sum().item()}")
print(f"Num Detections > 0.1: {(conf > 0.1).sum().item()}")

if (conf > 0.1).sum().item() == 0:
    print("❌ Model detected NOTHING (Confidence too low).")
else:
    print("✅ Model detected SOMETHING.")
