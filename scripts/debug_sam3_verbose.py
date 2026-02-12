#!/usr/bin/env python3
"""
Debug SAM3 Scores - Verbose
===========================
Prints ALL SAM3 boxes to debug coordinate issues.
"""

import sys
import os
from pathlib import Path
from PIL import Image

sys.path.append("/home/student/Toan/sam3")
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("SAM3 not found")
    sys.exit(1)

import torch

# Config
IMG_PATH = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/test/photo-0301-02/visible/000001.jpg")
if not IMG_PATH.exists():
    IMG_PATH = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/test/photo-0301-02/img1/000001.jpg")
    
GT_PATH = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/test/photo-0301-02/gt/gt.txt")
PROMPTS = ["person", "car"] # simplified for debug

def main():
    print(f"Analyzing: {IMG_PATH}")
    if not IMG_PATH.exists():
        print("Image not found!")
        return

    model = build_sam3_image_model()
    model.to("cuda")
    processor = Sam3Processor(model)
    
    image = Image.open(IMG_PATH).convert("RGB")
    w, h = image.size
    print(f"Image Size: {w}x{h}")
    
    inference_state = processor.set_image(image)
    
    # Load GT
    gt_boxes = []
    with open(GT_PATH) as f:
        for line in f:
            p = line.strip().split(',')
            fid = int(p[0])
            if fid == 1:
                tid = int(p[1])
                x, y, w, h_box = float(p[2]), float(p[3]), float(p[4]), float(p[5])
                gt_boxes.append((tid, [x, y, x+w, y+h_box]))
    
    print("-" * 60)
    print("Ground Truth Boxes (Frame 1):")
    for tid, box in gt_boxes:
        print(f"ID {tid}: {box}")

    print("-" * 60)
    print("SAM3 Predictions:")
    
    for prompt in PROMPTS:
        print(f"Prompt: '{prompt}'")
        out = processor.set_text_prompt(state=inference_state, prompt=prompt)
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        
        for i, box in enumerate(boxes):
            score = scores[i]
            if score > 0.1:
                print(f"  Box: {box.astype(int)}, Score: {score:.2f}")

if __name__ == "__main__":
    main()
