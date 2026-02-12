#!/usr/bin/env python3
"""
Debug SAM3 Scores
=================
Runs SAM3 on a single frame and prints RAW scores for all categories.
Helps diagnose why Persons are detected as Cars or vice versa.
"""

import sys
import os
from pathlib import Path
from PIL import Image

# Add SAM3 path
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
if not IMG_PATH.exists(): # Try img1
    IMG_PATH = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/test/photo-0301-02/img1/000001.jpg")
    
GT_PATH = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/test/photo-0301-02/gt/gt.txt")
PROMPTS = ["person", "car", "truck", "bus", "motorcycle"]

def main():
    print(f"Analyzing: {IMG_PATH}")
    
    # Load model
    model = build_sam3_image_model()
    model.to("cuda")
    processor = Sam3Processor(model)
    
    # Load Image
    image = Image.open(IMG_PATH).convert("RGB")
    inference_state = processor.set_image(image)
    
    # Load GT for frame 1
    gt_boxes = []
    with open(GT_PATH) as f:
        for line in f:
            p = line.strip().split(',')
            fid = int(p[0])
            if fid == 1:
                tid = int(p[1])
                x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
                gt_boxes.append((tid, [x, y, x+w, y+h]))

    print(f"Found {len(gt_boxes)} GT boxes in frame 1.")
    print("-" * 60)
    print(f"{'TrackID':<8} | {'Best Class':<12} | {'Score':<6} | {'All Scores'}")
    print("-" * 60)

    # Run Prompts
    # We collect outputs for ALL prompts
    all_preds = {} # prompt -> {boxes, scores}
    for prompt in PROMPTS:
        out = processor.set_text_prompt(state=inference_state, prompt=prompt)
        all_preds[prompt] = out

    # Match GTs
    for tid, gt_box in gt_boxes:
        # Check scores for this GT against all prompts
        best_prompt = "None"
        best_score = 0.0
        prompt_scores = {}
        
        for prompt in PROMPTS:
            out = all_preds[prompt]
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            
            # Find best matching box for this prompt
            max_iou = 0
            max_p_score = 0
            
            for i, pbox in enumerate(boxes):
                # IoU
                x1 = max(gt_box[0], pbox[0])
                y1 = max(gt_box[1], pbox[1])
                x2 = min(gt_box[2], pbox[2])
                y2 = min(gt_box[3], pbox[3])
                inter = max(0, x2-x1) * max(0, y2-y1)
                area_gt = (gt_box[2]-gt_box[0]) * (gt_box[3]-gt_box[1])
                area_p = (pbox[2]-pbox[0]) * (pbox[3]-pbox[1])
                union = area_gt + area_p - inter
                iou = inter / union if union > 0 else 0
                
                if iou > 0.5: # Strict Match
                    if scores[i] > max_p_score:
                        max_p_score = scores[i]
            
            prompt_scores[prompt] = max_p_score
            if max_p_score > best_score:
                best_score = max_p_score
                best_prompt = prompt
        
        # Format scores
        score_str = ", ".join([f"{p[:3]}:{s:.2f}" for p,s in prompt_scores.items()])
        print(f"{tid:<8} | {best_prompt:<12} | {best_score:.2f}   | {score_str}")

if __name__ == "__main__":
    main()
