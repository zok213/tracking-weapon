#!/usr/bin/env python3
"""
Visualize SAM3 vs GT (Side-by-Side)
===================================
Generates a split-screen video:
[ Left: SAM3 Prediction ] | [ Right: Original GT ]

Runs SAM3 inference LIVE on the sequence to ensure WYSIWYG.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add SAM3
sys.path.append("/home/student/Toan/sam3")
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("Error: SAM3 not found")
    sys.exit(1)

import torch

# Constants
VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")
OUTPUT_DIR = Path("/home/student/Toan/data/VT-MOT_SideBySide")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = ["person", "car", "truck", "bus", "motorcycle"]
CONF_THRESHOLD = 0.25 # Lower for viz to see what it's thinking

CLASS_COLORS = {
    # GT ID Colors
    1: (255, 0, 0),   # Blue
    2: (0, 255, 0),   # Green
    
    # SAM3 Label Colors
    "person": (0, 255, 255),  # Yellow
    "car": (255, 0, 0),       # Blue
    "truck": (0, 0, 255),     # Red
    "bus": (0, 128, 255),     # Orange
    "motorcycle": (0, 255, 0) # Green
}

def parse_gt(gt_path):
    data = {}
    if not gt_path.exists(): return {}
    with open(gt_path) as f:
        for line in f:
            p = line.strip().split(',')
            fid = int(p[0])
            tid = int(p[1])
            box = [float(p[2]), float(p[3]), float(p[4]), float(p[5])]
            cid = int(p[7])
            if fid not in data: data[fid] = []
            data[fid].append({"tid": tid, "box": box, "cid": cid})
    return data

def compute_iou(box1, box2):
    # box: x,y,w,h -> x1,y1,x2,y2
    b1 = [box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]]
    b2 = [box2[0], box2[1], box2[2], box2[3]] # pred is xyxy
    
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = box1[2]*box1[3]
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

def run_compare(seq_name):
    print(f"Loading SAM3 for {seq_name}...")
    model = build_sam3_image_model()
    model.to("cuda")
    processor = Sam3Processor(model)
    
    # Find sequence
    seq_path = None
    for split in ["train", "test", "val"]:
        p = VT_MOT_ROOT / "images" / split / seq_name
        if p.exists():
            seq_path = p
            break
            
    if not seq_path:
        print("Sequence not found")
        return

    gt_path = seq_path / "gt" / "gt.txt"
    img_dir = seq_path / "visible"
    if not img_dir.exists(): img_dir = seq_path / "img1"
    
    gt_data = parse_gt(gt_path)
    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    
    if not images: return
    
    # Setup Video
    im0 = cv2.imread(str(images[0]))
    h, w = im0.shape[:2]
    # Side by side: Width * 2
    out_path = OUTPUT_DIR / f"{seq_name}_Compare.mp4"
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w*2, h))
    
    print(f"Generating comparison video to {out_path}...")
    
    for img_file in tqdm(images):
        frame = cv2.imread(str(img_file))
        fid = int(img_file.stem)
        
        # --- LEFT: SAM3 ---
        frame_sam = frame.copy()
        
        # Ideally we only inference if there are GT boxes to check (simulating our pipeline)
        # But for Viz, let's just show what SAM3 sees on the GT boxes.
        
        gt_list = gt_data.get(fid, [])
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if gt_list:
            try:
                inference_state = processor.set_image(pil_img)
                
                # Check each GT box against prompts
                # We want to visualize the BEST match for each GT box
                
                for gt in gt_list:
                    best_match = {"class": "Unk", "score": 0.0}
                    
                    for prompt in PROMPTS:
                        sam_out = processor.set_text_prompt(state=inference_state, prompt=prompt)
                        boxes = sam_out["boxes"].cpu().numpy()
                        scores = sam_out["scores"].cpu().numpy()
                        
                        for i, pbox in enumerate(boxes):
                            score = scores[i]
                            if score < 0.1: continue
                            
                            iou = compute_iou(gt["box"], pbox)
                            if iou > 0.4:
                                match_prod = iou * score
                                if match_prod > best_match["score"]:
                                    best_match["score"] = match_prod
                                    best_match["class"] = prompt
                    
                    # Draw
                    bx, by, bw, bh = gt["box"]
                    x1, y1 = int(bx), int(by)
                    x2, y2 = int(bx+bw), int(by+bh)
                    
                    if best_match["score"] > CONF_THRESHOLD:
                        txt = f"{best_match['class']} {best_match['score']:.2f}"
                        col = CLASS_COLORS.get(best_match["class"], (255,255,255))
                    else:
                        txt = "Ignored"
                        col = (100, 100, 100) # Grey
                        
                    cv2.rectangle(frame_sam, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(frame_sam, txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            
            except Exception as e:
                print(e)

        cv2.putText(frame_sam, "SAM3 Prediction", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # --- RIGHT: GT ---
        frame_gt = frame.copy()
        for gt in gt_list:
            bx, by, bw, bh = gt["box"]
            x1, y1 = int(bx), int(by)
            x2, y2 = int(bx+bw), int(by+bh)
            
            cid = gt["cid"]
            col = CLASS_COLORS.get(cid, (255,255,255))
            txt = f"GT: {cid}"
            
            cv2.rectangle(frame_gt, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame_gt, txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        cv2.putText(frame_gt, "Original Ground Truth", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Concat
        combined = np.hstack((frame_sam, frame_gt))
        out.write(combined)
        
    out.release()
    print("Done")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: viz.py <seq_name>")
        sys.exit(1)
    run_compare(sys.argv[1])
