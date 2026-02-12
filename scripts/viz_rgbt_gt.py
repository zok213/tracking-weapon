#!/usr/bin/env python3
"""
Visualize RGBT Ground Truth (Side-by-Side)
==========================================
Generates a split-screen video:
[ Left: RGB + GT ] | [ Right: Thermal + GT ]

Purpose: Verify if 'Person' class (GT:1) includes detection of Motorcycles/Riders
in the 'wurenji' (Drone) subset.
"""

import cv2
import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Config
VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")
OUTPUT_DIR = Path("/home/student/Toan/data/VT-MOT_Viz_OneSubset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    1: (0, 255, 255),   # GT:1 = Person (Yellow)
    2: (0, 255, 0),     # GT:2 = Vehicle (Green) - usually Car?
    3: (0, 0, 255),     # GT:3 = ? (Red)
    4: (255, 0, 0),     # Blue
    5: (255, 0, 255)    # Magenta
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

def make_video(seq_name):
    # Find sequence
    seq_path = None
    for split in ["train", "test", "val"]:
        p = VT_MOT_ROOT / "images" / split / seq_name
        if p.exists():
            seq_path = p
            break
            
    if not seq_path:
        print(f"Sequence {seq_name} not found")
        return

    print(f"Processing {seq_name}...")
    
    gt_path = seq_path / "gt" / "gt.txt"
    rgb_dir = seq_path / "visible"
    if not rgb_dir.exists(): rgb_dir = seq_path / "img1"
    
    ir_dir = seq_path / "infrared"
    if not ir_dir.exists():
        print("No Infrared folder found!")
        return

    gt_data = parse_gt(gt_path)
    
    # Get images
    images = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
    if not images: return

    # Setup Video
    im0 = cv2.imread(str(images[0]))
    h, w = im0.shape[:2]
    
    out_path = OUTPUT_DIR / f"{seq_name}_RGBT_GT.mp4"
    # Side by side: Width * 2
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w*2, h))
    
    for img_file in tqdm(images):
        fid = int(img_file.stem)
        
        # Read RGB
        rgb = cv2.imread(str(img_file))
        
        # Read IR
        ir_name = img_file.name
        ir_path = ir_dir / ir_name
        if not ir_path.exists():
            ir_path = ir_dir / (img_file.stem + ".png")
        
        if ir_path.exists():
            ir = cv2.imread(str(ir_path)) # 3ch but grayscale
            if ir.shape[:2] != (h, w):
                ir = cv2.resize(ir, (w, h))
        else:
            ir = np.zeros_like(rgb)
            cv2.putText(ir, "NO IR", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        # Draw GT on both
        gt_list = gt_data.get(fid, [])
        
        for gt in gt_list:
            bx, by, bw, bh = gt["box"]
            cid = gt["cid"]
            tid = gt["tid"]
            
            p1 = (int(bx), int(by))
            p2 = (int(bx+bw), int(by+bh))
            col = CLASS_COLORS.get(cid, (255,255,255))
            
            # Label
            label = f"ID:{tid} C:{cid}"
            
            # Draw on RGB
            cv2.rectangle(rgb, p1, p2, col, 2)
            cv2.putText(rgb, label, (p1[0], p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            
            # Draw on IR
            cv2.rectangle(ir, p1, p2, col, 2)
            cv2.putText(ir, label, (p1[0], p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            
        # Titles
        cv2.putText(rgb, f"RGB: {seq_name}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(ir, f"Thermal (IR)", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        combined = np.hstack([rgb, ir])
        out.write(combined)
        
    out.release()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: script.py <seq_name1> [seq_name2 ...]")
        # Default behavior: Search for a wurenji sequence
        print("Searching for a wurenji sequence...")
        root = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/test")
        found = []
        for d in root.iterdir():
            if "wurenji" in d.name:
                found.append(d.name)
                if len(found) >= 1: break
        
        if found:
            print(f"Auto-processing: {found[0]}")
            make_video(found[0])
        else:
            print("No wurenji sequence found to default to.")
    else:
        for seq in sys.argv[1:]:
            make_video(seq)
