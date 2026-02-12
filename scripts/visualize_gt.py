#!/usr/bin/env python3
"""
Visualize VT-MOT Ground Truth
=============================
Renders original GT bounding boxes, Track IDs, and Class IDs onto video frames.
Helps auditing dataset quality.
"""

import cv2
import os
import glob
from tqdm import tqdm
import argparse
from pathlib import Path

# VT-MOT Config
VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")
OUTPUT_DIR = Path("/home/student/Toan/data/VT-MOT_GT_Viz")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color map for Classes (Arbitrary but distinct)
COLORS = {
    1: (255, 0, 0),    # Class 1: Blue (often Vehicle)
    2: (0, 255, 0),    # Class 2: Green (often Person?)
    3: (0, 0, 255),    # Class 3: Red
    4: (255, 255, 0),  # Cyan
    5: (255, 0, 255),  # Magenta
    0: (128, 128, 128) # Class 0: Gray
}
# Fallback color: White

def parse_gt(gt_path):
    # Dict: frame_id -> list of (tid, x, y, w, h, cid)
    data = {}
    if not gt_path.exists(): return {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8: continue
            fid = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            cid = int(parts[7]) # Class ID is usually 8th column (index 7)
            
            if fid not in data: data[fid] = []
            data[fid].append((tid, x, y, w, h, cid))
    return data

def visualize_sequence(seq_name):
    print(f"Processing {seq_name}...")
    
    # Locate Sequence
    seq_path = None
    for split in ["train", "test", "val"]:
        p = VT_MOT_ROOT / "images" / split / seq_name
        if p.exists():
            seq_path = p
            break
            
    if not seq_path:
        print(f"Sequence {seq_name} not found in VT-MOT")
        return

    # Paths
    gt_path = seq_path / "gt" / "gt.txt"
    img_dir = seq_path / "visible"
    if not img_dir.exists(): img_dir = seq_path / "img1"
    
    if not gt_path.exists() or not img_dir.exists():
        print("Missing GT or Images")
        return

    # Load Data
    frame_data = parse_gt(gt_path)
    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    
    if not images:
        print("No images found")
        return

    # Setup Video
    h, w = cv2.imread(str(images[0])).shape[:2]
    out_path = OUTPUT_DIR / f"{seq_name}_GT.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, 30.0, (w, h))

    # Render
    for img_file in tqdm(images):
        frame = cv2.imread(str(img_file))
        
        # Frame ID from filename usually (000001.jpg -> 1)
        try:
            fid = int(img_file.stem)
        except:
            continue
            
        if fid in frame_data:
            for item in frame_data[fid]:
                tid, x, y, bw, bh, cid = item
                
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + bw), int(y + bh)
                
                color = COLORS.get(cid, (255, 255, 255))
                
                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label (ID: Class)
                label = f"ID:{tid} C:{cid}"
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out.write(frame)

    out.release()
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, required=True)
    args = parser.parse_args()
    visualize_sequence(args.seq)
