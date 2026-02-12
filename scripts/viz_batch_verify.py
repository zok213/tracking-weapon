#!/usr/bin/env python3
"""
Bulk RGBT Visualization for User Verification
=============================================
Generates side-by-side RGBT videos for a specific list of sequences.
Packs them into a ZIP file for easy download.
"""

import cv2
import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import zipfile

# Config
VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")
OUTPUT_DIR = Path("/home/student/Toan/data/VT-MOT_Viz_Bulk")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Base list (Photo subset confirmed by user)
PHOTO_SUBSET = [
    "photo-0319-23", "photo-0319-18", "photo-0318-43", "photo-0318-39",
    "photo-0318-35", "photo-0318-32", "photo-0318-26", "photo-0318-27",
    "photo-0310-52", "photo-0310-51", "photo-0310-48", "photo-0310-42",
    "photo-0310-41", "photo-0310-40", "photo-0310-36", "photo-0310-34",
    "photo-0310-33", "photo-0310-28", 
    "photo-0306-01", "photo-0306-02"
]

VTUAV_SUBSET = [
    "Vtuav-02", "Vtuav-06", "Vtuav-17", "Vtuav-18"
]

# 2. Dynamic Discovery (Drone/UAV/RGBT)
def discover_sequences(root, keywords):
    found = []
    for split in ["train", "test", "val"]:
        d = root / "images" / split
        if not d.exists(): continue
        for p in d.iterdir():
            if not p.is_dir(): continue
            name = p.name
            # Case insensitive check
            if any(k.lower() in name.lower() for k in keywords):
                found.append(name)
    return sorted(list(set(found)))

# Keywords user asked for
DRONE_KEYWORDS = ["wurenji", "qiuxing", "RGBT", "VTUAV", "Vtuav"]

# Combine
ALL_SEQUENCES = sorted(list(set(PHOTO_SUBSET + VTUAV_SUBSET + discover_sequences(VT_MOT_ROOT, DRONE_KEYWORDS))))
TARGET_SEQUENCES = ALL_SEQUENCES 

print(f"Targeting {len(TARGET_SEQUENCES)} sequences: {TARGET_SEQUENCES}")

CLASS_COLORS = {
    1: (0, 255, 255),   # GT:1 = Person (Yellow)
    2: (0, 255, 0),     # GT:2 = Vehicle (Green)
    3: (0, 0, 255),     
    4: (255, 0, 0),     
    5: (255, 0, 255)    
}

def parse_gt(gt_path):
    data = {}
    if not gt_path.exists(): return {}
    with open(gt_path) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p)<8: continue
            fid = int(p[0])
            tid = int(p[1])
            box = [float(p[2]), float(p[3]), float(p[4]), float(p[5])]
            cid = int(p[7])
            if fid not in data: data[fid] = []
            data[fid].append({"tid": tid, "box": box, "cid": cid})
    return data

def make_video(seq_name):
    # Search for sequence
    seq_path = None
    for split in ["train", "test", "val"]:
        p = VT_MOT_ROOT / "images" / split / seq_name
        if p.exists():
            seq_path = p
            break
            
    if not seq_path:
        print(f"Skipping {seq_name} (Not found)")
        return None

    gt_path = seq_path / "gt" / "gt.txt"
    rgb_dir = seq_path / "visible"
    if not rgb_dir.exists(): rgb_dir = seq_path / "img1"
    ir_dir = seq_path / "infrared" # Assumed standard name
    
    if not rgb_dir.exists(): return None

    # Load GT
    gt_data = parse_gt(gt_path)
    
    # Images
    images = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
    if not images: return None

    # Setup Video
    im0 = cv2.imread(str(images[0]))
    h, w = im0.shape[:2]
    out_name = f"{seq_name}_Verify.mp4"
    out_path = OUTPUT_DIR / out_name
    
    # Limit to 500 frames max to keep file size small for download? No, user wants full check.
    # But for speed let's do max 300 frames (10 sec) or full if short.
    # User said "completely", let's do full.
    
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w*2, h))
    
    print(f"Rendering {seq_name}...")
    for img_file in tqdm(images, leave=False):
        fid = int(img_file.stem)
        rgb = cv2.imread(str(img_file))
        
        # Try finding IR
        ir_name = img_file.name
        ir_path = ir_dir / ir_name
        if not ir_path.exists(): ir_path = ir_dir / (img_file.stem + ".png")
        if not ir_path.exists(): ir_path = ir_dir / (img_file.stem + ".jpg")

        if ir_path.exists():
            ir = cv2.imread(str(ir_path))
            if ir.shape[:2] != (h,w): ir = cv2.resize(ir, (w,h))
        else:
            ir = np.zeros_like(rgb)
            cv2.putText(ir, "NO IR", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Draw Labels
        gt_list = gt_data.get(fid, [])
        for gt in gt_list:
            bx, by, bw, bh = gt["box"]
            x1, y1 = int(bx), int(by)
            x2, y2 = int(bx+bw), int(by+bh)
            col = CLASS_COLORS.get(gt["cid"], (200,200,200))
            
            # Label
            txt = f"ID:{gt['tid']} C:{gt['cid']}"
            
            cv2.rectangle(rgb, (x1,y1), (x2,y2), col, 2)
            cv2.putText(rgb, txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            
            cv2.rectangle(ir, (x1,y1), (x2,y2), col, 2)
            cv2.putText(ir, txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        combined = np.hstack([rgb, ir])
        out.write(combined)
        
    out.release()
    return out_path

def main():
    generated_files = []
    
    # Process
    for seq in TARGET_SEQUENCES:
        path = make_video(seq)
        if path: generated_files.append(path)
        
    # Zip
    if generated_files:
        zip_path = OUTPUT_DIR / "GT_Verification_Videos_v3.zip"
        print(f"Zipping {len(generated_files)} videos to {zip_path}...")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for f in generated_files:
                zf.write(f, f.name)
        print("Done.")

if __name__ == "__main__":
    main()
