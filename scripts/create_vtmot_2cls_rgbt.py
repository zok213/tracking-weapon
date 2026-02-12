#!/usr/bin/env python3
"""
VT-MOT High-Performance RGBT 2-Class Dataset Creator
====================================================
Generates 4-Channel RGBT .npy files for training.
Labels restricted to: 0=Human, 1=Car.

Features:
- Parallel Processing (utilizing all CPU cores)
- Direct GT Mapping (No AI)
- 4-Channel Stacking [B, G, R, T]
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse

# Config
VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")
OUTPUT_ROOT = Path("/home/student/Toan/data/VT-MOT_2cls_RGBT")

# Class Map
CLASS_MAP = {1: 0, 2: 1} # GT:1->Human, GT:2->Car

def process_single_pair(args):
    """Worker function to process one image pair"""
    rgb_path, ir_path, gt_boxes, out_img_dir, out_lbl_dir, fname_stem = args
    
    try:
        # 1. Image Processing
        rgb = cv2.imread(str(rgb_path)) # BGR
        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        
        if rgb is None or ir is None:
            return False

        # Resize IR to match RGB
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))
            
        # Stack [B, G, R, T]
        rgbt = np.dstack([rgb, ir])
        
        # Save NPY
        np.save(out_img_dir / f"{fname_stem}.npy", rgbt)
        
        # 2. Label Processing
        h, w = rgb.shape[:2]
        lines = []
        for box in gt_boxes:
            # x, y, w, h from MOT GT
            mx, my, mw, mh = box['box']
            cls_id = box['class']
            
            # YOLO Norm
            cx = (mx + mw/2) / w
            cy = (my + mh/2) / h
            nw = mw / w
            nh = mh / h
            
            # Clip
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))
            
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            
        # Save Label
        if lines:
            with open(out_lbl_dir / f"{fname_stem}.txt", 'w') as f:
                f.write('\n'.join(lines))
        
        return True
    
    except Exception as e:
        return False

def parse_gt_for_sequence(seq_path):
    """Returns dict: frame_id -> list of {box, class}"""
    gt_path = seq_path / "gt" / "gt.txt"
    data = {}
    if not gt_path.exists(): return {}
    
    with open(gt_path) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 8: continue
            fid = int(p[0])
            cid = int(p[7])
            
            if cid in CLASS_MAP:
                if fid not in data: data[fid] = []
                data[fid].append({
                    'box': [float(p[2]), float(p[3]), float(p[4]), float(p[5])],
                    'class': CLASS_MAP[cid]
                })
    return data

def main():
    if OUTPUT_ROOT.exists():
        print(f"Cleaning {OUTPUT_ROOT}...")
        shutil.rmtree(OUTPUT_ROOT)
    
    # Structure
    for split in ['train', 'val', 'test']:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    tasks = []
    
    print("Scanning sequences...")
    for split in ['train', 'val', 'test']:
        split_dir = VT_MOT_ROOT / "images" / split
        if not split_dir.exists(): continue
        
        for seq_path in split_dir.iterdir():
            if not seq_path.is_dir(): continue
            
            # Parse GT first
            gt_data = parse_gt_for_sequence(seq_path)
            if not gt_data: continue
            
            # Find Images
            rgb_dir = seq_path / "visible"
            if not rgb_dir.exists(): rgb_dir = seq_path / "img1" # Fallback
            
            ir_dir = seq_path / "infrared"
            
            if not rgb_dir.exists() or not ir_dir.exists():
                continue
                
            # Match RGB and IR
            rgb_imgs = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
            
            for rgb_p in rgb_imgs:
                fid = int(rgb_p.stem)
                
                # Check IR exists
                ir_p = ir_dir / rgb_p.name # Assuming same name
                if not ir_p.exists():
                    # Try extension swap
                    ir_p = ir_dir / (rgb_p.stem + ".png")
                    if not ir_p.exists(): continue
                
                # Check GT exists
                if fid not in gt_data: continue
                
                # Prepare Task
                # Unique name: seqname_frameid
                fname = f"{seq_path.name}_{fid:06d}"
                out_img_dir = OUTPUT_ROOT / "images" / split
                out_lbl_dir = OUTPUT_ROOT / "labels" / split
                
                tasks.append((rgb_p, ir_p, gt_data[fid], out_img_dir, out_lbl_dir, fname))
                
    print(f"Total pairs to process: {len(tasks)}")
    
    # Run Parallel
    workers = max(1, cpu_count() - 2)
    print(f"Starting pool with {workers} workers...")
    
    with Pool(workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_pair, tasks), total=len(tasks)))
        
    print(f"Done. Processed {sum(results)} images successfully.")
    
    # Create YAML
    with open(OUTPUT_ROOT / "dataset.yaml", 'w') as f:
        f.write(f"""
path: {OUTPUT_ROOT}
train: images/train
val: images/val
test: images/test
nc: 2
names: ['human', 'car']
""")
    print("Created dataset.yaml")

if __name__ == "__main__":
    main()
