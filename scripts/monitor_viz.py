#!/usr/bin/env python3
"""
Monitor SAM3 Progress & Auto-Visualize
======================================
1. Scans VT-MOT output directory.
2. Checks if a sequence is fully processed (label_count == image_count).
3. If complete & no video exists -> Generates visualization video.
4. Prints overall progress bar.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from tqdm import tqdm

VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")
OUTPUT_DIR = Path("/home/student/Toan/data/VT-MOT_4cls_SAM3")
LABEL_DIR = OUTPUT_DIR / "labels"
VIZ_DIR = OUTPUT_DIR / "viz"
SCRIPT_PATH = "/home/student/Toan/scripts/visualize_sam3_labels.py"

def get_sequences():
    seqs = {}
    for split in ["train", "test", "val"]:
        split_dir = VT_MOT_ROOT / "images" / split
        if split_dir.exists():
            for seq in split_dir.iterdir():
                if seq.is_dir():
                    # Count images
                    img_dir = seq / "visible"
                    if not img_dir.exists(): img_dir = seq / "img1"
                    if not img_dir.exists(): continue
                    
                    n_imgs = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
                    seqs[seq.name] = n_imgs
    return seqs

def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Scanning sequences...")
    all_seqs = get_sequences()
    total_seqs = len(all_seqs)
    
    while True:
        completed = 0
        
        # Check output
        if not LABEL_DIR.exists():
            print("Waiting for label directory creation...")
            time.sleep(10)
            continue
            
        for seq_name, n_imgs in all_seqs.items():
            seq_lbl_dir = LABEL_DIR / seq_name
            if not seq_lbl_dir.exists():
                continue
                
            n_lbls = len(list(seq_lbl_dir.glob("*.txt")))
            
            # Check completion (allow small margin for empty frames if any)
            if n_lbls >= n_imgs: # Done
                completed += 1
                
                # Check if video exists
                vid_path = VIZ_DIR / f"{seq_name}_sam3.mp4"
                if not vid_path.exists():
                    print(f"\n🎬 Generating video for {seq_name}...")
                    subprocess.run([sys.executable, SCRIPT_PATH, "--seq", seq_name], 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"✅ Video created: {vid_path}")
            
        # Status
        sys.stdout.write(f"\r🚀 Progress: {completed}/{total_seqs} sequences completed. (Auto-visualizing...)")
        sys.stdout.flush()
        
        if completed == total_seqs:
            print("\n🎉 All sequences processed!")
            break
            
        time.sleep(30) # Check every 30s

if __name__ == "__main__":
    main()
