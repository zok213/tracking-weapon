#!/usr/bin/env python3
"""
Batch SAM3 Dataset Analyzer
===========================
Sorts sequences into "Has Motorcycle" vs "No Motorcycle" folders based on SAM3 predictions.
Optimized for speed: Checks 10 random frames per sequence instead of processing all.

Workflow:
1. Load target list (Photo subset + Drone + RGBT).
2. For each sequence:
   - Sample 10 frames + GT.
   - Run SAM3: "Is there a motorcycle?"
   - If >2 confident motorcycle detections -> Classify as "Has Motorcycle".
3. Organize Result:
   - Create directories: `Analysis/Has_Motorcycle` and `Analysis/No_Motorcycle`.
   - Symlink (`ln -s`) the Verification Videos into these folders for easy review.
   - Generate a report `Dataset_Analysis_Report.md`.
"""

import sys
import os
import shutil
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add SAM3
sys.path.append("/home/student/Toan/sam3")
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import torch

# Paths
VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")
VIZ_VIDEO_DIR = Path("/home/student/Toan/data/VT-MOT_Viz_Bulk")
ANALYSIS_DIR = Path("/home/student/Toan/data/VT-MOT_Analysis_Sorted")

# Setup Folders
MOTO_DIR = ANALYSIS_DIR / "Has_Motorcycle"
NO_MOTO_DIR = ANALYSIS_DIR / "No_Motorcycle"
MOTO_DIR.mkdir(parents=True, exist_ok=True)
NO_MOTO_DIR.mkdir(parents=True, exist_ok=True)

# Target List (Same as Viz V3)
PHOTO_SUBSET = [
    "photo-0319-23", "photo-0319-18", "photo-0318-43", "photo-0318-39",
    "photo-0318-35", "photo-0318-32", "photo-0318-26", "photo-0318-27",
    "photo-0310-52", "photo-0310-51", "photo-0310-48", "photo-0310-42",
    "photo-0310-41", "photo-0310-40", "photo-0310-36", "photo-0310-34",
    "photo-0310-33", "photo-0310-28", 
    "photo-0306-01", "photo-0306-02"
]
VTUAV_SUBSET = ["Vtuav-02", "Vtuav-06", "Vtuav-17", "Vtuav-18"]
DRONE_KEYWORDS = ["wurenji", "qiuxing", "RGBT", "VTUAV", "Vtuav"]

def discover_sequences(root, keywords):
    found = []
    for split in ["train", "test", "val"]:
        d = root / "images" / split
        if not d.exists(): continue
        for p in d.iterdir():
            if not p.is_dir(): continue
            name = p.name
            if any(k.lower() in name.lower() for k in keywords):
                found.append(name)
    return sorted(list(set(found)))

TARGET_SEQS = sorted(list(set(PHOTO_SUBSET + VTUAV_SUBSET + discover_sequences(VT_MOT_ROOT, DRONE_KEYWORDS))))

# Config
FRAMES_PER_SEQ = 8 # Enough to catch a moto
PROMPTS = ["motorcycle", "person", "car"] # Priority
MOTO_CONF = 0.35

def parse_gt(gt_path):
    # Returns raw boxes per frame
    data = {}
    if not gt_path.exists(): return {}
    with open(gt_path) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p)<6: continue
            fid = int(p[0])
            tid = int(p[1])
            box = [float(p[2]), float(p[3]), float(p[4]), float(p[5])]
            if fid not in data: data[fid] = []
            data[fid].append({"box": box, "tid": tid})
    return data

def analyze_sequence(seq_name, processor, model_state):
    # Find Path
    seq_path = None
    for split in ["train", "test", "val"]:
        p = VT_MOT_ROOT / "images" / split / seq_name
        if p.exists():
            seq_path = p
            break
    if not seq_path: return "Missing"

    gt_path = seq_path / "gt" / "gt.txt"
    gt_data = parse_gt(gt_path)
    if not gt_data: return "NoGT"
    
    img_dir = seq_path / "visible"
    if not img_dir.exists(): img_dir = seq_path / "img1"
    
    # Pick frames
    all_fids = sorted(gt_data.keys())
    if not all_fids: return "Empty"
    
    # Stratified sampling
    samples = []
    if len(all_fids) <= FRAMES_PER_SEQ:
        samples = all_fids
    else:
        step = len(all_fids) // FRAMES_PER_SEQ
        samples = all_fids[::step][:FRAMES_PER_SEQ]
    
    moto_counts = 0
    person_counts = 0
    
    for fid in samples:
        # Load Image
        # Find file
        # Try jpg/png
        img_name = f"{fid:06d}.jpg"
        img_p = img_dir / img_name
        if not img_p.exists(): img_p = img_dir / f"{fid:06d}.png"
        if not img_p.exists(): continue
        
        try:
            pil_img = Image.open(img_p).convert("RGB")
            # Run SAM3
            inference_state = processor.set_image(pil_img)
            
            # Check for motorcycle
            output = processor.set_text_prompt(state=inference_state, prompt="motorcycle")
            scores = output["scores"]
            boxes = output["boxes"]
            if hasattr(scores, 'cpu'): scores = scores.cpu().numpy()
            if hasattr(boxes, 'cpu'): boxes = boxes.cpu().numpy()
            
            # Match with GT? Or just trust SAM3 existence?
            # User wants to know if dataset HAS moto. 
            # If SAM3 confident (>0.4) on "motorcycle", likely yes.
            # But let's intersect with GT to be sure we are classifying *labeled objects*.
            
            gt_boxes = gt_data[fid]
            
            for i, score in enumerate(scores):
                if score < MOTO_CONF: continue
                pbox = boxes[i] # x1,y1,x2,y2
                
                # Check Overlap with ANY GT
                matched = False
                for gt in gt_boxes:
                    gbox = gt["box"] # x,y,w,h
                    # Convert
                    x1 = max(pbox[0], gbox[0])
                    y1 = max(pbox[1], gbox[1])
                    x2 = min(pbox[2], gbox[0]+gbox[2])
                    y2 = min(pbox[3], gbox[1]+gbox[3])
                    area = max(0, x2-x1) * max(0, y2-y1)
                    if area > 0: # minimal overlap
                         matched = True
                         break
                
                if matched:
                    moto_counts += 1
                    
        except Exception as e:
            print(f"Err {seq_name} {fid}: {e}")
            
    if moto_counts >= 1: # Strict? Or loose? "Has Moto"
        return "Has_Motorcycle"
    else:
        return "No_Motorcycle"

def main():
    print("Loading SAM3... (Faster check mode)")
    model = build_sam3_image_model().cuda().eval()
    processor = Sam3Processor(model)
    
    results = {}
    
    print(f"Scanning {len(TARGET_SEQS)} sequences...")
    for seq in tqdm(TARGET_SEQS):
        res = analyze_sequence(seq, processor, None)
        results[seq] = res
        
        # Link Video
        vid_name = f"{seq}_Verify.mp4"
        vid_path = VIZ_VIDEO_DIR / vid_name
        
        dest_dir = None
        if res == "Has_Motorcycle": dest_dir = MOTO_DIR
        elif res == "No_Motorcycle": dest_dir = NO_MOTO_DIR
        
        if dest_dir and vid_path.exists():
            target_link = dest_dir / vid_name
            if target_link.exists(): target_link.unlink()
            target_link.symlink_to(vid_path)
            
    # Report
    summary_path = ANALYSIS_DIR / "Dataset_Sort_Report.md"
    with open(summary_path, 'w') as f:
        f.write("# SAM3 Dataset Analysis\n\n")
        f.write(f"Total Scanned: {len(TARGET_SEQS)}\n")
        has_moto = [s for s,r in results.items() if r=="Has_Motorcycle"]
        no_moto = [s for s,r in results.items() if r=="No_Motorcycle"]
        f.write(f"Has Motorcycle: {len(has_moto)}\n"       )
        f.write(f"No Motorcycle: {len(no_moto)}\n\n")
        f.write("## Has Motorcycle\n")
        for s in has_moto: f.write(f"- {s}\n")
        f.write("\n## No Motorcycle (Person Only?)\n")
        for s in no_moto: f.write(f"- {s}\n")
        
    print(f"Analysis Complete. Results in {ANALYSIS_DIR}")
    print(f"Motocycles: {len(has_moto)} | Clean: {len(no_moto)}")

if __name__ == "__main__":
    main()
