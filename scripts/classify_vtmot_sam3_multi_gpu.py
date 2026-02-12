#!/usr/bin/env python3
"""
VT-MOT Vehicle Sub-Classification using SAM3 (Multi-GPU) - UNIVERSAL
====================================================================
Classifies ALL objects (Class 1 & 2) into: car, truck, motorcycle, human.
Uses both GPUs.

Logic Update (Universal):
1. Process EVERY GT object regardless of original Class ID.
2. Prompts: ["person", "car", "truck", "bus", "motorcycle"]
3. If SAM3 predicts "person" -> Class 3.
4. If SAM3 predicts Vehicle -> Class 0/1/2.
5. If SAM3 confidence < 0.35 -> IGNORE (-1).
"""

import sys
import os
from pathlib import Path
import multiprocessing as mp
import time

# Add SAM3 to path
SAM3_PATH = "/home/student/Toan/sam3"
if SAM3_PATH not in sys.path:
    sys.path.append(SAM3_PATH)

import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
import traceback

# ===== Configuration =====
VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")
OUTPUT_DIR = Path("/home/student/Toan/data/VT-MOT_4cls_SAM3")
LOG_DIR = Path("/home/student/Toan/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Mappings
# 4-class target: 0=Moto, 1=Car, 2=Truck, 3=Human
CLASS_MAP = {
    "motorcycle": 0,
    "car": 1,
    "truck": 2,
    "bus": 2,
    "person": 3
}

PROMPTS = ["motorcycle", "car", "truck", "bus", "person"]
CONF_THRESHOLD = 0.35

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0: return 0
    return inter_area / union_area

def parse_mot_gt(gt_path):
    frames = {}
    if not gt_path.exists(): return {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7: continue
            fid = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            cid = int(parts[7])
            if fid not in frames: frames[fid] = []
            frames[fid].append((tid, x, y, w, h, cid))
    return frames

def worker_process(device_id, sequences, output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format=f'[GPU{device_id}] %(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_DIR / f"sam3_gpu{device_id}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(f"worker_{device_id}")
    
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        device = f"cuda:{device_id}"
        logger.info(f"Loading SAM3 on {device}...")
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        model = build_sam3_image_model() 
        if hasattr(model, "to"):
            model.to("cuda") 
            
        processor = Sam3Processor(model)
        logger.info("Model loaded!")
        
    except Exception as e:
        logger.error(f"Failed to load SAM3: {traceback.format_exc()}")
        return

    for split, seq_path in sequences:
        try:
            seq_name = seq_path.name
            gt_path = seq_path / "gt" / "gt.txt"
            
            img_dir = seq_path / "visible"
            if not img_dir.exists(): img_dir = seq_path / "img1"
            if not img_dir.exists(): continue
            
            frame_data = parse_mot_gt(gt_path)
            if not frame_data: continue
            
            # Clean Logic: Check if we have Class 1 OR 2
            # Actually, scan EVERYTHING.
            
            seq_out = getattr(output_dir, 'joinpath')("labels") / seq_name
            seq_out.mkdir(parents=True, exist_ok=True)
            
            sample_img = next(img_dir.glob("*.jpg"), None) or next(img_dir.glob("*.png"), None)
            if not sample_img: continue
            fname_len = len(sample_img.stem)
            fmt_str = f"{{:0{fname_len}d}}{sample_img.suffix}"
            
            sorted_frames = sorted(frame_data.keys())
            logger.info(f"Processing {seq_name} ({len(sorted_frames)} frames)")
            
            for fid in sorted_frames:
                img_path = img_dir / fmt_str.format(fid)
                if not img_path.exists(): continue
                
                gt_list = frame_data[fid]
                if not gt_list: continue

                final_labels = []
                
                try:
                    pil_image = Image.open(img_path).convert("RGB")
                    w, h = pil_image.size
                    
                    inference_state = processor.set_image(pil_image)
                    
                    track_scores = {gt[0]: {"score": 0.0, "class": None} for gt in gt_list}
                    
                    for prompt in PROMPTS:
                        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
                        pred_boxes = output["boxes"]
                        scores = output["scores"]
                        
                        if hasattr(pred_boxes, 'cpu'): pred_boxes = pred_boxes.cpu().numpy()
                        if hasattr(scores, 'cpu'): scores = scores.cpu().numpy()
                        
                        for i, pred_box in enumerate(pred_boxes):
                            score = scores[i]
                            if score < 0.2: continue 
                            
                            for gt in gt_list:
                                tid = gt[0]
                                vx, vy, vw, vh = gt[1], gt[2], gt[3], gt[4]
                                gt_box = [vx, vy, vx+vw, vy+vh]
                                
                                iou = compute_iou(gt_box, pred_box)
                                if iou > 0.4:
                                    # Weighted Score: IoU * Conf
                                    match_score = iou * score
                                    if match_score > track_scores[tid]["score"]:
                                        track_scores[tid]["score"] = match_score
                                        track_scores[tid]["class"] = prompt
                                        
                    # Generate Labels
                    for gt in gt_list:
                        tid, x, y, wb, hb, _ = gt
                        res = track_scores.get(tid)
                        
                        new_cid = -1
                        if res and res["score"] > CONF_THRESHOLD and res["class"]:
                            new_cid = CLASS_MAP.get(res["class"], -1)
                        
                        if new_cid != -1:
                            cx = (x + wb/2) / w
                            cy = (y + hb/2) / h
                            nw = wb / w
                            nh = hb / h
                            final_labels.append(f"{new_cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                            
                except Exception as e:
                    logger.error(f"Error classifying {img_path}: {e}")

                if final_labels:
                    out_path = seq_out / f"{fid:06d}.txt"
                    with open(out_path, 'w') as f:
                        f.write('\n'.join(final_labels))
                        
        except Exception as ex:
            logger.error(f"Sequence {seq_path.name} failed: {ex}")
            traceback.print_exc()

def main():
    sequences = []
    for split in ["train", "test", "val"]:
        split_dir = VT_MOT_ROOT / "images" / split
        if split_dir.exists():
            for seq in split_dir.iterdir():
                if seq.is_dir():
                    sequences.append((split, seq.absolute()))
    
    import random
    random.shuffle(sequences)
    
    mid = len(sequences) // 2
    chunk1 = sequences[:mid]
    chunk2 = sequences[mid:]
    if not chunk2 and chunk1: chunk2 = [chunk1.pop()] 

    print(f"Total sequences: {len(sequences)}")
    print(f"GPU 0: {len(chunk1)} seqs")
    print(f"GPU 1: {len(chunk2)} seqs")

    p1 = mp.Process(target=worker_process, args=(0, chunk1, OUTPUT_DIR))
    p2 = mp.Process(target=worker_process, args=(1, chunk2, OUTPUT_DIR))
    p1.start(); p2.start()
    p1.join(); p2.join()
    print("All done.")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
