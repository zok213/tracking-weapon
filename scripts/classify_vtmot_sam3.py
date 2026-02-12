#!/usr/bin/env python3
"""
VT-MOT Vehicle Sub-Classification using SAM3 (Whole-Frame Strategy)
===================================================================
Classifies generic "vehicle" annotations into: car, truck, motorcycle

Strategy:
1. Load full image (efficient batching)
2. Prompt SAM3 with ["car", "truck", "motorcycle", "bus"]
3. Match detected masks with Ground Truth bounding boxes (IoU)
4. Assign labels based on best match
"""

import sys
import os
from pathlib import Path

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

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("Error: Could not import SAM3. Make sure it is installed and in path.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== Configuration =====
VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")
OUTPUT_DIR = Path("/home/student/Toan/data/VT-MOT_4cls_SAM3")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

VEHICLE_TO_ID = {"motorcycle": 0, "car": 1, "truck": 2, "bus": 2}  # bus -> truck
PERSON_ID = 3
PROMPTS = ["motorcycle", "car", "truck", "bus"]

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
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

def box_ioa(box_gt, box_pred):
    """Compute Intersection over Area of GT (how much of GT is covered by pred)"""
    x1 = max(box_gt[0], box_pred[0])
    y1 = max(box_gt[1], box_pred[1])
    x2 = min(box_gt[2], box_pred[2])
    y2 = min(box_gt[3], box_pred[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    gt_area = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
    
    if gt_area == 0: return 0
    return inter_area / gt_area

class SAM3Classifier:
    def __init__(self):
        logger.info(f"Loading SAM3 model on {DEVICE}...")
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        logger.info("SAM3 loaded successfully")

    def process_image(self, image_path, gt_boxes):
        """
        Run SAM3 on full image and match matches with GT boxes.
        gt_boxes: list of (track_id, x, y, w, h)
        Returns: dict {track_id: class_id}
        """
        if not os.path.exists(image_path):
            return {}

        try:
            pil_image = Image.open(image_path).convert("RGB")
            inference_state = self.processor.set_image(pil_image)
            
            # Run prompts sequentially (SAM3 limits?) or check if batching supported
            # For now, sequential gives clearest results
            
            # Store best class for each GT box
            gt_results = {gt[0]: {"class": "car", "score": 0.0} for gt in gt_boxes} # Default to car
            
            for prompt_text in PROMPTS:
                output = self.processor.set_text_prompt(state=inference_state, prompt=prompt_text)
                pred_boxes = output["boxes"] # [N, 4] x1,y1,x2,y2
                scores = output["scores"]   # [N]
                
                # Convert to numpy/list
                if hasattr(pred_boxes, 'cpu'): pred_boxes = pred_boxes.cpu().numpy()
                if hasattr(scores, 'cpu'): scores = scores.cpu().numpy()
                
                # Normalize class name (bus -> truck)
                target_cls = prompt_text
                if target_cls == "bus": target_cls = "truck"
                
                # Match predictions to GT
                for i, pred_box in enumerate(pred_boxes):
                    score = scores[i]
                    if score < 0.2: continue
                    
                    for gt in gt_boxes:
                        track_id = gt[0]
                        gt_x1, gt_y1 = gt[1], gt[2]
                        gt_x2, gt_y2 = gt[1] + gt[3], gt[2] + gt[4]
                        gt_rect = [gt_x1, gt_y1, gt_x2, gt_y2]
                        
                        overlap = compute_iou(gt_rect, pred_box)
                        
                        # Update if this prompt gives a better match/score
                        if overlap > 0.4:
                            # Heuristic: Combine overlap and score
                            match_score = overlap * score
                            if match_score > gt_results[track_id]["score"]:
                                gt_results[track_id]["score"] = match_score
                                gt_results[track_id]["class"] = target_cls

            return gt_results
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {}

def parse_mot_gt(gt_path):
    # Returns dict: frame_id -> list of (track_id, x, y, w, h, class_id)
    frames = {}
    if not gt_path.exists(): return {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(',')
            fid = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            cid = int(parts[7])
            if fid not in frames: frames[fid] = []
            frames[fid].append((tid, x, y, w, h, cid))
    return frames

def main():
    classifier = SAM3Classifier()
    
    # Find sequences
    sequences = []
    for split in ["train", "test", "val"]:
        split_dir = VT_MOT_ROOT / "images" / split
        if split_dir.exists():
            for seq in split_dir.iterdir():
                if seq.is_dir():
                    sequences.append((split, seq))
    
    logger.info(f"Checking {len(sequences)} sequences")
    
    for split, seq_path in tqdm(sequences):
        seq_name = seq_path.name
        gt_path = seq_path / "gt" / "gt.txt"
        
        # Image dir
        img_dir = seq_path / "visible"
        if not img_dir.exists(): img_dir = seq_path / "img1"
        if not img_dir.exists(): continue
            
        # Parse GT
        frame_data = parse_mot_gt(gt_path)
        if not frame_data: continue
        
        # Output dir
        seq_out = OUTPUT_DIR / "labels" / seq_name
        seq_out.mkdir(parents=True, exist_ok=True)
        
        # Determine image extension and format
        sample_img = next(img_dir.glob("*.jpg"), None) or next(img_dir.glob("*.png"), None)
        if not sample_img: continue
        ext = sample_img.suffix
        
        # Determine number of digits in filename
        fname_len = len(sample_img.stem) # usually 6 (000001) or 4 (0001)
        fmt_str = f"{{:0{fname_len}d}}{ext}"

        # Process frames
        sorted_frames = sorted(frame_data.keys())
        
        # Optional: Skip frames? Process every frame?
        # SAM3 is fast enough? We'll see.
        
        for fid in tqdm(sorted_frames, desc=seq_name, leave=False):
            img_path = img_dir / fmt_str.format(fid)
            
            # Filter for vehicle boxes only (class 1)
            vehicles_gt = [x for x in frame_data[fid] if x[5] == 1]
            persons_gt = [x for x in frame_data[fid] if x[5] == 0] # or 2? VT-MOT usually 1=veh, 2=person? Config said 0=person, 1=vehicle.
            # Let's trust the config: 1=vehicle.
            
            # If no vehicles, just write persons
            final_labels = []
            
            # Get image dims
            h, w = 0, 0
            if os.path.exists(img_path):
                 # We need dims for normalization
                 # Reading image header is fast
                 with Image.open(img_path) as im: w, h = im.size
            else:
                continue

            # Classify vehicles
            if vehicles_gt:
                classification = classifier.process_image(img_path, vehicles_gt)
                
                for v in vehicles_gt:
                    tid, x, y, wb, hb, _ = v
                    cls_name = classification.get(tid, {"class": "car"})["class"]
                    cls_id = VEHICLE_TO_ID.get(cls_name, 1) # Default to car (1)
                    
                    # YOLO format
                    cx = (x + wb/2) / w
                    cy = (y + hb/2) / h
                    nw = wb / w
                    nh = hb / h
                    final_labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            # Add persons (Human = 3)
            # Annotations usually use class 1=car, 2=person? Or 1, 2? 
            # VT-MOT gt: "1,1,1" ... class 1.
            # Person sequences: "1,1,..."? No, usually distinct. The config said 0=person.
            # Let's simply remap whatever is NOT 1 (vehicle) to Human (3).
            
            for p in frame_data[fid]:
                if p[5] != 1: # Not vehicle
                     x, y, wb, hb = p[1], p[2], p[3], p[4]
                     cx = (x + wb/2) / w
                     cy = (y + hb/2) / h
                     nw = wb / w
                     nh = hb / h
                     final_labels.append(f"{PERSON_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            
            # Write label file
            if final_labels:
                out_path = seq_out / f"{fid:06d}.txt"
                with open(out_path, 'w') as f:
                    f.write('\n'.join(final_labels))

if __name__ == "__main__":
    main()
