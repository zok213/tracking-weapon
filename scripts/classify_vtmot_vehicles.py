#!/usr/bin/env python3
"""
VT-MOT Vehicle Sub-Classification using CLIP
=============================================
Classifies generic "vehicle" annotations into: car, truck, motorcycle

Uses CLIP for zero-shot classification of cropped vehicle regions.
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== Configuration =====
VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")
OUTPUT_DIR = Path("/home/student/Toan/data/VT-MOT_4cls")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Class mapping: Original VT-MOT -> Target 4-class
# VT-MOT: 0=person, 1=vehicle
# Target: 0=motorcycle, 1=car, 2=truck, 3=human
VEHICLE_CLASSES = ["motorcycle", "car", "truck", "bus"]
VEHICLE_TO_ID = {"motorcycle": 0, "car": 1, "truck": 2, "bus": 2}  # bus -> truck
PERSON_ID = 3  # human


class VehicleClassifier:
    """CLIP-based vehicle classifier"""
    
    def __init__(self, device=DEVICE):
        self.device = device
        logger.info(f"Loading CLIP model on {device}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Pre-compute text embeddings for vehicle classes
        self.class_prompts = [
            "a photo of a motorcycle",
            "a photo of a car",
            "a photo of a truck",
            "a photo of a bus"
        ]
        
        with torch.no_grad():
            text_inputs = self.processor(text=self.class_prompts, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            self.text_features = self.model.get_text_features(**text_inputs)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        
        logger.info("CLIP model loaded successfully!")
    
    def classify(self, image_crop: np.ndarray) -> tuple:
        """
        Classify a vehicle crop.
        
        Returns: (class_name, class_id, confidence)
        """
        # Preprocess image
        image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB) if len(image_crop.shape) == 3 else image_crop
        
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (image_features @ self.text_features.T).squeeze(0)
            probs = similarity.softmax(dim=0)
            
            best_idx = probs.argmax().item()
            confidence = probs[best_idx].item()
        
        class_name = VEHICLE_CLASSES[best_idx]
        class_id = VEHICLE_TO_ID[class_name]
        
        return class_name, class_id, confidence
    
    def classify_batch(self, crops: list) -> list:
        """Batch classification for efficiency"""
        if not crops:
            return []
        
        # Convert all crops to RGB
        rgb_crops = []
        for crop in crops:
            if len(crop.shape) == 3:
                rgb_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            else:
                rgb_crops.append(crop)
        
        inputs = self.processor(images=rgb_crops, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ self.text_features.T)
            probs = similarity.softmax(dim=-1)
            
            best_indices = probs.argmax(dim=-1).cpu().numpy()
            confidences = probs.max(dim=-1).values.cpu().numpy()
        
        results = []
        for idx, conf in zip(best_indices, confidences):
            class_name = VEHICLE_CLASSES[idx]
            class_id = VEHICLE_TO_ID[class_name]
            results.append((class_name, class_id, conf))
        
        return results


def parse_mot_gt(gt_path: Path) -> list:
    """
    Parse MOT gt.txt format.
    Format: frame_id, track_id, x, y, w, h, conf, class_id, visibility
    
    Returns: list of (frame_id, track_id, x, y, w, h, class_id)
    """
    annotations = []
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                class_id = int(parts[7]) if len(parts) > 7 else 1
                annotations.append((frame_id, track_id, x, y, w, h, class_id))
    return annotations


def process_sequence(seq_path: Path, classifier: VehicleClassifier, output_dir: Path, batch_size: int = 32):
    """
    Process a single VT-MOT sequence.
    
    1. Load gt.txt annotations
    2. For each vehicle bbox, crop and classify
    3. Save new 4-class YOLO labels
    """
    seq_name = seq_path.name
    gt_path = seq_path / "gt" / "gt.txt"
    rgb_path = seq_path / "img1"  # RGB images
    
    if not gt_path.exists():
        logger.warning(f"No gt.txt found for {seq_name}")
        return 0, 0
    
    # VT-MOT uses 'visible' folder for RGB images
    rgb_path = seq_path / "visible"
    if not rgb_path.exists():
        rgb_path = seq_path / "img1"
        if not rgb_path.exists():
            rgb_path = seq_path / "rgb"
            if not rgb_path.exists():
                logger.warning(f"No image folder found for {seq_name}")
                return 0, 0
    
    # Parse annotations
    annotations = parse_mot_gt(gt_path)
    if not annotations:
        return 0, 0
    
    # Group by frame
    frame_annotations = {}
    for ann in annotations:
        frame_id = ann[0]
        if frame_id not in frame_annotations:
            frame_annotations[frame_id] = []
        frame_annotations[frame_id].append(ann)
    
    # Output directory for this sequence
    seq_output = output_dir / "labels" / seq_name
    seq_output.mkdir(parents=True, exist_ok=True)
    
    processed = 0
    vehicles_classified = 0
    
    # Process each frame
    for frame_id, anns in tqdm(frame_annotations.items(), desc=f"Processing {seq_name}", leave=False):
        # Find image file
        img_candidates = [
            rgb_path / f"{frame_id:06d}.jpg",
            rgb_path / f"{frame_id:06d}.png",
            rgb_path / f"{frame_id:04d}.jpg",
            rgb_path / f"{frame_id:04d}.png",
        ]
        
        img_path = None
        for candidate in img_candidates:
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            continue
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_h, img_w = img.shape[:2]
        
        # Collect vehicle crops for batch processing
        vehicle_indices = []
        vehicle_crops = []
        yolo_labels = []
        
        for i, ann in enumerate(anns):
            frame_id, track_id, x, y, w, h, class_id = ann
            
            if class_id == 0:  # Person -> human (3)
                # Convert to YOLO format (normalized)
                cx = (x + w/2) / img_w
                cy = (y + h/2) / img_h
                nw = w / img_w
                nh = h / img_h
                yolo_labels.append(f"{PERSON_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                processed += 1
            
            elif class_id == 1:  # Vehicle -> needs classification
                # Crop vehicle region
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(img_w, int(x + w))
                y2 = min(img_h, int(y + h))
                
                if x2 > x1 and y2 > y1:
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        vehicle_indices.append(i)
                        vehicle_crops.append(crop)
        
        # Batch classify vehicles
        if vehicle_crops:
            # Process in batches
            for batch_start in range(0, len(vehicle_crops), batch_size):
                batch_end = min(batch_start + batch_size, len(vehicle_crops))
                batch_crops = vehicle_crops[batch_start:batch_end]
                batch_indices = vehicle_indices[batch_start:batch_end]
                
                results = classifier.classify_batch(batch_crops)
                
                for idx, (class_name, class_id, conf) in zip(batch_indices, results):
                    ann = anns[idx]
                    _, _, x, y, w, h, _ = ann
                    
                    # Convert to YOLO format
                    cx = (x + w/2) / img_w
                    cy = (y + h/2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    
                    yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    vehicles_classified += 1
                    processed += 1
        
        # Save YOLO label file
        if yolo_labels:
            label_file = seq_output / f"{frame_id:06d}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_labels))
    
    return processed, vehicles_classified


def main():
    """Main entry point"""
    logger.info("=" * 80)
    logger.info("VT-MOT Vehicle Sub-Classification Pipeline")
    logger.info("=" * 80)
    
    # Find all sequences
    sequences = []
    for split in ["test", "train", "val"]:
        split_path = VT_MOT_ROOT / "images" / split
        if split_path.exists():
            for seq in split_path.iterdir():
                if seq.is_dir():
                    sequences.append((split, seq))
    
    logger.info(f"Found {len(sequences)} sequences to process")
    
    # Initialize classifier
    classifier = VehicleClassifier()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process all sequences
    total_processed = 0
    total_vehicles = 0
    
    for split, seq_path in tqdm(sequences, desc="Processing sequences"):
        output_dir = OUTPUT_DIR / split
        processed, vehicles = process_sequence(seq_path, classifier, output_dir)
        total_processed += processed
        total_vehicles += vehicles
    
    logger.info("=" * 80)
    logger.info(f"COMPLETE!")
    logger.info(f"Total annotations processed: {total_processed:,}")
    logger.info(f"Vehicles classified: {total_vehicles:,}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
