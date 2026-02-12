#!/usr/bin/env python3
"""
Ensemble Inference for Dual RGBT Person Detection
Combines Near View and Far View MCF models for multi-scale person detection

Strategy:
- Near View model: Better for large/close persons (bbox area > 0.01)
- Far View model: Better for small/distant persons (bbox area <= 0.01)
- Uses Weighted Box Fusion (WBF) or NMS to merge predictions
"""
import os
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict


def boxes_to_list(results):
    """Convert YOLO results to list of boxes with scores."""
    boxes = []
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                boxes.append([x1, y1, x2, y2, conf, cls])
    return boxes


def nms_fusion(boxes_near, boxes_far, iou_threshold=0.5, score_boost_near_large=1.1, score_boost_far_small=1.1):
    """
    Merge predictions from Near View and Far View models with score boosting.
    
    Args:
        boxes_near: List of [x1, y1, x2, y2, score, class] from Near View model
        boxes_far: List of [x1, y1, x2, y2, score, class] from Far View model
        iou_threshold: IoU threshold for NMS
        score_boost_near_large: Boost for large boxes from Near View
        score_boost_far_small: Boost for small boxes from Far View
    """
    all_boxes = []
    
    # Process Near View boxes - boost large boxes
    for box in boxes_near:
        x1, y1, x2, y2, score, cls = box
        area = (x2 - x1) * (y2 - y1)
        # Relative area to image (assuming normalized or use absolute)
        if area > 0.01:  # Large box threshold
            score = min(score * score_boost_near_large, 1.0)
        all_boxes.append([x1, y1, x2, y2, score, cls, 'near'])
    
    # Process Far View boxes - boost small boxes
    for box in boxes_far:
        x1, y1, x2, y2, score, cls = box
        area = (x2 - x1) * (y2 - y1)
        if area <= 0.01:  # Small box threshold
            score = min(score * score_boost_far_small, 1.0)
        all_boxes.append([x1, y1, x2, y2, score, cls, 'far'])
    
    if not all_boxes:
        return []
    
    # Sort by score (descending)
    all_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # Apply NMS
    keep = []
    while all_boxes:
        best = all_boxes.pop(0)
        keep.append(best[:6])  # Remove source tag
        
        remaining = []
        for box in all_boxes:
            if compute_iou(best[:4], box[:4]) < iou_threshold:
                remaining.append(box)
        all_boxes = remaining
    
    return keep


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


class EnsembleDetector:
    """
    Ensemble detector combining Near View and Far View MCF models.
    """
    
    def __init__(self, near_weights, far_weights, device=0):
        """
        Initialize ensemble detector.
        
        Args:
            near_weights: Path to Near View model weights
            far_weights: Path to Far View model weights
            device: GPU device (0, 1, ...) or 'cpu'
        """
        print(f"Loading Near View model: {near_weights}")
        self.near_model = YOLO(str(near_weights))
        
        print(f"Loading Far View model: {far_weights}")
        self.far_model = YOLO(str(far_weights))
        
        self.device = device
        print(f"Ensemble detector ready on device: {device}")
    
    def predict(self, source, conf=0.25, iou_threshold=0.5, 
                boost_near_large=1.1, boost_far_small=1.1,
                use_tta=False, verbose=True):
        """
        Run ensemble prediction.
        
        Args:
            source: Image path, directory, or list of images
            conf: Confidence threshold
            iou_threshold: NMS IoU threshold for fusion
            boost_near_large: Score boost for large boxes from Near View
            boost_far_small: Score boost for small boxes from Far View
            use_tta: Enable Test-Time Augmentation
            verbose: Print detection counts
        
        Returns:
            List of fused predictions per image
        """
        # Run both models
        near_results = self.near_model.predict(
            source, conf=conf, device=self.device, augment=use_tta, verbose=False
        )
        far_results = self.far_model.predict(
            source, conf=conf, device=self.device, augment=use_tta, verbose=False
        )
        
        fused_results = []
        
        for i, (near_r, far_r) in enumerate(zip(near_results, far_results)):
            # Convert to box lists
            boxes_near = boxes_to_list([near_r])
            boxes_far = boxes_to_list([far_r])
            
            # Fuse predictions
            fused = nms_fusion(
                boxes_near, boxes_far, 
                iou_threshold=iou_threshold,
                score_boost_near_large=boost_near_large,
                score_boost_far_small=boost_far_small
            )
            
            fused_results.append({
                'boxes': fused,
                'near_count': len(boxes_near),
                'far_count': len(boxes_far),
                'fused_count': len(fused)
            })
            
            if verbose:
                print(f"Image {i}: Near={len(boxes_near)}, Far={len(boxes_far)}, Fused={len(fused)}")
        
        return fused_results


def main():
    """Example usage of ensemble detector."""
    project_root = Path("/home/student/Toan")
    runs_dir = project_root / "runs"
    
    # Model paths (update after training completes)
    near_weights = runs_dir / "near_view_mcf_phase2/weights/best.pt"
    far_weights = runs_dir / "far_view_mcf_phase2_v2/weights/best.pt"
    
    # Check if models exist
    if not near_weights.exists():
        near_weights = runs_dir / "near_view_mcf_phase1/weights/best.pt"
    if not far_weights.exists():
        far_weights = runs_dir / "far_view_mcf_phase1_v2/weights/best.pt"
    
    print("=" * 60)
    print("RGBT Ensemble Person Detector (Near View + Far View)")
    print("=" * 60)
    
    if not near_weights.exists() or not far_weights.exists():
        print("ERROR: Model weights not found. Complete training first.")
        print(f"  Near View: {near_weights} (exists: {near_weights.exists()})")
        print(f"  Far View: {far_weights} (exists: {far_weights.exists()})")
        return
    
    # Initialize ensemble
    detector = EnsembleDetector(near_weights, far_weights, device=0)
    
    # Example prediction
    test_image = project_root / "datasets/vtmot_near/images/val"
    if test_image.exists():
        results = detector.predict(
            str(test_image),
            conf=0.25,
            iou_threshold=0.5,
            boost_near_large=1.1,
            boost_far_small=1.1,
            use_tta=False
        )
        
        total_detections = sum(r['fused_count'] for r in results)
        print(f"\nTotal ensemble detections: {total_detections}")
    else:
        print(f"Test image path not found: {test_image}")


if __name__ == "__main__":
    main()
