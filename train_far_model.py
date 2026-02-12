#!/usr/bin/env python3
"""
Train Far View MCF Model (YOLOv11x RGBT)
Optimized for long-range person detection (Vtuav, wurenji, RGBT234, qiuxing sources)
Has stronger scale augmentation for small objects
"""
import os
import torch
from pathlib import Path
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
from ultralytics.utils import loss as loss_module

# ⭐ v2.7 RESEARCH FIX: Patch v8DetectionLoss.bbox_decode for device robustness
_original_bbox_decode = loss_module.v8DetectionLoss.bbox_decode

def _patched_bbox_decode(self, anchor_points, pred_dist):
    """Auto-move self.proj to match pred_dist device (v2.7 standardization)."""
    if self.use_dfl:
        if self.proj.device != pred_dist.device:
            self.proj = self.proj.to(pred_dist.device)
    return _original_bbox_decode(self, anchor_points, pred_dist)

loss_module.v8DetectionLoss.bbox_decode = _patched_bbox_decode
print("[OK] Applied v2.7 bbox_decode device patch")

class MCFTrainer(DetectionTrainer):
    """Custom trainer that preserves pre-loaded MCF model."""
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, mcf_model=None):
        self._mcf_model = mcf_model
        super().__init__(cfg, overrides, _callbacks)
    
    def setup_model(self):
        """Override to use pre-loaded MCF model instead of rebuilding from YAML."""
        if self._mcf_model is not None:
            self.model = self._mcf_model
            self.model.to(self.device)
            self.model.args = self.args
            print(f"[OK] Using pre-loaded MCF model on {self.device} (Namespace: {type(self.args)})")
            return self.model
        else:
            return super().setup_model()

def train():
    project_root = Path("/home/student/Toan")
    weights_path = project_root / "weights/LLVIP-yolo11x-RGBT-midfusion-MCF.pt"
    dataset_yaml = project_root / "datasets/vtmot_far/far_view.yaml"
    runs_dir = project_root / "runs"
    
    # Check GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Training FAR VIEW model (long-range person detection)")
    print(f"Dataset: {dataset_yaml}")
    
    # Load model
    print(f"Loading weights: {weights_path}")
    model = YOLO(str(weights_path))
    
    # Phase 1: Backbone Frozen (Warm-up)
    freeze_layers = [2, 3, 4, 5, 6, 17, 18, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
    
    # Check if we can resume Phase 1
    phase1_last = runs_dir / "far_view_mcf_phase1/weights/last.pt"
    if phase1_last.exists():
        print(f"Resuming Phase 1 from {phase1_last}")
        model = YOLO(str(phase1_last))
    
    phase1_config = {
        'model': str(weights_path),
        'data': str(dataset_yaml),
        'epochs': 20, 
        'imgsz': 640,
        'batch': 20,  # Optimized for RTX 4090
        'device': device,
        'use_simotm': 'RGBRGB6C',
        'channels': 6,
        'pairs_rgb_ir': ['_rgb_', '_ir_'],
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'warmup_epochs': 1,
        'freeze': freeze_layers,
        'project': str(runs_dir),
        'name': 'far_view_mcf_phase1',
        'exist_ok': True,
        'cache': 'ram',
        'workers': 12,
    }
    
    print("\n--- PHASE 1: Warm-up (Far View) ---")
    trainer1 = MCFTrainer(overrides=phase1_config, mcf_model=model.model)
    trainer1.train()
    
    # Phase 2: Full Fine-Tuning
    best_phase1 = runs_dir / "far_view_mcf_phase1/weights/best.pt"
    if not best_phase1.exists():
        best_phase1 = runs_dir / "far_view_mcf_phase1/weights/last.pt"
    
    print(f"\nLoading phase 1 best: {best_phase1}")
    model2 = YOLO(str(best_phase1))
    
    phase2_config = {
        'model': str(best_phase1),
        'data': str(dataset_yaml),
        'epochs': 50,
        'imgsz': 640,
        'batch': 20,  # Optimized for RTX 4090
        'device': device,
        'use_simotm': 'RGBRGB6C',
        'channels': 6,
        'pairs_rgb_ir': ['_rgb_', '_ir_'],
        'optimizer': 'SGD',
        'lr0': 0.0005,
        'lrf': 0.01,
        'cos_lr': True,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.1,
        'scale': 0.9,  # Far view: STRONGER scale augmentation for small objects
        'close_mosaic': 10,
        'freeze': None,
        'project': str(runs_dir),
        'name': 'far_view_mcf_phase2',
        'exist_ok': True,
        'cache': 'ram',
        'workers': 12,
    }
    
    print("\n--- PHASE 2: Full Fine-tuning (Far View) ---")
    trainer2 = MCFTrainer(overrides=phase2_config, mcf_model=model2.model)
    trainer2.train()
    
    print("\n✓ Far View MCF Training Complete!")

if __name__ == "__main__":
    train()
