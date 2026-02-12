#!/usr/bin/env python3
"""
Train Far View MCF Model (FLIR PRETRAINED)
Clean Dataset: vtmot_far
"""
import os
import torch
import psutil
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
    # Using FLIR weights as base
    weights_path = project_root / "weights/FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.01-warmup_epochs1-SGD.pt"
    
    # Force GPU 1 (or 0, depending on availability, defaulting to 1 for now if separate)
    device = 1 
    print(f"Using device: {device} (FLIR CLEAN DATASET - FAR VIEW)")
    
    dataset_yaml = project_root / "datasets/vtmot_far/far_view_clean.yaml"
    print(f"Dataset: {dataset_yaml}")
    
    runs_dir = project_root / "runs"
    
    # Disk Cache
    cache_mode = 'disk'
    print(f"[SMART CACHE] Using disk cache. Mode: '{cache_mode}'")
    
    # Load model (fresh)
    print(f"Loading weights: {weights_path}")
    model = YOLO(str(weights_path))
    
    # Phase 1: Backbone Frozen (Warm-up)
    freeze_layers = [2, 3, 4, 5, 6, 17, 18, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
    
    phase1_name = 'far_view_flir_clean_phase1'
    project_dir = runs_dir / "flir_clean" 
    
    phase1_config = {
        'model': str(weights_path),
        'data': str(dataset_yaml),
        'epochs': 20, 
        'imgsz': 640,
        'batch': 20, 
        'device': device,
        'use_simotm': 'RGBRGB6C',
        'channels': 6,
        'pairs_rgb_ir': ['_rgb_', '_ir_'],
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'warmup_epochs': 1,
        'freeze': freeze_layers,
        'project': str(project_dir),
        'name': phase1_name,
        'exist_ok': True,
        'cache': cache_mode,
        'workers': 12,
        'resume': False,
    }
    
    print("\n--- PHASE 1: Warm-up (Far View Clean) ---")
    trainer1 = MCFTrainer(overrides=phase1_config, mcf_model=model.model)
    trainer1.train()
    
    # Phase 2: Full Fine-Tuning
    phase2_name = 'far_view_flir_clean_phase2' 
    best_phase1 = project_dir / f"{phase1_name}/weights/best.pt"
    if not best_phase1.exists():
        best_phase1 = project_dir / f"{phase1_name}/weights/last.pt"
    
    print(f"\nLoading phase 1 best: {best_phase1}")
    model2 = YOLO(str(best_phase1))
    
    phase2_config = {
        'model': str(best_phase1),
        'data': str(dataset_yaml),
        'epochs': 50,
        'imgsz': 640,
        'batch': 20,
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
        'scale': 0.7, 
        'close_mosaic': 10,
        'freeze': None,
        'project': str(project_dir),
        'name': phase2_name,
        'exist_ok': True,
        'cache': cache_mode,
        'workers': 12,
    }
    
    print("\n--- PHASE 2: Full Fine-tuning (Far View Clean) ---")
    trainer2 = MCFTrainer(overrides=phase2_config, mcf_model=model2.model)
    trainer2.train()
    
    print("\n✓ Far View MCF Training Complete!")

if __name__ == "__main__":
    train()
