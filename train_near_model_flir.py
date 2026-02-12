#!/usr/bin/env python3
"""
Train Near View MCF Model (FLIR PRETRAINED EXPERIMENT)
Running on GPU 1
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
    runs_dir = project_root / "runs" # Re-added missing definition
    # CHANGED: FLIR Weights Initialization
    weights_path = project_root / "weights/FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.01-warmup_epochs1-SGD.pt"
    # CHANGED: Force GPU 1 (Resume Single)
    device = 1
    print(f"Using device: {device} (FLIR CLEAN DATASET)")
    
    # CHANGED: Dataset Path (Clean)
    # User request: "Use new Clean Dataset"
    dataset_yaml = project_root / "datasets/vtmot_near/near_view_clean.yaml"
    print(f"Dataset: {dataset_yaml}")
    
    # CHANGED: Smart Cache Logic
    # Dataset on disk is ~35GB (Compressed).
    cache_mode = 'disk'
    print(f"[SMART CACHE] Using disk cache (safer than RAM). Mode: '{cache_mode}'")
    
    # New Project Name for Clean Data
    phase1_name = 'near_view_flir_clean_phase1'
    project_dir = runs_dir / "flir_clean" # Group clean runs

    # Load model (RESUME from last.pt)
    last_ckpt = project_dir / f"{phase1_name}/weights/last.pt"
    if last_ckpt.exists():
        print(f"Resuming from checkpoint: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        resume_flag = True
    else:
        print(f"Loading weights: {weights_path}")
        model = YOLO(str(weights_path))
        resume_flag = False
    
    # Phase 1: Backbone Frozen (Warm-up)
    freeze_layers = [2, 3, 4, 5, 6, 17, 18, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
    
    phase1_config = {
        'model': str(last_ckpt) if resume_flag else str(weights_path),
        'data': str(dataset_yaml),
        'epochs': 20, 
        'imgsz': 640,
        'batch': 20,  # Optimized for RTX 4090 Single
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
        'resume': resume_flag, # Resume if checkpoint exists
    }
    
    print("\n--- PHASE 1: Warm-up (Clean Data) ---")
    # If resuming, let Trainer handle model loading (mcf_model=None). 
    # If starting fresh, pass the pre-loaded mcf_model.
    # trainer1 = MCFTrainer(overrides=phase1_config, mcf_model=None if resume_flag else model.model)
    # trainer1.train()
    print("[SKIP] Phase 1 Skipped by AI Engineer (Early Unfreeze Strategy). Using best.pt from previous run.")
    
    # Phase 2: Full Fine-Tuning
    phase2_name = 'near_view_flir_clean_phase2' 
    phase2_ckpt = runs_dir / "flir_clean" / phase2_name / "weights/last.pt"
    
    if phase2_ckpt.exists():
        print(f"\n[RESUME] Found Phase 2 checkpoint: {phase2_ckpt}")
        model2 = YOLO(str(phase2_ckpt))
        resume_phase2 = True
    else:
        best_phase1 = project_dir / f"{phase1_name}/weights/best.pt"
        if not best_phase1.exists():
            best_phase1 = project_dir / f"{phase1_name}/weights/last.pt"
        
        print(f"\n[START] Loading phase 1 best: {best_phase1}")
        model2 = YOLO(str(best_phase1))
        resume_phase2 = False
    
    phase2_config = {
        'model': str(phase2_ckpt) if resume_phase2 else str(best_phase1),
        'data': str(dataset_yaml),
        'epochs': 50,
        'imgsz': 640,
        'batch': 12, # Reduced further (16 failed)
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
        'resume': resume_phase2,
    }
    
    print("\n--- PHASE 2: Full Fine-tuning (FLIR Init) ---")
    trainer2 = MCFTrainer(overrides=phase2_config, mcf_model=None if resume_phase2 else model2.model)
    trainer2.train()
    
    print("\n✓ FLIR Experiment Training Complete!")

if __name__ == "__main__":
    train()
