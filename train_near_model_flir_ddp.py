#!/usr/bin/env python3
"""
Train Near View MCF Model (FLIR DDP ACCELERATED)
Running on GPU 0 + 1 (Distributed Data Parallel)
"""
import os
import psutil
from pathlib import Path
from ultralytics import YOLO
import mcf_utils  # Import the separate module

# Apply patch immediately on import
mcf_utils.apply_patch()

def train():
    project_root = Path("/home/student/Toan")
    # Original Weights (for reference or restart)
    weights_path = project_root / "weights/FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.01-warmup_epochs1-SGD.pt"
    dataset_yaml = project_root / "datasets/vtmot_near/near_view.yaml"
    runs_dir = project_root / "runs"
    
    # DDP Device Config
    device = [0, 1]
    print(f"Using device: {device} (FLIR DDP ACCELERATED)")
    
    # Smart Cache Logic
    # Dataset is ~988GB uncompressed. 
    # USER REQUEST: Do not cache (too large).
    cache_mode = False 
    print(f"[SMART CACHE] Disabled (Dataset too large). Mode: '{cache_mode}'")
    
    # ---------------------------------------------------------
    # Phase 1: Backbone Frozen (Warm-up)
    # ---------------------------------------------------------
    freeze_layers = [2, 3, 4, 5, 6, 17, 18, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
    phase1_name = 'near_view_mcf_flir_phase1'
    phase1_last = runs_dir / f"{phase1_name}/weights/last.pt"
    
    # Logic: Look for existing checkpoint to resume
    model_to_load = str(weights_path)
    if phase1_last.exists():
        print(f"Found Phase 1 checkpoint: {phase1_last}")
        model_to_load = str(phase1_last)
    
    print(f"Loading model: {model_to_load}")
    model = YOLO(model_to_load)

    # Config
    phase1_config = {
        'model': model_to_load,
        'data': str(dataset_yaml),
        'epochs': 20, 
        'imgsz': 640,
        'batch': 40,
        'device': device,
        'use_simotm': 'RGBRGB6C',
        'channels': 6,
        'pairs_rgb_ir': ['_rgb_', '_ir_'],
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'warmup_epochs': 1,
        'freeze': freeze_layers,
        'project': str(runs_dir),
        'name': phase1_name,
        'exist_ok': True,
        'cache': cache_mode,
        'workers': 12,
    }
    
    print("\n--- PHASE 1: Warm-up (FLIR DDP) ---")
    # Use imported MCFTrainer
    trainer1 = mcf_utils.MCFTrainer(overrides=phase1_config, mcf_model=model.model)
    trainer1.train()
    
    # ---------------------------------------------------------
    # Phase 2: Full Fine-Tuning
    # ---------------------------------------------------------
    phase2_name = 'near_view_mcf_flir_phase2'
    best_phase1 = runs_dir / f"{phase1_name}/weights/best.pt"
    if not best_phase1.exists():
        best_phase1 = runs_dir / f"{phase1_name}/weights/last.pt"
    
    print(f"\nLoading phase 1 best: {best_phase1}")
    model2 = YOLO(str(best_phase1))
    
    phase2_config = {
        'model': str(best_phase1),
        'data': str(dataset_yaml),
        'batch': 32, # Safe for 24GB VRAM (16 per GPU)
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
        'project': str(runs_dir),
        'name': phase2_name,
        'exist_ok': True,
        'cache': cache_mode,
        'workers': 12,
    }
    
    print("\n--- PHASE 2: Full Fine-tuning (FLIR DDP) ---")
    trainer2 = mcf_utils.MCFTrainer(overrides=phase2_config, mcf_model=model2.model)
    trainer2.train()
    
    print("\n✓ FLIR DDP Training Complete!")

if __name__ == "__main__":
    train()
