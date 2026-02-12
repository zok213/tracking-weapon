#!/usr/bin/env python3
"""
Framework-Based Training Script for VT-MOT Person-Only RGBT
===========================================================
Replaces Manual Loop with Ultralytics Native Engine (RGBTTrainer).
Uses 'yolo26x' for Large Model performance.
Uses 'AdamW' optimizer (proven in KUST4K).
"""

import os
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/student/Toan/tracking/stage1")
sys.path.insert(0, str(ROOT / 'scripts'))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Optional, let torch decide or user arg

import torch
from rgbt_pipeline_utils import RGBTTrainer
from rgbt_model_utils import create_rgbt_yolo, verify_rgbt_model
from ultralytics import YOLO

def train_vtmot_framework():
    """Train VT-MOT using RGBTTrainer (Framework Approach)."""
    
    print("=" * 80)
    print("STAGE 1C: VT-MOT FRAMEWORK TRAINING (RGBT YOLO26x)")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Paths
    project_dir = ROOT / 'runs' / 'vtmot_framework'
    name = 'v17_rgbt_yolo26x'
    
    # 1. Prepare Model (4-Channel Patching)
    # We create it, verify it, and save it to disk so Trainer can load it.
    model_save_path = ROOT / 'models' / 'yolo26x_rgbt_framework_init.pt'
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_save_path.exists():
        print(f"📦 Found existing 4ch init model: {model_save_path}")
        model_path = str(model_save_path)
    else:
        # Create fresh 4-channel model
        print("🆕 Creating fresh 4-channel YOLO26x model...")
        # Note: Ensure yolo26x.pt is available or downloaded
        model = create_rgbt_yolo(
            base_model='yolo26x.pt',
            save_path=str(model_save_path)
        )
        if not verify_rgbt_model(model):
            raise RuntimeError("Failed to create 4-channel model")
        model_path = str(model_save_path)

    # 2. Config (Aligned with train_improved.py but tweaked for VT-MOT)
    config = {
        # Explicitly define 4-channel Architecture
        'model': str(ROOT / 'configs/yolo26x_rgbt.yaml'),
        'pretrained': model_path, # Load our 4-channel weights
        
        'data': str(ROOT / 'configs/vtmot_rgbt.yaml'),
        'epochs': 50,
        'batch': 32,            # DDP MAX: 32 total / 2 GPUs = 16 per GPU (~17-19GB VRAM)
        'imgsz': 640,
        'device': [0, 1],       # Dual-GPU DDP
        'project': str(project_dir),
        'name': name,
        
        # Optimizer (AdamW - Proven in KUST4K)
        'optimizer': 'AdamW',
        'lr0': 0.001,           # Standard start
        'lrf': 0.01,            # Final LR factor
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        
        # Loss (Boost Person Class)
        'box': 7.5,
        'cls': 1.0,         # Slightly higher
        'dfl': 1.5,
        
        # RGBT Specifics
        'hsv_h': 0.0,       # Disable HSV (corrupts 4ch structure usually unless adapted)
        'hsv_s': 0.0,
        'hsv_v': 0.0,
        'mosaic': 1.0,      # Enable Mosaic (RGBTDataset handles buffer)
        'mixup': 0.0,       # Disable Mixup for stability first
        
        # Training
        'patience': 20,
        'save': True,
        'save_period': 5,
        'val': True,        # Enable validation (RGBTValidator)
        'plots': True,
        'verbose': True,
        'workers': 4,
        'amp': True,
        'deterministic': True,
        'seed': 42,
    }
    
    print("📋 Configuration:")
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    # 3. Train
    print("\n🚀 Starting Training using RGBTTrainer...")
    trainer = RGBTTrainer(overrides=config)
    try:
        results = trainer.train()
        print("\n✅ Training Complete.")
        return results
    except Exception as e:
        print(f"\n❌ Training Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_vtmot_framework.py = train_vtmot_framework # Name consistency
    train_vtmot_framework()
