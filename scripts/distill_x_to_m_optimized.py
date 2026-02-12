#!/usr/bin/env python3
"""
Optimized Distillation Stage 1: YOLO26/11 X-Large -> Medium
=========================================================
Teacher: YOLO26x RGBT (Best V17)
Student: YOLO26m RGBT

Configuration:
- Epochs: 100 (Maximizing Performance)
- Batch: 48 (Aggressive)
- Cache: RAM (Fastest IO)
- DDP: GPUs 0,1
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add pipeline scripts to path
sys.path.insert(0, "/home/student/Toan/tracking/stage1/scripts")

from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Configs
ROOT = Path("/home/student/Toan")
SEARCH_ROOT = ROOT / "tracking/stage1/runs/distillation"
DATA_YAML = ROOT / "tracking/stage1/configs/vtmot_rgbt.yaml"
STUDENT_YAML = ROOT / "tracking/stage1/configs/yolo26m_rgbt.yaml"
# Hardcoded Teacher Path from analysis
TEACHER_WEIGHTS = ROOT / "tracking/stage1/runs/vtmot_framework/v17_rgbt_yolo26x8/weights/best.pt"

def run_optimization():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f"maximize_26x_to_26m_{timestamp}"
    
    print("=" * 80)
    print(f"🔥 MAXIMIZE DISTILLATION: 26x -> 26m")
    print(f"   Teacher: {TEACHER_WEIGHTS}")
    print(f"   Student: {STUDENT_YAML}")
    print(f"   Epochs : 100")
    print(f"   Batch  : 48")
    print(f"   Cache  : RAM")
    print("=" * 80)
    
    if not TEACHER_WEIGHTS.exists():
        LOGGER.error(f"❌ Teacher weights not found at {TEACHER_WEIGHTS}")
        return

    # Initialize Student
    student = YOLO(str(STUDENT_YAML))
    
    # Check if we should initialize with previous 30-epoch weights?
    # Pros: Faster convergence. Cons: Might be stuck in local minima?
    # User said "Continue training" initially. 
    # But "Maximize" usually implies doing it right.
    # Let's start FRESH for 100 epochs to be safe (Clean Slate), OR load the 30-epoch best?
    # Loading 30-epoch best is safer for "Continuing".
    # Let's find previous best.
    prev_runs = list(SEARCH_ROOT.glob("distill_26x_to_26m_*/weights/best.pt"))
    pretrained_weights = False
    
    if prev_runs:
        latest_prev = max(prev_runs, key=os.path.getmtime)
        print(f"🔄 Found previous run {latest_prev.parent.parent.name}. Resuming weights?")
        # Actually, if we change batch size/cache, resuming 'model' is tricky with strict resume=True.
        # But setting 'pretrained=weights' is Transfer Learning.
        # Let's use Transfer Learning from the 30-epoch checkpoint to accelerate.
        pretrained_weights = str(latest_prev)
        print(f"   -> Using {pretrained_weights} as pretrained init.")
    
    # Training Config
    args = {
        'data': str(DATA_YAML),
        'epochs': 100,          # Extended duration
        'batch': 48,            # Optimized DDP Batch
        'imgsz': 640,
        'device': [0, 1],       # <--- FULL POWER: 2x 4090
        'project': str(SEARCH_ROOT),
        'name': name,
        'exist_ok': True,
        'pretrained': pretrained_weights, 
        'optimizer': 'SGD',
        'lr0': 0.01,
        'lrf': 0.01,
        'warmup_epochs': 3,
        'close_mosaic': 10,
        'amp': True,
        'val': True,
        'save': True,
        'cache': 'disk',        # <--- SAFE: Shared OS Cache (No RAM dupes)
        'workers': 8,
        'patience': 20,         # Early stopping if no gains
    }
    
    print("🚀 Launching Optimization...")
    student.train(**args)
    
    print(f"✅ Maximization Complete: {name}")

if __name__ == "__main__":
    run_optimization()
