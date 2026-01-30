#!/usr/bin/env python3
"""
KUST4K RGBT Dataset Creator
Creates 4-channel (RGB + Thermal) for YOLO training
Only uses the 537 paired RGB+TIR images
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

def create_kust4k_rgbt():
    kust4k_root = Path("/home/student/Toan/data/KUST4K/Kust4K")
    output_root = Path("/home/student/Toan/data/KUST4K_RGBT")
    labels_src = Path("/home/student/Toan/data/KUST4K_yolo/labels")
    
    rgb_dir = kust4k_root / "RGB"
    tir_dir = kust4k_root / "TIR"
    
    # Find paired images
    rgb_files = set(f.name for f in rgb_dir.glob("*.png"))
    tir_files = set(f.name for f in tir_dir.glob("*.png"))
    paired = sorted(rgb_files & tir_files)
    
    print(f"Found {len(paired)} paired RGBT images")
    
    # Create splits (80/10/10)
    np.random.seed(42)
    np.random.shuffle(paired)
    n = len(paired)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    splits = {
        'train': paired[:n_train],
        'val': paired[n_train:n_train+n_val],
        'test': paired[n_train+n_val:]
    }
    
    created = 0
    for split_name, images in splits.items():
        out_img_dir = output_root / "images" / split_name
        out_lbl_dir = output_root / "labels" / split_name
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCreating {split_name} ({len(images)} images)...")
        
        for img_name in tqdm(images, desc=split_name):
            rgb_path = rgb_dir / img_name
            tir_path = tir_dir / img_name
            
            # Read RGB (3 channels)
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb is None:
                continue
            
            # Read TIR (grayscale)
            tir = cv2.imread(str(tir_path), cv2.IMREAD_GRAYSCALE)
            if tir is None:
                continue
            
            # Resize TIR to match RGB if needed
            if tir.shape[:2] != rgb.shape[:2]:
                tir = cv2.resize(tir, (rgb.shape[1], rgb.shape[0]))
            
            # Stack to 4 channels: [B, G, R, T]
            rgbt = np.dstack([rgb, tir])  # Shape: (H, W, 4)
            
            # Save as NPY (preserves 4 channels)
            npy_name = img_name.replace('.png', '.npy')
            np.save(out_img_dir / npy_name, rgbt)
            
            # Copy corresponding label (from YOLO conversion)
            lbl_name = img_name.replace('.png', '.txt')
            for lbl_split in ['train', 'val', 'test']:
                lbl_src = labels_src / lbl_split / lbl_name
                if lbl_src.exists():
                    shutil.copy(lbl_src, out_lbl_dir / lbl_name)
                    break
            
            created += 1
    
    # Create YAML config
    config = f"""# KUST4K RGBT 4-Channel Dataset
# Pretrain dataset for thermal fusion learning

path: {output_root}
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: person
  1: vehicle

# 4-channel input
channels: 4  # [R, G, B, T]
"""
    yaml_path = output_root / "data_kust4k_rgbt.yaml"
    with open(yaml_path, 'w') as f:
        f.write(config)
    
    print(f"\n✅ Created {created} 4-channel RGBT images")
    print(f"✅ Config saved to: {yaml_path}")
    print(f"\nSplits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


if __name__ == "__main__":
    create_kust4k_rgbt()
