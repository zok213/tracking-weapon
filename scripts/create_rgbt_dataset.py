#!/usr/bin/env python3
"""
RGBT 4-Channel Dataset Preparation for VT-MOT
Creates 4-channel (RGB + Thermal) images for YOLO training

Night Vision Capability:
- RGB: 3 channels (visible light, fails in darkness)
- TIR: 1 channel (thermal infrared, works 24/7)
- Combined: 4 channels [R, G, B, T]
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

class RGBTDatasetCreator:
    """Create 4-channel RGBT dataset from VT-MOT"""
    
    def __init__(self, vtmot_root, output_root):
        self.vtmot_root = Path(vtmot_root)
        self.output_root = Path(output_root)
        
    def find_paired_images(self):
        """Find RGB-TIR paired images in VT-MOT"""
        pairs = []
        
        # VT-MOT structure varies, search for common patterns
        rgb_dirs = list(self.vtmot_root.glob('**/RGB')) + \
                   list(self.vtmot_root.glob('**/rgb')) + \
                   list(self.vtmot_root.glob('**/visible'))
        
        tir_dirs = list(self.vtmot_root.glob('**/TIR')) + \
                   list(self.vtmot_root.glob('**/tir')) + \
                   list(self.vtmot_root.glob('**/thermal')) + \
                   list(self.vtmot_root.glob('**/infrared'))
        
        print(f"Found {len(rgb_dirs)} RGB dirs, {len(tir_dirs)} TIR dirs")
        
        # Match pairs
        for rgb_dir in rgb_dirs:
            # Find corresponding TIR directory
            parent = rgb_dir.parent
            for tir_dir in tir_dirs:
                if tir_dir.parent == parent:
                    # Same parent, these are paired
                    rgb_images = set(f.stem for f in rgb_dir.glob('*.png'))
                    rgb_images.update(f.stem for f in rgb_dir.glob('*.jpg'))
                    
                    tir_images = set(f.stem for f in tir_dir.glob('*.png'))
                    tir_images.update(f.stem for f in tir_dir.glob('*.jpg'))
                    
                    common = rgb_images & tir_images
                    for stem in common:
                        rgb_path = rgb_dir / f"{stem}.png"
                        if not rgb_path.exists():
                            rgb_path = rgb_dir / f"{stem}.jpg"
                        tir_path = tir_dir / f"{stem}.png"
                        if not tir_path.exists():
                            tir_path = tir_dir / f"{stem}.jpg"
                        
                        if rgb_path.exists() and tir_path.exists():
                            pairs.append((rgb_path, tir_path))
        
        print(f"Found {len(pairs)} RGB-TIR pairs")
        return pairs
    
    def create_4channel_image(self, rgb_path, tir_path, output_path):
        """Combine RGB (3ch) + TIR (1ch) into 4-channel NPY"""
        # Read RGB
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            return False
        
        # Read TIR (grayscale)
        tir = cv2.imread(str(tir_path), cv2.IMREAD_GRAYSCALE)
        if tir is None:
            return False
        
        # Resize TIR to match RGB if needed
        if tir.shape[:2] != rgb.shape[:2]:
            tir = cv2.resize(tir, (rgb.shape[1], rgb.shape[0]))
        
        # Stack to 4 channels: [B, G, R, T]
        rgbt = np.dstack([rgb, tir])  # Shape: (H, W, 4)
        
        # Save as NPY (preserves 4 channels)
        np.save(output_path, rgbt)
        
        return True
    
    def create_dataset(self, split_ratio=(0.8, 0.1, 0.1)):
        """Create train/val/test splits of 4-channel data"""
        pairs = self.find_paired_images()
        
        if len(pairs) == 0:
            print("❌ No RGBT pairs found!")
            return
        
        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(pairs)
        
        n = len(pairs)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])
        
        splits = {
            'train': pairs[:n_train],
            'val': pairs[n_train:n_train+n_val],
            'test': pairs[n_train+n_val:]
        }
        
        stats = {'total': 0, 'success': 0}
        
        for split_name, split_pairs in splits.items():
            out_dir = self.output_root / 'images' / split_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nProcessing {split_name} ({len(split_pairs)} pairs)...")
            
            for rgb_path, tir_path in tqdm(split_pairs, desc=split_name):
                stats['total'] += 1
                
                out_name = f"{rgb_path.stem}.npy"
                out_path = out_dir / out_name
                
                if self.create_4channel_image(rgb_path, tir_path, out_path):
                    stats['success'] += 1
                    
                    # Copy label if exists
                    label_src = rgb_path.parent.parent / 'labels' / f"{rgb_path.stem}.txt"
                    if not label_src.exists():
                        label_src = rgb_path.parent.parent / 'annotations' / f"{rgb_path.stem}.txt"
                    
                    if label_src.exists():
                        label_dst = self.output_root / 'labels' / split_name
                        label_dst.mkdir(parents=True, exist_ok=True)
                        shutil.copy(label_src, label_dst / f"{rgb_path.stem}.txt")
        
        print(f"\n✅ Created {stats['success']}/{stats['total']} 4-channel images")
        
        # Create YAML config
        self.create_yaml_config()
    
    def create_yaml_config(self):
        """Create dataset config for 4-channel training"""
        config = f"""# VT-MOT RGBT 4-Channel Dataset
# For night vision: RGB (day) + Thermal (24/7)

path: {self.output_root}
train: images/train
val: images/val
test: images/test

# Classes (same as KUST4K)
nc: 2
names:
  0: person
  1: vehicle

# 4-channel input mode
channels: 4  # [R, G, B, T]
"""
        yaml_path = self.output_root / 'data_rgbt.yaml'
        with open(yaml_path, 'w') as f:
            f.write(config)
        print(f"✅ Created config: {yaml_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/home/student/Toan/data/VT-MOT')
    parser.add_argument('--output', default='/home/student/Toan/data/VT-MOT_RGBT')
    args = parser.parse_args()
    
    creator = RGBTDatasetCreator(args.input, args.output)
    creator.create_dataset()


if __name__ == '__main__':
    main()
