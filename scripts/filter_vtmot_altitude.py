#!/usr/bin/env python3
"""
VT-MOT Altitude Filter Script
Filters VT-MOT dataset to only include frames from 30-50m altitude range.
Based on DATASET_STRATEGY.md requirements.
"""

import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

class VTMOTAltitudeFilter:
    """Filter VT-MOT dataset by altitude metadata"""
    
    def __init__(self, vtmot_root, output_root, min_alt=30, max_alt=50):
        self.vtmot_root = Path(vtmot_root)
        self.output_root = Path(output_root)
        self.min_alt = min_alt
        self.max_alt = max_alt
        
    def estimate_altitude_from_bbox_size(self, label_file):
        """
        Estimate altitude from average person bbox size.
        Heuristic: At 30m, person ~40px height; at 50m, person ~25px height
        This is a fallback if no explicit altitude metadata exists.
        """
        if not label_file.exists():
            return None
        
        heights = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id == 0:  # person class
                        h = float(parts[4])  # normalized height
                        heights.append(h)
        
        if not heights:
            return None
        
        avg_height = np.mean(heights)
        
        # Altitude estimation (inverse relationship)
        # At 640x640, h=0.0625 (~40px) → ~30m; h=0.039 (~25px) → ~50m
        if avg_height > 0.08:  # Person too big → altitude < 30m
            return 25
        elif avg_height > 0.05:  # Good range
            return 40  # ~35-45m
        elif avg_height > 0.03:  # OK range
            return 50  # ~45-55m
        else:  # Person too small → altitude > 50m
            return 70
    
    def filter_dataset(self):
        """Filter VT-MOT to altitude range"""
        
        # VT-MOT structure: images/{split}/ and labels/{split}/
        splits = ['train', 'val', 'test']
        
        stats = {
            'total': 0,
            'kept': 0,
            'rejected': 0,
            'altitude_dist': {}
        }
        
        for split in splits:
            img_dir = self.vtmot_root / 'images' / split
            lbl_dir = self.vtmot_root / 'labels' / split
            
            if not img_dir.exists():
                print(f"Skip {split}: {img_dir} not found")
                continue
            
            out_img_dir = self.output_root / 'images' / split
            out_lbl_dir = self.output_root / 'labels' / split
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            print(f"\nProcessing {split} split ({len(images)} images)...")
            
            for img_path in tqdm(images, desc=split):
                stats['total'] += 1
                
                # Find corresponding label
                lbl_path = lbl_dir / (img_path.stem + '.txt')
                
                # Estimate altitude
                alt = self.estimate_altitude_from_bbox_size(lbl_path)
                
                if alt is None:
                    # No persons detected, keep for background diversity
                    alt = 40  # Assume mid-range
                
                # Track altitude distribution
                alt_bucket = (alt // 10) * 10
                stats['altitude_dist'][alt_bucket] = stats['altitude_dist'].get(alt_bucket, 0) + 1
                
                # Filter by altitude
                if self.min_alt <= alt <= self.max_alt:
                    stats['kept'] += 1
                    
                    # Copy image and label
                    shutil.copy(img_path, out_img_dir / img_path.name)
                    if lbl_path.exists():
                        shutil.copy(lbl_path, out_lbl_dir / lbl_path.name)
                    else:
                        # Create empty label file
                        (out_lbl_dir / (img_path.stem + '.txt')).touch()
                else:
                    stats['rejected'] += 1
        
        # Print stats
        print(f"\n✅ Altitude Filtering Complete")
        print(f"   Total: {stats['total']}")
        print(f"   Kept ({self.min_alt}-{self.max_alt}m): {stats['kept']} ({100*stats['kept']/max(stats['total'],1):.1f}%)")
        print(f"   Rejected: {stats['rejected']}")
        print(f"\n   Altitude Distribution:")
        for alt, count in sorted(stats['altitude_dist'].items()):
            bar = '█' * int(count / max(stats['altitude_dist'].values()) * 20)
            print(f"     {alt:3}m: {count:6} {bar}")
        
        return stats
    
    def create_yaml_config(self):
        """Create YOLO dataset config for filtered data"""
        config = f"""# VT-MOT Filtered Dataset (30-50m altitude)
# Auto-generated by filter_vtmot_altitude.py

path: {self.output_root}
train: images/train
val: images/val
test: images/test

# Classes
nc: 2
names:
  0: person
  1: vehicle
"""
        yaml_path = self.output_root / 'data_vtmot_filtered.yaml'
        with open(yaml_path, 'w') as f:
            f.write(config)
        print(f"\n✅ Created config: {yaml_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Filter VT-MOT by altitude')
    parser.add_argument('--input', default='/home/student/Toan/data/VT-MOT',
                        help='VT-MOT root directory')
    parser.add_argument('--output', default='/home/student/Toan/data/VT-MOT_filtered',
                        help='Output directory for filtered data')
    parser.add_argument('--min-alt', type=int, default=30, help='Minimum altitude (m)')
    parser.add_argument('--max-alt', type=int, default=50, help='Maximum altitude (m)')
    
    args = parser.parse_args()
    
    print(f"VT-MOT Altitude Filter")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Range:  {args.min_alt}-{args.max_alt}m")
    
    filterer = VTMOTAltitudeFilter(
        args.input, 
        args.output,
        args.min_alt,
        args.max_alt
    )
    
    filterer.filter_dataset()
    filterer.create_yaml_config()


if __name__ == '__main__':
    main()
