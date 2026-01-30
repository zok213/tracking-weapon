#!/usr/bin/env python3
"""
KUST4K Dataset Converter and Anchor Optimizer
Converts segmentation masks to YOLO format and optimizes anchors
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import json
from tqdm import tqdm
import shutil

class KUST4KConverter:
    def __init__(self, kust4k_root, output_root):
        self.kust4k_root = Path(kust4k_root) / "Kust4K"
        self.output_root = Path(output_root)
        
        # KUST4K class mapping (from segmentation colors)
        # Typical RGBT tracking: person=1, vehicle=2, etc.
        self.class_map = {
            1: 0,  # person → YOLO class 0
            2: 1,  # vehicle → YOLO class 1
        }
        
        # Split files
        self.splits = {
            'train': ['broken_in_train_day_528.txt', 'broken_in_train_night_316.txt'],
            'val': ['broken_in_val_day_74.txt', 'broken_in_val_night_46.txt'],
            'test': ['broken_in_test_day_151.txt', 'broken_in_test_night_91.txt'],
        }
        
    def get_split_images(self):
        """Read split files to get image assignments"""
        splits = {'train': [], 'val': [], 'test': []}
        
        # Get broken images to exclude
        broken = set()
        for split, files in self.splits.items():
            for f in files:
                fpath = self.kust4k_root / "infos" / f
                if fpath.exists():
                    with open(fpath) as fp:
                        broken.update(line.strip() for line in fp)
        
        # All images
        all_images = sorted(os.listdir(self.kust4k_root / "RGB"))
        
        # Simple split: 70% train, 15% val, 15% test
        valid_images = [img for img in all_images if img not in broken]
        n = len(valid_images)
        
        splits['train'] = valid_images[:int(n * 0.7)]
        splits['val'] = valid_images[int(n * 0.7):int(n * 0.85)]
        splits['test'] = valid_images[int(n * 0.85):]
        
        print(f"Dataset splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        return splits
    
    def mask_to_bboxes(self, mask_path):
        """Convert segmentation mask to bounding boxes"""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return []
        
        bboxes = []
        h, w = mask.shape
        
        # Find unique class IDs (non-zero)
        unique_classes = np.unique(mask)
        
        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue
            
            # Create binary mask for this class
            class_mask = (mask == class_id).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                
                # Filter small boxes (noise)
                if bw < 5 or bh < 5:
                    continue
                
                # Convert to YOLO format (normalized x_center, y_center, width, height)
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                norm_w = bw / w
                norm_h = bh / h
                
                # Map to YOLO class
                yolo_class = self.class_map.get(class_id, 0)  # Default to person
                
                bboxes.append({
                    'class': yolo_class,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': norm_w,
                    'height': norm_h,
                    'abs_w': bw,
                    'abs_h': bh,
                })
        
        return bboxes
    
    def convert_dataset(self):
        """Convert KUST4K to YOLO format"""
        splits = self.get_split_images()
        all_bboxes = []  # For anchor optimization
        
        for split_name, images in splits.items():
            print(f"\nConverting {split_name} split ({len(images)} images)...")
            
            # Create directories
            img_dir = self.output_root / "images" / split_name
            lbl_dir = self.output_root / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            
            for img_name in tqdm(images, desc=split_name):
                # Source paths
                rgb_path = self.kust4k_root / "RGB" / img_name
                mask_path = self.kust4k_root / "Seg_annos" / img_name
                
                if not rgb_path.exists():
                    continue
                
                # Convert mask to bboxes
                if mask_path.exists():
                    bboxes = self.mask_to_bboxes(mask_path)
                else:
                    bboxes = []
                
                # Copy RGB image
                dst_img = img_dir / img_name
                if not dst_img.exists():
                    shutil.copy(rgb_path, dst_img)
                
                # Write YOLO label file
                label_name = img_name.rsplit('.', 1)[0] + '.txt'
                label_path = lbl_dir / label_name
                
                with open(label_path, 'w') as f:
                    for bbox in bboxes:
                        f.write(f"{bbox['class']} {bbox['x_center']:.6f} {bbox['y_center']:.6f} {bbox['width']:.6f} {bbox['height']:.6f}\n")
                        all_bboxes.append((bbox['abs_w'], bbox['abs_h']))
        
        print(f"\n✅ Converted {len(all_bboxes)} bounding boxes total")
        return all_bboxes
    
    def optimize_anchors(self, bboxes, n_anchors=9):
        """K-means clustering for optimal anchors"""
        if len(bboxes) < n_anchors:
            print("Not enough bboxes for anchor optimization")
            return None
        
        # Convert to numpy
        sizes = np.array(bboxes)
        
        # K-means clustering
        print(f"\nOptimizing anchors with {len(sizes)} boxes...")
        kmeans = KMeans(n_clusters=n_anchors, random_state=42, n_init=10)
        kmeans.fit(sizes)
        
        # Get cluster centers (sorted by area)
        anchors = kmeans.cluster_centers_
        anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
        
        print("\n✅ Optimized Anchors (width, height):")
        for i, (w, h) in enumerate(anchors):
            print(f"   Anchor {i+1}: ({w:.1f}, {h:.1f})")
        
        return anchors.tolist()
    
    def create_yaml_config(self, anchors=None):
        """Create YOLO dataset config"""
        config = f"""# KUST4K Dataset Configuration for YOLO
# Auto-generated by convert_kust4k.py

path: {self.output_root}
train: images/train
val: images/val
test: images/test

# Classes
nc: 2
names:
  0: person
  1: vehicle

# Optimized anchors (optional)
"""
        if anchors:
            config += f"# anchors: {anchors}\n"
        
        yaml_path = self.output_root / "data_kust4k.yaml"
        with open(yaml_path, 'w') as f:
            f.write(config)
        
        print(f"\n✅ Created config: {yaml_path}")

def main():
    # Paths
    kust4k_root = "/home/student/Toan/data/KUST4K"
    output_root = "/home/student/Toan/data/KUST4K_yolo"
    
    # Convert
    converter = KUST4KConverter(kust4k_root, output_root)
    bboxes = converter.convert_dataset()
    
    # Optimize anchors
    anchors = converter.optimize_anchors(bboxes)
    
    # Create config
    converter.create_yaml_config(anchors)
    
    # Save anchors
    if anchors:
        anchor_path = Path(output_root) / "anchors.json"
        with open(anchor_path, 'w') as f:
            json.dump({
                'anchors': anchors,
                'n_boxes': len(bboxes),
                'altitude': '30-60m (KUST4K)',
            }, f, indent=2)
        print(f"✅ Saved anchors to {anchor_path}")

if __name__ == "__main__":
    main()
