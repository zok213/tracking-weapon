#!/usr/bin/env python3
"""
KUST4K RGBT 4-Channel Training
Custom training for RGB+Thermal fusion using NPY files
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import random

class RGBTDataset(Dataset):
    """4-channel RGBT dataset from NPY files"""
    
    def __init__(self, img_dir, lbl_dir, img_size=640, augment=True):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Find all NPY files
        self.images = sorted(self.img_dir.glob('*.npy'))
        print(f"Found {len(self.images)} RGBT images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load 4-channel image
        img_path = self.images[idx]
        img = np.load(img_path)  # Shape: (H, W, 4)
        
        # Load labels
        lbl_path = self.lbl_dir / (img_path.stem + '.txt')
        labels = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append([float(x) for x in parts[:5]])
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                for lbl in labels:
                    lbl[1] = 1 - lbl[1]  # Flip x_center
        
        # Normalize and transpose to (C, H, W)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # (4, H, W)
        
        # Convert labels to tensor
        if labels:
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)
        
        return torch.tensor(img), labels, str(img_path)


def create_4ch_yolo(pretrained_path, device='cuda:1'):
    """Create YOLO with 4-channel input"""
    from ultralytics import YOLO
    
    model = YOLO(pretrained_path)
    
    # Modify first conv layer
    first_conv = model.model.model[0].conv
    
    new_conv = nn.Conv2d(
        in_channels=4,  # RGB + Thermal
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None
    )
    
    # Initialize: copy RGB weights, avg for thermal
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = first_conv.weight
        new_conv.weight[:, 3:4, :, :] = first_conv.weight.mean(dim=1, keepdim=True)
        if first_conv.bias is not None:
            new_conv.bias = first_conv.bias
    
    # Replace in model
    model.model.model[0].conv = new_conv
    model.model.to(device)
    
    print(f"✅ Modified YOLO for 4-channel input")
    print(f"   First conv: {new_conv.weight.shape}")
    
    return model


def train_rgbt_simple():
    """Simple RGBT training loop"""
    from ultralytics import YOLO
    
    # For now, use standard YOLO training with RGB images
    # The 4-channel NPY requires custom dataloader integration
    
    # Train on the 3-channel version first as baseline
    model = YOLO('/home/student/Toan/runs/detect/checkpoints/teacher/yolo26x_kust4k/weights/best.pt')
    
    # Since Ultralytics doesn't support 4ch natively,
    # we'll document the approach for custom implementation
    print("Note: Native 4-channel requires custom dataloader")
    print("Using standard YOLO training for now")
    

if __name__ == '__main__':
    print("KUST4K RGBT Training")
    print("=" * 40)
    
    # Create dataset
    train_dataset = RGBTDataset(
        img_dir='/home/student/Toan/data/KUST4K_RGBT/images/train',
        lbl_dir='/home/student/Toan/data/KUST4K_RGBT/labels/train',
        img_size=640,
        augment=True
    )
    
    # Test loading
    img, labels, path = train_dataset[0]
    print(f"Image shape: {img.shape}")  # Should be (4, 640, 640)
    print(f"Labels shape: {labels.shape}")
    print(f"Path: {path}")
    
    # Create 4-ch model
    model = create_4ch_yolo(
        '/home/student/Toan/runs/detect/checkpoints/teacher/yolo26x_kust4k/weights/best.pt',
        device='cuda:1'
    )
    
    print("\n✅ RGBT setup complete!")
    print("Ready for custom training loop")
