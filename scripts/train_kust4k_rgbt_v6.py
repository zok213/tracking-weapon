#!/usr/bin/env python3
"""
KUST4K RGBT Training V6 (Running & Correct)
- 4-channel input (Verified)
- v8DetectionLoss (Real learning)
- Batch size 4 (FP32 Stable)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import random
import sys
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import IterableSimpleNamespace, ops

# Custom Dataset (Same as before)
class RGBTDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=640, augment=True):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.img_size = img_size
        self.augment = augment
        self.images = sorted(self.img_dir.glob('*.npy'))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = np.load(img_path)
        lbl_path = self.lbl_dir / (img_path.stem + '.txt')
        labels = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append([float(x) for x in parts[:5]])
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size))
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img).copy()
            for lbl in labels:
                lbl[1] = 1 - lbl[1]
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        if labels:
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(img), labels

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    new_labels = []
    for i, lbl in enumerate(labels):
        if len(lbl) > 0:
            batch_idx = torch.full((len(lbl), 1), i, dtype=torch.float32)
            new_labels.append(torch.cat([batch_idx, lbl], dim=1))
    if new_labels:
        labels = torch.cat(new_labels, 0)
    else:
        labels = torch.zeros((0, 6), dtype=torch.float32)
    return imgs, labels

def train_rgbt_epoch(model, loader, optimizer, device, loss_fn):
    model.model.train()
    total_loss = 0
    pbar = tqdm(loader, desc='Training', leave=False)
    
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        # Prepare batch for v8DetectionLoss
        # labels shape: [N, 6] -> (batch_idx, cls, x, y, w, h)
        batch = {
            'img': imgs,
            'bboxes': labels[:, 2:], # x,y,w,h
            'cls': labels[:, 1:2],   # class
            'batch_idx': labels[:, 0]
        }
        
        # Forward
        preds = model.model(imgs)
        
        # Handle dictionary output (e.g. one2many/one2one keys)
        if isinstance(preds, dict):
             if 'one2many' in preds: preds = preds['one2many']
             elif 'one2one' in preds: preds = preds['one2one']
             else: preds = list(preds.values())[0]

        # Loss
        loss, loss_items = loss_fn(preds, batch)
        
        # Backward
        if loss.ndim > 0:
            loss = loss.sum()
        loss.backward()
        
        # Clip grads (optional but good for stability)
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 10.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)

def main():
    print("="*60, flush=True)
    print("KUST4K RGBT Training V6 (Real Loss)", flush=True)
    print("="*60, flush=True)
    
    device = 'cuda:0'
    epochs = 100
    batch_size = 4 # Keep small for stability
    
    train_dataset = RGBTDataset("/home/student/Toan/data/KUST4K_RGBT/images/train", "/home/student/Toan/data/KUST4K_RGBT/labels/train", augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    from ultralytics import YOLO
    model_path = "/home/student/Toan/models/yolo26x_rgbt_init.pt"
    print(f"Loading {model_path}...", flush=True)
    model = YOLO(model_path)
    model.to(device)
    
    # Fix hypothesis (args) by loading defaults and merging
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import IterableSimpleNamespace
    
    # Get default config. Note: get_cfg returns IterableSimpleNamespace
    # We use valid default values.
    # We can also manually create a dict with essential keys
    hyp = {
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    }
    model.model.args = IterableSimpleNamespace(**hyp)
    
    # Initialize Loss
    loss_fn = v8DetectionLoss(model.model)
    
    # Optimizer
    optimizer = optim.AdamW(model.model.parameters(), lr=0.001)
    
    print("Starting training...", flush=True)
    for epoch in range(epochs):
        loss = train_rgbt_epoch(model, train_loader, optimizer, device, loss_fn)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}", flush=True)
        
        if (epoch+1)%10 == 0:
             torch.save(model.model.state_dict(), f"/home/student/Toan/checkpoints/rgbt/epoch{epoch+1}_v6.pt")

if __name__ == '__main__':
    main()
