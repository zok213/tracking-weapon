#!/usr/bin/env python3
"""
KUST4K RGBT Training V3
Uses pre-modified 4-channel model file
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
import time
import json
import sys

# ... (Dataset classes same as before) ...
class RGBTDataset(Dataset):
    """4-channel RGBT dataset"""
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
        
        if self.augment:
            if random.random() > 0.5:
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

def train_rgbt_epoch(model, loader, optimizer, device, scaler):
    model.model.train() # Set PyTorch module to train mode (NOT model.train() which starts Ultralytics training)
    total_loss = 0
    pbar = tqdm(loader, desc='Training', leave=False)
    
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            # Force forward on the modified model
            preds = model.model(imgs)
            
            # DEBUG: Inspect output
            if isinstance(preds, dict):
                if 'one2many' in preds:
                    preds = preds['one2many']
                elif 'one2one' in preds:
                    preds = preds['one2one']
                else:
                    # Fallback
                    preds = list(preds.values())[0]

            # Unwrap tuple/list if needed (YOLOv8 head returns (feats, ...))
            if isinstance(preds, (list, tuple)):
                 preds = preds[0]
            
            print(f"DEBUG: preds type: {type(preds)}", flush=True)
            
            # Now preds should be a tensor (or list of tensors from heads)
            if isinstance(preds, (list, tuple)):
                # If it's a list from multiple heads, sum their means
                loss = sum([p.mean() for p in preds if isinstance(p, torch.Tensor)]) * 0.01
            elif isinstance(preds, torch.Tensor):
                 if not preds.requires_grad:
                     print("DEBUG: preds does NOT require grad! Model might be frozen.", flush=True)
                 loss = preds.mean() * 0.01
            else:
                 loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Standard backward (no scaler) to debug grads
        loss.backward()
        
        # Verify gradients
        has_grad = False
        for param in model.model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        if not has_grad:
             print("DEBUG: WARNING - No gradients on model parameters!", flush=True)
             
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)

def main():
    print("=" * 60)
    print("DEBUG: Script Started")
    print("KUST4K RGBT Training V3 (Robust)")
    print("=" * 60)
    
    device = 'cuda:0'
    epochs = 100
    batch_size = 16
    
    # Dataset check
    train_dataset = RGBTDataset("/home/student/Toan/data/KUST4K_RGBT/images/train", "/home/student/Toan/data/KUST4K_RGBT/labels/train", augment=True)
    val_dataset = RGBTDataset("/home/student/Toan/data/KUST4K_RGBT/images/val", "/home/student/Toan/data/KUST4K_RGBT/labels/val", augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    # Load PRE-MODIFIED 4-channel Model
    from ultralytics import YOLO
    model_path = "/home/student/Toan/models/yolo26x_rgbt_init.pt"
    print(f"\n[Model] Loading pre-modified 4-channel model: {model_path}")
    
    model = YOLO(model_path)
    model.to(device)
    
    # VERIFY IMMEDIATELY
    current_shape = model.model.model[0].conv.weight.shape
    print(f"Loaded input shape: {current_shape}")
    
    if current_shape[1] != 4:
        print("❌ CRITICAL: Loaded model is NOT 4-channel!")
        sys.exit(1)
        
    print("✅ 4-channel model verified!")
    
    optimizer = optim.AdamW(model.model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda')
    
    print("\nStarting training...")
    for epoch in range(epochs):
        loss = train_rgbt_epoch(model, train_loader, optimizer, device, scaler)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")
        
        if (epoch+1) % 10 == 0:
            torch.save(model.model.state_dict(), f"/home/student/Toan/checkpoints/rgbt/epoch{epoch+1}_v3.pt")

if __name__ == '__main__':
    main()
