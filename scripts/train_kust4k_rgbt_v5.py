#!/usr/bin/env python3
"""
KUST4K RGBT Training V5 (Clean Rewrite)
No AMP, Direct Debugging
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

def train_rgbt_epoch(model, loader, optimizer, device):
    model.model.train()
    total_loss = 0
    pbar = tqdm(loader, desc='Training', leave=False)
    
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        
        # Forward (No AMP)
        preds = model.model(imgs)
        
        # Safe extraction
        if isinstance(preds, dict):
             if 'one2many' in preds: preds = preds['one2many']
             elif 'one2one' in preds: preds = preds['one2one']
             else: preds = list(preds.values())[0]
             
        if isinstance(preds, (list, tuple)):
             preds = preds[0]

        print(f"DEBUG LOOP: preds type {type(preds)}", flush=True)

        # Loss calculation
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        if isinstance(preds, (list, tuple)):
            loss = sum([p.mean() for p in preds if isinstance(p, torch.Tensor)]) * 0.01
        elif isinstance(preds, torch.Tensor):
             loss = preds.mean() * 0.01
        
        # Backward
        loss.backward()
        
        # Check gradients
        grad_norm = 0.0
        for p in model.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item()
        
        print(f"DEBUG LOOP: grad_norm={grad_norm:.6f}", flush=True)
        
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Break after 1 batch for debug purposes if we want fast feedback
        # But let's run full epoch
    
    return total_loss / len(loader)

def main():
    print("="*60, flush=True)
    print("KUST4K RGBT Training V5 (Clean)", flush=True)
    print("="*60, flush=True)
    
    device = 'cuda:0'
    epochs = 100
    batch_size = 4
    
    train_dataset = RGBTDataset("/home/student/Toan/data/KUST4K_RGBT/images/train", "/home/student/Toan/data/KUST4K_RGBT/labels/train", augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    from ultralytics import YOLO
    model_path = "/home/student/Toan/models/yolo26x_rgbt_init.pt"
    print(f"Loading {model_path}...", flush=True)
    model = YOLO(model_path)
    model.to(device)
    
    # Verify input channels
    c = model.model.model[0].conv.weight.shape[1]
    print(f"Input channels: {c}", flush=True)
    if c != 4:
        print("FAIL: Not 4 channels", flush=True)
        sys.exit(1)
        
    optimizer = optim.AdamW(model.model.parameters(), lr=0.001)
    
    print("Starting training...", flush=True)
    for epoch in range(epochs):
        loss = train_rgbt_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}", flush=True)
        
        if (epoch+1)%10 == 0:
             torch.save(model.model.state_dict(), f"/home/student/Toan/checkpoints/rgbt/epoch{epoch+1}_v5.pt")

if __name__ == '__main__':
    main()
