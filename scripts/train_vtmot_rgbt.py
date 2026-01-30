#!/usr/bin/env python3
"""
VT-MOT RGBT Fine-tuning (Phase 2)
- Uses best weights from Phase 1 (KUST4K)
- 4-channel input
- v8DetectionLoss
- Scaled up for 83K images
- Validation Enabled (Real Engineering)
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
from ultralytics.utils import IterableSimpleNamespace

# RGBT Dataset
class RGBTDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=640, augment=True):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.img_size = img_size
        self.augment = augment
        self.images = sorted(list(self.img_dir.glob('*.npy')))
        print(f"[Dataset] Found {len(self.images)} images in {img_dir}", flush=True)
    
    def __len__(self): return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
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
            
            if labels: labels = torch.tensor(labels, dtype=torch.float32)
            else: labels = torch.zeros((0, 5), dtype=torch.float32)
            return torch.tensor(img), labels
        except Exception as e:
            print(f"Error loading {img_path}: {e}", flush=True)
            return torch.zeros((4, self.img_size, self.img_size)), torch.zeros((0, 5))

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    new_labels = []
    for i, lbl in enumerate(labels):
        if len(lbl) > 0:
            batch_idx = torch.full((len(lbl), 1), i, dtype=torch.float32)
            new_labels.append(torch.cat([batch_idx, lbl], dim=1))
    if new_labels: labels = torch.cat(new_labels, 0)
    else: labels = torch.zeros((0, 6), dtype=torch.float32)
    return imgs, labels

def train_epoch(model, loader, optimizer, device, loss_fn):
    model.model.train()
    total_loss = 0
    pbar = tqdm(loader, desc='Training', leave=False, mininterval=5.0)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        batch = {
            'img': imgs,
            'bboxes': labels[:, 2:], 
            'cls': labels[:, 1:2],
            'batch_idx': labels[:, 0]
        }
        preds = model.model(imgs)
        if isinstance(preds, dict):
             if 'one2many' in preds: preds = preds['one2many']
             elif 'one2one' in preds: preds = preds['one2one']
             else: preds = list(preds.values())[0]

        loss, _ = loss_fn(preds, batch)
        if loss.ndim > 0: loss = loss.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 10.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)

def validate_epoch(model, loader, device, loss_fn):
    model.model.eval()
    
    # HACK: Force the Head to 'training' mode to return raw logits (dict)
    # instead of decoded boxes. This allows v8DetectionLoss to work.
    # We keep the rest of the model in eval() to freeze BN stats.
    head = model.model.model[-1]
    head.training = True
    
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation', leave=False, mininterval=5.0)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            batch = {
                'img': imgs,
                'bboxes': labels[:, 2:], 
                'cls': labels[:, 1:2],
                'batch_idx': labels[:, 0]
            }
            
            # Forward
            preds = model.model(imgs)
            
            # Handle output
            if isinstance(preds, dict):
                 if 'one2many' in preds: preds = preds['one2many']
                 elif 'one2one' in preds: preds = preds['one2one']
                 else: preds = list(preds.values())[0]

            loss, _ = loss_fn(preds, batch)
            if loss.ndim > 0: loss = loss.sum()
            total_loss += loss.item()
            
    # Reset head to eval just in case
    head.training = False
    return total_loss / len(loader)

def main():
    print("="*60, flush=True)
    print("VT-MOT RGBT Fine-tuning (Phase 2 - Optimized)", flush=True)
    print("="*60, flush=True)
    
    device = 'cuda:0'
    epochs = 50
    batch_size = 4
    
    train_img = "/home/student/Toan/data/VT-MOT_RGBT/images/train"
    train_lbl = "/home/student/Toan/data/VT-MOT_RGBT/labels/train"
    val_img = "/home/student/Toan/data/VT-MOT_RGBT/images/val"
    val_lbl = "/home/student/Toan/data/VT-MOT_RGBT/labels/val"
    
    train_dataset = RGBTDataset(train_img, train_lbl, augment=True)
    # Enable Validation
    val_dataset = RGBTDataset(val_img, val_lbl, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    from ultralytics import YOLO
    
    # Load Phase 1 Checkpoint (Epoch 90)
    phase1_ckpt = "/home/student/Toan/checkpoints/rgbt/epoch90_v6.pt"
    print(f"Loading Phase 1 weights: {phase1_ckpt}...", flush=True)
    
    model = YOLO("/home/student/Toan/models/yolo26x_rgbt_init.pt")
    state_dict = torch.load(phase1_ckpt)
    try:
        model.model.load_state_dict(state_dict)
        print("✅ Phase 1 weights loaded successfully", flush=True)
    except Exception as e:
        print(f"⚠️ Direct load failed: {e}", flush=True)
    
    model.to(device)
    
    # Setup Loss with Optimized Cls Weight
    hyp = {
        'box': 7.5, 
        'cls': 1.0, # Increased focus
        'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0,
        'label_smoothing': 0.0, 'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7,
        'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,
        'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5,
        'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0, 'warmup_epochs': 3.0,
        'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
    }
    model.model.args = IterableSimpleNamespace(**hyp)
    loss_fn = v8DetectionLoss(model.model)
    
    # Optimizer (Engineering Rate)
    optimizer = optim.AdamW(model.model.parameters(), lr=1.75e-4)
    
    print(f"Starting fine-tuning on {len(train_dataset)} train, {len(val_dataset)} val images...", flush=True)
    
    save_dir = Path("/home/student/Toan/checkpoints/rgbt_vtmot")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss = validate_epoch(model, val_loader, device, loss_fn)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.model.state_dict(), save_dir / "best.pt")
            print(f"  --> New Best Val Loss: {val_loss:.4f}", flush=True)
            
        if (epoch+1)%5 == 0:
             torch.save(model.model.state_dict(), save_dir / f"epoch{epoch+1}.pt")

if __name__ == '__main__':
    main()
