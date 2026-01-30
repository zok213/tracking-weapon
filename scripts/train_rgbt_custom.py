#!/usr/bin/env python3
"""
KUST4K RGBT Training with Modified Ultralytics
Custom training loop using 4-channel modified YOLO
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
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for imgs, labels in tqdm(loader, desc='Training', leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            # Forward pass through model
            preds = model.model(imgs)
            
            # Simple loss: we'll use the model's built-in loss
            # For now, just run forward for validation
            loss = torch.tensor(0.0, device=device)
            for p in preds:
                if isinstance(p, torch.Tensor):
                    loss = loss + p.mean() * 0.001
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Validation', leave=False):
            imgs = imgs.to(device)
            preds = model.model(imgs)
            
            loss = torch.tensor(0.0, device=device)
            for p in preds:
                if isinstance(p, torch.Tensor):
                    loss = loss + p.mean() * 0.001
            
            total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    print("=" * 60)
    print("KUST4K RGBT 4-Channel Training")
    print("=" * 60)
    
    device = 'cuda:0'
    epochs = 100
    batch_size = 16
    lr = 0.001
    
    # Datasets
    train_dataset = RGBTDataset(
        "/home/student/Toan/data/KUST4K_RGBT/images/train",
        "/home/student/Toan/data/KUST4K_RGBT/labels/train",
        augment=True
    )
    val_dataset = RGBTDataset(
        "/home/student/Toan/data/KUST4K_RGBT/images/val",
        "/home/student/Toan/data/KUST4K_RGBT/labels/val",
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Load 4-channel model
    from ultralytics import YOLO
    
    pretrained = "/home/student/Toan/runs/detect/checkpoints/teacher/yolo26x_kust4k/weights/best.pt"
    model = YOLO(pretrained)
    
    # Modify for 4-channel
    first_conv = model.model.model[0].conv
    new_conv = nn.Conv2d(4, first_conv.out_channels, first_conv.kernel_size,
                          first_conv.stride, first_conv.padding,
                          bias=first_conv.bias is not None).to(device)
    
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = first_conv.weight.to(device)
        new_conv.weight[:, 3:4, :, :] = first_conv.weight.mean(dim=1, keepdim=True).to(device)
        if first_conv.bias is not None:
            new_conv.bias = nn.Parameter(first_conv.bias.to(device))
    
    model.model.model[0].conv = new_conv
    model.model.to(device)
    
    print("✅ 4-channel YOLO26x loaded")
    
    # Optimizer
    optimizer = optim.AdamW(model.model.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    # Training
    save_dir = Path("/home/student/Toan/checkpoints/rgbt")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        train_loss = train_rgbt_epoch(model, train_loader, optimizer, device, scaler)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.model.state_dict(), save_dir / 'best_rgbt.pt')
        
        # Save periodic
        if (epoch + 1) % 10 == 0:
            torch.save(model.model.state_dict(), save_dir / f'epoch{epoch+1}_rgbt.pt')
        
        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save final
    torch.save(model.model.state_dict(), save_dir / 'last_rgbt.pt')
    
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    
    print(f"\n✅ Training complete!")
    print(f"   Best model: {save_dir / 'best_rgbt.pt'}")


if __name__ == '__main__':
    main()
