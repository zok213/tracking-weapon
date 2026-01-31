#!/usr/bin/env python3
"""
VT-MOT RGBT Fine-tuning (Phase 2) - DDP Version
- Distributed Data Parallel (Multi-GPU)
- SyncBatchNorm
- 2x RTX 4090 Utilization
- Total Batch Size: 16 (8 per GPU)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import random
import sys
import os
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import IterableSimpleNamespace
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# RGBT Dataset (Identical logic)
class RGBTDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=640, augment=True):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.img_size = img_size
        self.augment = augment
        self.images = sorted(list(self.img_dir.glob('*.npy')))
        # Only print on Rank 0
        if dist.is_initialized() and dist.get_rank() == 0:
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
            if dist.get_rank() == 0: print(f"Error loading {img_path}: {e}", flush=True)
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

def train_epoch(model, loader, optimizer, device, loss_fn, rank):
    model.train()
    total_loss = torch.zeros(1).to(device)
    
    if rank == 0:
        pbar = tqdm(loader, desc='Training', leave=False, mininterval=5.0)
    else:
        pbar = loader
        
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
        
        # DDP Model forward
        preds = model(imgs)
        
        # Unwrap logic (model is DDP wrapped? No, ultralytics model handles forward)
        # Wait, model(imgs) returns dict or tuple?
        # If DDP, model is the DDP wrapper. model(imgs) -> module(imgs) -> dict.
        
        if isinstance(preds, dict):
             if 'one2many' in preds: preds = preds['one2many']
             elif 'one2one' in preds: preds = preds['one2one']
             else: preds = list(preds.values())[0]

        loss, _ = loss_fn(preds, batch)
        if loss.ndim > 0: loss = loss.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        
        total_loss += loss.detach()
        
        if rank == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
    # Reduce loss across devices for logging
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    avg_loss = total_loss.item() / (dist.get_world_size() * len(loader))
    return avg_loss

def validate_epoch(model, loader, device, loss_fn, rank):
    # Important: accessing internal model in DDP is .module
    # We apply the HACK to .module
    
    # model is DDP(YOLO.model)
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()
    
    # HACK: Force Head to Training (Validation Fix)
    head = raw_model.model[-1]
    head.training = True
    
    total_loss = torch.zeros(1).to(device)
    
    with torch.no_grad():
        if rank == 0:
             pbar = tqdm(loader, desc='Validation', leave=False, mininterval=5.0)
        else:
             pbar = loader
             
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            batch = {
                'img': imgs,
                'bboxes': labels[:, 2:], 
                'cls': labels[:, 1:2],
                'batch_idx': labels[:, 0]
            }
            
            preds = model(imgs)
            if isinstance(preds, dict):
                 if 'one2many' in preds: preds = preds['one2many']
                 elif 'one2one' in preds: preds = preds['one2one']
                 else: preds = list(preds.values())[0]

            loss, _ = loss_fn(preds, batch)
            if loss.ndim > 0: loss = loss.sum()
            total_loss += loss.detach()
            
    head.training = False # Reset
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    avg_loss = total_loss.item() / (dist.get_world_size() * len(loader))
    return avg_loss

def main():
    # DDP Setup
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("="*60, flush=True)
        print(f"VT-MOT RGBT V2 - DISTRIBUTED TRAINING (DDP)", flush=True)
        print(f"GPUs: {world_size} | Batch (Per GPU): 8 | Total Batch: {8*world_size}", flush=True)
        print("="*60, flush=True)
    
    epochs = 50
    batch_size = 12 # Increased from 8 to 12 (Total 24) for max utilization
    
    train_img = "/home/student/Toan/data/VT-MOT_RGBT/images/train"
    train_lbl = "/home/student/Toan/data/VT-MOT_RGBT/labels/train"
    val_img = "/home/student/Toan/data/VT-MOT_RGBT/images/val"
    val_lbl = "/home/student/Toan/data/VT-MOT_RGBT/labels/val"
    
    dataset_train = RGBTDataset(train_img, train_lbl, augment=True)
    dataset_val = RGBTDataset(val_img, val_lbl, augment=False)
    
    sampler_train = DistributedSampler(dataset_train, shuffle=True)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    from ultralytics import YOLO
    
    # Initialize Model on all ranks (Architecture only)
    model_obj = YOLO("/home/student/Toan/models/yolo26x_rgbt_init.pt")
    
    # Load Weights on all ranks (CPU first to avoid OOM or race)
    phase1_ckpt = "/home/student/Toan/checkpoints/rgbt/epoch90_v6.pt"
    state_dict = torch.load(phase1_ckpt, map_location='cpu')
    try:
        model_obj.model.load_state_dict(state_dict)
        if rank == 0: print("✅ Phase 1 weights loaded successfully on Rank 0", flush=True)
    except Exception as e:
        if rank == 0: print(f"⚠️ Load failed: {e}", flush=True)
    
    model_obj.model.to(device)
    
    # SyncBatchNorm (Recommended for Multi-GPU)
    model_obj.model = nn.SyncBatchNorm.convert_sync_batchnorm(model_obj.model)
    
    # DDP Wrap
    model_ddp = DDP(model_obj.model, device_ids=[local_rank])
    
    # Setup Loss
    hyp = {
        'box': 7.5, 'cls': 1.0, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0,
        'label_smoothing': 0.0, 'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7,
        'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,
        'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5,
        'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0, 'warmup_epochs': 3.0,
        'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
    }
    model_ddp.module.args = IterableSimpleNamespace(**hyp)
    loss_fn = v8DetectionLoss(model_ddp.module)
    
    optimizer = optim.AdamW(model_ddp.parameters(), lr=1.75e-4)
    
    save_dir = Path("/home/student/Toan/checkpoints/rgbt_vtmot_ddp")
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # RESUME LOGIC
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Check for latest epoch
    latest_epoch = 0
    latest_ckpt = None
    if save_dir.exists():
        for p in save_dir.glob("epoch*.pt"):
            try:
                # epoch25.pt -> 25
                ep = int(p.stem.replace("epoch", ""))
                if ep > latest_epoch:
                    latest_epoch = ep
                    latest_ckpt = p
            except: pass
            
    if latest_ckpt:
        if rank == 0: print(f"🔄 Resuming from {latest_ckpt} (Epoch {latest_epoch})", flush=True)
        # Load weights
        state_dict = torch.load(latest_ckpt, map_location=device)
        model_ddp.module.load_state_dict(state_dict)
        start_epoch = latest_epoch # Start from next? No, if we saved epoch 25, we finished epoch 25.
        # So we should start from range(start_epoch, epochs) where start_epoch=25 means 0..24 are done.
        # Loop is range(epochs) -> 0,1,2...24, 25.
        # If we finished 25 (0-indexed logic?), let's assume saved "epoch25" means 25 epochs finished.
        # So next is epoch index 25 (which is 26th epoch).
        
        # NOTE: My save logic was: f"epoch{epoch+1}.pt"
        # So if epoch loop index was 24, we saved "epoch25.pt".
        # So we have finished 25 epochs.
        # Next loop index should be 25.
        pass
        
    for epoch in range(start_epoch, epochs):
        sampler_train.set_epoch(epoch) # Critical for shuffling
        
        train_loss = train_epoch(model_ddp, loader_train, optimizer, device, loss_fn, rank)
        val_loss = validate_epoch(model_ddp, loader_val, device, loss_fn, rank)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}", flush=True)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model_ddp.module.state_dict(), save_dir / "best.pt")
                print(f"  --> New Best Val (Saved)", flush=True)
                
            if (epoch+1)%5 == 0:
                torch.save(model_ddp.module.state_dict(), save_dir / f"epoch{epoch+1}.pt")
                
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
