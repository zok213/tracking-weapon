#!/usr/bin/env python3
"""
VI-ReID Distributed Training Script (DDP + RAM Cache)
=====================================================
Highly Optimized for Multi-GPU Training (Speed + Accuracy).

Features:
- DistributedDataParallel (DDP) for 2x GPU scaling.
- RAM Caching: Loads compressed JPG bytes into RAM to eliminate IO bottlenecks.
- Distributed Random Identity Sampler: Ensures PxK valid sampling across ranks.
- AMP (Automatic Mixed Precision).
- Resume Capability.

Usage:
    torchrun --nproc_per_node=2 train_vi_reid_ddp.py
"""

import os
import glob
import re
import random
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data.sampler import Sampler
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# --- CONFIG ---
DATA_ROOT = "/home/student/Toan/data/VT-MOT_ReID_Person_Only"
BATCH_SIZE = 256  # Per GPU (Total 512)
NUM_INSTANCES = 8 # Images per identity
IMG_SIZE = (256, 128)
LR = 0.00035 * 2 # Linearly scale learning rate for DDP (2 GPUs)
EPOCHS = 60
CHECKPOINT_DIR = "checkpoints_vi_ddp"
RESUME_PATH = "checkpoints_vi/agw_vtmot_best.pth" # Try to resume from single-gpu best

def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        # Fallback for debug
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# --- 1. OPTIMIZED DATASET (RAM CACHE) ---
class VIReIDDataset(Dataset):
    def __init__(self, root, transform=None, cache=True):
        self.root = root
        self.transform = transform
        self.cache = cache
        
        self.rgb_dir = os.path.join(root, 'bounding_box_train')
        self.ir_dir = os.path.join(root, 'ir_bounding_box_train')
        
        self.data_info = [] # (path/bytes, pid, camid, modality)
        
        self.pid_map = {}
        self.next_pid = 0
        
        print("🔍 Scanning dataset...")
        self._load_folder(self.rgb_dir, modality=0)
        self._load_folder(self.ir_dir, modality=1) # 1=IR
        
        self.pids = sorted(list(self.pid_map.values()))
        self.num_classes = len(self.pids)
        
        if dist.get_rank() == 0:
            print(f"📦 Dataset: {len(self.data_info)} images, {self.num_classes} IDs.")
            
        # RAM Caching
        if self.cache:
            if dist.get_rank() == 0:
                 print("🚀 Caching images to RAM...")
            
            # Using list storage
            self.cached_images = []
            
            # Use tqdm only on rank 0
            iterator = tqdm(self.data_info) if dist.get_rank() == 0 else self.data_info
            
            new_data_info = []
            for item in iterator:
                path, pid, camid, mod = item
                with open(path, 'rb') as f:
                    img_bytes = f.read()
                self.cached_images.append(img_bytes)
                new_data_info.append((len(self.cached_images)-1, pid, camid, mod))
            
            self.data_info = new_data_info
            if dist.get_rank() == 0:
                print(f"✅ Cached {len(self.cached_images)} images into RAM.")

    def _load_folder(self, folder, modality):
        if not os.path.exists(folder): return
        files = glob.glob(os.path.join(folder, '*.jpg'))
        for fpath in files:
            fname = os.path.basename(fpath)
            # Regex identify
            match = re.match(r'^(\d+)_c(\d+)', fname)
            if match:
                pid_str = match.group(1)
                cam_id = int(match.group(2))
                if pid_str not in self.pid_map:
                    self.pid_map[pid_str] = self.next_pid
                    self.next_pid += 1
                pid = self.pid_map[pid_str]
                self.data_info.append((fpath, pid, cam_id, modality))

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        item = self.data_info[idx]
        
        if self.cache:
            idx_cache, pid, camid, mod = item
            img_bytes = self.cached_images[idx_cache]
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        else:
            path, pid, camid, mod = item
            img = Image.open(path).convert('RGB')
            
        if self.transform:
            img = self.transform(img)
            
        return img, pid

# --- 2. DISTRIBUTED SAMPLER ---
class DistributedRandomIdentitySampler(Sampler):
    """
    DDP-aware PK Sampler.
    Each GPU gets a subset of identities.
    For each identity, sample K instances.
    """
    def __init__(self, data_source, batch_size, num_instances, num_replicas=None, rank=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package.")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package.")
            rank = dist.get_rank()
            
        self.num_replicas = num_replicas
        self.rank = rank
        
        # Build Index
        self.index_dic = defaultdict(list)
        # Handle cached vs non-cached structure
        for index, item in enumerate(self.data_source.data_info):
            pid = item[1] # pid is always index 1
            self.index_dic[pid].append(index)
            
        self.pids = list(self.index_dic.keys())
        
        # Split PIDs across ranks
        # Ensure total PIDs divisible by replicas? No, just split.
        self.num_samples = len(self.pids) // self.num_replicas
        
    def __iter__(self):
        # Deterministic shuffling based on epoch (set elsewhere)
        # Actually sampler needs set_epoch called manually or random seed
        indices = torch.randperm(len(self.pids)).tolist()
        
        # Subsample for this rank
        # pids for this rank
        # We want roughly equal number of PIDs per rank
        # e.g. Rank 0 gets [0, n/2], Rank 1 gets [n/2, n]
        
        # IMPORTANT: In DDP, each process must see DIFFERENT data?
        # Yes.
        
        my_indices = indices[self.rank:len(self.pids):self.num_replicas]
        
        final_idxs = []
        for pid_idx in my_indices:
            pid = self.pids[pid_idx]
            t_idxs = self.index_dic[pid]
            
            if len(t_idxs) < self.num_instances:
                t_idxs = np.random.choice(t_idxs, size=self.num_instances, replace=True)
            
            random.shuffle(t_idxs)
            # Sample K
            batch_idxs = t_idxs[:self.num_instances]
            final_idxs.extend(batch_idxs)
            
        return iter(final_idxs)

    def __len__(self):
        return self.num_samples * self.num_instances

# --- 3. MODEL (ResNet50) ---
class EmbedNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        self._init_params()

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features).view(features.size(0), -1)
        feat = self.bottleneck(features)
        if self.training:
            return self.classifier(feat), features # Logits, Features
        return feat

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

# --- 4. LOSS ---
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.loss = nn.MarginRankingLoss(margin=margin)
    def forward(self, inputs, targets):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, _ = torch.max(dist * mask.float(), dim=1)
        dist_an, _ = torch.min(dist * (1 - mask.float()) + mask.float() * 1e6, dim=1)
        y = dist_an.data.new().resize_as_(dist_an).fill_(1)
        return self.loss(dist_an, dist_ap, y)

# --- MAIN ---
def main():
    rank, local_rank, world_size = setup_ddp()
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(10),
        transforms.RandomCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5),
    ])
    
    dataset = VIReIDDataset(DATA_ROOT, transform=train_transform, cache=True)
    
    # DDP Sampler
    sampler = DistributedRandomIdentitySampler(dataset, BATCH_SIZE, NUM_INSTANCES, world_size, rank)
    
    loader = DataLoader(
        dataset, sampler=sampler, batch_size=BATCH_SIZE,
        num_workers=2, pin_memory=True
    )
    
    model = EmbedNetwork(dataset.num_classes).cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    triplet = TripletLoss()
    xent = nn.CrossEntropyLoss()
    
    start_epoch = 0
    
    # Resume
    if os.path.exists(RESUME_PATH):
        if rank == 0:
             print(f"🔄 Resuming from {RESUME_PATH}")
        checkpoint = torch.load(RESUME_PATH, map_location='cuda:0')
        # Handle single-gpu to ddp keys
        state_dict = checkpoint
        # Add 'module.' prefix if missing
        new_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('module.'):
                new_dict[f'module.{k}'] = v
            else:
                new_dict[k] = v
        model.load_state_dict(new_dict, strict=False)
        # Attempt to infer epoch? Not saved in simple script.
        # Assume start.
        
    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        print("🚀 DDP Training Started!")

    best_loss = 999.0
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        total_acc = 0
        steps = 0
        
        # DDP Sampler shuffle? Handled in __iter__ randomly
        
        pbar = tqdm(loader, disable=(rank!=0), desc=f"Ep {epoch+1}")
        for imgs, pids in pbar:
            imgs, pids = imgs.cuda(), pids.cuda()
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, feats = model(imgs)
            
            # Loss computation in FP32 to avoid NaN and Type mismatch
            with torch.amp.autocast('cuda', enabled=False):
                logits = logits.float()
                feats = feats.float()
                l_cls = xent(logits, pids)
                l_tri = triplet(feats, pids)
                loss = l_cls + l_tri
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics
            _, preds = torch.max(logits, 1)
            acc = (preds == pids).float().mean().item()
            
            # Reduce Loss/Acc for printing
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            
            total_loss += loss.item()
            total_acc += acc
            steps += 1
            
            if rank == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc*100:.1f}%"})
        
        if rank == 0:
            avg_loss = total_loss / steps
            avg_acc = total_acc / steps
            print(f"🏁 Ep {epoch+1}: Loss {avg_loss:.4f} | Acc {avg_acc*100:.2f}%")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/agw_best.pth")
                
    cleanup_ddp()

if __name__ == '__main__':
    main()
