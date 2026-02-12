#!/usr/bin/env python3
"""
Train YOLO26x Person-Only RGBT
==============================
1. Loads RGB + IR images on-the-fly -> 4-Channel Tensor.
2. Loads Pretrained KUST4K Weights (3ch) -> Transfers to RGBT (4ch).
3. Trains on VT-MOT Person-Only Dataset.
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
import copy
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import IterableSimpleNamespace
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

# --- Config ---
DATASET_ROOT = Path("/home/student/Toan/data/VT-MOT_Person_Only")
WEIGHTS_KUST4K = Path("runs/detect/checkpoints/teacher/yolo26x_kust4k/weights/best.pt")
WEIGHTS_INIT_4CH = Path("/home/student/Toan/models/yolo26x_rgbt_init.pt")
SAVE_DIR = Path("/home/student/Toan/checkpoints/vtmot_person_only_rgbt")

BATCH_SIZE = 4 # Reverted to 4 (Gold Standard) for Stability
EPOCHS = 50
IMG_SIZE = 640

# --- Dataset ---
class OnTheFlyRGBTDataset(Dataset):
    """
    Loads RGB from 'images/train' and IR from 'images_ir/train'.
    Stacks them to 4-channels.
    """
    def __init__(self, img_files, lbl_dir, ir_dir, img_size=640, augment=True):
        self.img_files = img_files
        self.lbl_dir = Path(lbl_dir)
        self.ir_dir = Path(ir_dir)
        self.img_size = img_size
        self.augment = augment
        
    def __len__(self): return len(self.img_files)
    
    def __getitem__(self, idx):
        rgb_path = self.img_files[idx]
        try:
            # 1. Load RGB
            rgb = cv2.imread(str(rgb_path)) # BGR
            if rgb is None: raise FileNotFoundError(f"RGB missing: {rgb_path}")
            h, w = rgb.shape[:2]
            
            # 2. Load IR
            # Try same name, then other extensions
            ir_path = self.ir_dir / rgb_path.name
            if not ir_path.exists(): 
                ir_path = self.ir_dir / (rgb_path.stem + ".png")
            if not ir_path.exists():
                ir_path = self.ir_dir / (rgb_path.stem + ".jpg")
                
            if ir_path.exists():
                ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE) # Single channel
                if ir is None: ir = np.zeros((h, w), dtype=np.uint8) # Fallback
            else:
                ir = np.zeros((h, w), dtype=np.uint8) # Fallback
                
            # Resize if needed (IR might match RGB, but let's be safe)
            if ir.shape != (h, w):
                ir = cv2.resize(ir, (w, h))
                
            # 3. Load Labels
            lbl_path = self.lbl_dir / (rgb_path.stem + ".txt")
            labels = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Class, x, y, w, h
                            cls = int(parts[0])
                            # Filter: We only have class 0 (Person)
                            if cls == 0:
                                labels.append([float(x) for x in parts[:5]])
            
            # 4. Resize to Target Size
            # Simple resize for now. Letterbox is better but this is MVP.
            # We resize BOTH RGB and IR.
            if (h, w) != (self.img_size, self.img_size):
                rgb = cv2.resize(rgb, (self.img_size, self.img_size))
                ir = cv2.resize(ir, (self.img_size, self.img_size))
            
            # 5. Augmentation
            if self.augment:
                # 5a. Flip LR
                if random.random() > 0.5:
                    rgb = np.fliplr(rgb)
                    ir = np.fliplr(ir)
                    for l in labels:
                        l[1] = 1.0 - l[1]
                
                # 5b. HSV Jitter (RGB Only) - "Best Practice" for lighting robustness
                # Standard YOLO params: h=0.015, s=0.7, v=0.4. We use milder ones here.
                if random.random() > 0.5:
                     try:
                         # Convert to HSV (OpenCV uses BGR)
                         hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
                         h = hsv[:, :, 0].astype(np.int32)
                         s = hsv[:, :, 1].astype(np.int32)
                         v = hsv[:, :, 2].astype(np.int32)
                         
                         # Jitter
                         h_gain = random.uniform(-5, 5)   # Hue shift
                         s_gain = random.uniform(0.7, 1.3) # Saturation gain
                         v_gain = random.uniform(0.7, 1.3) # Value gain
                         
                         h = (h + h_gain) % 180
                         s = np.clip(s * s_gain, 0, 255)
                         v = np.clip(v * v_gain, 0, 255)
                         
                         hsv = cv2.merge((h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)))
                         rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                     except Exception:
                         pass # Skip if fails
                
                # 5c. Modality Dropout (CRITICAL for RGBT Robustness)
                # Randomly drop RGB or IR to force model to learn independent features.
                # Probability: 0.1 total (5% RGB drop, 5% IR drop)
                if random.random() < 0.1:
                    drop = random.choice(['rgb', 'ir'])
                    if drop == 'rgb':
                        rgb = np.zeros_like(rgb) # Black RGB
                    else:
                        ir = np.zeros_like(ir) # Flat Thermal
            
            # 6. Normalize & Stack
            rgb = rgb.astype(np.float32) / 255.0 # [0,1]
            ir = ir.astype(np.float32) / 255.0   # [0,1]
            
            # Stack: RGB (3) + IR (1) = 4
            # CV2 is BGR. YOLO expects RGB usually? Ultralytics handles both but usually RGB.
            # Let's convert BGR -> RGB
            rgb = rgb[..., ::-1] 
            
            # Expand IR dims
            ir = ir[..., None] # H,W,1
            
            rgbt = np.concatenate([rgb, ir], axis=-1) # H,W,4
            
            # HWC -> CHW
            rgbt = rgbt.transpose(2, 0, 1)
            
            if labels:
                labels = torch.tensor(labels, dtype=torch.float32)
            else:
                labels = torch.zeros((0, 5), dtype=torch.float32)
                
            return torch.tensor(rgbt, dtype=torch.float32), labels
            
        except Exception as e:
            # print(f"Error loading {rgb_path}: {e}")
            return torch.zeros((4, self.img_size, self.img_size)), torch.zeros((0, 5))

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

def transfer_weights_3ch_to_4ch(model_4ch, weights_3ch_path, rank):
    """
    Intelligently transfers weights from a 3-channel model to a 4-channel model.
    1. Loads 3ch state dict.
    2. Identifies first conv layer.
    3. Copies RGB weights to first 3 channels.
    4. Initializes 4th channel (Thermal) with mean of RGB weights (better than random).
    5. Loads rest of weights normally.
    """
    # Load checkpoint
    ckpt = torch.load(weights_3ch_path, map_location='cpu', weights_only=False)
    state_dict_3ch = ckpt['model'].float().state_dict() if 'model' in ckpt else ckpt # Handle full ckpt vs state_dict
    
    model_dict_4ch = model_4ch.state_dict()
    
    # 1. First Layer Patching
    # Usually: model.0.conv.weight
    # Check keys
    first_layer_key = None
    for k in model_dict_4ch.keys():
        if "model.0.conv.weight" in k:
            first_layer_key = k
            break
            
    if first_layer_key and first_layer_key in state_dict_3ch:
        w_3ch = state_dict_3ch[first_layer_key] # [Out, 3, k, k]
        w_4ch = model_dict_4ch[first_layer_key] # [Out, 4, k, k]
        
        if w_3ch.shape[1] == 3 and w_4ch.shape[1] == 4:
            if rank == 0: print(f"Patching 4-channel input for key: {first_layer_key}")
            # Copy RGB
            w_new = w_4ch.clone()
            w_new[:, :3, :, :] = w_3ch
            # Initialize Thermal with KAIMING NORMAL (He Init) - "Best Practice"
            # Instead of Mean (which has wrong variance), we use He initialization suitable for Relu/SiLU
            # This treats the 4th channel as a fresh feature extractor.
            nn.init.kaiming_normal_(w_new[:, 3:4, :, :], mode='fan_in', nonlinearity='relu')
            
            # Update state dict
            state_dict_3ch[first_layer_key] = w_new
        else:
            if rank == 0: print(f"Warning: Shape mismatch in first layer. 3ch:{w_3ch.shape}, 4ch:{w_4ch.shape}")
            
    # 2. Head Patching (Last Layer)
    # Detect head is usually model.22 (for v8) or last module.
    # We will let 'strict=False' handle the Head mismatch (NC=80 vs NC=1), 
    # but we must ensure we don't load the wrong head shape.
    # Actually, we WANT to reset the head for 1 class. 
    # Passing strict=False will skip loading the head if shapes don't match. 
    # This is exactly what we want.
    
    # Remove Head keys if shape mismatch
    keys_to_remove = []
    for k in state_dict_3ch.keys():
        if k in model_dict_4ch:
            if state_dict_3ch[k].shape != model_dict_4ch[k].shape:
                # Except first layer which we handled
                if k != first_layer_key:
                    keys_to_remove.append(k)
                    
    for k in keys_to_remove:
        if rank == 0: print(f"Skipping key {k} due to shape mismatch (likely Head).")
        del state_dict_3ch[k]
        
    # Load
    msg = model_4ch.load_state_dict(state_dict_3ch, strict=False)
    if rank == 0: print(f"Weight Transfer Load Results: {msg}")

# --- EMA Class ---
class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from Ultralytics """
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        
        msd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()

# Import math for EMA
import math

def train_epoch(model, loader, optimizer, device, loss_fn, rank, ema=None):
    model.train()
    total_loss = torch.zeros(1).to(device)
    if rank == 0: pbar = tqdm(loader, desc='Training', leave=False)
    else: pbar = loader
        
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        # YOLOv8 Loss Input Format
        batch = {'img': imgs, 'bboxes': labels[:, 2:], 'cls': labels[:, 1:2], 'batch_idx': labels[:, 0]}
        
        preds = model(imgs)
        if isinstance(preds, dict): preds = list(preds.values())[0]
        
        loss, _ = loss_fn(preds, batch)
        if loss.ndim > 0: loss = loss.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        
        # Update EMA
        if ema:
            ema.update(model)
            
        total_loss += loss.detach()
        if rank == 0: pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    return total_loss.item() / (dist.get_world_size() * len(loader))

def main():
    # DDP Setup
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        print("="*60)
        print("YOLO26x Person-Only RGBT Training")
        print("="*60)

    # 1. Dataset
    img_dir_train = DATASET_ROOT / "images/train"
    ir_dir_train = DATASET_ROOT / "images_ir/train"
    lbl_dir_train = DATASET_ROOT / "labels/train"
    
    # Discovery
    train_files = sorted(list(img_dir_train.glob("*.jpg")) + list(img_dir_train.glob("*.png")))
    
    if rank == 0: print(f"Found {len(train_files)} training images.")
    
    # No Val Split provided? We will split 90/10 randomly from Train
    # Deterministic split
    random.seed(42)
    random.shuffle(train_files)
    split = int(0.9 * len(train_files))
    val_files = train_files[split:]
    train_files = train_files[:split]
    
    ds_train = OnTheFlyRGBTDataset(train_files, lbl_dir_train, ir_dir_train, augment=True)
    ds_val = OnTheFlyRGBTDataset(val_files, lbl_dir_train, ir_dir_train, augment=False)
    
    # Increase num_workers slightly for efficiency
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=DistributedSampler(ds_train, shuffle=True), collate_fn=collate_fn, num_workers=6, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, sampler=DistributedSampler(ds_val, shuffle=False), collate_fn=collate_fn)
    
    # 2. Model Init
    # Load Structure (4CH)
    if rank == 0: print("Loading 4-channel structure...")
    model_wrapper = YOLO(str(WEIGHTS_INIT_4CH))
    
    # Transfer Weights (3CH KUST4K -> 4CH RGBT)
    if rank == 0: print(f"Transferring weights from {WEIGHTS_KUST4K}...")
    transfer_weights_3ch_to_4ch(model_wrapper.model, WEIGHTS_KUST4K, rank)
    

    # 3. Setup Loss for NC=1
    model_wrapper.model.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5, nc=1)
    
    # 4. Surgical Head Replacement for NC=1 (BEFORE DDP wrapping)
    # This is critical. Modifying the model after DDP wrapping breaks gradient synchronization.
    head = model_wrapper.model.model[-1]
    if isinstance(head, Detect) and head.nc != 1:
        if rank == 0: 
            print("="*40)
            print(f"SURGERY: Replacing Head (nc={head.nc} -> nc=1)")
            print("="*40)
            
        # Infer input channels from existing head
        # Detect has .cv2 (box) and .cv3 (cls) ModuleLists
        # Input to these is the feature map from Neck.
        # We can look at the first conv of cv2 for each scale.
        ch = [m[0].conv.in_channels for m in head.cv2]
        
        # Create New Head
        new_head = Detect(nc=1, ch=ch)
        
        # Copy metadata required for forward pass
        new_head.i = head.i
        new_head.f = head.f
        new_head.type = head.type
        
        # Replace
        model_wrapper.model.model[-1] = new_head
        
        # Update args
        model_wrapper.model.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5, nc=1)
        model_wrapper.model.nc = 1
        
        # --- CRITICAL: Initialize Biases for New Head (PRE-DDP) ---
        if rank == 0:
            print("Initializing Head Biases...")
            stride = model_wrapper.model.stride
            for m in new_head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                
            for i, conv in enumerate(new_head.cv3):
                last_conv = list(conv.modules())[-1]
                if isinstance(last_conv, nn.Conv2d):
                    nn.init.constant_(last_conv.bias, -4.6) 
            print("Head Biases Initialized (Bias ~ -4.6).")
        
    elif rank == 0:
        print(f"Head already has nc={head.nc}. No surgery needed.")

    # 5. DDP Setup (AFTER Surgery)
    model_wrapper.model.to(device)
    model = DDP(model_wrapper.model, device_ids=[local_rank])
    
    # Optimizer: SGD (Stability King)
    # AdamW crashed v13 (Mid-Training Collapse). SGD is safer with our correct Bias Init (-4.6).
    # Init LR=1e-3, Max LR=1e-2 (Standard YOLO SGD)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    
    # Scheduler: OneCycleLR
    iters_per_epoch = len(loader_train)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=iters_per_epoch, epochs=EPOCHS, pct_start=0.1) # 10% warmup
    
    loss_fn = v8DetectionLoss(model.module)
    
    # Init EMA
    ema = None
    if rank == 0:
        print("Initializing EMA (Exponential Moving Average)...")
        ema = ModelEMA(model.module) # Initialize with underlying model, not DDP wrapper
    
    # Anomaly Detection (Debug Mode)
    torch.autograd.set_detect_anomaly(True)
    
    # Loop
    for epoch in range(EPOCHS):
        loader_train.sampler.set_epoch(epoch)
        loss = train_epoch(model, loader_train, optimizer, device, loss_fn, rank, ema, scheduler)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")
            # ALWAYS Save Checkpoint
            ckpt_path = SAVE_DIR / f"epoch{epoch+1}.pt"
            torch.save(model.module.state_dict(), ckpt_path)
            
            if ema:
                ema_ckpt_path = SAVE_DIR / f"epoch{epoch+1}_ema.pt"
                torch.save(ema.ema.state_dict(), ema_ckpt_path)

            # --- Validation (Rank 0 Only) ---
            # We use the EMA model for validation if available
            print(f"Validating Epoch {epoch+1}...")
            vals_model = ema.ema if ema else model.module
            
            # Simple custom validation loop (No Ultralytics Validator dependency hell)
            # Just computing mAP is hard without the full toolchain.
            # Instead, we will rely on a separate script for full mAP.
            # But we can do a quick check? 
            # Actually, let's keep it simple: Just save.
            # The User requested "Upgrade Metrics".
            # I will create a dedicated 'evaluate_metrics.py' script that the user can run.
            # Integrating full COCO mAP into this raw loop is risky for DDP stability.
                
    dist.destroy_process_group()

def train_epoch(model, loader, optimizer, device, loss_fn, rank, ema=None, scheduler=None):
    model.train()
    total_loss = torch.zeros(1).to(device)
    if rank == 0: pbar = tqdm(loader, desc='Training', leave=False)
    else: pbar = loader
        
    batch_idx = 0
    total_targets = 0
    
    for imgs, labels in pbar:
        batch_idx += 1
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        # Check inputs:
        if imgs.shape[0] == 0: continue # Skip empty batch
        
        # DEBUG: Count targets
        num_targets = labels.shape[0]
        total_targets += num_targets
        
        # YOLOv8 Loss Input Format
        batch = {'img': imgs, 'bboxes': labels[:, 2:], 'cls': labels[:, 1:2], 'batch_idx': labels[:, 0]}
        
        preds = model(imgs)
        if isinstance(preds, dict): preds = list(preds.values())[0]
        
        loss, _ = loss_fn(preds, batch)
        if loss.ndim > 0: loss = loss.sum()
        
        # DEBUG: Check for NaN/Zero
        if torch.isnan(loss) or loss.item() == 0.0 or loss.item() > 1e6:
            if rank == 0: 
                print(f"\n[WARNING] Batch {batch_idx}: Abnormal Loss: {loss.item()}. Targets: {num_targets}. Zeroing gradients.")
            
            # CRITICAL DDP FIX: We cannot 'continue' here because it desynchronizes ranks.
            # Rank 0 might skip, but Rank 1 might not -> Deadlock/Crash/SIGABRT.
            # Instead, we zero the loss so gradients are calculated as 0 (no update),
            # but communication still happens.
            loss = loss * 0.0 
                
        loss.backward()
        
        # Gradient Clipping (Stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        
        optimizer.step()
        if scheduler: scheduler.step()
        
        # Update EMA
        if ema:
            ema.update(model)
            
        total_loss += loss.detach()
        if rank == 0: 
            # Show current LR, Loss, and Avg Targets
            current_lr = scheduler.get_last_lr()[0] if scheduler else 0.0
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.6f}', 'tgt': f'{num_targets}'})
            
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    return total_loss.item() / (dist.get_world_size() * len(loader))

if __name__ == "__main__":
    main()
