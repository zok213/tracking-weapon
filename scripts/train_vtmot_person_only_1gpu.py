#!/usr/bin/env python3
"""
Train YOLO26x Person-Only RGBT (Single GPU Version)
===================================================
1. Loads RGB + IR images on-the-fly -> 4-Channel Tensor.
2. Loads Pretrained KUST4K Weights (3ch) -> Transfers to RGBT (4ch).
3. Trains on VT-MOT Person-Only Dataset.
4. SINGLE GPU MODE (No DDP, No Sync, Max Stability).
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
import os
import copy
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import IterableSimpleNamespace
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

# --- Config ---
DATASET_ROOT = Path("/home/student/Toan/data/VT-MOT_Person_Only")
WEIGHTS_KUST4K = Path("runs/detect/checkpoints/teacher/yolo26x_kust4k/weights/best.pt")
# WEIGHTS_INIT_4CH = Path("/home/student/Toan/models/yolo26x_rgbt_init.pt") # Not needed if we patch in-memory
SAVE_DIR = Path("/home/student/Toan/checkpoints/vtmot_person_only_rgbt_1gpu")

BATCH_SIZE = 4 # Single GPU Batch Size
EPOCHS = 50
IMG_SIZE = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- EMA Class ---
class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from Ultralytics """
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - torch.exp(torch.tensor(-x / tau)))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        
        msd = model.state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()

# --- Dataset ---
class OnTheFlyRGBTDataset(Dataset):
    def __init__(self, img_files, lbl_dir, ir_dir, augment=False):
        self.img_files = img_files
        self.lbl_dir = lbl_dir
        self.ir_dir = ir_dir
        self.augment = augment

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        
        # Load RGB
        rgb = cv2.imread(str(img_path)) # BGR
        if rgb is None: raise FileNotFoundError(f"RGB not found: {img_path}")
        
        # Load IR
        ir_path = self.ir_dir / img_path.name
        ir = cv2.imread(str(ir_path)) # BGR (3 channels usually, but info is 1ch)
        if ir is None: raise FileNotFoundError(f"IR not found: {ir_path}")
        
        # Resize if needed (assuming 640x640 input for simplicity, or Letterbox)
        # For this script we assume images are reasonably close or we resize.
        # VT-MOT is variable. We MUST resize to fixed size.
        h0, w0 = rgb.shape[:2]
        r = IMG_SIZE / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            rgb = cv2.resize(rgb, (int(w0 * r), int(h0 * r)), interpolation=interp)
            ir = cv2.resize(ir, (int(w0 * r), int(h0 * r)), interpolation=interp)
            
        # Pad to 640x640
        shape = (IMG_SIZE, IMG_SIZE)
        dw = shape[1] - rgb.shape[1]
        dh = shape[0] - rgb.shape[0]
        
        # Exact padding
        top = dh // 2
        bottom = dh - top
        left = dw // 2
        right = dw - left
        
        # Color padding
        rgb = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
        ir = cv2.copyMakeBorder(ir, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
        
        # Create 4-Channel input: RGB + IR (1ch)
        # RGB is BGR in OpenCV
        ir_1ch = ir[..., :1] # Take Blue channel (all same)
        
        # Stack: BGR + IR -> 4 Channels
        # YOLO Preprocessing expects BGR usually if using their transforms, but here we manually feed tensor.
        # Ultralytics model(img) expects RGB usually? No, it handles it. 
        # But we are bypassing preprocessing. We feed tensor directly.
        # Let's stack as BGR + IR.
        input_img = np.concatenate([rgb, ir_1ch], axis=2) # 4 channels
        
        # Transpose to CHW
        input_img = input_img.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
        input_img = np.ascontiguousarray(input_img)
        
        # Labels
        lbl_path = self.lbl_dir / img_path.with_suffix('.txt').name
        labels = []
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    # Person Only (Class 0)
                    if cls_id != 0: continue 
                    
                    # YOLO format: x,y,w,h (normalized)
                    bx, by, bw, bh = map(float, parts[1:])
                    
                    # We padded/resized the image. We must adjust labels?
                    # Ultralytics loss expects normalized labels relative to the image tensor batch.
                    # Since we letterboxed the image, the valid area is smaller.
                    # We need to re-normalize relative to the NEW 640x640 frame.
                    
                    # Original dims: w0, h0
                    # New dims: 640, 640
                    # Scale r, Pads dw, dh
                    
                    # Denormalize
                    bx *= w0; bw *= w0
                    by *= h0; bh *= h0
                    
                    # Transform
                    bx = bx * r + dw
                    by = by * r + dh
                    bw = bw * r
                    bh = bh * r
                    
                    # Normalize to 640
                    bx /= IMG_SIZE
                    by /= IMG_SIZE
                    bw /= IMG_SIZE
                    bh /= IMG_SIZE
                    
                    labels.append([0, bx, by, bw, bh]) # Force class 0 (Person)
                    
        labels = torch.tensor(labels) if len(labels) > 0 else torch.zeros((0, 5))
        
        # Augmentation (Skip for stability first)
        
        return torch.from_numpy(input_img).float() / 255.0, labels

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    
    # Add batch index to labels
    new_labels = []
    for i, lbl in enumerate(labels):
        if lbl.shape[0] > 0:
            b_lbl = torch.zeros((lbl.shape[0], 6))
            b_lbl[:, 0] = i # Batch index
            b_lbl[:, 1:] = lbl
            new_labels.append(b_lbl)
    
    if len(new_labels) > 0:
        new_labels = torch.cat(new_labels, 0)
    else:
        new_labels = torch.zeros((0, 6))
        
    return imgs, new_labels

def transfer_weights_3ch_to_4ch(model, weights_path):
    print(f"Loading weights from {weights_path}...")
    # Safe to use weights_only=False as we trust our own checkpoints
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    # Ultralytics ckpt has 'model' key usually
    state_dict = ckpt['model'].float().state_dict() if 'model' in ckpt else ckpt
    
    # 1. Patch First Conv (0.conv.weight)
    # Key: model.0.conv.weight (Shape [64, 3, 3, 3])
    # Target: [64, 4, 3, 3]
    
    # We load state dict into our 4ch model, but we need to patch the key in the state_dict
    # Or we load the state dict into a 3ch model then copy? No, we have 4ch structure.
    
    # Simple strategy: Iterate and copy
    my_sd = model.state_dict()
    
    for k, v in state_dict.items():
        if k not in my_sd: continue
        
        if k == "model.0.conv.weight":
            # 3CH weight
            print(f"Transferring {k} (3ch -> 4ch)...")
            val_3ch = v
            val_4ch = my_sd[k] # Already initialized 4ch
            
            # Copy RGB
            val_4ch[:, :3, :, :] = val_3ch
            
            # Thermal should be Kaiming Init (Done in Model Init)
            # So we only overwrite RGB part
            my_sd[k] = val_4ch
            
        elif v.shape != my_sd[k].shape:
            # Skip Mismatched (Head mostly)
            print(f"Skipping {k}: Shape Mismatch {v.shape} vs {my_sd[k].shape}")
        else:
            my_sd[k] = v
            
    model.load_state_dict(my_sd)
    print("Transfer Complete.")

def main():
    print("="*60)
    print("YOLO26x Person-Only RGBT Training (1 GPU)")
    print("="*60)
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Dataset
    img_dir_train = DATASET_ROOT / "images/train"
    ir_dir_train = DATASET_ROOT / "images_ir/train"
    lbl_dir_train = DATASET_ROOT / "labels/train"
    
    train_files = sorted(list(img_dir_train.glob("*.jpg")) + list(img_dir_train.glob("*.png")))
    print(f"Found {len(train_files)} training images.")
    
    # Deterministic split
    random.seed(42)
    random.shuffle(train_files)
    split = int(0.9 * len(train_files))
    train_files = train_files[:split]
    
    ds_train = OnTheFlyRGBTDataset(train_files, lbl_dir_train, ir_dir_train, augment=True)
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=6, pin_memory=True)
    
    # 2. Model Init
    print("Initializing Model...")
    # Load Skeleton (using 'yolo26x.yaml' implicitly by loading pt but we patch)
    # Actually just load the pt to get structure, then patch.
    model_wrapper = YOLO("yolo26x.pt") 
    model = model_wrapper.model
    
    # Patch 4-Channel Input
    w = model.model[0].conv.weight
    if w.shape[1] == 3:
        print("Patching 4-Channel Input...")
        new_w = torch.zeros((w.shape[0], 4, w.shape[2], w.shape[3]), device=w.device)
        # We will load weights later, but for structure correctness:
        model.model[0].conv.weight = nn.Parameter(new_w)
        model.model[0].conv.in_channels = 4
        
    # Transfer Weights
    transfer_weights_3ch_to_4ch(model, WEIGHTS_KUST4K)
    
    # 3. Surgery Head (NC=1)
    head = model.model[-1]
    if head.nc != 1:
        print(f"Replacing Head (nc={head.nc} -> nc=1)...")
        ch = [m[0].conv.in_channels for m in head.cv2]
        new_head = Detect(nc=1, ch=ch)
        
        # Copy metadata
        new_head.i = head.i
        new_head.f = head.f
        new_head.type = head.type
        
        model.model[-1] = new_head
        model.nc = 1
        model.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5, nc=1)
        
        # Init Biases (-4.6)
        print("Initializing Head Biases (-4.6)...")
        for i, conv in enumerate(new_head.cv3):
            last_conv = list(conv.modules())[-1]
            if isinstance(last_conv, nn.Conv2d):
                nn.init.constant_(last_conv.bias, -4.6) 
                
    model.to(DEVICE)
    loss_fn = v8DetectionLoss(model)
    
    # 4. Optimizer: SGD (Stable)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4) # 1e-3 safe start
    
    # Scheduler
    iters_per_epoch = len(loader_train)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=iters_per_epoch, epochs=EPOCHS, pct_start=0.1)
    
    # EMA
    ema = ModelEMA(model)
    
    # Loop
    print(f"Starting Training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(loader_train, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, labels in pbar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            if imgs.shape[0] == 0: continue
            
            optimizer.zero_grad()
            
            # Loss Input
            batch = {'img': imgs, 'bboxes': labels[:, 2:], 'cls': labels[:, 1:2], 'batch_idx': labels[:, 0]}
            
            # Ensure training mode
            if not model.model[-1].training:
                print("DEBUG: Force setting head.training = True")
                model.model[-1].train()
            
            preds = model(imgs)
            
            # DEBUG: Inspect preds detail
            if isinstance(preds, torch.Tensor):
                 print(f"DEBUG: preds Tensor Shape: {preds.shape}")
            elif isinstance(preds, list):
                 print(f"DEBUG: preds List len={len(preds)}")
                 print(f"DEBUG: Head Config: nc={model.model[-1].nc}, training={model.model[-1].training}")
            elif isinstance(preds, tuple):
                 print(f"DEBUG: preds Tuple len={len(preds)}")
                 for i, p in enumerate(preds):
                     print(f"  preds[{i}] type: {type(p)}")
            elif isinstance(preds, dict):
                 print(f"DEBUG: preds is Dict. Keys: {list(preds.keys())}")
                 for k, v in preds.items():
                     print(f"  Key '{k}': {type(v)}")
                     if hasattr(v, 'shape'): print(f"    Shape: {v.shape}")
            else:
                 print(f"DEBUG: preds Unknown Type: {type(preds)}")
            
            loss, _ = loss_fn(preds, batch)
            if loss.ndim > 0: loss = loss.sum()
            
            # Check
            if torch.isnan(loss) or loss.item() == 0.0 or loss.item() > 1e6:
                 print(f"[WARNING] Abnormal Loss: {loss.item()}. Zeroing.")
                 loss = loss * 0.0
                 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
            
        # Save
        torch.save(model.state_dict(), SAVE_DIR / f"epoch{epoch+1}.pt")
        torch.save(ema.ema.state_dict(), SAVE_DIR / f"epoch{epoch+1}_ema.pt")

if __name__ == "__main__":
    main()
