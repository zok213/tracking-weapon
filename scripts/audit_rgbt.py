import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import IterableSimpleNamespace
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Re-define minimal Dataset/Collate to check actual data flowing in
class RGBTDatasetCheck(torch.utils.data.Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=640):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.images = sorted(self.img_dir.glob('*.npy'))[:10] # Just check 10
        print(f"Checking {len(self.images)} samples...")
        
    def __len__(self): return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = np.load(img_path) # [H, W, 4]
        
        # Check Value Range
        if img.max() > 255:
            print(f"⚠️ Image {img_path.name} has values > 255 (Max: {img.max()})")
        
        # Label
        lbl_path = self.lbl_dir / (img_path.stem + '.txt')
        labels = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append([float(x) for x in parts[:5]]) # cls, x, y, w, h
        
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1) # [4, 640, 640]
        
        if labels: labels = torch.tensor(labels, dtype=torch.float32)
        else: labels = torch.zeros((0, 5), dtype=torch.float32)
        
        return torch.tensor(img), labels, str(img_path)

def collate_fn(batch):
    imgs, labels, paths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    new_labels = []
    for i, lbl in enumerate(labels):
        if len(lbl) > 0:
            batch_idx = torch.full((len(lbl), 1), i, dtype=torch.float32)
            new_labels.append(torch.cat([batch_idx, lbl], dim=1))
    if new_labels: labels = torch.cat(new_labels, 0)
    else: labels = torch.zeros((0, 6), dtype=torch.float32)
    return imgs, labels, paths

def audit_phase1():
    device = 'cuda:0'
    ckpt_path = "/home/student/Toan/checkpoints/rgbt/epoch90_v6.pt" # The stable one
    
    print("="*60)
    print("DEEP AUDIT: KUST4K RGBT Phase 1")
    print("="*60)
    
    # 1. VISUALIZATION CHECK
    ds = RGBTDatasetCheck(
        "/home/student/Toan/data/KUST4K_RGBT/images/train",
        "/home/student/Toan/data/KUST4K_RGBT/labels/train"
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    
    imgs, labels, paths = next(iter(loader))
    imgs = imgs.to(device)
    # labels: [N, 6] -> (batch_idx, cls, x, y, w, h)
    
    print("\n--- Data Statistics ---")
    print(f"Input Shape: {imgs.shape}")
    print(f"Input Range: [{imgs.min():.3f}, {imgs.max():.3f}] (Expected: [0.0, 1.0])")
    print(f"Labels Count: {len(labels)}")
    
    # Save a debug image
    # Take first image, RGB channels, draw boxes
    img0 = imgs[0].cpu().numpy().transpose(1, 2, 0) # [640, 640, 4]
    rgb = (img0[:, :, :3] * 255).astype(np.uint8).copy()
    tir = (img0[:, :, 3] * 255).astype(np.uint8).copy()
    
    # Draw boxes
    img_labels = labels[labels[:, 0] == 0]
    H, W = 640, 640
    for lbl in img_labels:
        cls, x, y, w, h = lbl[1:]
        x1 = int((x - w/2) * W)
        y1 = int((y - h/2) * H)
        x2 = int((x + w/2) * W)
        y2 = int((y + h/2) * H)
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(tir, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
    debug_path = "/home/student/Toan/audit_sample.jpg"
    # Side by side
    tir_colored = cv2.cvtColor(tir, cv2.COLOR_GRAY2BGR)
    vis = np.hstack([rgb, tir_colored])
    cv2.imwrite(debug_path, vis)
    print(f"Saved visualization to {debug_path} (Please Check!)")
    
    # 2. MODEL & LOSS CHECK
    print(f"\n--- Model & Loss Breakdown ---")
    # Load model
    model = YOLO("/home/student/Toan/models/yolo26x_rgbt_init.pt")
    # Load weights
    state_dict = torch.load(ckpt_path)
    model.model.load_state_dict(state_dict)
    model.to(device)
    model.model.train() # Enable train mode for loss calc
    
    # Setup Loss
    hyp = {
        'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0,
        'label_smoothing': 0.0, 'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7,
        'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,
        'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5,
        'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0, 'warmup_epochs': 3.0,
        'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
    }
    model.model.args = IterableSimpleNamespace(**hyp)
    loss_fn = v8DetectionLoss(model.model)
    
    # Forward Pass
    batch = {
        'img': imgs,
        'bboxes': labels[:, 2:], 
        'cls': labels[:, 1:2],
        'batch_idx': labels[:, 0]
    }
    
    preds = model.model(imgs)
    if isinstance(preds, dict):
         preds = list(preds.values())[0] # Try generic unwrap
    
    # Calculate Loss
    loss, loss_items = loss_fn(preds, batch)
    
    # loss_items is a tensor: (box, cls, dfl)
    box_loss = loss_items[0].item()
    cls_loss = loss_items[1].item()
    dfl_loss = loss_items[2].item()
    
    print(f"Total Loss Output (Tensor): {loss}")
    
    if loss.ndim > 0:
        total_loss_val = loss.sum().item()
        print(f"Summed Loss (used in training): {total_loss_val:.4f}")
    else:
        total_loss_val = loss.item()
        print(f"Scalar Loss: {total_loss_val:.4f}")
        
    print(f"\nBreakdown (Scaled by Batch Size?):")
    print(f"  Box Loss: {box_loss:.4f} (Weight 7.5)")
    print(f"  Cls Loss: {cls_loss:.4f} (Weight 0.5)")
    print(f"  DFL Loss: {dfl_loss:.4f} (Weight 1.5)")
    
    # Analyze Meaning
    print("\n--- Engineering Analysis ---")
    bs = 4
    per_img_loss = loss.sum().item() / bs
    print(f"Per-Image Loss: {per_img_loss:.4f}")
    
    if per_img_loss < 2.0:
        print("✅ VERDICT: Loss is LOW per image. The 7.1 is just Batch Summation.")
    else:
        print("⚠️ VERDICT: Loss is genuinely high per image. Model underfitting.")

if __name__ == '__main__':
    audit_phase1()
