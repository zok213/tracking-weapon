#!/usr/bin/env python3
"""
VI-ReID Training Script (10GB VRAM version)
Optimized for available GPU 0 memory.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import random
from tqdm import tqdm

# FORCE GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- 1. Dataset ---
class ReIDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Walk directories (format: root/pid/image.jpg)
        if not os.path.exists(root_dir):
             os.makedirs(root_dir, exist_ok=True)
             
        pids = sorted(os.listdir(root_dir))
        for pid_idx, pid in enumerate(pids):
            pid_path = os.path.join(root_dir, pid)
            if not os.path.isdir(pid_path): continue
            
            for img_file in os.listdir(pid_path):
                if img_file.endswith(('.jpg', '.png')):
                    self.images.append(os.path.join(pid_path, img_file))
                    self.labels.append(pid_idx) # Map PID string to int index
                    
        self.num_classes = len(pids)
        print(f"📦 Loaded {len(self.images)} images, {self.num_classes} identities.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 256, 128), label

# --- 2. Model (ResNet50 + GeM) ---
class EmbedNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Backbone
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) # Remove FC and AvgPool
        
        # GeM Pooling (Generalized Mean) - Better than Max/Avg
        self.pool = nn.AdaptiveAvgPool2d(1) 
        
        # Heads
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False) # Fix bias
        
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        self.bottleneck.apply(self._weights_init_kaiming)
        self.classifier.apply(self._weights_init_classifier)
        
    def forward(self, x):
        features = self.backbone(x) # (B, 2048, H, W)
        features = self.pool(features) # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1) # (B, 2048)
        
        feat = self.bottleneck(features)
        
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, features # Return logits + global features
        else:
            return feat # Inference embedding

    def _weights_init_kaiming(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def _weights_init_classifier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)

# --- 3. Loss Functions ---
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        inputs = inputs.float() # Stability: Force FP32 for distance calc
        n = inputs.size(0)
        # Compute pairwise distance matrix
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # Euclidean distance
        
        # Hard Negative Mining (Batch Hard)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, _ = torch.max(dist * mask.float(), dim=1) # Hardest positive
        dist_an, _ = torch.min(dist * (1 - mask.float()) + mask.float() * 1e6, dim=1) # Hardest negative
        
        y = dist_an.data.new().resize_as_(dist_an).fill_(1)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

# --- 4. Main Training Loop ---
def main():
    # Config
    DATA_DIR = "/home/student/Toan/stage1/data/vtmot_reid_generated/train"
    BATCH_SIZE = 64 # Fits in 10-12GB VRAM
    EPOCHS = 60 
    LR = 0.00035
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure checkpoints dir
    os.makedirs("checkpoints", exist_ok=True)
    
    # Validation Check
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"⚠️ Data directory empty or missing: {DATA_DIR}")
        print("Please run `prepare_vtmot_reid.py` first.")
        # Try to find prepare script and suggest running it
        return
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.4)),
    ])
    
    # Datasets
    full_dataset = ReIDDataset(DATA_DIR, transform=train_transform)
    if len(full_dataset) == 0:
        print("No images found.")
        return

    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # Init Model
    model = EmbedNetwork(num_classes=full_dataset.num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda')
    criterion_cls = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=0.3)
    
    print(f"\n🚀 Training on GPU 0 | {len(full_dataset)} images | {full_dataset.num_classes} IDs")
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits, features = model(images)
                loss_cls = criterion_cls(logits, labels)
                loss_tri = criterion_triplet(features, labels)
                loss = loss_cls + loss_tri
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, preds = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        train_acc = (100.0 * train_correct / (train_total + 1e-6))
        avg_loss = train_loss/len(train_loader)
        
        print(f"🏁 Ep {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {train_acc:.2f}%")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/reid_agw_resnet50_best.pth")
            
    print(f"\n✅ Finished. Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
