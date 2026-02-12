#!/usr/bin/env python3
"""
VI-ReID Validation Script
=========================
Evaluates the trained AGW Model on VT-MOT Test Set.
Metrics: Rank-1, Rank-5, Rank-10, mAP.
Modes: 
  - RGB to IR (Visible Query -> Thermal Gallery)
  - IR to RGB (Thermal Query -> Visible Gallery)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
import glob
import re
from tqdm import tqdm
from sklearn.preprocessing import normalize
from collections import defaultdict

# Config
DATA_ROOT = "/home/student/Toan/data/VT-MOT_ReID_Person_Only"
MODEL_PATH = "checkpoints_vi_ddp/agw_best.pth"
IMG_SIZE = (256, 128)
BATCH_SIZE = 256
DEVICE = "cuda"

# --- Model Definition (Must Match Training) ---
class EmbedNetwork(nn.Module):
    def __init__(self, num_classes=0):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features).view(features.size(0), -1)
        feat = self.bottleneck(features)
        return feat

def load_data(folder, verbose=True):
    files = glob.glob(os.path.join(folder, '*.jpg')) + glob.glob(os.path.join(folder, '*.png'))
    data = []
    pids = []
    camids = []
    
    for fpath in files:
        fname = os.path.basename(fpath)
        # Parse 0001_c1...
        match = re.match(r'^(\d+)_c(\d+)', fname)
        if match:
            pid = int(match.group(1))
            camid = int(match.group(2))
            data.append(fpath)
            pids.append(pid)
            camids.append(camid)
            
    if verbose:
        print(f"📦 Loaded {len(data)} images from {os.path.basename(folder)}")
    return data, np.array(pids), np.array(camids)

def extract_features(model, file_paths):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features = []
    
    # Batch processing
    for i in tqdm(range(0, len(file_paths), BATCH_SIZE), desc="Extracting"):
        batch_paths = file_paths[i:i+BATCH_SIZE]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            imgs.append(transform(img))
            
        batch = torch.stack(imgs).to(DEVICE)
        
        with torch.no_grad():
            feat = model(batch)
            features.append(feat.cpu().numpy())
            
    return np.concatenate(features)

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, rank=1):
    num_q, num_g = distmat.shape
    if num_g < rank:
        rank = num_g
        print(f"Note: Gallery size {num_g} < Rank {rank}")

    # Initialize aggregators
    all_cmc = np.zeros(num_g)
    all_ap = []
    
    # Process in chunks to avoid OOM during argsort/broadcasting
    chunk_size = 1000
    
    print(f"Computing Metrics (Chunked {chunk_size})...")
    
    for i in range(0, num_q, chunk_size):
        end = min(i + chunk_size, num_q)
        # Slicing creates a view or copy, manageable (1000 x 69000 float32 = 276MB)
        dist_chunk = distmat[i:end] 
        q_pids_chunk = q_pids[i:end]
        q_camids_chunk = q_camids[i:end]
        
        # Sort indices (1000 x 69000 int64 = 552MB) -> Safe
        indices = np.argsort(dist_chunk, axis=1)
        
        # Matches (1000 x 69000 bool = 69MB)
        matches = (g_pids[indices] == q_pids_chunk[:, np.newaxis])
        
        # Keep matches only (filter same cam/junk if needed)
        # Here we assume Cross-Modality = different cams implicitly or explicit check
        # For standard VI-ReID (SYSU), we use all gallery.
        
        # CMC
        # cumsum matches
        cmc = np.cumsum(matches, axis=1)
        cmc[cmc > 1] = 1
        all_cmc += np.sum(cmc, axis=0) # Accumulate sum for mean later
        
        # AP
        num_rel = np.sum(matches, axis=1)
        for j in range(len(num_rel)):
             if num_rel[j] > 0:
                 match = matches[j]
                 cumsum = np.cumsum(match)
                 precision = cumsum / np.arange(1, num_g + 1)
                 ap = np.sum(precision * match) / num_rel[j]
                 all_ap.append(ap)
                 
    # Finalize
    all_cmc = all_cmc / num_q
    mAP = np.mean(all_ap) if all_ap else 0
    
    return all_cmc, mAP

def evaluate_mode(model, query_dir, gallery_dir, mode_name="Test"):
    print(f"\n--- Evaluating Mode: {mode_name} ---")
    q_paths, q_pids, q_camids = load_data(query_dir)
    g_paths, g_pids, g_camids = load_data(gallery_dir)
    
    if len(q_paths) == 0 or len(g_paths) == 0:
        print("⚠️ Empty Query or Gallery.")
        return
    
    print("Extracting Query Features...")
    q_feats = extract_features(model, q_paths)
    print("Extracting Gallery Features...")
    g_feats = extract_features(model, g_paths)
    
    # Normalize (Cosine Distance = Euclidean on Normalized)
    q_feats = normalize(q_feats, norm='l2', axis=1)
    g_feats = normalize(g_feats, norm='l2', axis=1)
    
    # Euclidean Distance (on normalized vectors -> related to Cosine)
    # dist = 2 - 2 * dot_product
    # or just dot product and sort descending
    
    print("Computing Distance Matrix (Chunked)...")
    
    m, n = q_feats.shape[0], g_feats.shape[0]
    distmat = np.zeros((m, n), dtype=np.float32)
    
    # Chunk size for Query
    chunk_size = 2000 
    
    q_feats_t = torch.from_numpy(q_feats).to(DEVICE)
    g_feats_t = torch.from_numpy(g_feats).to(DEVICE)
    
    for i in tqdm(range(0, m, chunk_size), desc="DistMat"):
        end = min(i + chunk_size, m)
        q_chunk = q_feats_t[i:end]
        
        # dist = 2 - 2 * q @ g.T
        # Compute simulation on GPU
        sim = torch.mm(q_chunk, g_feats_t.t())
        dist_chunk = 2 - 2 * sim
        distmat[i:end] = dist_chunk.cpu().numpy()
        
    del q_feats_t, g_feats_t, sim, dist_chunk
    torch.cuda.empty_cache()
    
    print("Computing Metrics...")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    
    print(f"🏆 {mode_name} Results:")
    print(f"   mAP    : {mAP:.2%}")
    print(f"   Rank-1 : {cmc[0]:.2%}")
    print(f"   Rank-5 : {cmc[4]:.2%}")
    print(f"   Rank-10: {cmc[9]:.2%}")
    
    return mAP, cmc[0]

def main():
    print(f"Loading Model: {MODEL_PATH}")
    # We don't know num_classes of checkpoint, but for feature extraction it doesn't matter
    # except for loading weights. We must match architecture.
    # The checkpoint has 'classifier.weight' size.
    # We can load with strict=False to ignore classifier.
    
    model = EmbedNetwork(num_classes=1115).to(DEVICE) # Match checkpoint size
    state_dict = torch.load(MODEL_PATH)
    
    # Fix DDP prefix
    new_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_dict[name] = v
        
    model.load_state_dict(new_dict, strict=False)
    model.eval()
    
    # Mode 1: RGB Query -> IR Gallery
    evaluate_mode(
        model, 
        query_dir=os.path.join(DATA_ROOT, 'query'), 
        gallery_dir=os.path.join(DATA_ROOT, 'ir_bounding_box_test'),
        mode_name="RGB -> IR (Visible to Thermal)"
    )
    
    # Mode 2: IR Query -> RGB Gallery
    # Assuming 'ir_bounding_box_test' can be query too? 
    # Usually we use a subset. Let's use full set as per SYSU all-search.
    evaluate_mode(
        model, 
        query_dir=os.path.join(DATA_ROOT, 'ir_bounding_box_test'),  
        gallery_dir=os.path.join(DATA_ROOT, 'bounding_box_test'),
        mode_name="IR -> RGB (Thermal to Visible)"
    )

if __name__ == "__main__":
    main()
