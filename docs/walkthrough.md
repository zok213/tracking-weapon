
# Walkthrough: Far-View Pivot + GitHub Backup

## What Was Done

### 1. Training Assessment ✅

**Near-View Gated Fusion** (Epoch 7/20):

- mAP50: **0.635** (plateauing, +0.002 in last 2 epochs)
- mAP50-95: **0.326**
- Train loss decreasing well, but val mAP saturating
- **Decision**: Epoch 7 `best.pt` is sufficient for transfer learning

### 2. Far-View Training Script ✅

Created [train_far_model_gated.py](file:///home/student/Toan/train_far_model_gated.py):

- **Transfer Learning V2**: Near `best.pt` → Far domain adaptation
- Single-phase unfrozen training (SGD, cosine LR, 30 epochs)
- Strong small-object augmentations (scale=0.7, mosaic, mixup, copy_paste)
- Gate supervision + gradient clipping from proven near-view approach

### 3. GitHub Backup ✅

- **153 files committed** (19,928 lines of code)
- Successfully **pushed to** `github.com/zok213/tracking-weapon.git`
- Includes: all training scripts, analysis tools, dataset YAMLs

### 4. Dataset Packaging (Background)

- Parallel sharded zip script running (6 shards, 32 workers)
- Train: 4 parts, Val/Test: 2 parts

## Blocked: GPU Driver Mismatch 🚨

```
Kernel module:  NVIDIA 580.95.05
Userspace lib:  NVIDIA 580.126
```

**Root Cause**: NVIDIA driver was updated without system reboot. The kernel still loads the old driver module.

**Fix**: `sudo reboot` — then far-view training will start automatically when relaunched.

## After Reboot — Quick Start

```bash
cd /home/student/Toan
nohup python3 -u train_far_model_gated.py > runs/far_gated_train.log 2>&1 &
tail -f runs/far_gated_train.log
```

## Weight Strategy

| Weight | Location | Purpose |
|--------|----------|---------|
| Near best.pt | `runs/gated_experiment/near_view_gated_phase1/weights/best.pt` | Starting weights for far training |
| Backup copy | `weights/best_near_gated_phase1.pt` | Safety copy |
| FLIR pretrained | `weights/FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-*.pt` | Fallback if near best.pt missing |
