
# Implementation Plan – Pivot to Far-View Deployment + GitHub Backup

## 1. Current Training Assessment (AI Engineer Analysis)

### Epoch 7/20 Results (`vtmot_near`, Gated Mid-Fusion)

| Epoch | Train/Box | Train/Cls | mAP50 | mAP50-95 | Val/Box | Val/Cls | LR |
|-------|-----------|-----------|-------|----------|---------|---------|-----|
| 1     | 2.113     | 1.548     | 0.611 | 0.300    | 2.283   | 2.112   | 1e-3 |
| 4     | 1.520     | 0.995     | 0.633 | 0.322    | 2.192   | 1.742   | 8.5e-4 |
| 7     | 1.392     | 0.902     | **0.635** | **0.326** | 2.207 | 1.535   | 7e-4 |

### Verdict

> [!WARNING]
> **Training Loss** is decreasing well (2.11 → 1.39 box loss). However, **Validation mAP is plateauing** — only +0.002 improvement from epoch 5→7. This indicates the model is **near saturation** on this frozen-backbone warmup phase. Continuing all 20 epochs gives diminishing returns.

**Recommendation**: ✅ **Stop near training. Use `best.pt` (epoch 7) to pivot to far-view.**

**Why this is GOOD:**

- `best.pt` has learned RGBT feature fusion from near-view data
- Far-view data is 2x larger (284k vs ~100k images) — more diverse
- Transfer learning from near→far is a proven strategy (domain adaptation)

---

## 2. Strategy: Pivot to Far-View

### What Changes

| Aspect | Current (Near) | New (Far) |
|--------|---------------|-----------|
| Dataset | `vtmot_near` (~100k imgs) | `vtmot_far` (284k imgs) |
| YAML | `near_view_clean.yaml` | `far_view_clean.yaml` |
| Weights | FLIR pretrained → Random Gate | **best.pt from near** (Gate already learned!) |
| Goal | Research experiment | **Deployment-ready model** |

### Key Improvements for Far-View Training

1. **Use near `best.pt` as starting weights** (Transfer Learning V2)
   - The Gated Fusion layers are already trained — massive head start
   - Only fine-tune on far data for domain adaptation

2. **Use `far_view_clean.yaml`** (NOT the old `far_view.yaml`)
   - This has the correct test split we just added

3. **Unfrozen training** — no frozen backbone needed
   - The backbone is already fine-tuned from near-view

---

## 3. Proposed Changes

### Training Script

#### [NEW] [train_far_model_gated.py](file:///home/student/Toan/train_far_model_gated.py)

Create a new far-view training script based on the proven near-view gated approach:

- **Weights**: Load from `/home/student/Toan/runs/gated_experiment/near_view_gated_phase1/weights/best.pt`
- **Dataset**: `datasets/vtmot_far/far_view_clean.yaml`
- **Config**: Single phase (no frozen backbone), SGD optimizer, cosine LR
- **Epochs**: 30 (far dataset is 3x larger, fewer epochs needed)
- **Batch**: 16, ImgSz: 640
- **Augmentations**: Mosaic, MixUp, CopyPaste, Scale for small-object robustness

---

### GitHub Backup

#### [MODIFY] [.gitignore](file:///home/student/Toan/.gitignore)

Temporarily override `.gitignore` to include critical files:

- Add exception for `weights/best_near_gated_phase1.pt` (copy of best.pt)
- Add exception for `datasets/vtmot_far/far_view_clean.yaml`

#### Git LFS for Weights

Since `best.pt` is 339MB (exceeds GitHub's 100MB limit), use **Git LFS**:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

#### Commit Strategy

```
commit 1: Code + configs + YAML (lightweight)
commit 2: weights via LFS (best.pt only)
```

**Files to include:**

- All `*.py` scripts (training, verification, utils)
- `YOLOv11-RGBT/` (the modified ultralytics fork with GatedSpatialFusion_V3)
- `datasets/vtmot_far/far_view_clean.yaml`
- `gate_supervision.py`, `visualize_gates.py`
- `weights/best_near_gated_phase1.pt` (via LFS)

**Files to EXCLUDE (already in `.gitignore`):**

- `datasets/` images/labels (1.2TB — use Kaggle)
- `runs/` (20GB — checkpoint only the best)
- `tracking/` (98GB — separate concern)
- `data/` (source data)

---

## 4. Verification Plan

### Automated

- `git status` — confirm all critical files tracked
- `git push` — verify push succeeds
- Launch `train_far_model_gated.py` and confirm epoch 1 starts

### Manual

- After push, clone on another machine to verify repo completeness
- Confirm far training converges (mAP50 should exceed 0.6 within 5 epochs)
