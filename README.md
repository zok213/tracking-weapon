# VT-MOT Tracking Weapon 🎯

RGBT Multi-Object Tracking with Gated Mid-Fusion for Deployment.

## Quick Start (After Clone)

```bash
# 1. Install dependencies
pip install -e ./YOLOv11-RGBT

# 2. Download dataset from Kaggle
# Upload vtmot_far_*.zip parts, then extract:
# unzip vtmot_far_train_part1.zip -d datasets/vtmot_far/
# unzip vtmot_far_train_part2.zip -d datasets/vtmot_far/
# ... etc

# 3. Download weights (from original machine backup)
# Place in weights/ directory

# 4. Train far-view model
python3 train_far_model_gated.py
```

## Project Structure

```
├── train_far_model_gated.py    # 🎯 MAIN: Far-view deployment training
├── train_near_model_gated.py   # Near-view gated fusion experiment
├── gate_supervision.py         # Gate supervision loss module
├── visualize_gates.py          # Gate weight visualization
├── mcf_utils.py                # MCF utility functions
├── YOLOv11-RGBT/               # Modified Ultralytics with GatedSpatialFusion_V3
│   └── ultralytics/
│       └── nn/modules/block.py # ⭐ GatedSpatialFusion_V3 implementation
├── datasets/
│   ├── vtmot_far/far_view_clean.yaml  # Far-view dataset config
│   └── vtmot_near/near_view_clean.yaml
├── weights/                    # Pretrained weights (not in git, >100MB)
├── docs/                       # Architecture docs, analysis, walkthroughs
└── scripts/                    # Utility scripts
```

## Key Architecture

**Gated Spatial Fusion V3** — Custom RGBT fusion layer:

- Dual-branch attention (RGB + Thermal)
- MC-Dropout uncertainty estimation
- Learnable illumination scaling
- Gate supervision loss for convergence

## Training Strategy

1. **Near-view warmup** → best.pt (mAP50=0.635)
2. **Far-view fine-tune** → Transfer Learning V2 (near→far domain adaptation)

## Dataset

- **vtmot_far**: 284k images (Train 74%, Val 16%, Test 11%)
- Sources: VTuav, wurenji, qiuxing, RGBT234, photo sequences
- Single class: person (far-view detection)
