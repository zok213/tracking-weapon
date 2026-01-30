# VT-MOT: Vision-Thermal Multi-Object Tracking

Multi-modal object tracking system for Qualcomm QCS8550 edge deployment.

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/VT-MOT.git
cd VT-MOT

# Download datasets (manual - Google Drive)
# See data/README.md for links

# Train teacher model
python -c "
from ultralytics import YOLO
model = YOLO('yolo26x.pt')
model.train(data='configs/data_kust4k.yaml', epochs=100, batch=16, device=0)
"
```

## Project Structure

```
├── configs/         # YOLO dataset configs
├── scripts/         # Training & utility scripts
├── docs/            # Architecture documentation
├── data/            # Datasets (gitignored)
├── checkpoints/     # Model weights (gitignored)
└── logs/            # Training logs (gitignored)
```

## Key Components

1. **Detection**: YOLO26x → YOLO11m (Knowledge Distillation)
2. **Tracking**: ByteTrack with VI-ReID
3. **Fusion**: RGB + Thermal (CBAM at P3)
4. **Target**: Qualcomm QCS8550 (≤20.5ms latency)

## Datasets

| Dataset | Size | Altitude | Purpose |
|:--------|:-----|:---------|:--------|
| KUST4K | 4K images | 30-60m | Pretraining |
| VT-MOT | 166K images | Mixed | Main training |

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Dataflow](docs/DATAFLOW.md)
- [QCS8550 Guide](docs/qualcomm/QCS8550_SUMMARY.md)
- [Training Guide](docs/qualcomm/TRAINING_GUIDE.md)

## License

MIT
