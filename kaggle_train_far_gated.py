#!/usr/bin/env python3
"""
=============================================================================
VT-MOT Far-View Gated Training — Kaggle Notebook
=============================================================================
Run this notebook on Kaggle with GPU (P100/T4/A100) to train the
Gated Mid-Fusion model on the vtmot_far dataset.

PREREQUISITES (Kaggle Setup):
  1. Upload dataset zips as Kaggle Datasets:
     - Dataset 1: "vtmot-far-train" containing vtmot_far_train_part1-4.zip
     - Dataset 2: "vtmot-far-valtest" containing vtmot_far_val_test_part1-2.zip
     - Dataset 3: "vtmot-weights" containing best_near_gated_phase1.pt
       (and optionally FLIR_aligned3C-...pt)
  2. Add all 3 datasets to notebook inputs
  3. Enable GPU accelerator in notebook settings
  4. Set persistence to "Files" for checkpoint saving
=============================================================================
"""

# ============================================================
# CELL 1: Environment Setup
# ============================================================
import os
import subprocess
import sys

WORK_DIR = "/kaggle/working"
REPO_DIR = os.path.join(WORK_DIR, "tracking-weapon")
DATASET_DIR = os.path.join(WORK_DIR, "datasets/vtmot_far")

# Kaggle input paths (adjust dataset slugs to match YOUR upload names)
KAGGLE_INPUT = "/kaggle/input"
# Common patterns - adjust these to your actual Kaggle dataset names:
TRAIN_DATASET = os.path.join(KAGGLE_INPUT, "vtmot-far-train")
VALTEST_DATASET = os.path.join(KAGGLE_INPUT, "vtmot-far-valtest")
WEIGHTS_DATASET = os.path.join(KAGGLE_INPUT, "vtmot-weights")

print("=" * 70)
print("VT-MOT Far-View Gated Training — Kaggle Setup")
print("=" * 70)

# ============================================================
# CELL 2: Clone Repository
# ============================================================
if not os.path.exists(REPO_DIR):
    print("\n📥 Cloning repository...")
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/zok213/tracking-weapon.git",
        REPO_DIR
    ], check=True)
    print("✅ Repository cloned.")
else:
    print("✅ Repository already exists.")

# ============================================================
# CELL 3: Install Modified Ultralytics (YOLOv11-RGBT)
# ============================================================
print("\n📦 Installing YOLOv11-RGBT (modified Ultralytics)...")

# Install the modified ultralytics from local source
ultralytics_dir = os.path.join(REPO_DIR, "YOLOv11-RGBT")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-e", ultralytics_dir,
    "--quiet", "--no-deps"
], check=True)

# Install additional requirements
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "einops>=0.7", "timm>=0.9", "efficientnet-pytorch>=0.7.1",
    "albumentations>=1.0.3", "thop", "psutil",
    "--quiet"
], check=True)

print("✅ Dependencies installed.")

# Verify import
import torch
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

from ultralytics import YOLO
print(f"   Ultralytics imported successfully")

try:
    from ultralytics.nn.modules.block import GatedSpatialFusion_V3
    print(f"   ✅ GatedSpatialFusion_V3 available!")
except ImportError:
    print(f"   ❌ GatedSpatialFusion_V3 NOT found — check YOLOv11-RGBT installation")

# ============================================================
# CELL 4: Setup Dataset
# ============================================================
print("\n📂 Setting up dataset...")

os.makedirs(DATASET_DIR, exist_ok=True)

def find_and_extract_zips(search_dir, target_dir, pattern="vtmot_far"):
    """Find zip files and extract them to target directory."""
    import glob
    import zipfile
    
    zips = glob.glob(os.path.join(search_dir, "**", f"*{pattern}*.zip"), recursive=True)
    if not zips:
        # Try direct listing
        zips = glob.glob(os.path.join(search_dir, "*.zip"))
    
    print(f"   Found {len(zips)} zip files in {search_dir}")
    for z in sorted(zips):
        print(f"   📦 Extracting: {os.path.basename(z)} ({os.path.getsize(z) / (1024**3):.1f} GB)")
        with zipfile.ZipFile(z, 'r') as zf:
            zf.extractall(target_dir)
        print(f"      ✅ Done")

# Extract from Kaggle input datasets
for dataset_path in [TRAIN_DATASET, VALTEST_DATASET]:
    if os.path.exists(dataset_path):
        find_and_extract_zips(dataset_path, DATASET_DIR)
    else:
        print(f"   ⚠️ Dataset not found: {dataset_path}")
        print(f"      Available inputs: {os.listdir(KAGGLE_INPUT) if os.path.exists(KAGGLE_INPUT) else 'none'}")

# Verify dataset structure
for split in ["train", "val", "test"]:
    img_dir = os.path.join(DATASET_DIR, "images", split)
    lbl_dir = os.path.join(DATASET_DIR, "labels", split)
    if os.path.exists(img_dir):
        n_imgs = len(os.listdir(img_dir))
        n_lbls = len(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else 0
        print(f"   {split}: {n_imgs} images, {n_lbls} labels")
    else:
        print(f"   {split}: ❌ NOT FOUND at {img_dir}")

# ============================================================
# CELL 5: Create Dataset YAML (Dynamic Paths)
# ============================================================
yaml_content = f"""
path: {DATASET_DIR}
train: images/train
val: images/val
test: images/test
names:
  0: person
"""

yaml_path = os.path.join(DATASET_DIR, "far_view_kaggle.yaml")
with open(yaml_path, 'w') as f:
    f.write(yaml_content)
print(f"\n✅ Dataset YAML: {yaml_path}")

# ============================================================
# CELL 6: Setup Weights
# ============================================================
print("\n⚖️ Setting up weights...")

weights_dir = os.path.join(WORK_DIR, "weights")
os.makedirs(weights_dir, exist_ok=True)

best_pt = None
flir_pt = None

# Check Kaggle weights dataset
if os.path.exists(WEIGHTS_DATASET):
    import glob
    weight_files = glob.glob(os.path.join(WEIGHTS_DATASET, "**", "*.pt"), recursive=True)
    for wf in weight_files:
        basename = os.path.basename(wf)
        dest = os.path.join(weights_dir, basename)
        if not os.path.exists(dest):
            import shutil
            shutil.copy2(wf, dest)
            print(f"   📋 Copied: {basename} ({os.path.getsize(wf) / (1024**2):.0f} MB)")
        
        if "best_near_gated" in basename:
            best_pt = dest
        elif "FLIR" in basename:
            flir_pt = dest

# Check repo weights directory
repo_weights = os.path.join(REPO_DIR, "weights")
if os.path.exists(repo_weights):
    import glob
    for wf in glob.glob(os.path.join(repo_weights, "*.pt")):
        basename = os.path.basename(wf)
        if "best_near_gated" in basename:
            best_pt = wf

if best_pt:
    print(f"   ✅ Best weights: {best_pt}")
else:
    print(f"   ⚠️ No best_near_gated_phase1.pt found!")
    print(f"      Will use FLIR weights or random init as fallback.")

if flir_pt:
    print(f"   ✅ FLIR weights: {flir_pt}")

# ============================================================
# CELL 7: Training Script (Self-Contained)
# ============================================================
print("\n" + "=" * 70)
print("🚀 STARTING FAR-VIEW GATED MID-FUSION TRAINING")
print("=" * 70)

# Add repo to path for gate_supervision and visualize_gates
sys.path.insert(0, REPO_DIR)

from pathlib import Path
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import loss as loss_module

# v2.7 bbox_decode patch
_original_bbox_decode = loss_module.v8DetectionLoss.bbox_decode
def _patched_bbox_decode(self, anchor_points, pred_dist):
    if self.use_dfl:
        if self.proj.device != pred_dist.device:
            self.proj = self.proj.to(pred_dist.device)
    return _original_bbox_decode(self, anchor_points, pred_dist)
loss_module.v8DetectionLoss.bbox_decode = _patched_bbox_decode
print("[OK] Applied v2.7 bbox_decode device patch")


class MCFTrainer(DetectionTrainer):
    """Custom trainer with Gated Fusion support."""
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, mcf_model=None):
        self._mcf_model = mcf_model
        super().__init__(cfg, overrides, _callbacks)
    
    def setup_model(self):
        model = self._mcf_model if self._mcf_model is not None else super().setup_model()
        count = 0
        try:
            from ultralytics.nn.modules.block import GatedSpatialFusion_V3
            modules = model.modules() if hasattr(model, 'modules') else model.model.modules()
            for m in modules:
                if isinstance(m, GatedSpatialFusion_V3):
                    m.export_gates = True
                    count += 1
        except ImportError:
            pass
        print(f"[MCFTrainer] Gate export enabled on {count} layers.")
        
        if self._mcf_model is not None:
            self.model = model
            self.model.to(self.device)
            self.model.args = self.args
            return self.model
        return model

    def get_loss(self):
        loss = super().get_loss()
        try:
            from gate_supervision import GatedDetectionLoss
            print("[MCFTrainer] Gate supervision loss active.")
            return GatedDetectionLoss(self.model, loss)
        except ImportError:
            return loss


# Weight selection
device = 0
resume_flag = False
runs_dir = os.path.join(WORK_DIR, "runs")
project_dir = os.path.join(runs_dir, "far_gated_deployment")
run_name = "far_view_gated_kaggle"

last_ckpt = os.path.join(project_dir, run_name, "weights", "last.pt")

if os.path.exists(last_ckpt):
    print(f"[RESUME] Checkpoint found: {last_ckpt}")
    model = YOLO(last_ckpt)
    resume_flag = True
elif best_pt and os.path.exists(best_pt):
    print(f"[TRANSFER] Loading near-view best.pt: {best_pt}")
    model = YOLO(best_pt)
    print("✅ Transfer Learning V2: Near→Far domain adaptation")
elif flir_pt and os.path.exists(flir_pt):
    print(f"[FALLBACK] Loading FLIR pretrained weights: {flir_pt}")
    model_yaml = os.path.join(REPO_DIR, "YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11x-RGBT-gated-v3.yaml")
    model = YOLO(model_yaml)
    ckpt = torch.load(flir_pt, map_location='cpu')
    if 'model' in ckpt:
        chk_sd = ckpt['model'].state_dict()
        mdl_sd = model.model.state_dict()
        filtered = {k: v for k, v in chk_sd.items() if k in mdl_sd and v.shape == mdl_sd[k].shape}
        if filtered:
            model.model.load_state_dict(filtered, strict=False)
            print(f"✅ Transferred {len(filtered)} layers from FLIR.")
else:
    print("[INIT] No pretrained weights. Training from scratch (not recommended).")
    model_yaml = os.path.join(REPO_DIR, "YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11x-RGBT-gated-v3.yaml")
    model = YOLO(model_yaml)

# Gradient clipping
def on_train_start(trainer):
    if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
        orig_step = trainer.optimizer.step
        def clipped_step(closure=None):
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=10.0)
            return orig_step(closure)
        trainer.optimizer.step = clipped_step
        print("⚡ Gradient Clipping (norm=10.0) applied.")

# Kaggle-optimized config
# Note: Kaggle GPU has 16GB VRAM (P100/T4) — batch=8 is safer
config = {
    'model': last_ckpt if resume_flag else 'yolo11x.pt',
    'data': yaml_path,
    'epochs': 30,
    'imgsz': 640,
    'batch': 8,             # Kaggle-safe (16GB VRAM)
    'device': device,
    'use_simotm': 'RGBRGB6C',
    'channels': 6,
    'pairs_rgb_ir': ['_rgb_', '_ir_'],
    'optimizer': 'SGD',
    'lr0': 0.005,
    'lrf': 0.01,
    'cos_lr': True,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 2,
    'warmup_bias_lr': 0.05,
    'mosaic': 1.0,
    'mixup': 0.15,
    'copy_paste': 0.1,
    'scale': 0.7,
    'close_mosaic': 10,
    'patience': 15,
    'save_period': 5,
    'freeze': None,
    'project': project_dir,
    'name': run_name,
    'exist_ok': True,
    'cache': 'disk',        # Disk cache (Kaggle has limited RAM)
    'workers': 2,            # Kaggle limit
    'resume': resume_flag,
}

print(f"\n  Device:     GPU {device}")
print(f"  Dataset:    {yaml_path}")
print(f"  Epochs:     {config['epochs']}")
print(f"  Batch:      {config['batch']}")
print(f"  Optimizer:  SGD (lr=0.005, cosine)")
print(f"  Output:     {project_dir}/{run_name}")
print("=" * 70 + "\n")

trainer = MCFTrainer(
    overrides=config,
    mcf_model=None if resume_flag else model.model
)
trainer.add_callback("on_train_start", on_train_start)
trainer.train()

print("\n✅ Training Complete!")
print(f"   Best weights: {project_dir}/{run_name}/weights/best.pt")

# ============================================================
# CELL 8: Save Results (Kaggle Output)
# ============================================================
import shutil

output_dir = "/kaggle/working/output"
os.makedirs(output_dir, exist_ok=True)

# Copy best weights to output
best_weight = os.path.join(project_dir, run_name, "weights", "best.pt")
if os.path.exists(best_weight):
    shutil.copy2(best_weight, os.path.join(output_dir, "best_far_gated.pt"))
    print(f"✅ Best weights saved to: {output_dir}/best_far_gated.pt")

# Copy results
results_csv = os.path.join(project_dir, run_name, "results.csv")
if os.path.exists(results_csv):
    shutil.copy2(results_csv, os.path.join(output_dir, "results.csv"))
    print(f"✅ Results CSV saved to: {output_dir}/results.csv")

print("\n🎯 All outputs saved to /kaggle/working/output/")
print("   Download from Kaggle Output tab.")
