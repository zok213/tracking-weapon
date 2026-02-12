#!/usr/bin/env python3
"""
Train Far View Gated Mid-Fusion Model (Deployment-Ready)
=========================================================
Strategy: Transfer Learning V2 (Near→Far Domain Adaptation)
  - Starts from near-view best.pt (Gated Fusion already learned)
  - Fine-tunes on far-view dataset (284k images, larger & more diverse)
  - Single phase: full model unfrozen (backbone already fine-tuned)

AI Engineer Notes:
  - Far-view dataset is 3x larger → fewer epochs needed
  - SGD + Cosine LR for stable convergence on large dataset
  - Strong augmentations for small-object robustness (far = small targets)
"""
import os
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import loss as loss_module

# ⭐ v2.7 RESEARCH FIX: Patch v8DetectionLoss.bbox_decode for device robustness
_original_bbox_decode = loss_module.v8DetectionLoss.bbox_decode

def _patched_bbox_decode(self, anchor_points, pred_dist):
    """Auto-move self.proj to match pred_dist device (v2.7 standardization)."""
    if self.use_dfl:
        if self.proj.device != pred_dist.device:
            self.proj = self.proj.to(pred_dist.device)
    return _original_bbox_decode(self, anchor_points, pred_dist)

loss_module.v8DetectionLoss.bbox_decode = _patched_bbox_decode
print("[OK] Applied v2.7 bbox_decode device patch")


class MCFTrainer(DetectionTrainer):
    """Custom trainer that preserves pre-loaded MCF model with Gate Supervision."""
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, mcf_model=None):
        self._mcf_model = mcf_model
        super().__init__(cfg, overrides, _callbacks)
    
    def setup_model(self):
        """Override to use pre-loaded MCF model and enable Gate Export."""
        model = self._mcf_model if self._mcf_model is not None else super().setup_model()
        
        # Enable Gate Export for Supervision
        print("[MCFTrainer] Enabling 'export_gates' on GatedSpatialFusion_V3 layers...")
        count = 0
        try:
            from ultralytics.nn.modules.block import GatedSpatialFusion_V3
            modules = model.modules() if hasattr(model, 'modules') else model.model.modules()
            for m in modules:
                if isinstance(m, GatedSpatialFusion_V3):
                    m.export_gates = True
                    count += 1
        except ImportError:
            print("[MCFTrainer] GatedSpatialFusion_V3 not found, skipping gate export.")
        print(f"[MCFTrainer] Enabled gate export on {count} layers.")
        
        if self._mcf_model is not None:
            self.model = model
            self.model.to(self.device)
            self.model.args = self.args
            return self.model
        return model

    def get_loss(self):
        """Return GatedDetectionLoss wrapping default loss."""
        loss = super().get_loss()
        try:
            from gate_supervision import GatedDetectionLoss
            print("[MCFTrainer] Wrapping Loss with GatedDetectionLoss (Auxiliary Supervision Active)")
            return GatedDetectionLoss(self.model, loss)
        except ImportError:
            print("[MCFTrainer] gate_supervision not found, using standard loss.")
            return loss


def train():
    project_root = Path("/home/student/Toan")
    runs_dir = project_root / "runs"
    
    # ========================================
    # CONFIGURATION
    # ========================================
    
    # Device: Use GPU 1 (free) or fallback
    device = 1 if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 0
    print(f"Using device: {device} (FAR-VIEW DEPLOYMENT TRAINING)")
    
    # Dataset: Far-view (cleaned, with test split)
    dataset_yaml = project_root / "datasets/vtmot_far/far_view_clean.yaml"
    print(f"Dataset: {dataset_yaml}")
    
    # Cache mode
    cache_mode = 'disk'
    
    # Project structure
    experiment_name = 'far_gated_deployment'
    project_dir = runs_dir / experiment_name
    run_name = 'far_view_gated_v1'
    
    # ========================================
    # WEIGHT LOADING STRATEGY
    # ========================================
    
    last_ckpt = project_dir / f"{run_name}/weights/last.pt"
    resume_flag = False
    
    if last_ckpt.exists():
        # Resume from checkpoint
        print(f"[RESUME] Found checkpoint: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        resume_flag = True
    else:
        # Transfer Learning V2: Near best.pt → Far
        near_best = project_root / "runs/gated_experiment/near_view_gated_phase1/weights/best.pt"
        
        if near_best.exists():
            print(f"[TRANSFER] Loading Near-View Gated best.pt: {near_best}")
            model = YOLO(str(near_best))
            print("✅ Transfer Learning V2: Near→Far domain adaptation")
            print("   Gated Fusion layers are already trained from near-view!")
        else:
            # Fallback: Use FLIR pretrained weights with gated YAML
            print("[FALLBACK] Near best.pt not found. Using FLIR pretrained + Gated YAML.")
            model_yaml = project_root / "YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11x-RGBT-gated-v3.yaml"
            model = YOLO(str(model_yaml))
            
            flir_weights = project_root / "weights/FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.01-warmup_epochs1-SGD.pt"
            if flir_weights.exists():
                print(f"Loading FLIR weights: {flir_weights}")
                try:
                    ckpt = torch.load(str(flir_weights), map_location='cpu')
                    if 'model' in ckpt:
                        chk_sd = ckpt['model'].state_dict()
                        mdl_sd = model.model.state_dict()
                        filtered = {k: v for k, v in chk_sd.items() 
                                   if k in mdl_sd and v.shape == mdl_sd[k].shape}
                        if filtered:
                            model.model.load_state_dict(filtered, strict=False)
                            print(f"✅ Transferred {len(filtered)} layers from FLIR.")
                except Exception as e:
                    print(f"❌ Weight loading failed: {e}")
    
    # ========================================
    # GRADIENT CLIPPING CALLBACK
    # ========================================
    
    def on_train_start(trainer):
        """Apply gradient clipping to optimizer."""
        if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            orig_step = trainer.optimizer.step
            def clipped_step(closure=None):
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=10.0)
                return orig_step(closure)
            trainer.optimizer.step = clipped_step
            print("⚡ Gradient Clipping (norm=10.0) applied.")

    # ========================================
    # TRAINING CONFIG
    # ========================================
    
    config = {
        'model': str(last_ckpt) if resume_flag else str(model.model_name) if hasattr(model, 'model_name') else 'yolo11x.pt',
        'data': str(dataset_yaml),
        'epochs': 30,
        'imgsz': 640,
        'batch': 16,
        'device': device,
        'use_simotm': 'RGBRGB6C',
        'channels': 6,
        'pairs_rgb_ir': ['_rgb_', '_ir_'],
        'optimizer': 'SGD',
        'lr0': 0.005,          # Higher LR since we have good init weights
        'lrf': 0.01,           # Final LR factor (cosine annealing)
        'cos_lr': True,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 2,
        'warmup_bias_lr': 0.05,
        # Strong augmentations for FAR-VIEW (small targets)
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.1,
        'scale': 0.7,          # Aggressive scale for small-object robustness
        'close_mosaic': 10,    # Disable mosaic last 10 epochs for clean convergence
        'label_smoothing': 0.1,
        # Training params
        'patience': 15,        # Early stopping
        'save_period': 5,      # Save checkpoint every 5 epochs
        'freeze': None,        # No frozen layers — full fine-tuning
        'project': str(project_dir),
        'name': run_name,
        'exist_ok': True,
        'cache': cache_mode,
        'workers': 8,
        'resume': resume_flag,
    }
    
    print("\n" + "=" * 70)
    print("FAR-VIEW GATED MID-FUSION — DEPLOYMENT TRAINING")
    print("=" * 70)
    print(f"  Device:     GPU {device}")
    print(f"  Dataset:    {dataset_yaml}")
    print(f"  Epochs:     {config['epochs']}")
    print(f"  Batch:      {config['batch']}")
    print(f"  Optimizer:  {config['optimizer']} (lr={config['lr0']}, cosine)")
    print(f"  Augments:   Mosaic + MixUp + CopyPaste + Scale(0.7)")
    print(f"  Output:     {project_dir / run_name}")
    print("=" * 70 + "\n")
    
    # ========================================
    # LAUNCH TRAINING
    # ========================================
    
    trainer = MCFTrainer(
        overrides=config,
        mcf_model=None if resume_flag else model.model
    )
    
    # Register callbacks
    trainer.add_callback("on_train_start", on_train_start)
    
    # Gate Visualization (every 5 epochs)
    viz_image = str(project_root / "datasets/vtmot_far/images/val")
    
    def on_train_epoch_end(trainer):
        epoch = trainer.epoch + 1
        if epoch % 5 == 0:
            print(f"📊 Epoch {epoch}: Saving gate visualization...")
            try:
                from visualize_gates import visualize_gates
                # Find first val image
                import glob
                val_imgs = glob.glob(os.path.join(viz_image, "*_rgb_*.jpg"))
                if val_imgs:
                    save_path = trainer.save_dir / f"gate_viz_epoch_{epoch}.png"
                    visualize_gates(trainer.model, val_imgs[0], save_path=str(save_path))
            except Exception as e:
                print(f"⚠️ Visualization failed: {e}")
    
    trainer.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    trainer.train()
    
    print("\n✅ Far-View Gated Deployment Training Complete!")
    print(f"   Best weights: {project_dir / run_name / 'weights/best.pt'}")


if __name__ == "__main__":
    train()
