#!/usr/bin/env python3
"""
Train Near View Gated Mid-Fusion Model (EXPERIMENT GPU 0)
"""
import os
import torch
import psutil
from pathlib import Path
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
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
    """Custom trainer that preserves pre-loaded MCF model."""
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, mcf_model=None):
        self._mcf_model = mcf_model
        super().__init__(cfg, overrides, _callbacks)
    

    def setup_model(self):
        """Override to use pre-loaded MCF model and enable Gate Supervision."""
        model = self._mcf_model if self._mcf_model is not None else super().setup_model()
        
        # 🟢 GATED FUSION SETUP: Enable Gate Export for Supervision
        print("[MCFTrainer] Enabling 'export_gates' on GatedSpatialFusion_V3 layers...")
        count = 0
        from ultralytics.nn.modules.block import GatedSpatialFusion_V3
        modules = model.modules() if hasattr(model, 'modules') else model.model.modules()
        for m in modules:
            if isinstance(m, GatedSpatialFusion_V3):
                m.export_gates = True
                count += 1
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
        
        # Wrap with Gate Supervision
        from gate_supervision import GatedDetectionLoss
        print("[MCFTrainer] Wrapping Loss with GatedDetectionLoss (Auxiliary Supervision Active)")
        return GatedDetectionLoss(self.model, loss)

def train():
    project_root = Path("/home/student/Toan")
    runs_dir = project_root / "runs"
    
    # 🟢 DEVICE: GPU 0 (Free)
    device = 0
    print(f"Using device: {device} (GATED FUSION EXPERIMENT)")
    
    # Dataset
    dataset_yaml = project_root / "datasets/vtmot_near/near_view_clean.yaml"
    print(f"Dataset: {dataset_yaml}")
    
    # Cache
    cache_mode = 'disk'
    
    # Project Paths
    experiment_name = 'gated_experiment'
    project_dir = runs_dir / experiment_name
    
    # 🟢 MODEL INIT: Use custom GATED YAML (X-Large to match FLIR weights)
    model_yaml = project_root / "YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11x-RGBT-gated-v3.yaml"
    
    # Checkpoint logic
    phase1_name = 'near_view_gated_phase1'
    last_ckpt = project_dir / f"{phase1_name}/weights/last.pt"
    
    resume_flag = False
    
    if last_ckpt.exists():
        print(f"Resuming from checkpoint: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        resume_flag = True
    else:
        print(f"Initializing new Gated Model from: {model_yaml}")
        model = YOLO(str(model_yaml))
        
        # Load backbone weights from FLIR RGBT model (Best Transfer Learning Source)
        # This has learned RGBT features but uses standard concatenation
        # We load with strict=False to transfer backbone/head while initializing Gate randomly
        flir_weights = "/home/student/Toan/weights/FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.01-warmup_epochs1-SGD.pt"
        if os.path.exists(flir_weights):
            print(f"Transferring RGBT weights from: {flir_weights}")
            try:
                ckpt = torch.load(flir_weights, map_location='cpu')
                if 'model' in ckpt:
                    # Robust loading: Filter out size mismatches
                    chk_state_dict = ckpt['model'].state_dict()
                    model_state_dict = model.model.state_dict()
                    
                    filtered_dict = {}
                    skipped_layers = []
                    
                    for k, v in chk_state_dict.items():
                        if k in model_state_dict:
                            if v.shape == model_state_dict[k].shape:
                                filtered_dict[k] = v
                            else:
                                skipped_layers.append(f"{k} (chk: {v.shape} != mdl: {model_state_dict[k].shape})")
                    
                    if filtered_dict:
                        model.model.load_state_dict(filtered_dict, strict=False)
                        print(f"✅ Successfully transferred {len(filtered_dict)} layers from FLIR RGBT model!")
                        if skipped_layers:
                            print(f"⚠️ Skipped {len(skipped_layers)} layers due to shape mismatch (Expected for Gated/Head changes):")
                            for s in skipped_layers[:5]: # print first 5
                                print(f"   - {s}")
                            if len(skipped_layers) > 5: print("   ... and more.")
                    else:
                        print("❌ No matching layers found! Architecture might be too different.")
                        
                else:
                    print("⚠️ 'model' key not found in checkpoint.")
            except Exception as e:
                print(f"❌ Warning: Could not load pretrained weights: {e}")
        else:
            print(f"⚠️ Pretrained weight file not found: {flir_weights}")
            print("Falling back to random initialization (Not recommended).")
            
    
    # Phase 1: Frozen Backbone
    # Freeze standard YOLO backbone layers
    freeze_layers = [2, 3, 4, 5, 6, 17, 18, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
    
    # Bug #4: Custom Logic for Gate Supervision
    # We need to monkey-patch the Loss function or the Trainer to include gate loss
    # Since patching Trainer.train_step is hard, we will use a Callback approach or simply 
    # acknowledge that extending standard YOLO for this specific loss requires modifying the internal loop.
    # However, Ultralytics allows 'add_callback'.
    # A cleaner way for this script is to patch the v8DetectionLoss.__call__ 
    
    # Capture original call
    _origin_loss_call = loss_module.v8DetectionLoss.__call__
    
    def _patched_loss_call(self, preds, batch):
        """Adds Gate Supervision Loss to standard detection loss."""
        # Standard Loss
        loss, loss_items = _origin_loss_call(self, preds, batch)
        
        # Bug #4: Gate Supervision
        # Access model from self.model (it's bound to the loss object usually? No, passed in init)
        # We need to find the GatedSpatialFusion layers.
        # They stored 'last_gate_weights' during forward pass.
        
        gate_loss = torch.tensor(0.0, device=preds[0].device)
        found_gates = 0
        
        # Iterate modules to find V3 gates
        # self.model is the Deployed model (DistModel or similar). 
        # Need to be careful about unwrapping.
        # But wait, 'self' here is v8DetectionLoss. It doesn't usually hold the model reference continuously?
        # Actually it handles 'preds'.
        
        # Alternative: The model instance is GLOBAL in this script context if we are careful, 
        # but inside the trainer it's a copy.
        # Let's try to access via the batch? No.
        
        # Let's rely on the fact that 'preds' comes from the model.
        # If we can't easily access the model instance, we can't get the gate weights.
        # BUT: The 'model' variable in this script IS passed to the trainer.
        # Let's try to attach the model to the loss function during setup?
        
        return loss, loss_items

    # Okay, patching __call__ is tricky because we need the model instance.
    # Let's try a simpler approach: 
    # On 'on_train_batch_end', we can't affect loss.
    # On 'on_before_zero_grad', we can calculate extra loss and backward it manually?
    # YES. 'on_before_zero_grad' runs *before* optimizer step. 
    # But backward() has already been called on the main loss.
    # So we can calculate gate loss, call gate_loss.backward(), then optimizer step.
    # Pytorch accumulates gradients! This is the way.
    
    def on_train_batch_end(trainer):
        """
        Bug #4: Gate Supervision Loop
        Opt #3: Gradient Clipping
        """
        # 1. Gate Supervision
        # Get the model
        model = trainer.model
        
        # Get RGB images from batch
        # trainer.batch is available?
        # In 'on_train_batch_end', trainer.loss_items is available.
        # We need the input images 'batch["img"]'.
        
        # Warning: Ultralytics dataloader might verify logic.
        pass

    # Actually, let's redefine MCFTrainer to handle this properly via 'optimizer_step' override?
    # Or just use the simple patch approach where we inject a hook.
    
    # Realistically, implementing Bug #4 (Gate Loss) without breaking Ultralytics updates is hard.
    # BUT Opt #3 (Clip Grad) is easy via 'add_callback'.
    
    
    def on_train_start(trainer):
        # Opt #3: Gradient Clipping (Monkey-patching Optimizer)
        # Ultralytics doesn't expose 'on_before_optimizer_step', so we wrap the step function.
        if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            orig_step = trainer.optimizer.step
            def clipped_step(closure=None):
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=10.0)
                return orig_step(closure)
            trainer.optimizer.step = clipped_step
            print("⚡ [Opt #3] Gradient Clipping (norm=10.0) applied to Optimizer step.")
    
    phase1_config = {
        'model': str(last_ckpt) if resume_flag else str(model_yaml),
        'data': str(dataset_yaml),
        'epochs': 20, 
        'imgsz': 640,
        'batch': 16,  
        'device': device,
        'use_simotm': 'RGBRGB6C',
        'channels': 6,
        'pairs_rgb_ir': ['_rgb_', '_ir_'],
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'warmup_epochs': 1,
        'freeze': freeze_layers,
        'project': str(project_dir),
        'name': phase1_name,
        'exist_ok': True,
        'cache': cache_mode,
        'workers': 8,
        'resume': resume_flag,
    }
    
    print("\n--- PHASE 1: Warm-up (Gated Fusion Training) ---")
    trainer1 = MCFTrainer(overrides=phase1_config, mcf_model=None if resume_flag else model.model)
    

    # Opt #3: Register Gradient Clipping Hook
    trainer1.add_callback("on_train_start", on_train_start)
    
    # Opt #4: Gate Visualization Callback
    viz_image = "/home/student/Toan/datasets/vtmot_near/images/val/qiuxing-0306-07_000001_rgb_.jpg"
    
    def on_train_epoch_end(trainer):
        # Visualize every 5 epochs
        epoch = trainer.epoch + 1
        if epoch % 5 == 0:
            print(f"📊 Visualizing Gates for Epoch {epoch}...")
            save_path = trainer.save_dir / f"gate_viz_epoch_{epoch}.png"
            from visualize_gates import visualize_gates
            try:
                visualize_gates(trainer.model, viz_image, save_path=str(save_path))
            except Exception as e:
                 print(f"⚠️ Visualization failed: {e}")

    trainer1.add_callback("on_train_epoch_end", on_train_epoch_end)

    trainer1.train()
    
    # Phase 2: Full Fine-Tuning
    phase2_name = 'near_view_gated_phase2' 
    phase2_ckpt = runs_dir / experiment_name / phase2_name / "weights/last.pt"
    
    resume_phase2 = False
    if phase2_ckpt.exists():
        print(f"\n[RESUME] Found Phase 2 checkpoint: {phase2_ckpt}")
        model2 = YOLO(str(phase2_ckpt))
        resume_phase2 = True
    else:
        best_phase1 = project_dir / f"{phase1_name}/weights/best.pt"
        if not best_phase1.exists():
            best_phase1 = project_dir / f"{phase1_name}/weights/last.pt"
        
        print(f"\n[START] Loading phase 1 best: {best_phase1}")
        model2 = YOLO(str(best_phase1))
        resume_phase2 = False
    
    phase2_config = {
        'model': str(phase2_ckpt) if resume_phase2 else str(best_phase1),
        'data': str(dataset_yaml),
        'epochs': 50,
        'imgsz': 640,
        'batch': 12, 
        'device': device,
        'use_simotm': 'RGBRGB6C',
        'channels': 6,
        'pairs_rgb_ir': ['_rgb_', '_ir_'],
        'optimizer': 'SGD',
        'lr0': 0.0005,
        'lrf': 0.01,
        'cos_lr': True,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.1,
        'scale': 0.7, 
        'close_mosaic': 10,
        'freeze': None,
        'project': str(project_dir),
        'name': phase2_name,
        'exist_ok': True,
        'cache': cache_mode,
        'workers': 8,
        'resume': resume_phase2,
    }
    
    print("\n--- PHASE 2: Full Fine-tuning (Gated) ---")
    trainer2 = MCFTrainer(overrides=phase2_config, mcf_model=None if resume_phase2 else model2.model)
    

    # Opt #3: Register Gradient Clipping Hook
    trainer2.add_callback("on_train_start", on_train_start)
    
    # Opt #4: Gate Visualization Callback
    trainer2.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    trainer2.train()
    
    print("\n✓ Gated Fusion Experiment Complete!")

if __name__ == "__main__":
    train()
