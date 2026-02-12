
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss


class GatedDetectionLoss(nn.Module):
    """
    Wrapper around v8DetectionLoss to add Gate Supervision Loss.
    
    Principles:
    1. Illumination-aware: Dark -> RGB low weight, IR high weight.
    2. Modality-consistency: Bright -> RGB high weight.
    3. Safety: Clears 'active_gate_weights' immediately to prevent Deepcopy crashes.
    """
    def __init__(self, model, original_loss):
        super().__init__()

        self.model = model
        self.original_loss = original_loss
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def __getattr__(self, name):
        # Proxy attribute access to original loss (e.g., hyperparams, dfl, etc.)
        return getattr(self.original_loss, name)
        
    def __call__(self, preds, batch):
        # 1. Compute original detection loss
        loss, loss_items = self.original_loss(preds, batch)
        
        # 2. Compute Gate Supervision Loss
        gate_loss = 0.0
        
        # Get RGB image from batch (B, 6, H, W)
        imgs = batch['img'] 
        # RGB is first 3 channels
        rgb_imgs = imgs[:, :3, :, :]
        
        # Calculate Illumination (Global mean per image)
        # (B, 3, H, W) -> (B, 1, 1, 1)
        illum = rgb_imgs.mean(dim=(1, 2, 3), keepdim=True)
        
        # Target Generation
        # If illum < 0.2 (Night): Target RGB=0.1, IR=0.9
        # If illum > 0.5 (Day): Target RGB=0.6, IR=0.6 (or balanced)
        
        # We use a soft linear mapping for targets
        # Target RGB weight = Illumination (clamped 0.1 to 0.9)
        target_rgb = torch.clamp(illum, 0.05, 0.9)
        
        # Target IR weight = 1.0 - (Illum * 0.5)
        # If Illum=0.0 -> IR=1.0
        # If Illum=1.0 -> IR=0.5
        target_ir = torch.clamp(1.0 - illum * 0.5, 0.5, 1.0)
        
        # Normalize targets to sum to 1 (Approx)
        t_sum = target_rgb + target_ir + 1e-6
        target_rgb = target_rgb / t_sum
        target_ir = target_ir / t_sum
        
        # Iterate over modules to find active gates
        # We look for GatedSpatialFusion_V3 modules
        count = 0
        from ultralytics.nn.modules.block import GatedSpatialFusion_V3
        
        modules = self.model.modules() if hasattr(self.model, 'modules') else self.model.model.modules()
        
        for m in modules:
            if isinstance(m, GatedSpatialFusion_V3):
                # Check directly for attribute to avoid crashes
                if hasattr(m, 'active_gate_weights') and m.active_gate_weights is not None:
                    # shape: (B, 2, H, W)
                    gw = m.active_gate_weights
                    
                    # Average pooling to get global gate state
                    # We supervise the *mean* behavior, not every pixel (allows local variation)
                    gate_mean = gw.mean(dim=(2, 3)) # (B, 2)
                    
                    g_rgb = gate_mean[:, 0:1]
                    g_ir = gate_mean[:, 1:2]
                    
                    # MSE Loss against targets
                    # Note: target_rgb is (B, 1, 1, 1) -> view to (B, 1)
                    t_rgb = target_rgb.view(-1, 1)
                    t_ir = target_ir.view(-1, 1)
                    
                    # Loss
                    l_rgb = F.mse_loss(g_rgb, t_rgb)
                    l_ir = F.mse_loss(g_ir, t_ir)
                    
                    gate_loss += (l_rgb + l_ir)
                    
                    # CRITICAL SAFETY: Clear the weights from the model immediately!
                    # This prevents the graph from staying attached to 'self', 
                    # which causes deepcopy errors in EMA.
                    m.active_gate_weights = None
                    count += 1
        
        # Scale the gate loss (Auxiliary task)
        # 0.05 is a conservative weight (5% impact)
        if count > 0:
            total_gate_loss = gate_loss * 0.05
            loss += total_gate_loss
            
            # Update loss items for logging?
            # loss_items is (box, cls, dfl). 
            # We can't resize it easily without breaking loggers.
            # So we just add to the total 'loss' which is what backprop uses.
            # The 'box/cls/dfl' logs will remain same, but 'train/box_loss' etc won't sum to total_loss.
            # That's acceptable for auxiliary loss.
        
        return loss, loss_items
