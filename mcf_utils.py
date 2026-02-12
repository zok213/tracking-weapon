import torch
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

def apply_patch():
    loss_module.v8DetectionLoss.bbox_decode = _patched_bbox_decode
    print("[MCF UTILS] Applied v2.7 bbox_decode device patch")

class MCFTrainer(DetectionTrainer):
    """Custom trainer that preserves pre-loaded MCF model."""
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, mcf_model=None):
        self._mcf_model = mcf_model
        super().__init__(cfg, overrides, _callbacks)
    
    def setup_model(self):
        """Override to use pre-loaded MCF model instead of rebuilding from YAML."""
        if self._mcf_model is not None:
            self.model = self._mcf_model
            self.model.args = self.args
            print(f"[MCF UTILS] Using pre-loaded MCF model (Namespace: {type(self.args)})")
            return self.model
        else:
            return super().setup_model()
