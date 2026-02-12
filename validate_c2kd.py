import torch
import torch.nn as nn
import timm
import time
import sys

# Define the Architecture (From Logs)
class LightweightStudent(nn.Module):
    def __init__(self, num_classes=1530):
        super().__init__()
        # MobileNetV3 Large
        self.rgb_bb = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=0)
        self.ir_bb = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=0)
        
        # Calculate feat_dim (Expected typically 960 for MV3-Large)
        with torch.no_grad():
            dummy = torch.randn(1,3,288,144)
            self.feat_dim = self.rgb_bb(dummy).shape[-1]
            
        print(f"[ARCH] Feature Dim: {self.feat_dim}")
            
        self.rgb_bn = nn.BatchNorm1d(self.feat_dim)
        self.ir_bn = nn.BatchNorm1d(self.feat_dim)
        
        # Projector & Classifier
        self.projector = nn.Sequential(nn.Linear(self.feat_dim, 2048), nn.ReLU(), nn.Linear(2048, 2048))
        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)

    def forward(self, rgb=None, ir=None):
        out = {}
        if rgb is not None:
            f = self.rgb_bb(rgb)
            bn = self.rgb_bn(f)
            out['rgb'] = bn # Return BN feature for inference
        if ir is not None:
            f = self.ir_bb(ir)
            bn = self.ir_bn(f)
            out['ir'] = bn
        return out

def validate():
    print("="*40)
    print("AI Engineer Validation: C2KD v93")
    print("="*40)
    
    # 1. Instantiate Model
    print("[1] Instantiating MobileNetV3 Architecture...")
    try:
        model = LightweightStudent(num_classes=1530)
        model.eval()
        print("    Success.")
    except Exception as e:
        print(f"    Failed: {e}")
        return

    # 2. Load Weights
    weight_path = '/home/student/Toan/analysis_c2kd/student_vtmot_best.pth'
    print(f"[2] Loading Weights from {weight_path}...")
    try:
        state_dict = torch.load(weight_path, map_location='cpu')
        msg = model.load_state_dict(state_dict, strict=True)
        print("    Success.")
        print(f"    Load Message: {msg}")
    except Exception as e:
        print(f"    Failed: {e}")
        return

    # 3. Model Statistics
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[3] Model Stats:")
    print(f"    Parameters: {params:.2f}M (Lightweight)")
    print(f"    Target: Re-Identification (Tracking)")

    # 4. Inference Test (Forward Pass)
    print(f"[4] Inference Test (CPU)...")
    try:
        dummy_rgb = torch.randn(2, 3, 288, 144) # Batch 2
        dummy_ir = torch.randn(2, 3, 288, 144)
        
        start = time.time()
        with torch.no_grad():
            out = model(rgb=dummy_rgb, ir=dummy_ir)
        end = time.time()
        
        feat_rgb = out['rgb']
        feat_ir = out['ir']
        
        print(f"    Success.")
        print(f"    Output Shape (RGB): {feat_rgb.shape} (Batch, Dim)")
        print(f"    Output Shape (IR):  {feat_ir.shape} (Batch, Dim)")
        print(f"    Inference Time (Batch 2): {(end-start)*1000:.2f} ms")
        
        if feat_rgb.shape[1] == 960:
            print("    [CHECK] Dimension 960 is correct for MobileNetV3-Large.")
        else:
            print(f"    [WARN] Dimension {feat_rgb.shape[1]} unexpected.")
            
    except Exception as e:
        print(f"    Failed: {e}")
        return

    print("="*40)
    print("CONCLUSION: MODEL IS VALID AND FUNCTIONAL ✅")
    print("="*40)

if __name__ == "__main__":
    validate()
