
import sys
import os
import torch

# Add the local ultralytics to path so custom modules can be pickled/unpickled
sys.path.append('/home/student/Toan/YOLOv11-RGBT')

from ultralytics.nn.modules import GatedSpatialFusion

def inspect_weights(path):
    print(f"--- Inspecting {path} ---")
    try:
        ckpt = torch.load(path, map_location='cpu')
        model = ckpt.get('model')
        if model:
            print(f"Model Type: {type(model)}")
            # Check for GatedSpatialFusion layers
            has_gate = False
            for m in model.modules():
                if isinstance(m, GatedSpatialFusion):
                    has_gate = True
                    print(f"Found GatedSpatialFusion: {m}")
                    break
            
            if has_gate:
                print("✅ specific Gated Fusion architecture detected.")
            else:
                print("❌ No GatedSpatialFusion module found (might be standard YOLO or different fusion).")
                
            # Check input channels of the first layer
            first_layer = list(model.modules())[0] # roughly
            # a safer way for YOLO: model.model[0]
            try:
                p1 = model.model[0]
                if hasattr(p1, 'conv'):
                    print(f"First layer config: {p1.conv.in_channels} input channels")
                elif hasattr(p1, 'c1'):
                     print(f"First layer c1: {p1.c1}")
            except:
                print("Could not analyze first layer channels easily.")
                
        else:
            print("No 'model' key in checkpoint.")
            
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    weights = [
        '/home/student/Toan/weights/FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.01-warmup_epochs1-SGD.pt',
        '/home/student/Toan/models/yolo26x_rgbt_init.pt',
        '/home/student/Toan/models/yolo26x.pt'
    ]
    for w in weights:
        if os.path.exists(w):
            inspect_weights(w)
        else:
            print(f"File not found: {w}")
