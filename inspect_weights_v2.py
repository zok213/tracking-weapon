
import sys
import os
import torch

# Add the local ultralytics to path 
sys.path.append('/home/student/Toan/YOLOv11-RGBT')

# Import the specific module class so pickle can find it
from ultralytics.nn.modules.block import GatedSpatialFusion
# Also need to make sure 'ultralytics.nn.modules' generally works if pickle references that
import ultralytics.nn.modules

def inspect_weights(path):
    print(f"--- Inspecting {path} ---")
    try:
        # Load with weights_only=False because we need to unpickle custom classes
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        print("Checkpoint loaded.")
        
        # Check keys
        print(f"Keys: {ckpt.keys()}")
        
        model = ckpt.get('model')
        if model:
            print(f"Model Type: {type(model)}")
            
            # DFS or recursive search for GatedSpatialFusion
            has_gate = False
            for name, m in model.named_modules():
                if isinstance(m, GatedSpatialFusion):
                    has_gate = True
                    print(f"✅ Found GatedSpatialFusion at: {name}")
                    # Inspect dropout prob if possible
                    if hasattr(m, 'dropout_prob'):
                        print(f"   dropout_prob: {m.dropout_prob}")
                    break
            
            if not has_gate:
                print("❌ No GatedSpatialFusion module found.")
            
            # Check input channels (first conv)
            try:
                # model.model is usually the Sequential container in YOLO
                first_layer = model.model[0]
                if hasattr(first_layer, 'conv'):
                    print(f"First Layer Channels: {first_layer.conv.in_channels}")
                else:
                    print(f"First layer type: {type(first_layer)}")
            except Exception as e:
                print(f"Could not check first layer: {e}")
                
        else:
            print("No 'model' key in checkpoint.")
            
    except Exception as e:
        print(f"FAILED to load/inspect: {e}")

if __name__ == "__main__":
    weights = [
        # The user's specific weights
        '/home/student/Toan/weights/FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.01-warmup_epochs1-SGD.pt',
        '/home/student/Toan/weights/LLVIP-yolo11x-RGBT-midfusion-MCF.pt', 
        '/home/student/Toan/weights/M3FD-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.001-warmup_epochs1-Adam.pt',
        '/home/student/Toan/models/yolo26x_rgbt_init.pt',
        '/home/student/Toan/models/yolo26x.pt'
    ]
    
    for w in weights:
        if os.path.exists(w):
            inspect_weights(w)
        else:
            print(f"File not found: {w}")
