import torch
from pathlib import Path

MODEL_PATH = "/home/student/Toan/tracking/stage1/models/yolo26x_rgbt_framework_init.pt"

def inspect_model():
    print(f"Loading {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, weights_only=False)
    
    if isinstance(ckpt, dict):
        model = ckpt.get('model')
    else:
        model = ckpt
        
    print(f"Model Type: {type(model)}")
    
    # Check first layer
    first_layer = model.model[0]
    print(f"Layer 0 Conv In-Channels: {first_layer.conv.in_channels}")
    
    # Check YAML/Args
    if hasattr(model, 'yaml'):
        print(f"Model.yaml: {model.yaml}")
    else:
        print("Model.yaml: None")
        
    if hasattr(model, 'args'):
        print(f"Model.args: {model.args}")
        
    # Check for 'ch' key in yaml
    if hasattr(model, 'yaml') and isinstance(model.yaml, dict):
        print(f"YAML ch: {model.yaml.get('ch')}")

if __name__ == "__main__":
    inspect_model()
