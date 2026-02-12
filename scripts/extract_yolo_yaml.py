import torch
import yaml
from pathlib import Path

CKPT_PATH = "/home/student/Toan/tracking/stage1/models/yolo26x_rgbt_framework_init.pt"
OUTPUT_YAML = "/home/student/Toan/tracking/stage1/configs/yolo26x_rgbt.yaml"

def extract_yaml():
    print(f"Loading {CKPT_PATH}...")
    ckpt = torch.load(CKPT_PATH, weights_only=False)
    
    if isinstance(ckpt, dict):
        model = ckpt.get('model')
    else:
        model = ckpt
        
    if hasattr(model, 'yaml') and isinstance(model.yaml, dict):
        config = model.yaml.copy()
        
        # Force Channel Count
        config['ch'] = 4
        config['channels'] = 4
        
        # Verify Backbone input
        # config['backbone'][0] -> [-1, 1, Conv, [64, 3, 2]]
        # We should NOT verify explicit args if checks rely on 'ch'.
        # But if we can, let's update it to be safe.
        # args usually: [out_ch, k, s] (in_ch detected from previous)
        # So [64, 3, 2] = 64 out, 3x3 kernel, 2 stride.
        # It seems implicit.
        
        print(f"Extracted Config: {config.keys()}")
        print(f"Setting ch=4")
        
        # Save
        with open(OUTPUT_YAML, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
            
        print(f"✅ Saved architecture to {OUTPUT_YAML}")
        
    else:
        print("❌ Model has no yaml attribute!")

if __name__ == "__main__":
    extract_yaml()
