import torch
import torch.nn as nn
from ultralytics import YOLO

def debug_mod():
    print("Loading YOLO...")
    model = YOLO('/home/student/Toan/runs/detect/checkpoints/teacher/yolo26x_kust4k/weights/best.pt')
    
    print("\nOriginal 1st layer:")
    print(model.model.model[0].conv.weight.shape)
    
    # Modify
    first_conv = model.model.model[0].conv
    new_conv = nn.Conv2d(4, first_conv.out_channels, 3, 2, 1).cuda()
    
    # Replace
    model.model.model[0].conv = new_conv
    model.model.to('cuda')
    
    print("\nModified 1st layer (via reference):")
    print(model.model.model[0].conv.weight.shape)
    
    # Check if persistent in forward pass
    print("\nChecking forward pass...")
    try:
        dummy = torch.randn(1, 4, 640, 640).cuda()
        res = model.model(dummy)
        print("✅ Forward pass successful!")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")

if __name__ == '__main__':
    debug_mod()
