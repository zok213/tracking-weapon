#!/usr/bin/env python3
"""
YOLO 4-Channel Training Script
Modifies YOLO to accept 4-channel input (RGB + Thermal)

Night Vision: Thermal IR works in complete darkness
- Day: RGB dominant
- Night: Thermal dominant
- Combined: 24/7 operation
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path

def modify_yolo_for_4ch(model_path='yolo26x.pt', output_path='yolo26x_4ch.pt'):
    """
    Modify YOLO first conv layer: 3 channels → 4 channels
    Preserves RGB weights, initializes thermal channel
    """
    print(f"Loading model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Find first conv layer
    # Ultralytics YOLO structure: model.model.model[0].conv
    first_conv = None
    for name, module in model.model.model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            first_conv = module
            first_conv_name = name
            break
    
    if first_conv is None:
        print("❌ Could not find first conv layer with 3 channels")
        return None
    
    print(f"Found first conv: {first_conv_name}")
    print(f"  Shape: {first_conv.weight.shape}")  # [out_ch, 3, kH, kW]
    
    # Create new conv with 4 input channels
    out_channels = first_conv.out_channels
    kernel_size = first_conv.kernel_size
    stride = first_conv.stride
    padding = first_conv.padding
    
    new_conv = nn.Conv2d(
        in_channels=4,  # RGB + Thermal
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=first_conv.bias is not None
    )
    
    # Initialize weights
    with torch.no_grad():
        # Copy RGB weights
        new_conv.weight[:, :3, :, :] = first_conv.weight
        
        # Initialize thermal channel
        # Option 1: Average of RGB channels (luminance-like)
        thermal_init = first_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:4, :, :] = thermal_init
        
        # Copy bias if exists
        if first_conv.bias is not None:
            new_conv.bias = first_conv.bias
    
    print(f"  New shape: {new_conv.weight.shape}")  # [out_ch, 4, kH, kW]
    
    # Replace in model (this is tricky with Ultralytics)
    # For now, save the modified weights
    torch.save({
        'original_conv_weight': first_conv.weight,
        'new_conv_weight': new_conv.weight,
        'new_conv_bias': new_conv.bias if new_conv.bias is not None else None,
        'conv_name': first_conv_name,
        'note': '4-channel input for RGBT fusion'
    }, output_path)
    
    print(f"✅ Saved 4-channel weights to: {output_path}")
    print(f"   Use these weights to modify the model during training init")
    
    return new_conv


def train_rgbt(data_yaml, model_path='yolo26x.pt', epochs=80, batch=16, device=0):
    """
    Train YOLO on 4-channel RGBT data
    """
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    
    # Note: Ultralytics doesn't natively support 4-channel
    # Would need custom dataloader and model modification
    # This is a template for the approach
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        device=device,
        imgsz=640,
        project='checkpoints/teacher',
        name='yolo26x_rgbt',
        save_period=2,
        patience=20,
    )
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/student/Toan/yolo26x.pt')
    parser.add_argument('--output', default='/home/student/Toan/models/yolo26x_4ch_weights.pt')
    args = parser.parse_args()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    modify_yolo_for_4ch(args.model, args.output)
