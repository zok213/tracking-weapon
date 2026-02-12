
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os
import argparse
import glob
import cv2
import numpy as np

# Add project root
sys.path.append(os.getcwd())

from ultralytics.nn.modules.block import GatedSpatialFusion_V3
from ultralytics import YOLO


def visualize_gates(model_or_path, image_path, save_path='gate_viz.png'):
    # Handle Model Loading
    if isinstance(model_or_path, str):
        print(f"Loading model from {model_or_path}...")
        try:
            model = YOLO(model_or_path).model
        except:
            print("Could not load as YOLO object, trying direct load...")
            return
    else:
        # Assume it's a model object (YOLO wrapper or nn.Module)
        if hasattr(model_or_path, 'model'):
             model = model_or_path.model 
        else:
             model = model_or_path


    # Enable gate export
    print("Enabling gate export...")
    for m in model.modules():
        if isinstance(m, GatedSpatialFusion_V3):
            m.export_gates = True
            
    model.eval()
    
    # Load Image
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found.")
        # Create dummy
        print("Using dummy dark image.")
        img = torch.zeros(1, 3, 640, 640)
    else:
        print(f"Loading {image_path}...")
        img_cv = cv2.imread(image_path)
        img_cv = cv2.resize(img_cv, (640, 640))
        img = torch.from_numpy(img_cv).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
    device = next(model.parameters()).device
    img = img.to(device)
    
    # Run Forward
    print("Running forward pass...")
    with torch.no_grad():
        _ = model(img)
        
    # Extract Gates
    print("Extracting gates...")
    gate_data = []
    
    levels = []
    rgb_weights = []
    ir_weights = []
    
    layer_idx = 0
    for m in model.modules():
        if isinstance(m, GatedSpatialFusion_V3):
            if m.active_gate_weights is not None:
                # Shape: (B, 2, H, W)
                gw = m.active_gate_weights
                avg_rgb = gw[:, 0].mean().item()
                avg_ir = gw[:, 1].mean().item()
                
                levels.append(f"Layer {layer_idx}")
                rgb_weights.append(avg_rgb)
                ir_weights.append(avg_ir)
                layer_idx += 1
                
                # Clear to be safe
                m.active_gate_weights = None
                
    # Plot
    if not levels:
        print("No gates found! (Maybe model doesn't have GatedSpatialFusion_V3?)")
        return

    illum_val = img.mean().item()
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(levels))
    width = 0.35
    
    plt.bar(x - width/2, rgb_weights, width, label='RGB Gate', color='blue', alpha=0.7)
    plt.bar(x + width/2, ir_weights, width, label='IR Gate', color='red', alpha=0.7)
    
    plt.ylabel('Gate Weight')
    plt.title(f'Gate Weights per Layer (Illumination: {illum_val:.3f})')
    plt.xticks(x, levels)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(True, axis='y')
    
    print(f"Saving visualization to {save_path}...")
    plt.savefig(save_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='runs/gated_experiment/weights/last.pt', help='Path to model weights')
    parser.add_argument('--image', type=str, default='dummy', help='Path to test image')
    args = parser.parse_args()
    
    visualize_gates(args.model, args.image)
