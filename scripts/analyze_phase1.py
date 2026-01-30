import torch
import torch.nn as nn
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_weights():
    # Paths
    init_path = "/home/student/Toan/models/yolo26x_rgbt_init.pt"
    trained_path = "/home/student/Toan/checkpoints/rgbt/epoch100_v6.pt" # Last
    trained_path_90 = "/home/student/Toan/checkpoints/rgbt/epoch90_v6.pt" # Before spike
    
    print("="*60)
    print("PHASE 1 FORENSIC ANALYSIS")
    print("="*60)
    
    # 1. Load Initial Model
    print(f"Loading Init: {init_path}")
    model_init = YOLO(init_path)
    w_init = model_init.model.model[0].conv.weight.data.clone() # [64, 4, 3, 3]
    
    # 2. Load Trained Model (Epoch 90 - Clean)
    print(f"Loading Trained (Epoch 90): {trained_path_90}")
    # Need to load structure first
    model_trained = YOLO(init_path) 
    state_dict = torch.load(trained_path_90) # Load state dict directly
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    model_trained.model.load_state_dict(state_dict)
    w_trained = model_trained.model.model[0].conv.weight.data
    
    # 3. Calculate Change
    delta = torch.abs(w_trained - w_init)
    
    print("\n--- Weight Change Analysis ---")
    rgb_change = delta[:, :3, :, :].mean().item()
    thermal_change = delta[:, 3, :, :].mean().item()
    
    print(f"Mean Absolute Change (RGB):     {rgb_change:.6f}")
    print(f"Mean Absolute Change (Thermal): {thermal_change:.6f}")
    
    ratio = thermal_change / (rgb_change + 1e-9)
    print(f"Thermal/RGB Learning Ratio:     {ratio:.4f}")
    
    if ratio < 0.1:
        print("\n❌ CRITICAL: MODALITY COLLAPSE DETECTED")
        print("The model largely ignored the thermal channel.")
    elif ratio > 0.5:
        print("\n✅ HEALTHY: Significant thermal learning detected.")
    else:
        print("\n⚠️ WARNING: Low thermal learning.")
        
    print("\n--- First Filter Gradients (Proxy) ---")
    # We don't have grads saved, but we can look at weight magnitude
    print(f"Init Thermal Norm: {w_init[:, 3].norm().item():.4f}")
    print(f"Trained Thermal Norm: {w_trained[:, 3].norm().item():.4f}")

    # 4. Analyze Last Epoch Spike (Epoch 100)
    print(f"\nLoading Spiked Model (Epoch 100): {trained_path}")
    try:
        model_spike = YOLO(init_path)
        model_spike.model.load_state_dict(torch.load(trained_path))
        w_spike = model_spike.model.model[0].conv.weight.data
        
        spike_delta = torch.abs(w_spike - w_trained).mean().item()
        print(f"Change Epoch 90->100: {spike_delta:.6f}")
        
        if spike_delta > 1.0:
             print("❌ MASSIVE weight shift in last 10 epochs (Explosion)")
        else:
             print("ℹ️ Weights stable, loss spike might be data/batch specific")
             
    except Exception as e:
        print(f"Could not load epoch 100: {e}")

if __name__ == '__main__':
    analyze_weights()
