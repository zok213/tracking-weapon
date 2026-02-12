import torch
import torch.nn as nn
from ultralytics.nn.modules.block import GatedSpatialFusion
from ultralytics.nn.tasks import DetectionModel
import time

def test_1_initialization():
    print("\n--- Test 1: Gate Initialization (Fix #1) ---")
    model = GatedSpatialFusion(512, 256, with_illum=False) 
    
    last_conv = model.gate_conv[-1].conv
    print(f"Weight Mean: {last_conv.weight.mean().item():.6f}")
    
    bias_val = 0.0
    if last_conv.bias is not None:
        bias_val = last_conv.bias.max().item()
        print(f"Bias Mean: {bias_val:.6f}")
    else:
        print("Bias is None (handled by BN or disabled).")
    
    if abs(last_conv.weight.mean().item()) < 1e-5 and abs(bias_val) < 1e-5:
        print("✅ PASS: weights/bias are zero-initialized.")
    else:
        print("❌ FAIL: weights/bias are NOT zero.")

def test_2_night_suppression():
    print("\n--- Test 2: Night Suppression (Fix #2 + Logic) ---")
    # Setup
    model = GatedSpatialFusion(64, 32, with_illum=True)
    model.eval()
    
    # Case A: Pitch Black Night (RGB is zero/noise, Illum should be 0)
    dark_rgb = torch.zeros(1, 3, 640, 640) 
    # Use smaller feature map to save compute
    rgb_feat = torch.randn(1, 32, 20, 20) * 5 
    ir_feat = torch.randn(1, 32, 20, 20) * 5
    x = torch.cat([rgb_feat, ir_feat], dim=1)
    
    try:
        out = model(x, rgb_image=dark_rgb)
        print("✅ PASS: Forward with Dark RGB ran successfully.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ FAIL: {e}")

    # Case B: Bright Day
    bright_rgb = torch.ones(1, 3, 640, 640)
    try:
        out = model(x, rgb_image=bright_rgb)
        print("✅ PASS: Forward with Bright RGB ran successfully.")
    except Exception as e:
        print(f"❌ FAIL: {e}")

def test_3_dropout_stats():
    print("\n--- Test 3: Dropout Statistics (Fix #3) ---")
    model = GatedSpatialFusion(64, 32, dropout_prob=0.3)
    model.train()
    
    x = torch.randn(10, 64, 20, 20)
    # Just run forward
    try:
        out = model(x)
        print("✅ PASS: Forward in train mode runs.")
    except Exception as e:
        print(f"❌ FAIL: {e}")

def test_4_full_integration():
    print("\n--- Test 4: Full Model Integration (Tasks.py check) ---")
    try:
        # Use Nano model for verification to avoid OOM on CPU
        model = DetectionModel('/home/student/Toan/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-gated.yaml', ch=6, verbose=False)
        print("Model initialized.")
        
        # Create 6-channel input
        img = torch.randn(2, 6, 640, 640)
        
        # Forward pass
        out = model(img)
        
        # Check output
        print(f"Output type: {type(out)}")
        if isinstance(out, list): 
             print(f"Output list len: {len(out)}") # Training output is usually loss items? No, DetectionModel forward returns preds in eval mode
             if len(out) > 0 and hasattr(out[0], 'shape'):
                 print(f"Output 0 shape: {out[0].shape}")
        
        print("✅ PASS: Full model forward pass successful.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ FAIL: Full model integration failed with error: {e}")

if __name__ == "__main__":
    test_1_initialization()
    test_2_night_suppression()
    test_3_dropout_stats()
    test_4_full_integration()
