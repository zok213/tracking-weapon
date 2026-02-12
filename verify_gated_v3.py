
import torch
import torch.nn as nn
from ultralytics.nn.modules.block import GatedSpatialFusion_V3
from ultralytics.nn.tasks import DetectionModel
import sys
import os

# Redirect standard output to unbuffered
sys.stdout.reconfigure(line_buffering=True)


import torch
import torch.nn as nn
from ultralytics.nn.modules.block import GatedSpatialFusion_V3
from ultralytics.nn.tasks import DetectionModel
import sys
import os

# Redirect standard output to unbuffered
sys.stdout.reconfigure(line_buffering=True)

def test_1_mc_dropout_convergence():
    print("\n--- Test 1: MC-Dropout Sample Count Verification (Fix #1) ---")
    model = GatedSpatialFusion_V3(512, 256)
    model.eval()  # Important: Test in eval mode

    # Use ONES to ensure dropout variance is significant
    # (Dropout variance scales with square of magnitude)
    dummy_feat = torch.ones(4, 256, 40, 40)

    # Test different sample counts
    for n in [2, 5, 20]:
        uncertainties = []
        for _ in range(5):  # Repeat 5 times
            unc = model.estimate_uncertainty(dummy_feat, model.rgb_dropout, n_samples=n)
            uncertainties.append(unc.mean().item())
        
        std = torch.tensor(uncertainties).std().item()
        mean_unc = torch.tensor(uncertainties).mean().item()
        print(f"n={n:2d}: mean={mean_unc:.4f}, std={std:.4f}")
        
        # Check: std should decrease as n increases
        if n >= 20:
            if std < 0.05:
                # Also verify uncertainty is non-negligible (>0.01) with 'ones' input
                if mean_unc > 0.01:
                    print(f"✅ PASS: n={n} stable and confident (std={std:.4f}, mean={mean_unc:.4f})")
                else:
                    print(f"❌ FAIL: Uncertainty is suspiciously low ({mean_unc:.4f})")
                    return False
            else:
                print(f"❌ FAIL: n={n} still has high variance (std={std:.4f})")
                return False
    return True

def test_2_eval_mode_uq():
    print("\n--- Test 2: Eval Mode Uncertainty Check (Fix #2) ---")
    model = GatedSpatialFusion_V3(512, 256)
    # Use ONES
    dummy_feat = torch.ones(4, 256, 40, 40)

    # Test in EVAL mode
    model.eval()
    unc_eval = model.estimate_uncertainty(dummy_feat, model.rgb_dropout, n_samples=20)

    print(f"Eval mode uncertainty mean: {unc_eval.mean():.4f}")
    if unc_eval.mean() > 0.01:
        print("✅ PASS: Uncertainty estimation works in eval mode")
    else:
        print("❌ FAIL: Uncertainty is zero/low in eval mode!")
        return False
    return True

def test_3_night_suppression():
    print("\n--- Test 3: Night Suppression (Fix #2 + Logic) ---")
    model = GatedSpatialFusion_V3(256, 256)
    model.eval()

    # Simulate pitch black night
    dark_rgb_img = torch.zeros(4, 3, 640, 640)
    noise_rgb_feat = torch.randn(4, 128, 40, 40) * 0.01  # Garbage
    valid_ir_feat = torch.randn(4, 128, 40, 40) * 1.0    # Good signal

    fused = model([noise_rgb_feat, valid_ir_feat], dark_rgb_img)
    
    # We can't access gate weights directly unless we modify forward to return them.
    # But we added 'last_gate_weights' storage in the module!
    gate = model.last_gate_weights
    
    if gate is None:
        print("❌ FAIL: last_gate_weights not stored.")
        return False

    rgb_weight = gate[:, 0].mean().item()
    ir_weight = gate[:, 1].mean().item()

    print(f"Dark scene - RGB weight: {rgb_weight:.3f}, IR weight: {ir_weight:.3f}")
    
    # Expect RGB < 0.25 (it should be suppressed due to dark image)
    if rgb_weight < 0.25 and ir_weight > 0.75:
        print("✅ PASS: Night suppression working correctly")
    else:
        print(f"❌ FAIL: RGB weight too high ({rgb_weight:.3f})")
        return False
    return True

def test_4_token_gradients():
    print("\n--- Test 4: Token Gradient Flow (Fix #3 Magnitude) ---")
    model = GatedSpatialFusion_V3(512, 256)
    model.train()

    # Forward with token (simulated by manually expanding, 
    # since specific dropout path is stochastic)
    rgb_token = model.E_rgb.expand(4, 256, 40, 40)
    ir_feat = torch.randn(4, 256, 40, 40)
    dummy_rgb_img = torch.randn(4, 3, 640, 640)

    # We need to manually call forward but force token usage?
    # No, just pass token as input.
    # forward expects (rgb, ir).
    
    fused = model([rgb_token, ir_feat], dummy_rgb_img)
    loss = fused.sum()
    loss.backward()

    if model.E_rgb.grad is None:
        print("❌ FAIL: No gradient on E_rgb")
        return False
        
    token_grad = model.E_rgb.grad.abs().mean().item()
    print(f"Token gradient magnitude: {token_grad:.6f}")
    
    if token_grad > 1e-5:
        print("✅ PASS: Learnable tokens receiving gradients")
    else:
        print("❌ FAIL: Token gradient too small (dead parameter)")
        return False
    return True

def test_5_full_integration():
    print("\n--- Test 5: Full V3 Integration (YAML + Tasks.py) ---")
    try:
        # Load the X-Large V3 Config
        cfg_path = '/home/student/Toan/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11x-RGBT-gated-v3.yaml'
        model = DetectionModel(cfg_path, ch=6, verbose=False)
        print("Model initialized from YAML.")
        
        fusion_layers = [m for m in model.modules() if isinstance(m, GatedSpatialFusion_V3)]
        print(f"Found {len(fusion_layers)} GatedSpatialFusion_V3 layers.")
        
        if len(fusion_layers) == 3:
             print("✅ PASS: Correct number of V3 layers found.")
        else:
             print(f"❌ FAIL: Expected 3, found {len(fusion_layers)}")
             return False

        # Forward pass
        img = torch.randn(2, 6, 640, 640)
        out = model(img)
        print("✅ PASS: Forward pass successful.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ FAIL: Integration test crashed: {e}")
        return False
    return True

if __name__ == "__main__":
    results = [
        test_1_mc_dropout_convergence(),
        test_2_eval_mode_uq(),
        test_3_night_suppression(),
        test_4_token_gradients(),
        test_5_full_integration()
    ]
    
    if all(results):
        print("\n🏆 ALL ENGINEERING TESTS PASSED. READY FOR DEPLOYMENT.")
        exit(0)
    else:
        print("\n💥 SOME TESTS FAILED.")
        exit(1)
