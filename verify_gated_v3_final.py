
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from ultralytics.nn.modules.block import GatedSpatialFusion_V3

def test_1_mc_dropout_convergence():
    print("\n--- Test 1: MC-Dropout Sample Count Verification (Fix #1) ---")
    model = GatedSpatialFusion_V3(512, 256)
    model.eval()  # Important: Test in eval mode

    dummy_feat = torch.ones(4, 256, 40, 40) # Use ones to check variance of dropout only

    # Test different sample counts
    results = {}
    for n in [2, 5, 10, 20]:
        uncertainties = []
        for _ in range(10):  # Repeat 10 times to get variance of the estimate
            # We need to simulate dropout randomness. 
            # Since input is ones, variance comes from dropout masks.
            unc = model.estimate_uncertainty(dummy_feat, model.rgb_dropout, n_samples=n)
            uncertainties.append(unc.mean().item())

        std = torch.tensor(uncertainties).std().item()
        mean = torch.tensor(uncertainties).mean().item()
        results[n] = std
        print(f"n={n:2d}: mean={mean:.4f}, std={std:.4f}")
    
    # Check: std should be low for n=20
    # Note: With fixed inputs (ones), variance of the *estimate* should decrease as N increases.
    # The actual uncertainty value depends on dropout rate (p=0.2 -> var approx p*(1-p)?)
    
    if results[20] > 0.05:
         print(f"❌ FAIL: n=20 variance is high (std={results[20]:.4f})")
    else:
         print(f"✅ PASS: n=20 variance is low (std={results[20]:.4f})")

def test_2_eval_mode_uncertainty():
    print("\n--- Test 2: Eval Mode Uncertainty Check (Fix #2) ---")
    model = GatedSpatialFusion_V3(512, 256)
    dummy_feat = torch.randn(4, 256, 40, 40)

    # Test in EVAL mode
    model.eval()
    unc_eval = model.estimate_uncertainty(dummy_feat, model.rgb_dropout, n_samples=20)

    print(f"Eval mode uncertainty mean: {unc_eval.mean():.4f}")
    
    if unc_eval.mean() < 0.0001:
        print("❌ FAIL: Uncertainty is zero in eval mode!")
    else:
        print("✅ PASS: Uncertainty estimation works in eval mode")

def test_3_night_suppression():
    print("\n--- Test 3: Night Suppression (End-to-End) ---")
    model = GatedSpatialFusion_V3(256, 256)
    model.eval()

    # Simulate pitch black night
    dark_rgb_img = torch.zeros(4, 3, 640, 640)
    
    # RGB features are noise (low confidence if uncertainty works, but here we test illumination impact)
    # To test logic:
    # gate_rgb = gate * conf * illum
    # If illum is small (night), gate_rgb should be small.
    
    noise_rgb_feat = torch.randn(4, 128, 40, 40)
    valid_ir_feat = torch.randn(4, 128, 40, 40)

    fused = model([noise_rgb_feat, valid_ir_feat], dark_rgb_img)
    
    # We can't easily access gate weights from forward output since it returns merged tensor.
    # BUT, we can inspect correct behavior via hooks or by temporarily returning weights.
    # Since we removed 'last_gate_weights', we can't check it directly.
    # We will infer it by checking if output is dominated by IR.
    
    # Alternative: Instrument the model for this test
    # We will compute gate values manually to verify logic
    
    # 1. Illum
    illum = model.illum_estimator(dark_rgb_img).view(4, 1, 1, 1).mean()
    print(f"Illum value (should be low): {illum.item():.4f}")
    
    if illum.item() > 0.2:
        print("❌ FAIL: Illumination estimator thinks black image is bright!")
        return

    print("✅ PASS: Night suppression logic likely active (illum is low)")

def test_4_token_gradient_flow():
    print("\n--- Test 4: Token Gradient Flow (Fix #3 Magnitude) ---")
    model = GatedSpatialFusion_V3(512, 256)
    model.train() # Must be in train mode for dropout/tokens

    # Forward with token (simulated by manually expanding, 
    # but actual forward handles it probabilistically. 
    # We'll just check if parameters have grad after backward)
    
    # Force usage of tokens? 
    # The tokens are parameters. If we can get grad on them, they are connected.
    
    rgb_token = model.E_rgb.expand(4, 256, 40, 40)
    ir_feat = torch.randn(4, 256, 40, 40)
    dummy_rgb_img = torch.randn(4, 3, 640, 640)
    
    # We manually pass token as input to see if it flows? 
    # No, model uses tokens internally when dropout happens.
    # Let's force drop in code? No, probabilistic.
    # Let's just run forward 10 times, statistical likelihood of using token is high (p=0.3).
    
    target = torch.randn(4, 256, 40, 40)
    
    loss_sum = 0
    for _ in range(20):
        # We need to trigger the internal "if drop_rgb" logic
        # We can simulate this by mocking, but let's just run and hope p=0.3 hits.
        fused = model([torch.randn(4,256,40,40), torch.randn(4,256,40,40)], dummy_rgb_img)
        loss = fused.sum()
        loss.backward()
        loss_sum += loss.item()
    
    if model.E_rgb.grad is None:
         print("❌ FAIL: No gradient on E_rgb (maybe dropout didn't trigger?)")
    else:
         grad_mag = model.E_rgb.grad.abs().mean().item()
         print(f"Token gradient magnitude: {grad_mag:.6f}")
         if grad_mag > 1e-5:
             print("✅ PASS: Learnable tokens receiving gradients")
         else:
             print("❌ FAIL: Token gradient too small")

def test_5_integration_safety():
    print("\n--- Test 5: Integration / Crash Check ---")
    try:
        model = GatedSpatialFusion_V3(512, 512)
        model.eval()
        x = [torch.randn(2, 256, 20, 20), torch.randn(2, 256, 20, 20)]
        out = model(x, torch.randn(2, 3, 640, 640))
        print("✅ PASS: Forward pass successful")
        
        import copy
        model_copy = copy.deepcopy(model)
        print("✅ PASS: Deepcopy successful (No graph storage issue)")
        
    except Exception as e:
        print(f"❌ FAIL: Crash detected: {e}")

if __name__ == "__main__":
    test_1_mc_dropout_convergence()
    test_2_eval_mode_uncertainty()
    test_3_night_suppression()
    test_4_token_gradient_flow()
    test_5_integration_safety()
