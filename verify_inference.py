from ultralytics import YOLO
import os
import cv2

# Weights
weights_path = "/home/student/Toan/weights/FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.01-warmup_epochs1-SGD.pt"

# Image Source
source_dir = "/home/student/Toan/data_processing/dataset_roboflow/video_47/images"
# Get first RGB image
files = sorted([f for f in os.listdir(source_dir) if "_rgb.jpg" in f])
if not files:
    print("❌ No images found")
    exit()

img_path = os.path.join(source_dir, files[0])
img_ir_path = img_path.replace("_rgb.jpg", "_ir.jpg")

print(f"Loading weights: {weights_path}")
try:
    model = YOLO(weights_path)
    print("✅ Model loaded successfully")
    
    # Run Inference (Dual Stream)
    # The model expects a list of [rgb, ir] or similar structure if using custom loader.
    # But standard YOLO predict() might not handle dual-stream inputs automatically for local files without a custom dataset loader.
    # However, Ultralytics 'predict' usually takes a source.
    # If the model is modified to accept a list/tuple in forward, passing a list of numpy arrays might work?
    # Or, we manually construct the input batch.
    
    # Custom Inference Loop
    import torch
    
    # Read Images
    im_rgb = cv2.imread(img_path)
    im_ir = cv2.imread(img_ir_path)
    
    # Preprocess (Resize to 640x640?)
    # The model probably expects 640.
    # But let's try using model.predict with a custom source logic?
    # No, model.predict() is complex. 
    # Let's try passing the image inputs directly if possible.
    # Simpler: We create a dummy dataset class or just try to pass a list of 2 images?
    # If we pass a list [img_rgb, img_ir], standard YOLO treats it as a batch of 2 images.
    # We need to pass ONE item that contains BOTH.
    
    # Approach: Use the model directly.
    # 1. Preprocess
    from ultralytics.data.augment import LetterBox
    
    target_size = 640
    pre = LetterBox(target_size, auto=False, stride=32)
    
    im_rgb_p = pre(image=im_rgb)
    im_ir_p = pre(image=im_ir)
    
    # Stack? No, Multi-Stream needs separate inputs to the forward pass.
    # Input shape: x = [rgb, ir]
    
    # Convert to Tensor
    t_rgb = torch.from_numpy(im_rgb_p).to(model.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t_ir = torch.from_numpy(im_ir_p).to(model.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # IF IR is 3-channel (read by cv2), take 1 channel? 
    # The model expect 3-channel IR? Filename says "aligned3C".
    # Check config... usually 1ch. "aligned3C" might mean converted to 3C.
    # Let's assume 3C for now as cv2 reads 3C.
    
    print(f"Inference input shapes: RGB {t_rgb.shape}, IR {t_ir.shape}")
    
    # Forward
    # The model.model is the PyTorch module.
    # It expects predictions.
    results = model.model([t_rgb, t_ir])
    
    print("✅ Inference ran successfully!")
    print(f"Output type: {type(results)}")
    # results is usually preds, (feats...)
    
    print("Done test.")

except Exception as e:
    print(f"❌ Inference failed: {e}")
