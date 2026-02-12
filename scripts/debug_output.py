#!/usr/bin/env python3
"""
Debug script to inspect model output format
"""

import torch
import numpy as np
from pathlib import Path

def main():
    device = 'cuda:0'
    
    # Load model
    from ultralytics import YOLO
    model_obj = YOLO("/home/student/Toan/models/yolo26x_rgbt_init.pt")
    state_dict = torch.load("/home/student/Toan/checkpoints/rgbt_vtmot_ddp/best.pt", map_location=device)
    model_obj.model.load_state_dict(state_dict)
    model_obj.model.to(device)
    model_obj.model.eval()
    
    # Load a sample image
    val_img_dir = Path("/home/student/Toan/data/VT-MOT_RGBT/images/val")
    img_path = list(val_img_dir.glob('*.npy'))[0]
    
    img = np.load(img_path)
    print(f"Image shape: {img.shape}")
    
    # Preprocess
    import cv2
    img_resized = cv2.resize(img, (640, 640)) if img.shape[:2] != (640, 640) else img
    img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    print(f"Input tensor shape: {img_tensor.shape}")
    
    # Inference
    with torch.no_grad():
        preds = model_obj.model(img_tensor)
    
    print(f"\n--- Model Output Analysis ---")
    print(f"Type: {type(preds)}")
    
    if isinstance(preds, dict):
        print(f"Keys: {preds.keys()}")
        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, (list, tuple)):
                print(f"  {k}: len={len(v)}")
                for i, item in enumerate(v[:3]):
                    if isinstance(item, torch.Tensor):
                        print(f"    [{i}]: shape={item.shape}")
    elif isinstance(preds, (list, tuple)):
        print(f"Length: {len(preds)}")
        for i, item in enumerate(preds[:3]):
            if isinstance(item, torch.Tensor):
                print(f"  [{i}]: shape={item.shape}")
    elif isinstance(preds, torch.Tensor):
        print(f"Shape: {preds.shape}")
        
    # Try to decode predictions
    print(f"\n--- Decoding Predictions ---")
    
    # Use ultralytics built-in postprocessing
    from ultralytics.utils.ops import non_max_suppression
    
    # Get raw output
    if isinstance(preds, dict):
        if 'one2many' in preds:
            raw = preds['one2many']
        else:
            raw = list(preds.values())[0]
    elif isinstance(preds, (list, tuple)):
        raw = preds[0]
    else:
        raw = preds
    
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
        
    print(f"Raw prediction shape: {raw.shape}")
    
    # Check values
    print(f"Min: {raw.min():.4f}, Max: {raw.max():.4f}, Mean: {raw.mean():.4f}")
    
    # Try NMS
    # For YOLOv8/11, output is (batch, 4+nc, num_predictions) 
    # or (batch, num_predictions, 4+nc)
    
    if raw.shape[1] < raw.shape[2]:
        # Transpose to (batch, num_preds, 4+nc)
        raw_t = raw.permute(0, 2, 1)
    else:
        raw_t = raw
        
    print(f"Transposed shape: {raw_t.shape}")
    
    # Try NMS
    try:
        nms_out = non_max_suppression(raw, conf_thres=0.001, iou_thres=0.45)
        print(f"NMS output: {len(nms_out)} images")
        for i, det in enumerate(nms_out):
            print(f"  Image {i}: {det.shape if isinstance(det, torch.Tensor) else len(det)} detections")
            if len(det) > 0:
                print(f"    Sample: {det[0]}")
    except Exception as e:
        print(f"NMS failed: {e}")
    
    # Check ground truth
    lbl_dir = Path("/home/student/Toan/data/VT-MOT_RGBT/labels/val")
    lbl_path = lbl_dir / (img_path.stem + '.txt')
    print(f"\n--- Ground Truth ({lbl_path}) ---")
    if lbl_path.exists():
        with open(lbl_path) as f:
            lines = f.readlines()
        print(f"Num GT boxes: {len(lines)}")
        for line in lines[:5]:
            print(f"  {line.strip()}")
    else:
        print("No label file found!")

if __name__ == '__main__':
    main()
