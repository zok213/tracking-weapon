#!/usr/bin/env python3
"""
Compare Scale 0.50 vs Scale 0.66 side-by-side
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

# Parameters
video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
rgb_path = video_dir / "VID_EO_34.mp4"
ir_path = video_dir / "VID_IR_34.mp4"
output_path = Path("/home/student/Toan/results/comparison_0.50_vs_0.66.mp4")

def align(rgb, ir_shape, scale):
    h_rgb, w_rgb = rgb.shape[:2]
    h_ir, w_ir = ir_shape[:2]
    ir_aspect = w_ir / h_ir
    
    crop_h = int(h_rgb * scale)
    crop_w = int(crop_h * ir_aspect)
    
    x = (w_rgb - crop_w) // 2
    y = (h_rgb - crop_h) // 2
    
    mn = min(x, y, w_rgb-x-crop_w, h_rgb-y-crop_h)
    if mn < 0: # Handle edge case if scale > 1 (not expected here)
        return cv2.resize(rgb, (w_ir, h_ir))

    rgb_crop = rgb[y:y+crop_h, x:x+crop_w]
    return cv2.resize(rgb_crop, (w_ir, h_ir))

def main():
    rgb_cap = cv2.VideoCapture(str(rgb_path))
    ir_cap = cv2.VideoCapture(str(ir_path))
    
    fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    w_ir = int(ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_ir = int(ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Combined width: IR + RGB(0.5) + RGB(0.66)
    out_w = w_ir * 3
    out_h = h_ir
    
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                            fps, (out_w, out_h))
    
    print(f"Generating comparison video: {output_path}")
    
    for i in range(300): # 300 frames ~ 10 seconds
        ret1, rgb = rgb_cap.read()
        ret2, ir = ir_cap.read()
        if not ret1 or not ret2: break
        
        # Ensure IR is 3ch
        if len(ir.shape)==2: ir = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        
        # Scale 0.50
        rgb_050 = align(rgb, ir.shape, 0.50)
        cv2.putText(rgb_050, "Scale 0.50 (User Request)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Scale 0.66
        rgb_066 = align(rgb, ir.shape, 0.66)
        cv2.putText(rgb_066, "Scale 0.66 (Best Fit)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # IR Reference
        cv2.putText(ir, "IR Reference", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        combined = np.hstack([rgb_050, rgb_066, ir])
        writer.write(combined)
        
        if i % 50 == 0: print(f"Processing frame {i}...")
        
    writer.release()
    print("Done!")

if __name__ == "__main__":
    main()
