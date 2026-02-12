#!/usr/bin/env python3
"""
Scale Sweep Video (0.40 - 0.50)
Offset: Fixed (0, 60)
Visualization: Alpha Blend (Opacity 0.6 RGB, 0.4 IR)
Frame: 381
"""
import cv2
import numpy as np
from pathlib import Path

def crop_and_scale(rgb, w_ir, h_ir, scale):
    h_rgb, w_rgb = rgb.shape[:2]
    ir_aspect = w_ir / h_ir
    crop_h = int(h_rgb * scale)
    crop_w = int(crop_h * ir_aspect)
    x = (w_rgb - crop_w) // 2
    y = (h_rgb - crop_h) // 2
    if x < 0 or y < 0: return cv2.resize(rgb, (w_ir, h_ir))
    rgb_crop = rgb[y:y+crop_h, x:x+crop_w]
    return cv2.resize(rgb_crop, (w_ir, h_ir))

def main():
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    output_path = Path("/home/student/Toan/results/sweep_scale_40_50_alpha.mp4")
    
    cap_rgb = cv2.VideoCapture(str(video_dir / "VID_EO_34.mp4"))
    cap_ir = cv2.VideoCapture(str(video_dir / "VID_IR_34.mp4"))
    
    # Target Frame 381
    target_frame = 381
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ts = cap_rgb.get(cv2.CAP_PROP_POS_MSEC)
    cap_ir.set(cv2.CAP_PROP_POS_MSEC, ts)
    
    ret, rgb = cap_rgb.read()
    ret2, ir = cap_ir.read()
    if not ret or not ret2: return

    if len(ir.shape)==2: ir_3ch = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    else: ir_3ch = ir
    
    h, w = ir_3ch.shape[:2]
    
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                            30, (w, h))
    print(f"Generating Scale Sweep 0.40->0.50: {output_path}")
    
    # Scale range
    scales = np.arange(0.40, 0.501, 0.001) # 100 frames
    
    # Create a nice ping-pong loop? Or just linear?
    # Let's do Linear 0.40 -> 0.50 -> 0.40 so they can watch it back and forth
    scales = np.concatenate([scales, scales[::-1]])
    
    offset_x = 0
    offset_y = 60
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    
    for s in scales:
        # 1. Scale
        rgb_scaled = crop_and_scale(rgb, w, h, s)
        
        # 2. Shift (Fixed)
        rgb_shifted = cv2.warpAffine(rgb_scaled, M, (w, h))
        
        # 3. Alpha Blend (0.6 RGB, 0.4 IR)
        # Note: rgb_shifted has black borders where shifted. 
        # We might want to keep IR bright there.
        
        # Standard blending
        blended = cv2.addWeighted(rgb_shifted, 0.6, ir_3ch, 0.4, 0)
        
        cv2.putText(blended, f"Scale: {s:.3f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(blended, f"Offset: (0, 60)", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(blended, "Opacity: 0.6", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                   
        writer.write(blended)
            
    writer.release()
    print("Done!")

if __name__ == "__main__":
    main()
