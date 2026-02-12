#!/usr/bin/env python3
"""
Ultra-Fine Alignment Sweep
Range: x=[9, 11], y=[64, 65]
Step: 0.1
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
    output_path = Path("/home/student/Toan/results/sweep_11_9_64_65.mp4")
    
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
    
    # Base Scale 0.40
    rgb_base = crop_and_scale(rgb, w, h, 0.40)
    
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                            30, (w, h))
    print(f"Generating Sweep for x[9-11], y[64-65]: {output_path}")
    
    # Raster scan
    # y from 64 to 65
    # x from 9 to 11
    step = 0.1
    
    # Range inclusive
    for dy in np.arange(64, 65 + step, step):
        # Scan x
        xs = np.arange(9, 11 + step, step)
         # Zigzag for smoothness
        if int(dy*10)%2 == 1: xs = xs[::-1]
        
        for dx in xs:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            rgb_shifted = cv2.warpAffine(rgb_base, M, (w, h))
            
            overlay = np.zeros_like(rgb_shifted)
            overlay[:,:,0] = rgb_shifted[:,:,0]
            overlay[:,:,1] = rgb_shifted[:,:,1]
            overlay[:,:,2] = ir_3ch[:,:,2]
            
            cv2.putText(overlay, f"dx={dx:.1f}, dy={dy:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(overlay, "Scale 0.40 | Frame 381", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                       
            writer.write(overlay)
            
    writer.release()
    print("Done!")

if __name__ == "__main__":
    main()
