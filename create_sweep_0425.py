#!/usr/bin/env python3
"""
Alignment Sweep
Scale: 0.425
Range: x=[0], y=[60, 65]
Step: 0.2
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
    output_path = Path("/home/student/Toan/results/sweep_scale0425_0_60_65.mp4")
    
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
    
    # Scale 0.425
    rgb_base = crop_and_scale(rgb, w, h, 0.425)
    
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                            30, (w, h))
    print(f"Generating Sweep Scale 0.425: {output_path}")
    
    step = 0.2
    
    # y from 60 to 65
    for dy in np.arange(60, 65 + step, step):
        # x fixed at 0 (or small range? User said "0").
        # Let's do just 0 to be precise to request.
        dx = 0
        
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        rgb_shifted = cv2.warpAffine(rgb_base, M, (w, h))
        
        overlay = np.zeros_like(rgb_shifted)
        overlay[:,:,0] = rgb_shifted[:,:,0]
        overlay[:,:,1] = rgb_shifted[:,:,1]
        overlay[:,:,2] = ir_3ch[:,:,2]
        
        cv2.putText(overlay, f"Scale 0.425 | dx={dx}, dy={dy:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                   
        writer.write(overlay)
            
    writer.release()
    print("Done!")

if __name__ == "__main__":
    main()
