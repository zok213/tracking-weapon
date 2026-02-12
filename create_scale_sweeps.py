#!/usr/bin/env python3
"""
Multi-Scale Alignment Sweep
Scales: 0.41, 0.415, 0.42
Range: x=[0, 7], y=[64, 67]
Step: 0.2 (Coarser step to keep video length manageable? Or 0.1?)
Let's use 0.25 for speed and smoothness.
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

def generate_sweep(scale, x_range, y_range, output_path, video_dir, target_frame=381):
    cap_rgb = cv2.VideoCapture(str(video_dir / "VID_EO_34.mp4"))
    cap_ir = cv2.VideoCapture(str(video_dir / "VID_IR_34.mp4"))
    
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ts = cap_rgb.get(cv2.CAP_PROP_POS_MSEC)
    cap_ir.set(cv2.CAP_PROP_POS_MSEC, ts)
    
    ret, rgb = cap_rgb.read()
    ret2, ir = cap_ir.read()
    if not ret or not ret2: return

    if len(ir.shape)==2: ir_3ch = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    else: ir_3ch = ir
    
    h, w = ir_3ch.shape[:2]
    
    rgb_base = crop_and_scale(rgb, w, h, scale)
    
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                            30, (w, h))
    print(f"Generating Sweep Scale {scale}: {output_path}")
    
    step = 0.25
    
    for dy in np.arange(y_range[0], y_range[1] + step, step):
        xs = np.arange(x_range[0], x_range[1] + step, step)
        if int(dy*4)%2 == 1: xs = xs[::-1]
        
        for dx in xs:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            rgb_shifted = cv2.warpAffine(rgb_base, M, (w, h))
            
            overlay = np.zeros_like(rgb_shifted)
            overlay[:,:,0] = rgb_shifted[:,:,0]
            overlay[:,:,1] = rgb_shifted[:,:,1]
            overlay[:,:,2] = ir_3ch[:,:,2]
            
            cv2.putText(overlay, f"Scale {scale} | dx={dx:.2f}, dy={dy:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Range x{x_range} y{y_range}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                       
            writer.write(overlay)
            
    writer.release()

def main():
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    
    scales = [0.41, 0.415, 0.42]
    x_range = (0, 7)
    y_range = (64, 67)
    
    for s in scales:
        name = f"sweep_scale{str(s).replace('.','')}_0_7_64_67.mp4"
        out_path = Path("/home/student/Toan/results") / name
        generate_sweep(s, x_range, y_range, out_path, video_dir)
        
    print("All done!")

if __name__ == "__main__":
    main()
