#!/usr/bin/env python3
"""
Video Generation Package
1. Focused Sweep: Static frame @ 6s, scanning offsets around (0, 65).
2. Fixed Offset Run: 10s video playing with Scale 0.40 & Offset (0, 65) applied.
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

def make_focused_sweep():
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    output_path = Path("/home/student/Toan/results/alignment_sweep_focus_6s.mp4")
    
    cap_rgb = cv2.VideoCapture(str(video_dir / "VID_EO_34.mp4"))
    cap_ir = cv2.VideoCapture(str(video_dir / "VID_IR_34.mp4"))
    
    # Seek to 6s
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    target = int(fps * 6)
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, target)
    ts = cap_rgb.get(cv2.CAP_PROP_POS_MSEC)
    cap_ir.set(cv2.CAP_PROP_POS_MSEC, ts)
    
    ret, rgb = cap_rgb.read()
    ret2, ir = cap_ir.read()
    if not ret or not ret2: return

    if len(ir.shape)==2: ir = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    h, w = ir.shape[:2]
    
    # Scale 0.40 Base
    rgb_base = crop_and_scale(rgb, w, h, 0.40)
    
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                            30, (w, h))
    print(f"Generating Focus Sweep: {output_path}")
    
    # Scan pattern: Raster scan roughly around dy=65
    # dy: 45 to 85. dx: -20 to 20.
    for dy in range(45, 86, 2):
        for dx in range(-20, 21, 5):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            rgb_shifted = cv2.warpAffine(rgb_base, M, (w, h))
            
            overlay = np.zeros_like(rgb_shifted)
            overlay[:,:,0] = rgb_shifted[:,:,0]
            overlay[:,:,1] = rgb_shifted[:,:,1]
            overlay[:,:,2] = ir[:,:,2]
            
            cv2.putText(overlay, f"Scale 0.40 | dx={dx} dy={dy}", (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            writer.write(overlay)
            
    writer.release()

def make_fixed_offset_video():
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    output_path = Path("/home/student/Toan/results/video_scale_040_offset_65.mp4")
    
    cap_rgb = cv2.VideoCapture(str(video_dir / "VID_EO_34.mp4"))
    cap_ir = cv2.VideoCapture(str(video_dir / "VID_IR_34.mp4"))
    
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    w = int(cap_ir.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_ir.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                            fps, (w, h))
    print(f"Generating Fixed Offset Video: {output_path}")
    
    # Center target: dx=0, dy=65
    dx, dy = 0, 65
    
    for i in range(300): # 10s
        ret, rgb = cap_rgb.read()
        ret2, ir = cap_ir.read()
        if not ret or not ret2: break
        
        if len(ir.shape)==2: ir = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        
        rgb_scaled = crop_and_scale(rgb, w, h, 0.40)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        rgb_shifted = cv2.warpAffine(rgb_scaled, M, (w, h))
        
        overlay = np.zeros_like(rgb_shifted)
        overlay[:,:,0] = rgb_shifted[:,:,0]
        overlay[:,:,1] = rgb_shifted[:,:,1]
        overlay[:,:,2] = ir[:,:,2]
        
        cv2.putText(overlay, f"Scale 0.40 | Fixed Offset (0, 65)", (10,30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        writer.write(overlay)
        
    writer.release()

if __name__ == "__main__":
    make_focused_sweep()
    make_fixed_offset_video()
