#!/usr/bin/env python3
"""
Alignment Sweep Video
Objective: Visualize Scale 0.40 RGB 'sliding' over IR to check if alignment is possible.
Frame: 6s mark.
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
    output_path = Path("/home/student/Toan/results/alignment_sweep_040_6s.mp4")
    
    cap_rgb = cv2.VideoCapture(str(video_dir / "VID_EO_34.mp4"))
    cap_ir = cv2.VideoCapture(str(video_dir / "VID_IR_34.mp4"))
    
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    target_frame = int(fps * 6) # 6 seconds
    
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    # IR might need sync adjustment, but assuming roughly synced
    # Let's align IR timestamp
    rgb_ts = cap_rgb.get(cv2.CAP_PROP_POS_MSEC)
    cap_ir.set(cv2.CAP_PROP_POS_MSEC, rgb_ts)
    
    ret, rgb = cap_rgb.read()
    ret2, ir = cap_ir.read()
    
    if not ret or not ret2:
        print("Error reading frames")
        return
        
    h_ir, w_ir = ir.shape[:2]
    if len(ir.shape) == 2:
        ir_3ch = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    else:
        ir_3ch = ir

    # Base Crop Scale 0.40
    rgb_base = crop_and_scale(rgb, w_ir, h_ir, 0.40)
    
    # Create Sweep Video
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                            30, (w_ir, h_ir))
    
    print(f"Generating Alignment Sweep: {output_path}")
    
    # Spiral search or Grid scan?
    # Let's do a Grid Scan: X from -100 to 100, Y from -50 to 50
    # Current best guess offset might be near 0, but let's show wide range.
    
    # We will animate X then step Y. 
    # Or circle? Circle is nicer to watch.
    
    shifts = []
    # Create a spiral path
    for r in range(0, 150, 2): # Radius 0 to 150
        theta = r * 0.5 # Spiral angle
        dx = int(r * np.cos(theta))
        dy = int(r * np.sin(theta))
        shifts.append((dx, dy))
        
    for dx, dy in shifts:
        # Shift RGB
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        rgb_shifted = cv2.warpAffine(rgb_base, M, (w_ir, h_ir))
        
        # Overlay
        overlay = np.zeros_like(rgb_shifted)
        overlay[:,:,0] = rgb_shifted[:,:,0] # Cyan
        overlay[:,:,1] = rgb_shifted[:,:,1] # Cyan
        overlay[:,:,2] = ir_3ch[:,:,2]      # Red (IR)
        
        cv2.putText(overlay, f"Shift: dx={dx}, dy={dy}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, f"Scale 0.40 @ 6s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                   
        writer.write(overlay)
        
    writer.release()
    print("Done!")

if __name__ == "__main__":
    main()
