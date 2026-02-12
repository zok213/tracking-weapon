#!/usr/bin/env python3
"""
Fine Alignment Sweep Video
Center: (15, 61)
Range: +/- 3 pixels
Pattern: Spiral scan for smooth visual verification.
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
    output_path = Path("/home/student/Toan/results/fine_sweep_15_61.mp4")
    
    cap_rgb = cv2.VideoCapture(str(video_dir / "VID_EO_34.mp4"))
    cap_ir = cv2.VideoCapture(str(video_dir / "VID_IR_34.mp4"))
    
    # Target Frame 6s
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    target = int(fps * 6)
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, target)
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
    print(f"Generating Fine Sweep: {output_path}")
    
    cx, cy = 15, 61
    
    # Spiral around center
    # Radius 0 to 4 pixels
    # Angle step dense
    
    trajectory = []
    
    # 1. Hold at center
    for _ in range(15): trajectory.append((cx, cy))
    
    # 2. Spiral out
    max_r = 4.0
    for r in np.arange(0, max_r, 0.1):
        theta = r * 3.0 # radians
        dx = cx + r * np.cos(theta)
        dy = cy + r * np.sin(theta)
        trajectory.append((dx, dy))
        
    # 3. Spiral in
    for r in np.arange(max_r, 0, -0.1):
        theta = r * 3.0
        dx = cx + r * np.cos(theta)
        dy = cy + r * np.sin(theta)
        trajectory.append((dx, dy))

    # 4. Raster scan precise
    for dy in np.arange(cy-2, cy+2.1, 0.5):
        # Scan left-right
        xs = np.arange(cx-2, cx+2.1, 0.2)
        if int(dy*2)%2==1: xs = xs[::-1] # Zigzag
        for dx in xs:
            trajectory.append((dx, dy))

    for dx, dy in trajectory:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        rgb_shifted = cv2.warpAffine(rgb_base, M, (w, h))
        
        overlay = np.zeros_like(rgb_shifted)
        overlay[:,:,0] = rgb_shifted[:,:,0]
        overlay[:,:,1] = rgb_shifted[:,:,1]
        overlay[:,:,2] = ir_3ch[:,:,2]
        
        cv2.putText(overlay, f"({dx:.2f}, {dy:.2f})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, f"Scale 0.40", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                   
        writer.write(overlay)
            
    writer.release()
    print("Done!")

if __name__ == "__main__":
    main()
