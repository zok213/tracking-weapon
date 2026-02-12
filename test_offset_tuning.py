#!/usr/bin/env python3
"""
Fine-Tuning Offset at Scale 0.40
User identified 'dy=65' as a promising area.
Generating a grid of offsets around dx=0, dy=65 to find the perfect match.
Frame: 6s (Frame 180).
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
    output_dir = Path("/home/student/Toan/results/offset_tuning_040")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap_rgb = cv2.VideoCapture(str(video_dir / "VID_EO_34.mp4"))
    cap_ir = cv2.VideoCapture(str(video_dir / "VID_IR_34.mp4"))
    
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    target_frame = int(fps * 6) # 6 seconds
    
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    rgb_ts = cap_rgb.get(cv2.CAP_PROP_POS_MSEC)
    cap_ir.set(cv2.CAP_PROP_POS_MSEC, rgb_ts)
    
    ret, rgb = cap_rgb.read()
    ret2, ir = cap_ir.read()
    
    if not ret or not ret2: return
    
    h_ir, w_ir = ir.shape[:2]
    if len(ir.shape) == 2:
        ir_3ch = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    else:
        ir_3ch = ir

    # Base Crop Scale 0.40
    rgb_base = crop_and_scale(rgb, w_ir, h_ir, 0.40)
    
    # Grid Search Parameters
    # User liked dy=65. Let's search around it.
    dx_range = list(range(-20, 21, 10)) # -20, -10, 0, 10, 20
    dy_range = list(range(45, 86, 10))  # 45, 55, 65, 75, 85
    
    results = []
    
    for dy in dy_range:
        row_imgs = []
        for dx in dx_range:
            # Shift RGB
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            rgb_shifted = cv2.warpAffine(rgb_base, M, (w_ir, h_ir))
            
            # Overlay
            overlay = np.zeros_like(rgb_shifted)
            overlay[:,:,0] = rgb_shifted[:,:,0] # Cyan
            overlay[:,:,1] = rgb_shifted[:,:,1] # Cyan
            overlay[:,:,2] = ir_3ch[:,:,2]      # Red (IR)
            
            label = f"dx={dx}, dy={dy}"
            cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            row_imgs.append(overlay)
        
        # Stack row
        results.append(np.hstack(row_imgs))
        
    # Stack all rows
    final_grid = np.vstack(results)
    
    output_file = output_dir / "offset_grid_040_focus.jpg"
    cv2.imwrite(str(output_file), final_grid)
    print(f"Saved offset grid to {output_file}")

if __name__ == "__main__":
    main()
