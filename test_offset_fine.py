#!/usr/bin/env python3
"""
Sub-Pixel Offset Tuning (Step 0.5)
User Request: Scale 0.40, Center (15, 61), Step 0.5.
Range: +/- 2 pixels around center.
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
    output_dir = Path("/home/student/Toan/results/offset_fine_15_61")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap_rgb = cv2.VideoCapture(str(video_dir / "VID_EO_34.mp4"))
    cap_ir = cv2.VideoCapture(str(video_dir / "VID_IR_34.mp4"))
    
    # 6s Mark
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    target_frame = int(fps * 6)
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    rgb_ts = cap_rgb.get(cv2.CAP_PROP_POS_MSEC)
    cap_ir.set(cv2.CAP_PROP_POS_MSEC, rgb_ts)
    
    ret, rgb = cap_rgb.read()
    ret2, ir = cap_ir.read()
    if not ret or not ret2: return
    
    if len(ir.shape)==2: ir_3ch = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    else: ir_3ch = ir
    
    h_ir, w_ir = ir.shape[:2]
    
    # Base Scale 0.40
    rgb_base = crop_and_scale(rgb, w_ir, h_ir, 0.40)
    
    # Grid: dx [13, 17], dy [59, 63], Step 0.5
    dx_center, dy_center = 15, 61
    span = 2.0
    step = 0.5
    
    dx_vals = np.arange(dx_center - span, dx_center + span + step, step)
    dy_vals = np.arange(dy_center - span, dy_center + span + step, step)
    
    rows = []
    
    for dy in dy_vals:
        cols = []
        for dx in dx_vals:
            # Sub-pixel shift using warpAffine
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            rgb_shifted = cv2.warpAffine(rgb_base, M, (w_ir, h_ir), flags=cv2.INTER_LINEAR)
            
            # Overlay
            overlay = np.zeros_like(rgb_shifted)
            overlay[:,:,0] = rgb_shifted[:,:,0] # Cyan
            overlay[:,:,1] = rgb_shifted[:,:,1] # Cyan
            overlay[:,:,2] = ir_3ch[:,:,2]      # Red (IR)
            
            label = f"({dx:.1f}, {dy:.1f})"
            cv2.putText(overlay, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Draw Center Crosshair to help check alignment?
            # cv2.drawMarker(overlay, (w_ir//2, h_ir//2), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=10)
            
            cols.append(overlay)
        rows.append(np.hstack(cols))
        
    final_grid = np.vstack(rows)
    
    out_file = output_dir / "fine_grid_15_61.jpg"
    cv2.imwrite(str(out_file), final_grid)
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
