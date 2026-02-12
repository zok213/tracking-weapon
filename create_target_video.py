#!/usr/bin/env python3
"""
Target Video: Scales 0.40, 0.41, 0.42
User Request: Focus on 0.4-0.42. Create video with RGB and IR.
Format:
  [Overlay 0.40] [Overlay 0.41]
  [Overlay 0.42] [IR Reference]
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
    
    # Safety
    if x < 0 or y < 0: return cv2.resize(rgb, (w_ir, h_ir))
    
    rgb_crop = rgb[y:y+crop_h, x:x+crop_w]
    return cv2.resize(rgb_crop, (w_ir, h_ir))

def create_overlay(rgb_aligned, ir_3ch, label):
    overlay = np.zeros_like(rgb_aligned)
    overlay[:,:,0] = rgb_aligned[:,:,0] # Cyan
    overlay[:,:,1] = rgb_aligned[:,:,1] # Cyan
    overlay[:,:,2] = ir_3ch[:,:,2]      # Red (IR)
    
    cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return overlay

def main():
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    output_path = Path("/home/student/Toan/results/video_scale_040_042.mp4")
    
    cap_rgb = cv2.VideoCapture(str(video_dir / "VID_EO_34.mp4"))
    cap_ir = cv2.VideoCapture(str(video_dir / "VID_IR_34.mp4"))
    
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    w_ir = int(cap_ir.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_ir = int(cap_ir.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output layout: 2x2 grid
    out_w = w_ir * 2
    out_h = h_ir * 2
    
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                            fps, (out_w, out_h))
    
    print(f"Creating video: {output_path}")
    
    # Process 300 frames (approx 10s)
    frames_to_process = 300
    
    for i in range(frames_to_process):
        ret1, rgb = cap_rgb.read()
        ret2, ir = cap_ir.read()
        
        if not ret1 or not ret2: break
        
        if len(ir.shape) == 2:
            ir_3ch = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        else:
            ir_3ch = ir
            
        # Resize IR for grid slot if needed (it's already 640x512)
        
        # Scale 0.40
        rgb_40 = crop_and_scale(rgb, w_ir, h_ir, 0.40)
        ov_40 = create_overlay(rgb_40, ir_3ch, "Scale 0.40")
        
        # Scale 0.41
        rgb_41 = crop_and_scale(rgb, w_ir, h_ir, 0.41)
        ov_41 = create_overlay(rgb_41, ir_3ch, "Scale 0.41")
        
        # Scale 0.42
        rgb_42 = crop_and_scale(rgb, w_ir, h_ir, 0.42)
        ov_42 = create_overlay(rgb_42, ir_3ch, "Scale 0.42")
        
        # IR Reference
        ir_vis = ir_3ch.copy()
        cv2.putText(ir_vis, "IR Reference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Stack 2x2
        top = np.hstack([ov_40, ov_41])
        bot = np.hstack([ov_42, ir_vis])
        grid = np.vstack([top, bot])
        
        writer.write(grid)
        
        if i % 50 == 0: print(f"Processing frame {i}")
        
    writer.release()
    print("Done!")

if __name__ == "__main__":
    main()
