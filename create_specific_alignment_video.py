#!/usr/bin/env python3
"""
Specific Alignment Video Generator
User Request: Scale 0.40, Offset dx=15, dy=61.
Outputs:
1. Overlay Video (Red-Cyan)
2. Side-by-Side Video
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
    out_overlay = Path("/home/student/Toan/results/video_040_dx15_dy61_overlay.mp4")
    out_side = Path("/home/student/Toan/results/video_040_dx15_dy61_sidebyside.mp4")
    
    cap_rgb = cv2.VideoCapture(str(video_dir / "VID_EO_34.mp4"))
    cap_ir = cv2.VideoCapture(str(video_dir / "VID_IR_34.mp4"))
    
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    w_ir = int(cap_ir.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_ir = int(cap_ir.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Writers
    writer_overlay = cv2.VideoWriter(str(out_overlay), cv2.VideoWriter_fourcc(*'mp4v'), 
                                    fps, (w_ir, h_ir))
    writer_side = cv2.VideoWriter(str(out_side), cv2.VideoWriter_fourcc(*'mp4v'), 
                                 fps, (w_ir*2, h_ir))
    
    print(f"Generating Overlay: {out_overlay}")
    print(f"Generating Side-by-Side: {out_side}")
    
    scale = 0.40
    dx, dy = 15, 61
    
    # Process 10 seconds (300 frames)
    for i in range(300):
        ret1, rgb = cap_rgb.read()
        ret2, ir = cap_ir.read()
        if not ret1 or not ret2: break
        
        if len(ir.shape) == 2:
            ir_3ch = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        else:
            ir_3ch = ir
            
        # 1. Scale
        rgb_scaled = crop_and_scale(rgb, w_ir, h_ir, scale)
        
        # 2. Shift (Offset)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        rgb_shifted = cv2.warpAffine(rgb_scaled, M, (w_ir, h_ir))
        
        # 3. Create Overlay
        overlay = np.zeros_like(rgb_shifted)
        overlay[:,:,0] = rgb_shifted[:,:,0] # Cyan
        overlay[:,:,1] = rgb_shifted[:,:,1] # Cyan
        overlay[:,:,2] = ir_3ch[:,:,2]      # Red (IR)
        
        cv2.putText(overlay, f"Scale 0.40 | ({dx}, {dy})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        writer_overlay.write(overlay)
        
        # 4. Create Side-by-Side (RGB Shifted vs IR)
        # Draw label on RGB
        rgb_vis = rgb_shifted.copy()
        cv2.putText(rgb_vis, "RGB (Aligned)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw label on IR
        ir_vis = ir_3ch.copy()
        cv2.putText(ir_vis, "IR Reference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        side_by_side = np.hstack([rgb_vis, ir_vis])
        writer_side.write(side_by_side)
        
        if i % 50 == 0: print(f"Processing frame {i}")
        
    writer_overlay.release()
    writer_side.release()
    print("Done!")

if __name__ == "__main__":
    main()
