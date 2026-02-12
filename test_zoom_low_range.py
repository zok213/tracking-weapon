#!/usr/bin/env python3
"""
Zoom Test: 0.45 - 0.55 Precision Range
User specific request to investigate this range.
"""
import cv2
import numpy as np
from pathlib import Path

def main():
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    output_dir = Path("/home/student/Toan/results/zoom_tests_045_055")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frame 250 (good visibility of person and car)
    cap_rgb = cv2.VideoCapture(str(video_dir / "VID_EO_34.mp4"))
    cap_ir = cv2.VideoCapture(str(video_dir / "VID_IR_34.mp4"))
    
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 250)
    cap_ir.set(cv2.CAP_PROP_POS_FRAMES, 250)
    
    ret, rgb_frame = cap_rgb.read()
    ret2, ir_frame = cap_ir.read()
    
    cap_rgb.release()
    cap_ir.release()
    
    if not ret or not ret2:
        print("Error reading frames")
        return

    h_ir, w_ir = ir_frame.shape[:2]
    if len(ir_frame.shape) == 2:
        ir_3ch = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:
        ir_3ch = ir_frame
        
    scales = np.arange(0.45, 0.56, 0.01) # 0.45 to 0.55 inclusive
    
    for scale in scales:
        h_rgb, w_rgb = rgb_frame.shape[:2]
        ir_aspect = w_ir / h_ir
        
        crop_h = int(h_rgb * scale)
        crop_w = int(crop_h * ir_aspect)
        
        x = (w_rgb - crop_w) // 2
        y = (h_rgb - crop_h) // 2
        
        rgb_crop = rgb_frame[y:y+crop_h, x:x+crop_w]
        rgb_resized = cv2.resize(rgb_crop, (w_ir, h_ir))
        
        # Red-Cyan Overlay 
        overlay = np.zeros_like(rgb_resized)
        overlay[:,:,0] = rgb_resized[:,:,0] # Blue
        overlay[:,:,1] = rgb_resized[:,:,1] # Green
        overlay[:,:,2] = ir_3ch[:,:,2]      # Red (IR)
        
        label = f"Scale {scale:.2f}"
        cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        cv2.imwrite(str(output_dir / f"overlay_{scale:.2f}.jpg"), overlay)
        print(f"Generated {label}")

    # Grid
    image_paths = sorted(list(output_dir.glob("overlay_*.jpg")))
    images = [cv2.imread(str(p)) for p in image_paths]
    
    cols = 4
    rows = (len(images) + cols - 1) // cols
    grid_rows = []
    
    for r in range(rows):
        row_imgs = images[r*cols : (r+1)*cols]
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(images[0]))
        grid_rows.append(np.hstack(row_imgs))
        
    final_grid = np.vstack(grid_rows)
    cv2.imwrite(str(output_dir / "grid_045_055.jpg"), final_grid)
    print(f"Saved grid to {output_dir}/grid_045_055.jpg")

if __name__ == "__main__":
    main()
