#!/usr/bin/env python3
"""
Precision Zoom Testing - Round 2

User feedback: 0.66 might not be correct, wants to check around 0.5 again.
Objective: Test scales 0.45 to 0.75 in 0.02 increments with clear overlay grid.
"""
import cv2
import numpy as np
from pathlib import Path


def extract_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def crop_and_resize(rgb_frame, ir_w, ir_h, scale):
    h_rgb, w_rgb = rgb_frame.shape[:2]
    ir_aspect = ir_w / ir_h
    
    base_crop_h = h_rgb
    base_crop_w = int(base_crop_h * ir_aspect)
    
    crop_h = int(base_crop_h * scale)
    crop_w = int(base_crop_w * scale)
    
    x = (w_rgb - crop_w) // 2
    y = (h_rgb - crop_h) // 2
    
    rgb_cropped = rgb_frame[y:y+crop_h, x:x+crop_w]
    rgb_resized = cv2.resize(rgb_cropped, (ir_w, ir_h))
    
    return rgb_resized, (crop_w, crop_h)


def main():
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    output_dir = Path("/home/student/Toan/results/zoom_tests_precision")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rgb_frame = extract_frame(video_dir / "VID_EO_34.mp4", 250)
    ir_frame = extract_frame(video_dir / "VID_IR_34.mp4", 250)
    
    h_ir, w_ir = ir_frame.shape[:2]
    if len(ir_frame.shape) == 2:
        ir_3ch = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:
        ir_3ch = ir_frame
    
    # Test wider range to cover user's suggestion (0.5)
    # 0.45 to 0.75
    scales = np.arange(0.45, 0.76, 0.02)
    
    for scale in scales:
        rgb_scaled, _ = crop_and_resize(rgb_frame, w_ir, h_ir, scale)
        
        # Red-Cyan Overlay (Red=IR, Cyan=RGB)
        # Ideally: If perfectly aligned, overlapping areas are grey/white
        overlay = np.zeros_like(rgb_scaled)
        overlay[:,:,0] = rgb_scaled[:,:,0]  # Blue (RGB)
        overlay[:,:,1] = rgb_scaled[:,:,1]  # Green (RGB)
        overlay[:,:,2] = ir_3ch[:,:,2]     # Red (IR)
        
        cv2.putText(overlay, f"{scale:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imwrite(str(output_dir / f"overlay_{scale:.2f}.jpg"), overlay)
        print(f"Generated scaling: {scale:.2f}")

    # Create large grid
    image_paths = sorted(list(output_dir.glob("overlay_*.jpg")))
    images = [cv2.imread(str(p)) for p in image_paths]
    
    # Grid: 4 columns
    cols = 4
    rows = (len(images) + cols - 1) // cols
    
    grid_rows = []
    for r in range(rows):
        row_imgs = images[r*cols : (r+1)*cols]
        # Pad if last row incomplete
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(images[0]))
        grid_rows.append(np.hstack(row_imgs))
        
    final_grid = np.vstack(grid_rows)
    cv2.imwrite(str(output_dir / "master_grid.jpg"), final_grid)
    print(f"\n✓ Saved precision zoom tests to: {output_dir}")


if __name__ == "__main__":
    main()
