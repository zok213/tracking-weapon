#!/usr/bin/env python3
"""
Refined Zoom Level Testing - Fine-grained steps around 0.6-0.75
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
    output_dir = Path("/home/student/Toan/results/zoom_tests_refined")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rgb_frame = extract_frame(video_dir / "VID_EO_34.mp4", 200)
    ir_frame = extract_frame(video_dir / "VID_IR_34.mp4", 200)
    
    h_ir, w_ir = ir_frame.shape[:2]
    if len(ir_frame.shape) == 2:
        ir_3ch = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:
        ir_3ch = ir_frame
    
    # Fine-grained testing around 0.6-0.75
    scales = [0.55, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74]
    
    for scale in scales:
        rgb_scaled, crop_dims = crop_and_resize(rgb_frame, w_ir, h_ir, scale)
        
        # Add label
        label = f"Scale {scale:.2f} ({crop_dims[0]}x{crop_dims[1]})"
        cv2.putText(rgb_scaled, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Overlay with red-cyan channels for better visibility
        # RGB blue+green channels + IR red channel
        overlay = np.zeros_like(rgb_scaled)
        overlay[:,:,0] = ir_3ch[:,:,0]  # Blue from IR
        overlay[:,:,1] = rgb_scaled[:,:,1]  # Green from RGB
        overlay[:,:,2] = rgb_scaled[:,:,2]  # Red from RGB
        
        # Add IR reference overlay
        ir_labeled = ir_3ch.copy()
        cv2.putText(ir_labeled, "IR Ref", (w_ir-80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
        combined = np.hstack([rgb_scaled, overlay, ir_labeled])
        cv2.imwrite(str(output_dir / f"scale_{scale:.2f}_combined.jpg"), combined)
        print(f"Saved: scale={scale:.2f}")
    
    # Create mega comparison grid (3x3)
    rows = []
    for i in range(0, len(scales), 3):
        images = []
        for scale in scales[i:i+3]:
            rgb_scaled, _ = crop_and_resize(rgb_frame, w_ir, h_ir, scale)
            overlay = np.zeros_like(rgb_scaled)
            overlay[:,:,0] = ir_3ch[:,:,0]
            overlay[:,:,1] = rgb_scaled[:,:,1]
            overlay[:,:,2] = rgb_scaled[:,:,2]
            cv2.putText(overlay, f"{scale:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            images.append(overlay)
        while len(images) < 3:
            images.append(np.zeros_like(overlay))
        rows.append(np.hstack(images))
    
    grid = np.vstack(rows)
    cv2.imwrite(str(output_dir / "overlay_grid_refined.jpg"), grid)
    print(f"\n✓ Saved refined zoom tests to: {output_dir}")
    print("Look for scale where car edges align in RED-CYAN overlay")


if __name__ == "__main__":
    main()
