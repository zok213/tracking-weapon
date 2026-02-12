#!/usr/bin/env python3
"""
Systematic Zoom Level Testing

The IR camera is more zoomed in than the current RGB crop.
This script tests different zoom levels to find the exact match.

Key insight: Objects (like the car) should appear SAME SIZE in both RGB and IR.
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
    """
    Crop RGB at given scale and resize to IR dimensions.
    scale=1.0 means full aspect-ratio crop (2700x2160)
    scale=0.5 means half size crop (1350x1080)
    """
    h_rgb, w_rgb = rgb_frame.shape[:2]
    
    # IR aspect ratio
    ir_aspect = ir_w / ir_h  # 1.25
    
    # Base crop at full height, adjusted for aspect ratio
    base_crop_h = h_rgb  # 2160
    base_crop_w = int(base_crop_h * ir_aspect)  # 2700
    
    # Apply scale
    crop_h = int(base_crop_h * scale)
    crop_w = int(base_crop_w * scale)
    
    # Center crop
    x = (w_rgb - crop_w) // 2
    y = (h_rgb - crop_h) // 2
    
    rgb_cropped = rgb_frame[y:y+crop_h, x:x+crop_w]
    rgb_resized = cv2.resize(rgb_cropped, (ir_w, ir_h))
    
    return rgb_resized, (crop_w, crop_h)


def create_comparison_grid(rgb_frame, ir_frame, output_dir):
    """Create visual comparison for different zoom levels."""
    h_ir, w_ir = ir_frame.shape[:2]
    
    if len(ir_frame.shape) == 2:
        ir_3ch = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:
        ir_3ch = ir_frame
    
    # Test scales from 0.5 to 1.0
    scales = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    results = []
    for scale in scales:
        rgb_scaled, crop_dims = crop_and_resize(rgb_frame, w_ir, h_ir, scale)
        
        # Add label
        label = f"Scale {scale:.2f} ({crop_dims[0]}x{crop_dims[1]})"
        cv2.putText(rgb_scaled, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Side-by-side comparison
        comparison = np.hstack([rgb_scaled, ir_3ch])
        
        # Overlay
        overlay = cv2.addWeighted(rgb_scaled, 0.5, ir_3ch, 0.5, 0)
        
        cv2.imwrite(str(output_dir / f"zoom_scale_{scale:.2f}.jpg"), comparison)
        cv2.imwrite(str(output_dir / f"zoom_overlay_{scale:.2f}.jpg"), overlay)
        
        results.append((scale, rgb_scaled))
        print(f"Created: scale={scale:.2f}, crop={crop_dims}")
    
    # Create grid of all scales
    # 3 rows x 3 columns
    grid_rows = []
    for i in range(0, len(results), 3):
        row_images = [r[1] for r in results[i:i+3]]
        while len(row_images) < 3:
            row_images.append(np.zeros_like(row_images[0]))
        grid_rows.append(np.hstack(row_images))
    
    grid = np.vstack(grid_rows)
    cv2.imwrite(str(output_dir / "zoom_grid_all_scales.jpg"), grid)
    
    # Add IR reference
    ir_labeled = ir_3ch.copy()
    cv2.putText(ir_labeled, "IR Reference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(str(output_dir / "ir_reference.jpg"), ir_labeled)
    
    print(f"\n✓ Saved {len(scales)} zoom test images to: {output_dir}")
    return results


def main():
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    output_dir = Path("/home/student/Toan/results/zoom_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frame with visible reference objects (car)
    rgb_path = video_dir / "VID_EO_34.mp4"
    ir_path = video_dir / "VID_IR_34.mp4"
    
    frame_idx = 200  # Frame with car and people visible
    
    print("=" * 80)
    print("SYSTEMATIC ZOOM LEVEL TESTING")
    print("=" * 80)
    print(f"Frame: {frame_idx}")
    print("Testing scales: 0.5 to 1.0")
    print("Look for: Car and people should appear SAME SIZE in RGB and IR")
    print("=" * 80)
    
    rgb_frame = extract_frame(rgb_path, frame_idx)
    ir_frame = extract_frame(ir_path, frame_idx)
    
    if rgb_frame is None or ir_frame is None:
        print("ERROR: Could not extract frames")
        return
    
    create_comparison_grid(rgb_frame, ir_frame, output_dir)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("1. Review images in: " + str(output_dir))
    print("2. Find scale where CAR appears same size in RGB and IR")
    print("3. That scale = correct zoom level for alignment")
    print("=" * 80)


if __name__ == "__main__":
    main()
