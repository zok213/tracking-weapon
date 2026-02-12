#!/usr/bin/env python3
"""
Manual Interactive Alignment Test

Creates multiple crop variations with different offsets
to find the correct RGB region that matches IR.
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


def test_offset(rgb_frame, ir_frame, x_offset, y_offset, crop_w, crop_h, output_path):
    """Test a specific offset and create comparison."""
    h_rgb, w_rgb = rgb_frame.shape[:2]
    h_ir, w_ir = ir_frame.shape[:2]
    
    # Calculate crop position
    x = (w_rgb - crop_w) // 2 + x_offset
    y = (h_rgb - crop_h) // 2 + y_offset
    
    # Bounds
    x = max(0, min(x, w_rgb - crop_w))
    y = max(0, min(y, h_rgb - crop_h))
    
    # Crop and resize
    rgb_crop = rgb_frame[y:y+crop_h, x:x+crop_w]
    rgb_resized = cv2.resize(rgb_crop, (w_ir, h_ir))
    
    # Ensure IR is 3-channel
    if len(ir_frame.shape) == 2:
        ir_3ch = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:
        ir_3ch = ir_frame
    
    # Add labels
    label = f"x_off={x_offset}, y_off={y_offset}"
    cv2.putText(rgb_resized, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(ir_3ch, "IR Reference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Create comparison
    comparison = np.hstack([rgb_resized, ir_3ch])
    
    # Overlay
    overlay = cv2.addWeighted(rgb_resized, 0.5, ir_3ch, 0.5, 0)
    
    cv2.imwrite(str(output_path.with_suffix('.jpg')), comparison)
    cv2.imwrite(str(output_path.with_name(output_path.stem + '_overlay.jpg')), overlay)
    
    return comparison


def main():
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    output_dir = Path("/home/student/Toan/results/alignment_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rgb_path = video_dir / "VID_EO_34.mp4"
    ir_path = video_dir / "VID_IR_34.mp4"
    
    # Extract same frame
    frame_idx = 200
    rgb_frame = extract_frame(rgb_path, frame_idx)
    ir_frame = extract_frame(ir_path, frame_idx)
    
    h_rgb, w_rgb = rgb_frame.shape[:2]  # 3840x2160
    h_ir, w_ir = ir_frame.shape[:2]      # 640x512
    
    print("=" * 80)
    print("MANUAL ALIGNMENT TEST GRID")
    print("=" * 80)
    print(f"RGB: {w_rgb}x{h_rgb}")
    print(f"IR: {w_ir}x{h_ir}")
    
    # Match IR aspect ratio: 640/512 = 1.25
    # Crop height = full RGB height, crop width = height * 1.25
    crop_h = h_rgb  # 2160
    crop_w = int(crop_h * (w_ir / h_ir))  # 2700
    
    print(f"\nCrop size: {crop_w}x{crop_h}")
    
    # Test wide range of offsets
    # X can range from -(w_rgb-crop_w)/2 to +(w_rgb-crop_w)/2
    # Y can range from -(h_rgb-crop_h)/2 to +(h_rgb-crop_h)/2
    max_x_offset = (w_rgb - crop_w) // 2
    max_y_offset = (h_rgb - crop_h) // 2  # This is 0 since crop_h = h_rgb
    
    print(f"Max X offset: ±{max_x_offset}")
    print(f"Max Y offset: ±{max_y_offset}")
    
    # Test grid of offsets
    all_comparisons = []
    
    for x_pct in [-1.0, -0.5, 0, 0.5, 1.0]:
        row = []
        x_offset = int(max_x_offset * x_pct)
        
        for y_abs in [0]:  # Since max_y_offset is 0
            name = f"x{x_pct:+.1f}_y{y_abs}"
            output_path = output_dir / f"test_{name}"
            
            comparison = test_offset(rgb_frame, ir_frame, x_offset, y_abs, 
                                     crop_w, crop_h, output_path)
            row.append(comparison)
            print(f"  Created: {name}")
    
    # Create single row grid since y_offset is 0
    # Test smaller crop sizes to find zoom level
    print("\n--- Testing different crop scales ---")
    
    for scale in [0.25, 0.5, 0.75, 1.0]:
        scaled_crop_w = int(crop_w * scale)
        scaled_crop_h = int(crop_h * scale)
        
        if scaled_crop_w > w_rgb or scaled_crop_h > h_rgb:
            continue
        
        max_x = (w_rgb - scaled_crop_w) // 2
        max_y = (h_rgb - scaled_crop_h) // 2
        
        for x_pct in [-0.5, 0, 0.5]:
            x_offset = int(max_x * x_pct)
            for y_pct in [-0.5, 0, 0.5]:
                y_offset = int(max_y * y_pct)
                
                name = f"scale{scale}_x{x_pct:+.1f}_y{y_pct:+.1f}"
                output_path = output_dir / f"test_{name}"
                
                test_offset(rgb_frame, ir_frame, x_offset, y_offset,
                           scaled_crop_w, scaled_crop_h, output_path)
                print(f"  Created: {name}")
    
    print(f"\n✓ All test images saved to: {output_dir}")
    print("Review images to find best alignment, then update inference script.")


if __name__ == "__main__":
    main()
