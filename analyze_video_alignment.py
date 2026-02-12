#!/usr/bin/env python3
"""
Analyze RGB and IR video pairs to determine alignment requirements.
Extracts video properties and sample frames for visual comparison.
"""
import cv2
import os
import numpy as np
from pathlib import Path


def analyze_video(video_path):
    """Get video properties."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    props = {
        'path': str(video_path),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }
    cap.release()
    return props


def extract_sample_frame(video_path, frame_idx=100):
    """Extract a sample frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def main():
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    output_dir = Path("/home/student/Toan/results/alignment_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_pairs = [
        ("VID_EO_34.mp4", "VID_IR_34.mp4"),
        ("VID_EO_35.mp4", "VID_IR_35.mp4"),
        ("VID_EO_36.mp4", "VID_IR_36.mp4"),
    ]
    
    print("=" * 80)
    print("RGB-IR VIDEO ALIGNMENT ANALYSIS")
    print("=" * 80)
    
    for rgb_name, ir_name in video_pairs:
        rgb_path = video_dir / rgb_name
        ir_path = video_dir / ir_name
        
        print(f"\n{'='*80}")
        print(f"Pair: {rgb_name} + {ir_name}")
        print("=" * 80)
        
        # Analyze videos
        rgb_props = analyze_video(rgb_path)
        ir_props = analyze_video(ir_path)
        
        if not rgb_props or not ir_props:
            print("ERROR: Could not analyze videos")
            continue
        
        print(f"\nRGB Video:")
        print(f"  Resolution: {rgb_props['width']} x {rgb_props['height']}")
        print(f"  FPS: {rgb_props['fps']:.1f}")
        print(f"  Frames: {rgb_props['frames']}")
        print(f"  Duration: {rgb_props['duration']:.1f}s")
        
        print(f"\nIR Video:")
        print(f"  Resolution: {ir_props['width']} x {ir_props['height']}")
        print(f"  FPS: {ir_props['fps']:.1f}")
        print(f"  Frames: {ir_props['frames']}")
        print(f"  Duration: {ir_props['duration']:.1f}s")
        
        # Calculate aspect ratios
        rgb_aspect = rgb_props['width'] / rgb_props['height']
        ir_aspect = ir_props['width'] / ir_props['height']
        
        print(f"\nAspect Ratios:")
        print(f"  RGB: {rgb_aspect:.3f} ({rgb_props['width']}:{rgb_props['height']})")
        print(f"  IR: {ir_aspect:.3f} ({ir_props['width']}:{ir_props['height']})")
        
        # Calculate crop needed to match IR FOV
        # Assuming IR has narrower FOV, we need to crop RGB center
        print(f"\n--- ALIGNMENT RECOMMENDATION ---")
        
        if rgb_props['width'] > ir_props['width'] or rgb_props['height'] > ir_props['height']:
            # Calculate center crop for RGB to match IR aspect ratio
            target_aspect = ir_aspect
            
            if rgb_aspect > target_aspect:
                # RGB is wider, crop width
                new_width = int(rgb_props['height'] * target_aspect)
                crop_x = (rgb_props['width'] - new_width) // 2
                crop_y = 0
                crop_w = new_width
                crop_h = rgb_props['height']
            else:
                # RGB is taller, crop height
                new_height = int(rgb_props['width'] / target_aspect)
                crop_x = 0
                crop_y = (rgb_props['height'] - new_height) // 2
                crop_w = rgb_props['width']
                crop_h = new_height
            
            print(f"  RGB needs CENTER CROP:")
            print(f"    Original: {rgb_props['width']}x{rgb_props['height']}")
            print(f"    Crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
            print(f"    Final size: {crop_w}x{crop_h}")
        
        # Extract and save sample frames for visual comparison
        rgb_frame = extract_sample_frame(rgb_path, 100)
        ir_frame = extract_sample_frame(ir_path, 100)
        
        if rgb_frame is not None and ir_frame is not None:
            # Save original frames
            cv2.imwrite(str(output_dir / f"{rgb_name.replace('.mp4', '_original.jpg')}"), rgb_frame)
            cv2.imwrite(str(output_dir / f"{ir_name.replace('.mp4', '_original.jpg')}"), ir_frame)
            
            # Create side-by-side comparison
            h_rgb, w_rgb = rgb_frame.shape[:2]
            h_ir, w_ir = ir_frame.shape[:2]
            
            # Resize both to same height for comparison
            target_h = 360
            rgb_resized = cv2.resize(rgb_frame, (int(w_rgb * target_h / h_rgb), target_h))
            
            if len(ir_frame.shape) == 2:
                ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
            ir_resized = cv2.resize(ir_frame, (int(w_ir * target_h / h_ir), target_h))
            
            # Add labels
            cv2.putText(rgb_resized, f"RGB {w_rgb}x{h_rgb}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(ir_resized, f"IR {w_ir}x{h_ir}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Combine
            combined = np.hstack([rgb_resized, ir_resized])
            cv2.imwrite(str(output_dir / f"comparison_{rgb_name.replace('.mp4', '.jpg')}"), combined)
            print(f"\n  Sample frames saved to: {output_dir}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
