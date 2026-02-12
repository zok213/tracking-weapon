#!/usr/bin/env python3
"""
Feature-Based RGB-IR Alignment

The RGB and IR cameras appear to have different optical axes.
This script uses feature matching to find the correct homography
transformation between RGB and IR frames.

Also includes manual offset finding for debugging.
"""
import cv2
import numpy as np
from pathlib import Path


def extract_frame(video_path, frame_idx=100):
    """Extract a frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def try_feature_matching(rgb_frame, ir_frame):
    """
    Attempt feature-based alignment using ORB.
    This may not work well between RGB and thermal due to different modalities.
    """
    # Convert to grayscale
    rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    if len(ir_frame.shape) == 3:
        ir_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
    else:
        ir_gray = ir_frame
    
    # Resize RGB to roughly match IR for feature detection
    scale = ir_gray.shape[0] / rgb_gray.shape[0]
    rgb_scaled = cv2.resize(rgb_gray, None, fx=scale, fy=scale)
    
    # ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(rgb_scaled, None)
    kp2, des2 = orb.detectAndCompute(ir_gray, None)
    
    if des1 is None or des2 is None:
        print("  No features detected!")
        return None
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    print(f"  Found {len(matches)} matches")
    
    if len(matches) < 4:
        return None
    
    # Get matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H


def analyze_structural_alignment(rgb_frame, ir_frame, output_dir):
    """
    Analyze structural similarities to determine alignment.
    Uses edge detection for cross-modal matching.
    """
    h_rgb, w_rgb = rgb_frame.shape[:2]
    h_ir, w_ir = ir_frame.shape[:2]
    
    print(f"\nStructural Analysis:")
    print(f"  RGB: {w_rgb}x{h_rgb}")
    print(f"  IR: {w_ir}x{h_ir}")
    
    # Edge detection on both
    rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    if len(ir_frame.shape) == 3:
        ir_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
    else:
        ir_gray = ir_frame
    
    rgb_edges = cv2.Canny(rgb_gray, 50, 150)
    ir_edges = cv2.Canny(ir_gray, 50, 150)
    
    # Save edge images
    cv2.imwrite(str(output_dir / "rgb_edges.jpg"), rgb_edges)
    cv2.imwrite(str(output_dir / "ir_edges.jpg"), ir_edges)
    
    # Try different crop regions and find best correlation
    # Scale factor: RGB is 6x larger than IR in width
    scale_factor = w_rgb / w_ir  # 6.0
    
    print(f"\n  Scale factor RGB/IR: {scale_factor:.2f}")
    
    # The IR view corresponds to some region of RGB
    # Crop size in RGB pixels that maps to IR size
    crop_w = int(w_ir * scale_factor)  # Could be various values
    crop_h = int(h_ir * scale_factor)
    
    print(f"  If 1:1 scale, RGB crop should be: {crop_w}x{crop_h}")
    
    # But cameras likely have different focal lengths
    # Let's test multiple scale factors
    
    best_score = -1
    best_params = None
    
    # Test different crop sizes (different assumed focal length relationships)
    for crop_scale in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        # Crop dimensions in RGB space
        test_crop_w = int(w_ir * scale_factor * crop_scale)
        test_crop_h = int(h_ir * scale_factor * crop_scale)
        
        if test_crop_w > w_rgb or test_crop_h > h_rgb:
            continue
        if test_crop_w < 100 or test_crop_h < 100:
            continue
        
        # Test different offsets
        for x_offset_pct in [-0.2, -0.1, 0, 0.1, 0.2]:
            for y_offset_pct in [-0.2, -0.1, 0, 0.1, 0.2]:
                # Calculate crop position
                x = int((w_rgb - test_crop_w) / 2 + w_rgb * x_offset_pct)
                y = int((h_rgb - test_crop_h) / 2 + h_rgb * y_offset_pct)
                
                # Bounds check
                x = max(0, min(x, w_rgb - test_crop_w))
                y = max(0, min(y, h_rgb - test_crop_h))
                
                # Crop and resize
                rgb_crop = rgb_gray[y:y+test_crop_h, x:x+test_crop_w]
                rgb_resized = cv2.resize(rgb_crop, (w_ir, h_ir))
                
                # Compare edges
                rgb_crop_edges = cv2.Canny(rgb_resized, 50, 150)
                
                # Correlation score
                score = np.sum(rgb_crop_edges & ir_edges) / (np.sum(ir_edges) + 1)
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'crop_scale': crop_scale,
                        'x_offset_pct': x_offset_pct,
                        'y_offset_pct': y_offset_pct,
                        'x': x, 'y': y,
                        'crop_w': test_crop_w, 'crop_h': test_crop_h
                    }
    
    print(f"\n  Best alignment parameters:")
    print(f"    Crop scale: {best_params['crop_scale']}")
    print(f"    X offset: {best_params['x_offset_pct']*100:.0f}%")
    print(f"    Y offset: {best_params['y_offset_pct']*100:.0f}%")
    print(f"    Crop region: x={best_params['x']}, y={best_params['y']}")
    print(f"    Crop size: {best_params['crop_w']}x{best_params['crop_h']}")
    print(f"    Match score: {best_score:.4f}")
    
    # Create visualization of best alignment
    x, y = best_params['x'], best_params['y']
    crop_w, crop_h = best_params['crop_w'], best_params['crop_h']
    
    rgb_crop = rgb_frame[y:y+crop_h, x:x+crop_w]
    rgb_aligned = cv2.resize(rgb_crop, (w_ir, h_ir))
    
    if len(ir_frame.shape) == 2:
        ir_3ch = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:
        ir_3ch = ir_frame
    
    # Overlay for comparison
    overlay = cv2.addWeighted(rgb_aligned, 0.5, ir_3ch, 0.5, 0)
    
    cv2.imwrite(str(output_dir / "best_alignment_rgb.jpg"), rgb_aligned)
    cv2.imwrite(str(output_dir / "best_alignment_overlay.jpg"), overlay)
    
    # Side by side
    sidebyside = np.hstack([rgb_aligned, ir_3ch])
    cv2.imwrite(str(output_dir / "best_alignment_comparison.jpg"), sidebyside)
    
    return best_params


def manual_alignment_grid(rgb_frame, ir_frame, output_dir):
    """
    Create a grid of different crop positions for manual inspection.
    """
    h_rgb, w_rgb = rgb_frame.shape[:2]
    h_ir, w_ir = ir_frame.shape[:2]
    
    if len(ir_frame.shape) == 2:
        ir_3ch = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:
        ir_3ch = ir_frame
    
    # Test 9 different positions (3x3 grid)
    results = []
    
    # Use full width crop with IR aspect ratio
    crop_h = h_rgb
    crop_w = int(crop_h * (w_ir / h_ir))
    
    for x_pct in [0.0, 0.5, 1.0]:  # Left, Center, Right
        for y_pct in [0.0, 0.5, 1.0]:  # Top, Center, Bottom
            x = int((w_rgb - crop_w) * x_pct)
            y = int((h_rgb - crop_h) * y_pct)
            
            x = max(0, min(x, w_rgb - crop_w))
            y = max(0, min(y, h_rgb - crop_h))
            
            rgb_crop = rgb_frame[y:y+crop_h, x:x+crop_w]
            rgb_resized = cv2.resize(rgb_crop, (w_ir, h_ir))
            
            label = f"x={x_pct:.1f},y={y_pct:.1f}"
            cv2.putText(rgb_resized, label, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            results.append(rgb_resized)
    
    # Create 3x3 grid
    grid = []
    for i in range(3):
        row = np.hstack(results[i*3:(i+1)*3])
        grid.append(row)
    grid = np.vstack(grid)
    
    cv2.imwrite(str(output_dir / "position_grid.jpg"), grid)
    
    # Add IR reference
    cv2.putText(ir_3ch, "IR Reference", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(str(output_dir / "ir_for_comparison.jpg"), ir_3ch)
    
    print(f"\n  Position grid saved: position_grid.jpg")


def main():
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    output_dir = Path("/home/student/Toan/results/alignment_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rgb_path = video_dir / "VID_EO_34.mp4"
    ir_path = video_dir / "VID_IR_34.mp4"
    
    print("=" * 80)
    print("FEATURE-BASED RGB-IR ALIGNMENT ANALYSIS")
    print("=" * 80)
    
    frame_idx = 200
    rgb_frame = extract_frame(rgb_path, frame_idx)
    ir_frame = extract_frame(ir_path, frame_idx)
    
    if rgb_frame is None or ir_frame is None:
        print("ERROR: Could not extract frames!")
        return
    
    # Try feature matching (unlikely to work well for RGB-IR)
    print("\n1. Attempting feature-based matching...")
    H = try_feature_matching(rgb_frame, ir_frame)
    if H is not None:
        print("  Found homography matrix!")
        print(H)
    else:
        print("  Feature matching failed (expected for cross-modal)")
    
    # Structural alignment analysis
    print("\n2. Structural alignment analysis...")
    best_params = analyze_structural_alignment(rgb_frame, ir_frame, output_dir)
    
    # Create position grid for manual verification
    print("\n3. Creating position grid for manual verification...")
    manual_alignment_grid(rgb_frame, ir_frame, output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nBest found parameters:")
    print(f"  Crop: x={best_params['x']}, y={best_params['y']}")
    print(f"  Size: {best_params['crop_w']}x{best_params['crop_h']}")
    print(f"\nReview images in: {output_dir}")
    print("  - best_alignment_comparison.jpg")
    print("  - best_alignment_overlay.jpg")
    print("  - position_grid.jpg")


if __name__ == "__main__":
    main()
