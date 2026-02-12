#!/usr/bin/env python3
"""
Advanced RGB-IR Alignment Analysis

The IR camera likely captures ONLY the CENTER portion of the RGB scene.
This script tests multiple alignment hypotheses:
1. IR = center 640x512 of RGB (1:1 pixel mapping)
2. IR = scaled center region of RGB
3. IR = offset center with specific scale factor

Creates visual comparisons to determine correct alignment.
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


def crop_center(img, crop_w, crop_h):
    """Crop center region of image."""
    h, w = img.shape[:2]
    x = (w - crop_w) // 2
    y = (h - crop_h) // 2
    return img[y:y+crop_h, x:x+crop_w]


def create_alignment_tests(rgb_frame, ir_frame, output_dir):
    """
    Test multiple alignment hypotheses and create visual comparisons.
    """
    h_rgb, w_rgb = rgb_frame.shape[:2]  # 3840x2160
    h_ir, w_ir = ir_frame.shape[:2]     # 640x512
    
    # Ensure IR is 3-channel
    if len(ir_frame.shape) == 2:
        ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    
    print(f"RGB: {w_rgb}x{h_rgb}")
    print(f"IR: {w_ir}x{h_ir}")
    
    results = []
    
    # HYPOTHESIS 1: IR = exact center 640x512 of RGB
    # (1:1 pixel mapping, IR sees a tiny portion of RGB)
    print("\n=== HYPOTHESIS 1: Center 640x512 crop (1:1 pixels) ===")
    crop1 = crop_center(rgb_frame, w_ir, h_ir)
    # Resize to compare (both at 640x512)
    compare1 = np.hstack([crop1, ir_frame])
    cv2.imwrite(str(output_dir / "test1_center_640x512.jpg"), compare1)
    results.append(("1:1 center crop", crop1, 640, 512))
    
    # HYPOTHESIS 2: IR = center with 2x zoom (crop 1280x1024, scale to 640x512)
    print("=== HYPOTHESIS 2: Center 1280x1024 scaled to 640x512 ===")
    crop2_raw = crop_center(rgb_frame, 1280, 1024)
    crop2 = cv2.resize(crop2_raw, (w_ir, h_ir))
    compare2 = np.hstack([crop2, ir_frame])
    cv2.imwrite(str(output_dir / "test2_center_1280x1024_scaled.jpg"), compare2)
    results.append(("2x zoom center", crop2, 1280, 1024))
    
    # HYPOTHESIS 3: IR = center with 3x zoom (crop 1920x1536, scale to 640x512)
    print("=== HYPOTHESIS 3: Center 1920x1536 scaled to 640x512 ===")
    crop3_raw = crop_center(rgb_frame, 1920, 1536)
    crop3 = cv2.resize(crop3_raw, (w_ir, h_ir))
    compare3 = np.hstack([crop3, ir_frame])
    cv2.imwrite(str(output_dir / "test3_center_1920x1536_scaled.jpg"), compare3)
    results.append(("3x zoom center", crop3, 1920, 1536))
    
    # HYPOTHESIS 4: IR = center with 4x zoom (crop 2560x2048, scale to 640x512)
    # Note: 2048 > 2160, so we use max height
    print("=== HYPOTHESIS 4: Center 2560x2048 scaled (adjusted for height) ===")
    crop4_w = min(2560, w_rgb)
    crop4_h = min(2048, h_rgb)
    crop4_raw = crop_center(rgb_frame, crop4_w, crop4_h)
    crop4 = cv2.resize(crop4_raw, (w_ir, h_ir))
    compare4 = np.hstack([crop4, ir_frame])
    cv2.imwrite(str(output_dir / "test4_center_2560x2048_scaled.jpg"), compare4)
    results.append(("4x zoom center", crop4, crop4_w, crop4_h))
    
    # HYPOTHESIS 5: Full RGB resized to 640x512 (no crop, just scale)
    print("=== HYPOTHESIS 5: Full RGB scaled to 640x512 ===")
    crop5 = cv2.resize(rgb_frame, (w_ir, h_ir))
    compare5 = np.hstack([crop5, ir_frame])
    cv2.imwrite(str(output_dir / "test5_full_rgb_scaled.jpg"), compare5)
    results.append(("full RGB scaled", crop5, w_rgb, h_rgb))
    
    # HYPOTHESIS 6: Center with aspect ratio match (2700x2160 -> 640x512)
    print("=== HYPOTHESIS 6: Aspect-matched center crop 2700x2160 ===")
    # Match IR aspect ratio 640:512 = 1.25
    target_aspect = w_ir / h_ir
    new_w = int(h_rgb * target_aspect)  # 2700
    crop6_raw = crop_center(rgb_frame, new_w, h_rgb)
    crop6 = cv2.resize(crop6_raw, (w_ir, h_ir))
    compare6 = np.hstack([crop6, ir_frame])
    cv2.imwrite(str(output_dir / "test6_aspect_matched_2700x2160.jpg"), compare6)
    results.append(("aspect-matched", crop6, new_w, h_rgb))
    
    # Create a summary grid
    print("\n=== Creating summary grid ===")
    grid_rows = []
    for i in range(0, len(results), 2):
        row_images = []
        for j in range(2):
            if i + j < len(results):
                name, img, _, _ = results[i + j]
                # Add label
                labeled = img.copy()
                cv2.putText(labeled, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                row_images.append(labeled)
        if row_images:
            grid_rows.append(np.hstack(row_images))
    
    # Pad rows to same width
    max_w = max(r.shape[1] for r in grid_rows)
    padded_rows = []
    for row in grid_rows:
        if row.shape[1] < max_w:
            pad = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8)
            row = np.hstack([row, pad])
        padded_rows.append(row)
    
    grid = np.vstack(padded_rows)
    
    # Add IR reference on the right
    ir_reference = ir_frame.copy()
    cv2.putText(ir_reference, "IR Reference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.imwrite(str(output_dir / "alignment_grid.jpg"), grid)
    cv2.imwrite(str(output_dir / "ir_reference.jpg"), ir_reference)
    
    print(f"\n✓ All test images saved to: {output_dir}")
    return results


def main():
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    output_dir = Path("/home/student/Toan/results/alignment_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test with video pair 34 (best results previously)
    rgb_path = video_dir / "VID_EO_34.mp4"
    ir_path = video_dir / "VID_IR_34.mp4"
    
    print("=" * 80)
    print("ADVANCED RGB-IR ALIGNMENT ANALYSIS")
    print("=" * 80)
    print(f"RGB: {rgb_path}")
    print(f"IR: {ir_path}")
    
    # Extract frames at same index
    frame_idx = 200  # Try different frame for better analysis
    rgb_frame = extract_frame(rgb_path, frame_idx)
    ir_frame = extract_frame(ir_path, frame_idx)
    
    if rgb_frame is None or ir_frame is None:
        print("ERROR: Could not extract frames!")
        return
    
    print(f"\nExtracted frame {frame_idx} from both videos")
    
    # Test different alignments
    results = create_alignment_tests(rgb_frame, ir_frame, output_dir)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Open the test images in: " + str(output_dir))
    print("2. Compare each RGB crop with the IR reference")
    print("3. Find which hypothesis shows matching scene content")
    print("4. Use that crop configuration for inference")


if __name__ == "__main__":
    main()
