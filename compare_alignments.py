#!/usr/bin/env python3
"""
Extract and compare frames from different alignment methods.
"""
import cv2
from pathlib import Path


def extract_frame_from_video(video_path, frame_idx=200):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def main():
    results_dir = Path("/home/student/Toan/results")
    output_dir = results_dir / "alignment_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get frames from both methods
    videos = [
        ("Parallax 0.5", results_dir / "video_inference_parallax/parallax_corrected_VID_EO_34.mp4"),
        ("Aspect-ratio", results_dir / "video_inference_aligned/mcf_VID_EO_34.mp4"),
    ]
    
    for name, path in videos:
        if path.exists():
            frame = extract_frame_from_video(path, 200)
            if frame is not None:
                output_path = output_dir / f"{name.replace(' ', '_')}_frame200.jpg"
                cv2.imwrite(str(output_path), frame)
                print(f"Saved: {output_path}")
    
    # Also extract from raw videos to show alignment regions
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    rgb_path = video_dir / "VID_EO_34.mp4"
    ir_path = video_dir / "VID_IR_34.mp4"
    
    rgb_frame = extract_frame_from_video(rgb_path, 200)
    ir_frame = extract_frame_from_video(ir_path, 200)
    
    if rgb_frame is not None and ir_frame is not None:
        h_rgb, w_rgb = rgb_frame.shape[:2]
        h_ir, w_ir = ir_frame.shape[:2]
        
        # Draw different crop regions on RGB
        marked = rgb_frame.copy()
        
        # 1. Scale 0.5 center crop (1350x1080)
        crop_h = 1080
        crop_w = 1350
        x1, y1 = (w_rgb - crop_w) // 2, (h_rgb - crop_h) // 2
        cv2.rectangle(marked, (x1, y1), (x1+crop_w, y1+crop_h), (0, 0, 255), 3)  # RED
        cv2.putText(marked, "Scale 0.5 (1350x1080)", (x1 + 10, y1 + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 2. Aspect-ratio crop (2700x2160)
        crop_h2 = h_rgb
        crop_w2 = int(h_rgb * (w_ir / h_ir))  # 2700
        x2, y2 = (w_rgb - crop_w2) // 2, 0
        cv2.rectangle(marked, (x2, y2), (x2+crop_w2, y2+crop_h2), (0, 255, 0), 3)  # GREEN
        cv2.putText(marked, "Aspect-ratio (2700x2160)", (x2 + 10, y2 + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 3. Full frame
        cv2.rectangle(marked, (0, 0), (w_rgb-1, h_rgb-1), (255, 255, 0), 3)  # CYAN
        cv2.putText(marked, "Full RGB (3840x2160)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Resize for display
        scale = 800 / h_rgb
        marked_small = cv2.resize(marked, None, fx=scale, fy=scale)
        cv2.imwrite(str(output_dir / "crop_regions_comparison.jpg"), marked_small)
        print(f"Saved: {output_dir / 'crop_regions_comparison.jpg'}")


if __name__ == "__main__":
    main()
