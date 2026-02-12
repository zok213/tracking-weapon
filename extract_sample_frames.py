#!/usr/bin/env python3
"""Extract sample frames from output videos using OpenCV."""
import cv2
from pathlib import Path


def extract_frame(video_path, frame_idx, output_path):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")
        return True
    return False


def main():
    output_dir = Path("/home/student/Toan/results/alignment_comparison")
    output_dir.mkdir(exist_ok=True)
    
    videos = [
        ("/home/student/Toan/results/video_inference_v6_corrected/corrected_VID_EO_34_xoff-100.mp4", 
         output_dir / "v6_corrected_frame.jpg", 250),
        ("/home/student/Toan/results/video_inference_v6_corrected/offset_tests/offset_+0.mp4",
         output_dir / "v6_offset0_frame.jpg", 50),
    ]
    
    for video_path, output_path, frame_idx in videos:
        if Path(video_path).exists():
            extract_frame(video_path, frame_idx, output_path)


if __name__ == "__main__":
    main()
