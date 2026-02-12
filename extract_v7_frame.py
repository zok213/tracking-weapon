#!/usr/bin/env python3
"""Extract sample frame from v7 output for viewing."""
import cv2
from pathlib import Path


def main():
    video_path = Path("/home/student/Toan/results/video_inference_v7_final/final_VID_EO_34_scale0.66.mp4")
    output_path = Path("/home/student/Toan/results/alignment_comparison/v7_scale066_sample.jpg")
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 250)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
