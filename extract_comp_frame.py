#!/usr/bin/env python3
"""Extract sample frame from comparison video."""
import cv2
from pathlib import Path

def main():
    video_path = Path("/home/student/Toan/results/comparison_0.50_vs_0.66.mp4")
    output_path = Path("/home/student/Toan/results/comparison_frame.jpg")
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200) # Frame with car visible
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
