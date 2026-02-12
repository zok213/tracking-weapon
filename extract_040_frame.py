#!/usr/bin/env python3
"""Extract sample frame from 0.40-0.42 video."""
import cv2
from pathlib import Path

def main():
    video_path = Path("/home/student/Toan/results/video_scale_040_042.mp4")
    output_path = Path("/home/student/Toan/results/sample_040_042.jpg")
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 250) # Car/Person visible
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
