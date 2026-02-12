#!/usr/bin/env python3
"""Extract sample from specific alignment video."""
import cv2
from pathlib import Path

def main():
    video_path = Path("/home/student/Toan/results/video_040_dx15_dy61_overlay.mp4")
    output_path = Path("/home/student/Toan/results/sample_040_dx15_dy61.jpg")
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200) # Car/Person visible area
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
