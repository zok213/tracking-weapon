#!/usr/bin/env python3
"""Extract sample frame from FLIR Unfiltered video."""
import cv2
from pathlib import Path

def main():
    video_path = Path("/home/student/Toan/results/flir_unfiltered/FLIR_unfiltered.mp4")
    output_path = Path("/home/student/Toan/results/sample_flir_unfiltered.jpg")
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150) # Input Frame ~150 (actually 150 of 500 processed)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
