#!/usr/bin/env python3
"""Extract sample frame from FLIR Heuristic Fix video."""
import cv2
from pathlib import Path

def main():
    video_path = Path("/home/student/Toan/results/flir_heuristic_fix/FLIR_heuristic_shift_30pct.mp4")
    output_path = Path("/home/student/Toan/results/sample_flir_heuristic_fix.jpg")
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150) # Approx frame
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
