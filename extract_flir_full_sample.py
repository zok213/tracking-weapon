#!/usr/bin/env python3
"""Extract sample frame from FLIR Heuristic Full Video (Conf 0.10)."""
import cv2
from pathlib import Path

def main():
    video_path = Path("/home/student/Toan/results/flir_heuristic_fix/FLIR_heuristic_full_frame200.mp4")
    output_path = Path("/home/student/Toan/results/sample_flir_full_heuristic.jpg")
    
    cap = cv2.VideoCapture(str(video_path))
    # Target original Frame 451.
    # Video starts at 200. So 451 is index 251.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 251) 
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
