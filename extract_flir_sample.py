#!/usr/bin/env python3
"""Extract sample from FLIR Person-Only video."""
import cv2
from pathlib import Path

def main():
    video_path = Path("/home/student/Toan/results/flir_person_only/FLIR_person_only.mp4")
    output_path = Path("/home/student/Toan/results/sample_flir_person.jpg")
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 450) # The frame where Custom detected 4 people
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
