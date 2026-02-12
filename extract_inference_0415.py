#!/usr/bin/env python3
"""Extract sample frame from Scale 0.415 inference output."""
import cv2
from pathlib import Path

def main():
    video_path = Path("/home/student/Toan/results/video_inference_scale0415/det_VID_EO_34_0415.mp4")
    output_path = Path("/home/student/Toan/results/sample_inference_0415.jpg")
    
    cap = cv2.VideoCapture(str(video_path))
    # Frame 250 (Action area)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 250) 
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
