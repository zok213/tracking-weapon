#!/usr/bin/env python3
"""Extract sample frame from final comparison grid."""
import cv2
from pathlib import Path

def main():
    video_path = Path("/home/student/Toan/results/final_comparison_0415/comparison_grid_0415.mp4")
    output_path = Path("/home/student/Toan/results/sample_final_grid_0415.jpg")
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150) # Action
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
