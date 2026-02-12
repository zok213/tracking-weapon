#!/usr/bin/env python3
"""Extract sample frame from 3-model comparison grid."""
import cv2
from pathlib import Path

def main():
    video_path = Path("/home/student/Toan/results/prediction_grid_3models/models_comparison_grid.mp4")
    output_path = Path("/home/student/Toan/results/sample_3models_grid.jpg")
    
    cap = cv2.VideoCapture(str(video_path))
    # Frame 150 of the *output video* (which started at original frame 300)
    # So this is roughly original frame 450
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150) 
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
