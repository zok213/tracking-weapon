#!/usr/bin/env python3
"""Extract sample frames from LLVIP and M3FD benchmarks."""
import cv2
from pathlib import Path

def extract(video_path, out_path):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150) # Action area
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(str(out_path), frame)
        print(f"Saved: {out_path}")

def main():
    root = Path("/home/student/Toan/results/pretrained_bench_0415")
    
    extract(root / "det_VID_EO_34_LLVIP.mp4", root / "sample_LLVIP.jpg")
    extract(root / "det_VID_EO_34_M3FD.mp4", root / "sample_M3FD.jpg")

if __name__ == "__main__":
    main()
