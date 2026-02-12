#!/usr/bin/env python3
"""Check FLIR model class names."""
import sys
sys.path.insert(0, "/home/student/Toan/YOLOv11-RGBT")
from ultralytics import YOLO

def main():
    path = "/home/student/Toan/weights/FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.01-warmup_epochs1-SGD.pt"
    model = YOLO(path)
    print("Class Names:")
    print(model.names)

if __name__ == "__main__":
    main()
