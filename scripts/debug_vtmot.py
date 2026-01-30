import cv2
import numpy as np
from pathlib import Path

def debug_vtmot():
    # Pick a sample that failed (or likely exists)
    rgb_path = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/train/photo-0306-01/visible/000001.jpg")
    tir_path = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/train/photo-0306-01/infrared/000001.jpg")
    
    print(f"Testing pair:\nRGB: {rgb_path}\nTIR: {tir_path}")
    
    if not rgb_path.exists():
        print("❌ RGB path does not exist")
        # Try finding ANY jpg
        web_path = list(Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/train").rglob("*.jpg"))[0]
        print(f"Found existing: {web_path}")
    
    rgb = cv2.imread(str(rgb_path))
    print(f"RGB read: {'Success' if rgb is not None else 'Failed'}")
    if rgb is not None: print(f"RGB shape: {rgb.shape}")
    
    tir = cv2.imread(str(tir_path), cv2.IMREAD_GRAYSCALE)
    print(f"TIR read: {'Success' if tir is not None else 'Failed'}")
    if tir is not None: print(f"TIR shape: {tir.shape}")

if __name__ == '__main__':
    debug_vtmot()
