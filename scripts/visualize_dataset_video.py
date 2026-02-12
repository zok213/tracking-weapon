import cv2
import numpy as np
from pathlib import Path
import random
import os
from tqdm import tqdm

# Config
DATA_ROOT = Path("/home/student/Toan/data/VT-MOT_Person_Only")
IMG_DIR = DATA_ROOT / "images/train"
IR_DIR = DATA_ROOT / "images_ir/train"
LBL_DIR = DATA_ROOT / "labels/train"
OUTPUT_VIDEO = "/home/student/Toan/dataset_verification_video.mp4"
NUM_FRAMES = 300 # Visualize 10 seconds (30fps)

def draw_boxes(img, label_path, color=(0, 255, 0)):
    if not label_path.exists():
        return img
    
    h, w = img.shape[:2]
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            
            # Denormalize
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"ID:{cls}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def main():
    # 1. Find a Sequence
    # Get all images
    all_imgs = sorted(list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png")))
    if not all_imgs:
        print("No images found!")
        return

    # Group by sequence name (prefix before _XXXXXX)
    # e.g. photo-0306-01_000001.jpg -> photo-0306-01
    sequences = {}
    for p in all_imgs:
        stem = p.stem
        seq_name = "_".join(stem.split("_")[:-1])
        if seq_name not in sequences:
            sequences[seq_name] = []
        sequences[seq_name].append(p)
        
    # Pick a random sequence with enough frames
    valid_seqs = [s for s in sequences.keys() if len(sequences[s]) > 50]
    if not valid_seqs:
        seq_name = list(sequences.keys())[0] # Fallback
    else:
        seq_name = random.choice(valid_seqs)
        
    print(f"Visualizing Sequence: {seq_name}")
    frames = sorted(sequences[seq_name])[:NUM_FRAMES]
    
    # 2. Setup Video Writer
    sample_img = cv2.imread(str(frames[0]))
    h, w = sample_img.shape[:2]
    # Output: RGB | IR -> Width * 2
    out_w = w * 2
    out_h = h
    fps = 30
    
    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    
    # 3. Process
    for img_path in tqdm(frames, desc="Generating Video"):
        # Paths
        fname = img_path.name
        ir_path = IR_DIR / fname
        lbl_path = LBL_DIR / img_path.with_suffix(".txt").name
        
        # Load Images
        rgb = cv2.imread(str(img_path))
        ir = cv2.imread(str(ir_path)) # Load as 3ch for stacking
        
        if rgb is None or ir is None:
            continue
            
        # Draw GT
        rgb = draw_boxes(rgb, lbl_path, color=(0, 255, 0)) # Green on RGB
        ir = draw_boxes(ir, lbl_path, color=(0, 0, 255))   # Red on Thermal
        
        # Add labels "RGB" / "Thermal"
        cv2.putText(rgb, "RGB (GT: Green)", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(ir, "Thermal (GT: Red)", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Stack
        frame = np.hstack([rgb, ir])
        writer.write(frame)
        
    writer.release()
    print(f"Video saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
