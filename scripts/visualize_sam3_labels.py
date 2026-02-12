import cv2
import os
import glob
from tqdm import tqdm
import argparse
from pathlib import Path

# Configuration
COLORS = {
    0: (0, 255, 0),    # Motorcycle: Green
    1: (255, 0, 0),    # Car: Blue
    2: (0, 0, 255),    # Truck: Red
    3: (0, 255, 255)   # Human: Yellow
}
CLASS_NAMES = {
    0: "Motorcycle",
    1: "Car",
    2: "Truck",
    3: "Human"
}

def visualize_sequence(seq_name, output_dir, label_root, vt_mot_root):
    # Find image directory
    # Try train, test, val
    img_dir = None
    for split in ["train", "test", "val"]:
        p = Path(vt_mot_root) / "images" / split / seq_name / "visible"
        if p.exists():
            img_dir = p
            break
        p = Path(vt_mot_root) / "images" / split / seq_name / "img1"
        if p.exists():
            img_dir = p
            break
            
    if not img_dir:
        print(f"Error: Could not find images for {seq_name}")
        return

    # Label directory
    lbl_dir = Path(label_root) / seq_name
    if not lbl_dir.exists():
        print(f"Error: No labels found for {seq_name} at {lbl_dir}")
        return

    # Get images
    images = sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")))
    if not images:
        print("No images found")
        return

    # Setup Video Writer
    h, w = cv2.imread(images[0]).shape[:2]
    out_path = os.path.join(output_dir, f"{seq_name}_sam3.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))

    print(f"Processing {seq_name} ({len(images)} frames) -> {out_path}")

    for img_path in tqdm(images):
        frame = cv2.imread(img_path)
        
        # Read label
        fid = int(Path(img_path).stem) # 000001 -> 1
        # Label files might be 000001.txt or 1.txt? 
        # My script generated 000001.txt matching image name.
        lbl_path = lbl_dir / f"{Path(img_path).stem}.txt"
        
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                
                # De-normalize
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                color = COLORS.get(cls_id, (255, 255, 255))
                label = CLASS_NAMES.get(cls_id, str(cls_id))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out.write(frame)

    out.release()
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, required=True, help="Sequence name")
    args = parser.parse_args()
    
    VT_MOT_ROOT = "/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT"
    LABEL_ROOT = "/home/student/Toan/data/VT-MOT_4cls_SAM3/labels"
    OUTPUT_DIR = "/home/student/Toan/data/VT-MOT_4cls_SAM3/viz"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    visualize_sequence(args.seq, OUTPUT_DIR, LABEL_ROOT, VT_MOT_ROOT)
