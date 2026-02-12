
import os
import shutil
import glob
import cv2
import yaml
import random
from pathlib import Path
from tqdm import tqdm

# Config
SOURCE_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images") # We found 'test' here. 
# We will scan BOTH train and test if they exist.
OUTPUT_ROOT = Path("/home/student/Toan/data/VT-MOT_Visible_Person_Drone")
KEYWORDS = ["qiuxing", "photo", "lasher", "RGBT"]
TARGET_CLASS_ID = 1 # Person in GT
YOLO_CLASS_ID = 0   # Person in YOLO
SPLIT_RATIO = 0.8 # 80% Train

def create_dataset():
    if OUTPUT_ROOT.exists():
        print(f"⚠️ Removing existing output: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)
    
    # Create Dirs
    for split in ["train", "val"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Gather Sequences
    sequences = []
    # Search in both 'train' and 'test' folders if they exist
    for sub in ["train", "test"]:
        p = SOURCE_ROOT / sub
        if not p.exists(): continue
        
        for seq_path in p.iterdir():
            if not seq_path.is_dir(): continue
            if any(k in seq_path.name for k in KEYWORDS):
                # Verify content
                rgb_path = seq_path / "rgb"
                gt_path = seq_path / "gt" / "gt.txt"
                if rgb_path.exists() and gt_path.exists():
                    sequences.append(seq_path)

    print(f"✅ Found {len(sequences)} matching sequences.")
    random.shuffle(sequences)

    # 2. Process
    split_idx = int(len(sequences) * SPLIT_RATIO)
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]
    
    global_img_count = 0

    def process_split(seq_list, split_name):
        nonlocal global_img_count
        print(f"Processing {split_name}: {len(seq_list)} sequences...")
        
        for seq in tqdm(seq_list):
            rgb_dir = seq / "rgb"
            gt_file = seq / "gt" / "gt.txt"
            
            # Read first image to get dims
            first_img_path = next(rgb_dir.glob("*.jpg"))
            img = cv2.imread(str(first_img_path))
            if img is None: continue
            img_h, img_w = img.shape[:2]
            
            # Load GT
            # Format: frame, id, x, y, w, h, conf, cls, vis
            with open(gt_file, 'r') as f:
                lines = f.readlines()
            
            # Group by frame
            frame_anns = {}
            for line in lines:
                parts = line.strip().split(',')
                try:
                    frame_idx = int(parts[0])
                    cls_id = int(parts[7])
                    
                    if cls_id == TARGET_CLASS_ID:
                        # Convert to YOLO
                        x1 = float(parts[2])
                        y1 = float(parts[3])
                        w = float(parts[4])
                        h = float(parts[5])
                        
                        cx = (x1 + w / 2) / img_w
                        cy = (y1 + h / 2) / img_h
                        nw = w / img_w
                        nh = h / img_h
                        
                        # Clip
                        cx = max(0, min(1, cx))
                        cy = max(0, min(1, cy))
                        nw = max(0, min(1, nw))
                        nh = max(0, min(1, nh))
                        
                        if frame_idx not in frame_anns: frame_anns[frame_idx] = []
                        frame_anns[frame_idx].append(f"{YOLO_CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                except:
                    continue
            
            # Copy Files
            # Frame names are typically 000001.jpg etc.
            # Filename in rgb dir might not match frame index strictly?
            # VisDrone usually matches. Let's assume standard integer names.
            
            image_files = sorted(list(rgb_dir.glob("*.jpg")))
            for img_path in image_files:
                try:
                    # Parse frame ID from filename "0000123.jpg" -> 123
                    frame_id = int(img_path.stem)
                except:
                    continue
                
                # Check if we have annotations?
                # YOLO allows empty label files for bg.
                # But for speed, maybe skip empty?
                # User said "detect only class 1 human".
                # Standard practice: Include background frames to reduce FP?
                # Let's keep ALL frames, but only write label file if ann exists.
                # If no label file, YOLO treats as empty (bg).
                
                # Copy Image
                out_name = f"{seq.name}_{img_path.name}"
                out_img_path = OUTPUT_ROOT / "images" / split_name / out_name
                shutil.copy(img_path, out_img_path)
                
                # Write Label
                out_lbl_path = OUTPUT_ROOT / "labels" / split_name / out_name.replace('.jpg', '.txt')
                if frame_id in frame_anns:
                    with open(out_lbl_path, 'w') as lf:
                        lf.write('\n'.join(frame_anns[frame_id]))
                
                global_img_count += 1

    process_split(train_seqs, "train")
    process_split(val_seqs, "val")
    
    # 3. Create YAML
    data_yaml = {
        'path': str(OUTPUT_ROOT),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'person'}
    }
    
    with open(OUTPUT_ROOT / "dataset.yaml", 'w') as f:
        yaml.dump(data_yaml, f)
        
    print(f"✅ Dataset Created! Total Images: {global_img_count}")
    print(f"   Config: {OUTPUT_ROOT}/dataset.yaml")

if __name__ == "__main__":
    create_dataset()
