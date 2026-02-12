import os
import shutil
import random
from pathlib import Path
import configparser

def convert_vtmot_to_yolo():
    data_root = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/test")
    output_root = Path("/home/student/Toan/datasets/vtmot_rgbt")
    
    # Create directories
    for split in ['train', 'val']:
        (output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    seqs = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    random.seed(42)
    random.shuffle(seqs)
    
    split_idx = int(len(seqs) * 0.8)
    train_seqs = seqs[:split_idx]
    val_seqs = seqs[split_idx:]
    
    print(f"Total sequences: {len(seqs)}")
    print(f"Train on: {len(train_seqs)}, Val on: {len(val_seqs)}")
    
    for split, split_seqs in [('train', train_seqs), ('val', val_seqs)]:
        for seq in split_seqs:
            seq_path = data_root / seq
            seq_info = configparser.ConfigParser()
            seq_info.read(seq_path / 'seqinfo.ini')
            
            w = int(seq_info['Sequence']['imWidth'])
            h = int(seq_info['Sequence']['imHeight'])
            
            # Read gt.txt
            gt_path = seq_path / 'gt' / 'gt.txt'
            labels = {} # frame -> list of yolo bboxes
            
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if not parts: continue
                    frame = int(parts[0])
                    # id = parts[1]
                    bx, by, bw, bh = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    active = int(parts[6])
                    cls = int(parts[7])
                    
                    if cls == 1 and active == 1: # Human only
                        # Convert to YOLO (center_x, center_y, width, height) normalized
                        cx = (bx + bw/2) / w
                        cy = (by + bh/2) / h
                        nw = bw / w
                        nh = bh / h
                        
                        if frame not in labels: labels[frame] = []
                        labels[frame].append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            
            # Copy images and write labels
            for mod in ['visible', 'infrared']:
                mod_suffix = '_rgb_' if mod == 'visible' else '_ir_'
                img_dir = seq_path / mod
                for img_file in sorted(img_dir.glob('*.jpg')):
                    frame_num = int(img_file.stem)
                    new_name = f"{seq}_{frame_num:06d}{mod_suffix}.jpg"
                    
                    # Target paths
                    target_img = output_root / 'images' / split / new_name
                    target_lbl = output_root / 'labels' / split / f"{seq}_{frame_num:06d}.txt"
                    
                    # Copy image
                    shutil.copy(img_file, target_img)
                    
                    # Write label (once per frame pair)
                    if not target_lbl.exists():
                        frame_labels = labels.get(frame_num, [])
                        with open(target_lbl, 'w') as f:
                            f.write("\n".join(frame_labels))

    print("Conversion complete.")

if __name__ == "__main__":
    convert_vtmot_to_yolo()
