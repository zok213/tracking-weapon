
import os
import glob
import textwrap
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import configparser

import os
import glob
import textwrap
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import configparser
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define constants
SOURCE_BASE = "/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/test"
DEST_BASE = "/home/student/Toan/datasets/vtmot_far"
DEST_IMAGES_PATH = os.path.join(DEST_BASE, "images", "test")
DEST_LABELS_PATH = os.path.join(DEST_BASE, "labels", "test")
VALID_CLASS_IDS = [1, 2] # Pedestrian, Person on Vehicle

def process_single_sequence(seq):
    """Processes a single sequence: copies images and converts labels."""
    seq_path = os.path.join(SOURCE_BASE, seq)
    if not os.path.exists(seq_path):
        return f"WARNING: Sequence {seq} not found"
        
    # 1. Read Seq Info
    ini_path = os.path.join(seq_path, "seqinfo.ini")
    width = 1920
    height = 1080
    if os.path.exists(ini_path):
        config = configparser.ConfigParser()
        config.read(ini_path)
        if 'Sequence' in config:
            width = int(config['Sequence'].get('imWidth', 1920))
            height = int(config['Sequence'].get('imHeight', 1080))
    
    # 2. Process GT
    gt_path = os.path.join(seq_path, "gt", "gt.txt")
    gt_data = {} 
    
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6: continue
                try:
                    frame_idx = int(parts[0])
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    class_id = int(parts[7]) if len(parts) > 7 else 1
                    
                    if class_id in VALID_CLASS_IDS:
                            if frame_idx not in gt_data:
                                gt_data[frame_idx] = []
                            gt_data[frame_idx].append([0, x, y, w, h]) # Map to class 0
                            
                except ValueError:
                    continue

    # 3. Process Images
    files_copied = 0
    for modality, suffix in [("infrared", "ir"), ("visible", "rgb")]:
        mod_path = os.path.join(seq_path, modality)
        if not os.path.exists(mod_path):
            continue
            
        images = sorted(glob.glob(os.path.join(mod_path, "*.jpg")))
        
        for img_path in images:
            basename = os.path.basename(img_path)
            name_part = os.path.splitext(basename)[0]
            try:
                frame_idx = int(name_part)
            except ValueError:
                continue
                
            new_name = f"{seq}_{frame_idx:06d}_{suffix}_.jpg"
            dest_img_path = os.path.join(DEST_IMAGES_PATH, new_name)
            
            # Copy Image
            if not os.path.exists(dest_img_path):
                shutil.copy2(img_path, dest_img_path)
                files_copied += 1
                
            # Create Label
            if frame_idx in gt_data:
                dest_lbl_path = os.path.join(DEST_LABELS_PATH, new_name.replace(".jpg", ".txt"))
                with open(dest_lbl_path, 'w') as f_lbl:
                    for item in gt_data[frame_idx]:
                        cls, x, y, w_box, h_box = item
                        cx = (x + w_box / 2) / width
                        cy = (y + h_box / 2) / height
                        nw = w_box / width
                        nh = h_box / height
                        cx = max(0, min(1, cx))
                        cy = max(0, min(1, cy))
                        nw = max(0, min(1, nw))
                        nh = max(0, min(1, nh))
                        f_lbl.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                        
    return f"Finished {seq} ({files_copied} files)"

def process_test_data():
    test_sequences = [
        "Vtuav-02", "Vtuav-06", "Vtuav-17", "Vtuav-18",
        "wurenji-0302-01", "wurenji-0302-11", "wurenji-0302-15", "wurenji-0302-20",
        "wurenji-0303-07", "wurenji-0303-09", "wurenji-0303-16", "wurenji-0303-17", "wurenji-0303-22",
        "wurenji-0304-05", "wurenji-0304-07", "wurenji-0304-15", "wurenji-0304-22", "wurenji-0304-29",
        "wurenji-0305-09", "wurenji-0305-10", "wurenji-0305-13", "wurenji-0305-17"
    ]

    os.makedirs(DEST_IMAGES_PATH, exist_ok=True)
    os.makedirs(DEST_LABELS_PATH, exist_ok=True)
    
    print(f"Processing {len(test_sequences)} test sequences in PARALLEL...")
    print(f"Source: {SOURCE_BASE}")
    print(f"Dest: {DEST_BASE}")
    
    # Use ProcessPoolExecutor for CPU/IO parallelism
    # We can use multiple cores to copy faster
    max_workers = min(len(test_sequences), os.cpu_count() or 4) * 2 # IO bound, can oversubscribe
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_sequence, seq): seq for seq in test_sequences}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            # print(result) # Optional: print result

    print("Done adding test data.")

if __name__ == "__main__":
    process_test_data()
