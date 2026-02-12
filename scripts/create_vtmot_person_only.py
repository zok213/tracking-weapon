#!/usr/bin/env python3
"""
Create VT-MOT Person-Only RGBT Dataset (Manual Fixes + IR Support)
==================================================================
1. Selects sequences based on Viz Bulk logic.
2. Applies HARDCODED fixes for known label errors.
3. Filters to keep ONLY Class 1 (Person).
4. Symlinks both Visible and Infrared images.
5. Generates standard YOLO labels.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml

# CONFIG
VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")
OUTPUT_DIR = Path("/home/student/Toan/data/VT-MOT_Person_Only")

# 1. Sequence Selection Logic
PHOTO_SUBSET = [
    "photo-0319-23", "photo-0319-18", "photo-0318-43", "photo-0318-39",
    "photo-0318-35", "photo-0318-32", "photo-0318-26", "photo-0318-27",
    "photo-0310-52", "photo-0310-51", "photo-0310-48", "photo-0310-42",
    "photo-0310-41", "photo-0310-40", "photo-0310-36", "photo-0310-34",
    "photo-0310-33", "photo-0310-28", 
    "photo-0306-01", "photo-0306-02"
]
VTUAV_SUBSET = ["Vtuav-02", "Vtuav-06", "Vtuav-17", "Vtuav-18"]
DRONE_KEYWORDS = ["wurenji", "qiuxing", "RGBT", "VTUAV", "Vtuav"]

def discover_sequences(root, keywords):
    found = []
    for split in ["train", "test", "val"]:
        d = root / "images" / split
        if not d.exists(): continue
        for p in d.iterdir():
            if not p.is_dir(): continue
            name = p.name
            if any(k.lower() in name.lower() for k in keywords):
                found.append(name)
    return sorted(list(set(found)))

ALL_TARGETS = sorted(list(set(PHOTO_SUBSET + VTUAV_SUBSET + discover_sequences(VT_MOT_ROOT, DRONE_KEYWORDS))))

# 2. Manual Fix Logic
def apply_manual_fixes(seq_name, tid, cid):
    if seq_name == "wurenji-0304-07" and tid in [15, 25]: return 2
    if seq_name == "wurenji-0303-16" and tid == 39: return 2
    return cid

# 3. Processing
def process_dataset():
    # Clean Start
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    
    (OUTPUT_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images_ir" / "train").mkdir(parents=True, exist_ok=True) # IR Folder
    (OUTPUT_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
    
    print(f"Targeting {len(ALL_TARGETS)} sequences.")
    
    global_img_count = 0
    
    for seq_name in tqdm(ALL_TARGETS):
        seq_path = None
        for split in ["train", "test", "val"]:
            p = VT_MOT_ROOT / "images" / split / seq_name
            if p.exists():
                seq_path = p
                break
        if not seq_path: continue
        
        gt_path = seq_path / "gt" / "gt.txt"
        if not gt_path.exists(): continue
        
        # Parse GT
        frame_labels = {} 
        with open(gt_path) as f:
            for line in f:
                parts = line.strip().split(',')
                fid = int(parts[0])
                tid = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                cid = int(parts[7])
                
                cid = apply_manual_fixes(seq_name, tid, cid)
                
                if cid != 1: continue # ONLY PERSON
                
                if fid not in frame_labels: frame_labels[fid] = []
                frame_labels[fid].append((x, y, w, h))

        # Process Images
        rgb_dir = seq_path / "visible"
        if not rgb_dir.exists(): rgb_dir = seq_path / "img1"
        
        ir_dir = seq_path / "infrared" # Try standard IR
        
        images = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
        
        for img_file in images:
            fid = int(img_file.stem)
            
            # Destination Names
            dst_name = f"{seq_name}_{img_file.name}"
            dst_img = OUTPUT_DIR / "images" / "train" / dst_name
            dst_lbl = OUTPUT_DIR / "labels" / "train" / (dst_name.rsplit('.', 1)[0] + ".txt")
            
            # IR Image Finding
            ir_file = None
            if ir_dir.exists():
                # Try exact name match or extension swap
                cand = ir_dir / img_file.name
                if not cand.exists(): cand = ir_dir / (img_file.stem + ".png")
                if not cand.exists(): cand = ir_dir / (img_file.stem + ".jpg")
                if cand.exists(): ir_file = cand
            
            # Symlink RGB
            os.symlink(img_file, dst_img)
            
            # Symlink IR (if exists)
            if ir_file:
                dst_ir = OUTPUT_DIR / "images_ir" / "train" / dst_name
                # Ensure extension match? Usually safe to keep original ext.
                # But if RGB is .jpg and IR is .png, naming might be tricky if code assumes exact name.
                # Let's keep original filename for IR symlink.
                # Wait, loader usually does `replace('images', 'images_ir')` and keeps same name?
                # Or checks extensions. Safer to ensure name identity if possible.
                # Let's symlink with original extension but check if our loader can handle it.
                # My planned loader will assume same filename.
                # So if RGB is .jpg and IR is .png, we might have issue if we look for .jpg in IR folder.
                # Force rename extension? No, images are binary.
                # Let's just symlink as is, and update loader to be smart.
                if ir_file.suffix != img_file.suffix:
                    # Rename symlink to match RGB suffix? No, that breaks decoding.
                    # We will create symlink with original IR name.
                    # And loader must find it.
                    # Actually, for simplicity, most robust loaders check extensions.
                    os.symlink(ir_file, OUTPUT_DIR / "images_ir" / "train" / dst_name.replace(img_file.suffix, ir_file.suffix))
                else:
                    os.symlink(ir_file, OUTPUT_DIR / "images_ir" / "train" / dst_name)
            
            # Write Label
            boxes = frame_labels.get(fid, [])
            if boxes:
                # Need dimensions
                import cv2
                img = cv2.imread(str(img_file))
                if img is None: continue 
                ih, iw = img.shape[:2]
                
                with open(dst_lbl, 'w') as out_f:
                    for (x, y, w, h) in boxes:
                        xc = (x + w/2) / iw
                        yc = (y + h/2) / ih
                        nw = w / iw
                        nh = h / ih
                        out_f.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
            else:
                with open(dst_lbl, 'w') as out_f: pass
            
            global_img_count += 1
            
    print(f"Created dataset with {global_img_count} images.")
    
    yaml_content = {
        'path': str(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/train',
        'names': {0: 'Person'}
    }
    
    with open(OUTPUT_DIR / "dataset.yaml", 'w') as f:
        yaml.dump(yaml_content, f)

if __name__ == "__main__":
    process_dataset()
