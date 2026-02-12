#!/usr/bin/env python3
"""
VT-MOT to YOLO 2-Class Converter (Human + Car)
==============================================
Direct conversion from MOT GT format to YOLO format.
No AI needed - uses verified GT mappings:
  - GT Class 1 → YOLO Class 0 (Human)
  - GT Class 2 → YOLO Class 1 (Car/Vehicle)

Outputs:
  - /home/student/Toan/data/VT-MOT_2cls/
    ├── train/images/
    ├── train/labels/
    ├── val/images/
    ├── val/labels/
    ├── test/images/
    └── test/labels/
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Config
VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")
OUTPUT_DIR = Path("/home/student/Toan/data/VT-MOT_2cls")

# Class Mapping (Verified by user)
CLASS_MAP = {
    1: 0,  # GT:1 (Human) → YOLO Class 0
    2: 1,  # GT:2 (Vehicle) → YOLO Class 1
}

# Statistics
stats = defaultdict(int)

def parse_gt(gt_path):
    """Parse MOT GT file to dict: frame_id -> list of boxes"""
    data = {}
    if not gt_path.exists():
        return {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            fid = int(parts[0])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            cid = int(parts[7])
            
            if cid not in CLASS_MAP:
                stats['skipped_unknown_class'] += 1
                continue
                
            if fid not in data:
                data[fid] = []
            data[fid].append({
                'box': [x, y, w, h],
                'class': CLASS_MAP[cid]
            })
    return data

def get_image_size(img_path):
    """Get image dimensions without loading full image"""
    import struct
    import imghdr
    
    with open(img_path, 'rb') as f:
        head = f.read(24)
        if imghdr.what(None, head) == 'jpeg':
            f.seek(0)
            size = 2
            ftype = 0
            while not 0xc0 <= ftype <= 0xcf or ftype in (0xc4, 0xc8, 0xcc):
                f.seek(size, 1)
                byte = f.read(1)
                while ord(byte) == 0xff:
                    byte = f.read(1)
                ftype = ord(byte)
                size = struct.unpack('>H', f.read(2))[0] - 2
            f.seek(1, 1)
            h, w = struct.unpack('>HH', f.read(4))
            return w, h
        elif imghdr.what(None, head) == 'png':
            w, h = struct.unpack('>ii', head[16:24])
            return w, h
    
    # Fallback: use PIL
    from PIL import Image
    with Image.open(img_path) as img:
        return img.size

def process_sequence(seq_path, split, out_base):
    """Process a single sequence"""
    seq_name = seq_path.name
    
    # Find GT and images
    gt_path = seq_path / "gt" / "gt.txt"
    img_dir = seq_path / "visible"
    if not img_dir.exists():
        img_dir = seq_path / "img1"
    if not img_dir.exists() or not gt_path.exists():
        return 0
    
    # Parse GT
    gt_data = parse_gt(gt_path)
    if not gt_data:
        return 0
    
    # Output dirs
    out_img = out_base / split / "images" / seq_name
    out_lbl = out_base / split / "labels" / seq_name
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    
    # Get sample image for size
    sample = next(img_dir.glob("*.jpg"), None) or next(img_dir.glob("*.png"), None)
    if not sample:
        return 0
    
    try:
        img_w, img_h = get_image_size(sample)
    except:
        from PIL import Image
        with Image.open(sample) as img:
            img_w, img_h = img.size
    
    # Process frames
    count = 0
    fname_fmt = f"{{:0{len(sample.stem)}d}}{sample.suffix}"
    
    for fid, boxes in gt_data.items():
        img_name = fname_fmt.format(fid)
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
        
        # Write YOLO labels
        lines = []
        for box in boxes:
            x, y, w, h = box['box']
            cls = box['class']
            
            # Convert to YOLO format (center x, center y, width, height - normalized)
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            
            # Validate
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
                stats['skipped_invalid_box'] += 1
                continue
            
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats[f'class_{cls}'] += 1
        
        if lines:
            # Copy image
            dst_img = out_img / img_name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            
            # Write label
            lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
            with open(out_lbl / lbl_name, 'w') as f:
                f.write('\n'.join(lines))
            
            count += 1
    
    return count

def main():
    print("=" * 60)
    print("VT-MOT → YOLO 2-Class Converter")
    print("=" * 60)
    print(f"Input:  {VT_MOT_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Classes: 0=Human (GT:1), 1=Car (GT:2)")
    print("=" * 60)
    
    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    
    # Process all splits
    total_frames = 0
    for split in ["train", "test", "val"]:
        split_dir = VT_MOT_ROOT / "images" / split
        if not split_dir.exists():
            continue
        
        seqs = list(split_dir.iterdir())
        print(f"\n[{split.upper()}] Processing {len(seqs)} sequences...")
        
        for seq_path in tqdm(seqs):
            if seq_path.is_dir():
                n = process_sequence(seq_path, split, OUTPUT_DIR)
                total_frames += n
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total frames:     {total_frames:,}")
    print(f"Human (Class 0):  {stats['class_0']:,}")
    print(f"Car (Class 1):    {stats['class_1']:,}")
    print(f"Skipped (unknown): {stats['skipped_unknown_class']:,}")
    print(f"Skipped (invalid): {stats['skipped_invalid_box']:,}")
    print(f"\nOutput: {OUTPUT_DIR}")
    
    # Create data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(f"""# VT-MOT 2-Class Dataset
path: {OUTPUT_DIR}
train: train/images
val: val/images
test: test/images

nc: 2
names: ['human', 'car']
""")
    print(f"Created: {yaml_path}")

if __name__ == "__main__":
    main()
