
import os
import shutil
import random
import configparser
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_ROOT = Path("/home/student/Toan/datasets/VTMOT_train_extracted/data1/Datasets/Tracking/MOT/VTMOT/images/train")
OUTPUT_ROOT = Path("/home/student/Toan/datasets")
NUM_WORKERS = 16  # Parallel workers

# Dataset definitions
DATASET_RULES = {
    "vtmot_near": {
        "include": ["photo", "qiuxing", "RGBT234"],
        "exclude": ["LasHeR", "photo-0319"],
        "description": "Near View (Close-Range)"
    },
    "vtmot_far": {
        "include": ["Vtuav", "wurenji", "photo-0319", "RGBT234"],
        "exclude": ["LasHeR"],
        "description": "Far View (Long-Range)"
    }
}

# ---------------------

def parse_seqinfo(seq_path):
    """Parses seqinfo.ini to get image dimensions."""
    config = configparser.ConfigParser()
    ini_path = seq_path / "seqinfo.ini"
    if not ini_path.exists():
        return None
    config.read(ini_path)
    try:
        width = int(config['Sequence']['imWidth'])
        height = int(config['Sequence']['imHeight'])
        ext = config['Sequence']['imExt']
        return width, height, ext
    except KeyError:
        return None

def process_sequence_task(args):
    """Wrapper for resizing/copying a single sequence."""
    seq_path, output_base, split = args
    
    seq_name = seq_path.name
    res = parse_seqinfo(seq_path)
    if not res:
        return f"Skipping {seq_name}: Invalid seqinfo.ini"
    width, height, ext = res

    # setup output structure (exist_ok=True for parallel safety)
    img_out = output_base / "images" / split
    lbl_out = output_base / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # Read GT
    gt_path = seq_path / "gt" / "gt.txt"
    if not gt_path.exists():
        return f"Skipping {seq_name}: No gt.txt"

    # Parse Frame Labels
    frame_labels = {} # frame_idx -> list of YOLO lines
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8: continue
            
            frame_idx = int(parts[0])
            class_id = int(parts[7])
            
            # Filter: Class 1 (Person) ONLY
            if class_id != 1:
                continue

            x1_pixel = float(parts[2])
            y1_pixel = float(parts[3])
            w_pixel = float(parts[4])
            h_pixel = float(parts[5])

            x_center = (x1_pixel + w_pixel / 2) / width
            y_center = (y1_pixel + h_pixel / 2) / height
            w_norm = w_pixel / width
            h_norm = h_pixel / height

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))
            
            # YOLO class 0
            yolo_line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            
            if frame_idx not in frame_labels:
                frame_labels[frame_idx] = []
            frame_labels[frame_idx].append(yolo_line)

    # Copy files
    visible_dir = seq_path / "visible"
    infrared_dir = seq_path / "infrared"
    
    rgb_files = sorted(list(visible_dir.glob(f"*{ext}")))
    count = 0
    
    for rgb_file in rgb_files:
        try:
           frame_idx = int(rgb_file.stem)
        except ValueError:
            continue
            
        ir_file = infrared_dir / rgb_file.name
        if not ir_file.exists():
            continue

        out_name_base = f"{seq_name}_{rgb_file.stem}"
        rgb_out = img_out / f"{out_name_base}_rgb_{ext}"
        ir_out = img_out / f"{out_name_base}_ir_{ext}"
        
        shutil.copy2(rgb_file, rgb_out)
        shutil.copy2(ir_file, ir_out)

        lines = frame_labels.get(frame_idx, [])
        lbl_rgb_out = lbl_out / f"{out_name_base}_rgb_.txt"
        lbl_ir_out = lbl_out / f"{out_name_base}_ir_.txt"
        
        with open(lbl_rgb_out, 'w') as f:
            f.write("\n".join(lines))
        with open(lbl_ir_out, 'w') as f:
            f.write("\n".join(lines))
        count += 1
        
    return f"Processed {seq_name}: {count} pairs"

def main():
    if not SOURCE_ROOT.exists():
        print(f"Error: Source root not found: {SOURCE_ROOT}")
        return

    print("Scanning source directory...")
    all_seqs = sorted([p for p in SOURCE_ROOT.iterdir() if p.is_dir()])
    print(f"Found {len(all_seqs)} sequences total.")

    for dataset_name, rules in DATASET_RULES.items():
        print(f"\n--- Generating {dataset_name} ({rules['description']}) ---")
        dest_root = OUTPUT_ROOT / dataset_name
        
        if dest_root.exists():
            print(f"Cleaning existing {dest_root}...")
            shutil.rmtree(dest_root)
        dest_root.mkdir(parents=True)

        selected_seqs = []
        for seq in all_seqs:
            name = seq.name
            is_excluded = any(name.startswith(ex) for ex in rules["exclude"])
            if is_excluded: continue
            is_included = any(name.startswith(inc) for inc in rules["include"])
            if is_included: selected_seqs.append(seq)

        print(f"Selected {len(selected_seqs)} sequences for {dataset_name}.")
        
        random.seed(42)
        random.shuffle(selected_seqs)
        split_idx = int(len(selected_seqs) * 0.8)
        train_seqs = selected_seqs[:split_idx]
        val_seqs = selected_seqs[split_idx:]
        
        print(f"Split: {len(train_seqs)} Train, {len(val_seqs)} Val.")

        # Prepare tasks
        tasks = []
        for seq in train_seqs:
            tasks.append((seq, dest_root, "train"))
        for seq in val_seqs:
            tasks.append((seq, dest_root, "val"))

        print(f"Running {len(tasks)} tasks with {NUM_WORKERS} workers...")
        
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_sequence_task, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Building {dataset_name}"):
                pass # just update bar

    print("\n✓ Dataset Generation Complete!")

if __name__ == "__main__":
    main()
