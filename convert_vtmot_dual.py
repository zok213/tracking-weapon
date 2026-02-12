#!/usr/bin/env python3
"""
VTMOT Dual Dataset Converter
Creates two specialized YOLO datasets from VTMOT_train:
1. Near View: LasHeR, photo, qiuxing, RGBT234
2. Far View: photo (subset), qiuxing, RGBT234, Vtuav, wurenji

Filters Class 1 (Person) only, clips coordinates to [0,1].
"""
import os
import shutil
from pathlib import Path
from configparser import ConfigParser
from tqdm import tqdm

# ============ CONFIGURATION ============
SOURCE_ROOT = Path("/home/student/Toan/datasets/VTMOT_train_extracted/data1/Datasets/Tracking/MOT/VTMOT/images/train")
OUTPUT_BASE = Path("/home/student/Toan/datasets")

# Dataset 1: Near View (close-range detection)
NEAR_VIEW_PREFIXES = ["LasHeR", "photo", "qiuxing", "RGBT234"]
NEAR_VIEW_OUTPUT = OUTPUT_BASE / "vtmot_near"

# Dataset 2: Far View (long-range detection)  
FAR_VIEW_PREFIXES = ["Vtuav", "wurenji", "RGBT234", "qiuxing"]  # photo excluded for now, can add later
FAR_VIEW_OUTPUT = OUTPUT_BASE / "vtmot_far"

VAL_RATIO = 0.15  # 15% for validation
# =======================================


def get_sequence_prefix(seq_name: str) -> str:
    """Extract prefix from sequence name (e.g., 'LasHeR-001' -> 'LasHeR')"""
    for prefix in ["LasHeR", "RGBT234", "Vtuav", "wurenji", "qiuxing", "photo"]:
        if seq_name.startswith(prefix):
            return prefix
    return seq_name.split("-")[0]


def parse_seqinfo(seq_dir: Path) -> dict:
    """Read seqinfo.ini to get image dimensions."""
    seqinfo_path = seq_dir / "seqinfo.ini"
    if not seqinfo_path.exists():
        return {"width": 640, "height": 512}  # default
    
    config = ConfigParser()
    config.read(seqinfo_path)
    return {
        "width": int(config.get("Sequence", "imWidth", fallback=640)),
        "height": int(config.get("Sequence", "imHeight", fallback=512)),
    }


def convert_mot_to_yolo(gt_path: Path, img_width: int, img_height: int) -> dict:
    """
    Convert MOT gt.txt to YOLO format, filtering Class 1 (Person) only.
    Returns: {frame_id: [(class_id, x_center, y_center, width, height), ...]}
    """
    frame_labels = {}
    
    if not gt_path.exists():
        return frame_labels
    
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            
            frame_id = int(parts[0])
            # track_id = int(parts[1])  # Not needed for detection
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            # parts[6] is confidence, parts[7] is class_id
            class_id = int(parts[7]) if len(parts) > 7 else 1
            
            # Filter: Only Class 1 (Person)
            if class_id != 1:
                continue
            
            # Convert to YOLO format (normalized center + wh)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            
            # Clip coordinates to [0, 1] for reliability
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.001, min(1.0, w_norm))
            h_norm = max(0.001, min(1.0, h_norm))
            
            # Adjust if bbox extends beyond image
            if x_center - w_norm / 2 < 0:
                w_norm = x_center * 2
            if x_center + w_norm / 2 > 1:
                w_norm = (1 - x_center) * 2
            if y_center - h_norm / 2 < 0:
                h_norm = y_center * 2
            if y_center + h_norm / 2 > 1:
                h_norm = (1 - y_center) * 2
            
            # YOLO class is 0 (single class: person)
            if frame_id not in frame_labels:
                frame_labels[frame_id] = []
            frame_labels[frame_id].append((0, x_center, y_center, w_norm, h_norm))
    
    return frame_labels


def process_sequence(seq_dir: Path, output_dir: Path, split: str) -> tuple:
    """Process a single sequence, copy images and create labels."""
    seq_name = seq_dir.name
    img_info = parse_seqinfo(seq_dir)
    img_width, img_height = img_info["width"], img_info["height"]
    
    # Parse ground truth
    gt_path = seq_dir / "gt" / "gt.txt"
    frame_labels = convert_mot_to_yolo(gt_path, img_width, img_height)
    
    visible_dir = seq_dir / "visible"
    infrared_dir = seq_dir / "infrared"
    
    if not visible_dir.exists() or not infrared_dir.exists():
        return 0, 0
    
    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    img_count = 0
    label_count = 0
    
    for img_file in visible_dir.glob("*.jpg"):
        frame_id = int(img_file.stem)
        ir_file = infrared_dir / img_file.name
        
        if not ir_file.exists():
            continue
        
        # Output names with MCF suffixes
        base_name = f"{seq_name}_{img_file.stem}"
        rgb_name = f"{base_name}_rgb_.jpg"
        ir_name = f"{base_name}_ir_.jpg"
        label_name = f"{base_name}_rgb_.txt"
        
        # Copy images
        shutil.copy2(img_file, images_out / rgb_name)
        shutil.copy2(ir_file, images_out / ir_name)
        img_count += 1
        
        # Write label
        if frame_id in frame_labels:
            with open(labels_out / label_name, "w") as f:
                for cls, xc, yc, w, h in frame_labels[frame_id]:
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            label_count += 1
        else:
            # Write empty label for background images
            (labels_out / label_name).touch()
    
    return img_count, label_count


def create_dataset(prefixes: list, output_dir: Path, name: str):
    """Create a complete dataset from sequences matching given prefixes."""
    print(f"\n{'='*60}")
    print(f"Creating {name} Dataset")
    print(f"Sources: {', '.join(prefixes)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    # Find matching sequences
    all_sequences = list(SOURCE_ROOT.iterdir())
    matching_sequences = []
    for seq in all_sequences:
        if seq.is_dir():
            prefix = get_sequence_prefix(seq.name)
            if prefix in prefixes:
                matching_sequences.append(seq)
    
    print(f"Found {len(matching_sequences)} matching sequences")
    
    # Split into train/val
    import random
    random.seed(42)
    random.shuffle(matching_sequences)
    val_count = int(len(matching_sequences) * VAL_RATIO)
    val_sequences = set(seq.name for seq in matching_sequences[:val_count])
    
    total_images = 0
    total_labels = 0
    
    for seq in tqdm(matching_sequences, desc=f"Processing {name}"):
        split = "val" if seq.name in val_sequences else "train"
        imgs, lbls = process_sequence(seq, output_dir, split)
        total_images += imgs
        total_labels += lbls
    
    print(f"\n✓ {name} Dataset Complete:")
    print(f"  Images: {total_images} pairs")
    print(f"  Labels: {total_labels} (non-empty)")
    
    return total_images, total_labels


def create_yaml(output_dir: Path, name: str):
    """Create dataset YAML for YOLOv11-RGBT."""
    yaml_content = f"""# {name} Dataset for YOLOv11-RGBT MCF
path: {output_dir}
train: images/train
val: images/val

nc: 1
names:
  0: person

# MCF-specific settings
pairs_rgb_ir: ['_rgb_', '_ir_']
"""
    yaml_path = output_dir / f"{name.lower().replace(' ', '_')}.yaml"
    yaml_path.write_text(yaml_content)
    print(f"✓ Created: {yaml_path}")
    return yaml_path


def main():
    print("=" * 60)
    print("VTMOT Dual Dataset Converter")
    print("=" * 60)
    print(f"Source: {SOURCE_ROOT}")
    
    # Check source exists
    if not SOURCE_ROOT.exists():
        print(f"ERROR: Source not found: {SOURCE_ROOT}")
        return
    
    # Create Near View dataset
    near_imgs, near_lbls = create_dataset(NEAR_VIEW_PREFIXES, NEAR_VIEW_OUTPUT, "Near View")
    near_yaml = create_yaml(NEAR_VIEW_OUTPUT, "Near View")
    
    # Create Far View dataset
    far_imgs, far_lbls = create_dataset(FAR_VIEW_PREFIXES, FAR_VIEW_OUTPUT, "Far View")
    far_yaml = create_yaml(FAR_VIEW_OUTPUT, "Far View")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Near View: {near_imgs} image pairs, {near_lbls} labels")
    print(f"Far View:  {far_imgs} image pairs, {far_lbls} labels")
    print(f"\nYAMLs:")
    print(f"  {near_yaml}")
    print(f"  {far_yaml}")


if __name__ == "__main__":
    main()
