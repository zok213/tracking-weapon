import os
import cv2
import shutil
import configparser
import numpy as np
from tqdm import tqdm
from glob import glob

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_vtmot_sequence(seq_path, output_root):
    seq_name = os.path.basename(seq_path)
    gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
    ini_path = os.path.join(seq_path, 'seqinfo.ini')
    
    rgb_dir_name = 'visible'
    ir_dir_name = 'infrared'
    
    # Read seqinfo if exists to verify paths
    if os.path.exists(ini_path):
        config = configparser.ConfigParser()
        config.read(ini_path)
        if 'Sequence' in config:
            if 'vimDir' in config['Sequence']:
                rgb_dir_name = config['Sequence']['vimDir']
            if 'irmDir' in config['Sequence']:
                ir_dir_name = config['Sequence']['irmDir']
    
    rgb_folder = os.path.join(seq_path, rgb_dir_name)
    ir_folder = os.path.join(seq_path, ir_dir_name)
    
    if not os.path.exists(gt_path):
        print(f"Skipping {seq_name}: GT file not found at {gt_path}")
        return 0, 0

    if not os.path.exists(rgb_folder) or not os.path.exists(ir_folder):
        print(f"Skipping {seq_name}: Image folders not found ({rgb_folder}, {ir_folder})")
        return 0, 0

    # Read GT
    # Format: frame, id, left, top, width, height, conf, class, visibility
    gt_data = {}
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame_idx = int(parts[0])
            obj_id = int(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            cls_id = int(parts[7]) if len(parts) > 7 else 1 # Default to 1 if missing
            
            if frame_idx not in gt_data:
                gt_data[frame_idx] = []
            gt_data[frame_idx].append({
                'id': obj_id,
                'bbox': [x1, y1, w, h],
                'class': cls_id
            })

    num_ids_generated = 0
    num_crops_generated = 0
    
    # Track statistics
    seq_ids = set()

    sorted_frames = sorted(gt_data.keys())
    
    for frame_idx in sorted_frames:
        img_name = f"{frame_idx:06d}.jpg"
        rgb_path = os.path.join(rgb_folder, img_name)
        ir_path = os.path.join(ir_folder, img_name)
        
        if not os.path.exists(rgb_path) or not os.path.exists(ir_path):
            continue
            
        rgb_img = cv2.imread(rgb_path)
        ir_img = cv2.imread(ir_path)
        
        if rgb_img is None or ir_img is None:
            continue
            
        img_h, img_w = rgb_img.shape[:2]
        
        detections = gt_data[frame_idx]
        for det in detections:
            obj_id = det['id']
            x, y, w, h = det['bbox']
            cls_id = det['class']
            
            # Sanity check bbox
            x1, y1 = int(max(0, x)), int(max(0, y))
            x2, y2 = int(min(img_w, x + w)), int(min(img_h, y + h))
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Filter small crops (e.g. < 32x64)
            if (x2 - x1) < 20 or (y2 - y1) < 30:
                continue

            # Create Global ID: seq_name_class_id
            # We explicitly include class ID in the folder name to allow filtering later
            # Typically ReID is class-agnostic if trained on persons, but VTMOT has cars.
            # Using 'seq_id' as unique identifier folder
            
            global_id = f"{seq_name}_{obj_id}"
            save_dir = os.path.join(output_root, 'train', global_id)
            create_dir(save_dir)
            
            crop_name_base = f"{global_id}_c{cls_id}_f{frame_idx}"
            
            rgb_crop = rgb_img[y1:y2, x1:x2]
            ir_crop = ir_img[y1:y2, x1:x2]
            
            # Save crops
            cv2.imwrite(os.path.join(save_dir, f"{crop_name_base}_rgb.jpg"), rgb_crop)
            cv2.imwrite(os.path.join(save_dir, f"{crop_name_base}_thermal.jpg"), ir_crop)
            
            if global_id not in seq_ids:
                seq_ids.add(global_id)
                num_ids_generated += 1
            num_crops_generated += 2

    return num_ids_generated, num_crops_generated

def main():
    # Configuration
    DATA_ROOT = 'stage1/data/VTMOT_test/data1/Datasets/Tracking/MOT/VTMOT/images/test'
    OUTPUT_ROOT = 'stage1/data/vtmot_reid_generated'
    
    if os.path.exists(OUTPUT_ROOT):
        print(f"Output root {OUTPUT_ROOT} exists. Cleaning up...")
        shutil.rmtree(OUTPUT_ROOT)
    create_dir(OUTPUT_ROOT)
    create_dir(os.path.join(OUTPUT_ROOT, 'train'))
    
    sequences = sorted(glob(os.path.join(DATA_ROOT, '*')))
    print(f"Found {len(sequences)} sequences in {DATA_ROOT}")
    
    total_ids = 0
    total_crops = 0
    
    for seq_path in tqdm(sequences):
        if not os.path.isdir(seq_path):
            continue
            
        n_ids, n_crops = process_vtmot_sequence(seq_path, OUTPUT_ROOT)
        total_ids += n_ids
        total_crops += n_crops
        
    print(f"Processing Complete.")
    print(f"Total Identities: {total_ids}")
    print(f"Total Crops (RGB+IR): {total_crops}")
    print(f"Data saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
