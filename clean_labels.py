#!/usr/bin/env python3
"""
Clean YOLO labels by clipping coordinates to [0,1] range.
This fixes the 'non-normalized or out of bounds coordinates' warnings.
"""
import os
from pathlib import Path

def clip_label(value, min_val=0.0, max_val=1.0):
    return max(min_val, min(max_val, float(value)))

def clean_labels(labels_dir):
    labels_path = Path(labels_dir)
    fixed_count = 0
    total_files = 0
    
    for label_file in labels_path.glob("*.txt"):
        total_files += 1
        lines = []
        modified = False
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    lines.append(line.strip())
                    continue
                
                cls = parts[0]
                cx, cy, w, h = map(float, parts[1:5])
                
                # Check if any coordinate is out of bounds
                if cx < 0 or cx > 1 or cy < 0 or cy > 1 or w < 0 or w > 1 or h < 0 or h > 1:
                    modified = True
                
                # Clip coordinates
                cx = clip_label(cx)
                cy = clip_label(cy)
                w = clip_label(w)
                h = clip_label(h)
                
                # Also ensure center + half_size doesn't exceed bounds
                # Reduce width/height if they would exceed the boundary
                if cx + w/2 > 1.0:
                    w = (1.0 - cx) * 2
                if cx - w/2 < 0.0:
                    w = cx * 2
                if cy + h/2 > 1.0:
                    h = (1.0 - cy) * 2
                if cy - h/2 < 0.0:
                    h = cy * 2
                
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
        if modified:
            fixed_count += 1
            with open(label_file, 'w') as f:
                f.write("\n".join(lines))
    
    return fixed_count, total_files

if __name__ == "__main__":
    base_dir = Path("/home/student/Toan/datasets/vtmot_rgbt/labels")
    
    for split in ['train', 'val']:
        labels_dir = base_dir / split
        print(f"Cleaning {split} labels...")
        fixed, total = clean_labels(labels_dir)
        print(f"  Fixed {fixed}/{total} files")
    
    print("\nDone! Labels have been cleaned.")
    
    # Delete cache files so they are regenerated
    import shutil
    cache_train = base_dir / "train.cache"
    cache_val = base_dir / "val.cache"
    for cache in [cache_train, cache_val]:
        if cache.exists():
            cache.unlink()
            print(f"Deleted cache: {cache}")
