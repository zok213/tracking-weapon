#!/usr/bin/env python3
"""
Audit VT-MOT Class IDs
======================
Scans GT files from different naming conventions (LasHeR, photo, wurenji, etc.)
and prints the distribution of Class IDs (column 8).
"""

from pathlib import Path
import glob
from collections import defaultdict
import os

VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")

def scan_gt():
    # Group by prefix
    groups = defaultdict(list)
    
    for split in ["train", "test", "val"]:
        img_dir = VT_MOT_ROOT / "images" / split
        if not img_dir.exists(): continue
        
        for p in img_dir.iterdir():
            if p.is_dir():
                prefix = p.name.split('-')[0]
                groups[prefix].append(p)
    
    print(f"Found groups: {list(groups.keys())}")
    
    # Sample 1-2 from each group
    for prefix, paths in groups.items():
        print(f"\n--- Group: {prefix} ---")
        for i, seq_path in enumerate(paths[:2]): # Check first 2
            gt_path = seq_path / "gt" / "gt.txt"
            if not gt_path.exists():
                print(f"  {seq_path.name}: No GT")
                continue
                
            # Count classes
            classes = defaultdict(int)
            with open(gt_path) as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 8:
                        cid = parts[7]
                        classes[cid] += 1
            
            print(f"  {seq_path.name}: {dict(classes)}")

if __name__ == "__main__":
    scan_gt()
