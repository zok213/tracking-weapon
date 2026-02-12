import os
from pathlib import Path

root = Path("/home/student/Toan/datasets/vtmot_rgbt")

def fix_split(split):
    print(f"Fixing {split}...")
    img_dir = root / 'images' / split
    lbl_dir = root / 'labels' / split
    manifest_path = root / f"{split}.txt"
    
    # 1. Rename labels
    labels = list(lbl_dir.glob("*.txt"))
    for lbl in labels:
        if not lbl.stem.endswith("_rgb_"):
            new_name = lbl.parent / f"{lbl.stem}_rgb_.txt"
            lbl.rename(new_name)
    
    # 2. Generate manifest (absolute paths)
    rgb_images = sorted(list(img_dir.glob("*_rgb_.jpg")))
    with open(manifest_path, 'w') as f:
        for img in rgb_images:
            f.write(f"{img.absolute()}\n")
    
    print(f"Renamed {len(labels)} labels and created manifest with {len(rgb_images)} images for {split}")

fix_split('train')
fix_split('val')
