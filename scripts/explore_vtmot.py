
import os
from pathlib import Path

ROOT = Path("/home/student/Toan/data/VT-MOT")

def explore():
    print(f"Scanning {ROOT}...")
    keywords = ["qiuxing", "photo", "lasher", "RGBT"]
    found_seqs = []
    
    for path in ROOT.rglob("*"):
        if path.is_dir():
             # Check if it looks like a sequence
             if any(k in path.name for k in keywords):
                 # Check for images
                 rgb = path / "rgb"
                 ir = path / "ir"
                 if rgb.exists():
                     found_seqs.append(path)
                     print(f"✅ Found Sequence: {path}")
                     # Look for GT
                     # GT might be in 'gt/gt.txt' or '../gt/gt.txt' or similar
                     gt_candidates = [
                         path / "gt/gt.txt",
                         path.parent / "gt/gt.txt",
                         path / "labels.txt"
                     ]
                     for gt in gt_candidates:
                         if gt.exists():
                             print(f"   -> GT Found: {gt}")
                             # Print first line
                             with open(gt) as f:
                                 print(f"   -> GT Head: {f.readline().strip()}")
                             break
                     else:
                         print("   -> ⚠️ No GT found in standard locations.")

    print(f"\nTotal Matching Sequences: {len(found_seqs)}")

if __name__ == "__main__":
    explore()
