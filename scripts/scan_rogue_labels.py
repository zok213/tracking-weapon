from pathlib import Path

DATASET_ROOT = Path("/home/student/Toan/data/VT-MOT_Person_Only/labels/train")
rogue_count = 0
scanned_count = 0

print(f"Scanning {DATASET_ROOT}...")

for lbl_file in DATASET_ROOT.rglob("*.txt"):
    scanned_count += 1
    with open(lbl_file, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if not parts: continue
            cls_id = int(parts[0])
            if cls_id != 0:
                print(f"ROGUE FOUND: {lbl_file.name} (Line {idx+1}): Class {cls_id}")
                rogue_count += 1
                if rogue_count >= 10:
                    print("Stopping after 10 faults.")
                    exit(1)

if rogue_count == 0:
    print(f"✅ CLEAN. Scanned {scanned_count} files. All Class IDs are 0.")
else:
    print(f"❌ FOUND {rogue_count} rogue files.")
