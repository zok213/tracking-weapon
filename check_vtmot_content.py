
import os
import glob
from pathlib import Path

def check_content():
    # 1. User Provided Lists
    test_sequences = [
        "Vtuav-02", "Vtuav-06", "Vtuav-17", "Vtuav-18",
        "wurenji-0302-01", "wurenji-0302-11", "wurenji-0302-15", "wurenji-0302-20",
        "wurenji-0303-07", "wurenji-0303-09", "wurenji-0303-16", "wurenji-0303-17", "wurenji-0303-22",
        "wurenji-0304-05", "wurenji-0304-07", "wurenji-0304-15", "wurenji-0304-22", "wurenji-0304-29",
        "wurenji-0305-09", "wurenji-0305-10", "wurenji-0305-13", "wurenji-0305-17"
    ]

    train_sequences = [
        "Vtuav-01", "Vtuav-03", "Vtuav-04", "Vtuav-05", "Vtuav-07", "Vtuav-08", "Vtuav-09", "Vtuav-10",
        "Vtuav-11", "Vtuav-12", "Vtuav-13", "Vtuav-14", "Vtuav-15", "Vtuav-16", "Vtuav-19", "Vtuav-20",
        "wurenji-0302-02", "wurenji-0302-03", "wurenji-0302-04", "wurenji-0302-05", "wurenji-0302-06",
        "wurenji-0302-07", "wurenji-0302-08", "wurenji-0302-09", "wurenji-0302-10", "wurenji-0302-12",
        "wurenji-0302-13", "wurenji-0302-14", "wurenji-0302-17", "wurenji-0302-18", "wurenji-0302-19",
        "wurenji-0302-21", "wurenji-0302-22",
        "wurenji-0303-01", "wurenji-0303-02", "wurenji-0303-03", "wurenji-0303-04", "wurenji-0303-05",
        "wurenji-0303-06", "wurenji-0303-08", "wurenji-0303-10", "wurenji-0303-11", "wurenji-0303-12",
        "wurenji-0303-13", "wurenji-0303-14", "wurenji-0303-15", "wurenji-0303-18", "wurenji-0303-19",
        "wurenji-0303-20", "wurenji-0303-21",
        "wurenji-0304-01", "wurenji-0304-02", "wurenji-0304-03", "wurenji-0304-04", "wurenji-0304-06",
        "wurenji-0304-08", "wurenji-0304-09", "wurenji-0304-10", "wurenji-0304-11", "wurenji-0304-12",
        "wurenji-0304-13", "wurenji-0304-14", "wurenji-0304-16", "wurenji-0304-17", "wurenji-0304-18",
        "wurenji-0304-19", "wurenji-0304-20", "wurenji-0304-21", "wurenji-0304-23", "wurenji-0304-24",
        "wurenji-0304-25", "wurenji-0304-26", "wurenji-0304-27", "wurenji-0304-28", "wurenji-0304-30",
        "wurenji-0304-31",
        "wurenji-0305-01", "wurenji-0305-02", "wurenji-0305-03", "wurenji-0305-04", "wurenji-0305-05",
        "wurenji-0305-06", "wurenji-0305-07", "wurenji-0305-08", "wurenji-0305-11", "wurenji-0305-12",
        "wurenji-0305-14", "wurenji-0305-15", "wurenji-0305-16",
        "photo-0319-01", "photo-0319-02", "photo-0319-03", "photo-0319-04", "photo-0319-05", "photo-0319-06",
        "photo-0319-08", "photo-0319-09", "photo-0319-10", "photo-0319-11", "photo-0319-12", "photo-0319-13",
        "photo-0319-14", "photo-0319-15", "photo-0319-16", "photo-0319-17", "photo-0319-19", "photo-0319-20",
        "photo-0319-21", "photo-0319-22"
    ]

    target_dir = "/home/student/Toan/datasets/vtmot_far/images"
    
    print(f"Checking target directory: {target_dir}")
    
    # 2. Scan vtmot_far
    # Assuming structure is images/train/*.jpg and images/val/*.jpg
    # Filenames are like: sequence_name_frame_cam.jpg
    
    found_sequences = set()
    
    # Check train
    train_files = glob.glob(os.path.join(target_dir, "train", "*.jpg"))
    # Check val
    val_files = glob.glob(os.path.join(target_dir, "val", "*.jpg"))
    
    all_files = train_files + val_files
    print(f"Found {len(all_files)} total images in vtmot_far.")
    
    for fpath in all_files:
        basename = os.path.basename(fpath)
        # Parse sequence name
        # Format usually: sequence_name_frame_suffix
        # e.g. photo-0319-05_000001_ir_.jpg
        # The sequence name allows hyphens. The separator to frame number is usually the LAST underscore before frame?
        # Or maybe the first underscore?
        # Let's try splitting by underscore.
        
        parts = basename.split('_')
        # Reconstruct sequence name (everything before the frame number part)
        # Usually it's the first part for things like photo-0319-05
        # but what about wurenji-0302-01? It has hyphens.
        # "photo-0319-05_000001_ir_.jpg" -> parts[0] is "photo-0319-05" which matches.
        
        seq_name = parts[0]
        found_sequences.add(seq_name)
        
    print(f"Found {len(found_sequences)} unique sequences in vtmot_far.")
    
    # 3. Verify Test Sequences
    print("\n--- Checking TEST Sequences ---")
    missing_test = []
    present_test = []
    
    for seq in test_sequences:
        if seq in found_sequences:
            present_test.append(seq)
        else:
            missing_test.append(seq)
            
    print(f"Test Sequences Present: {len(present_test)}/{len(test_sequences)}")
    print(f"Test Sequences Missing: {len(missing_test)}/{len(test_sequences)}")
    if missing_test:
        print("Missing Test Sequences:")
        for s in missing_test:
            print(f"  - {s}")
            
    # 4. Verify Train Sequences
    print("\n--- Checking TRAIN Sequences ---")
    missing_train = []
    present_train = []
    
    for seq in train_sequences:
        if seq in found_sequences:
            present_train.append(seq)
        else:
            missing_train.append(seq)
            
    print(f"Train Sequences Present: {len(present_train)}/{len(train_sequences)}")
    print(f"Train Sequences Missing: {len(missing_train)}/{len(train_sequences)}")
    if missing_train:
        print("Missing Train Sequences (First 10):")
        for s in missing_train[:10]:
            print(f"  - {s}")
    
    # 5. Conclusion
    if len(missing_test) == len(test_sequences):
        print("\n[CONCLUSION] NO Test sequences found in vtmot_far.")
        print("This confirms vtmot_far is purely the TRAINING/VAL split.")
    elif len(missing_test) > 0:
        print("\n[CONCLUSION] PARTIAL Test sequences found.")
    else:
        print("\n[CONCLUSION] ALL Test sequences found.")

if __name__ == "__main__":
    check_content()
