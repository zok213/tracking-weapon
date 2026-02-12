
import os
import glob
import zipfile
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

# Configuration
DATASET_ROOT = "/home/student/Toan/datasets/vtmot_far"
OUTPUT_DIR = "/home/student/Toan/datasets"
MAX_SIZE_GB = 6.0  # Target size per shard (e.g., 6GB to easily fit 20GB limit/upload chunks)
COMPRESSION = zipfile.ZIP_STORED # Fastest, no CPU overhead

def get_all_files(directory, rel_start):
    """Recursively gets all files with relative paths."""
    file_list = []
    root_path = Path(directory)
    # Check if dir exists
    if not root_path.exists():
        print(f"Directory not found: {directory}")
        return []
        
    for p in root_path.rglob("*"):
        if p.is_file():
            # Get path relative to start (vtmot_far root)
            rel_path = p.relative_to(rel_start)
            file_list.append((str(p), str(rel_path), p.stat().st_size))
    return file_list

def create_zip_shard(shard_id, shard_name, files, output_path):
    """Creates a single zip file from a list of files."""
    try:
        with zipfile.ZipFile(output_path, 'w', COMPRESSION) as zf:
            for abs_path, rel_path, _ in files:
                zf.write(abs_path, rel_path)
        return (shard_id, True, f"Created {shard_name}: {len(files)} files")
    except Exception as e:
        return (shard_id, False, str(e))

def pack_dataset():
    print(f"Scanning files in {DATASET_ROOT}...")
    
    # 1. Define Groups
    groups = {
        "train": [
            os.path.join(DATASET_ROOT, "images", "train"),
            os.path.join(DATASET_ROOT, "labels", "train")
        ],
        "val_test": [
            os.path.join(DATASET_ROOT, "images", "val"),
            os.path.join(DATASET_ROOT, "labels", "val"),
            os.path.join(DATASET_ROOT, "images", "test"),
            os.path.join(DATASET_ROOT, "labels", "test")
        ]
    }
    
    # Always include yaml in one of them, or both? Let's put in val_test (smaller)
    yaml_file = os.path.join(DATASET_ROOT, "far_view_clean.yaml")
    
    tasks = []
    
    for group_name, distinct_paths in groups.items():
        print(f"Collecting files for {group_name}...")
        all_files = [] # (abs, rel, size)
        
        for dpath in distinct_paths:
            files = get_all_files(dpath, DATASET_ROOT)
            all_files.extend(files)
            
        if group_name == "val_test":
             if os.path.exists(yaml_file):
                 all_files.append((yaml_file, "far_view_clean.yaml", os.path.getsize(yaml_file)))

        # Sort by path to ensure deterministic order (though multiprocessing might shuffle completion)
        all_files.sort(key=lambda x: x[1])
        
        total_size = sum(f[2] for f in all_files)
        total_files = len(all_files)
        
        print(f"  Group {group_name}: {total_files} files, {total_size / (1024**3):.2f} GB")
        
        # Determine number of shards
        num_shards = math.ceil(total_size / (MAX_SIZE_GB * 1024**3))
        # Ensure at least 1
        num_shards = max(1, num_shards)
        
        print(f"  Splitting {group_name} into {num_shards} shards...")
        
        # Split files roughly equally
        chunk_size = math.ceil(total_files / num_shards)
        
        for i in range(num_shards):
            start = i * chunk_size
            end = start + chunk_size
            shard_files = all_files[start:end]
            
            shard_name = f"vtmot_far_{group_name}_part{i+1}.zip"
            output_path = os.path.join(OUTPUT_DIR, shard_name)
            
            tasks.append((shard_name, shard_files, output_path))

    # 2. Execute Parallel
    print(f"\nStarting parallel compression with {os.cpu_count()} workers...")
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for i, (name, files, path) in enumerate(tasks):
            futures.append(executor.submit(create_zip_shard, i, name, files, path))
            
        for future in tqdm(as_completed(futures), total=len(futures)):
            shard_id, success, msg = future.result()
            print(f"[{'OK' if success else 'FAIL'}] {msg}")

if __name__ == "__main__":
    pack_dataset()
