#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Override sys.argv to force specific sequence testing if needed, or just import logic
# We will just reuse the worker function logic but on a single sequence provided manually

# Define sequence
TEST_SEQ_PATH = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/test/photo-0301-02")
OUTPUT_DIR = Path("/home/student/Toan/data/VT-MOT_4cls_SAM3")

# Import the worker function from our main script
# (Need to make sure it can be imported or just replicate calling it)
sys.path.append("/home/student/Toan/scripts")
import classify_vtmot_sam3_multi_gpu as cls_script

print(f"Testing logic on: {TEST_SEQ_PATH}")

# Call worker process directly (synchronously)
# Device 0, List of [(split, path)], Output Dir
seq_tuple = ("test", TEST_SEQ_PATH)
cls_script.worker_process(0, [seq_tuple], OUTPUT_DIR)

print("Test processing done. Now visualizing...")

# Run viz
os.system(f"python3 /home/student/Toan/scripts/visualize_sam3_labels.py --seq photo-0301-02")
