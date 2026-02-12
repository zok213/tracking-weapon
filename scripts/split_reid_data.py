import os
import glob
import shutil
import random
import re

DATA_ROOT = "/home/student/Toan/data/VT-MOT_ReID_Person_Only"
TRAIN_RGB = os.path.join(DATA_ROOT, "bounding_box_train")
TRAIN_IR = os.path.join(DATA_ROOT, "ir_bounding_box_train")
TEST_RGB = os.path.join(DATA_ROOT, "bounding_box_test")
TEST_IR = os.path.join(DATA_ROOT, "ir_bounding_box_test")
QUERY = os.path.join(DATA_ROOT, "query") # Usually RGB query

# Create directories
for d in [TEST_RGB, TEST_IR, QUERY]:
    os.makedirs(d, exist_ok=True)

# Get all PIDs
files = glob.glob(os.path.join(TRAIN_RGB, "*.jpg"))
pids = set()
for f in files:
    fname = os.path.basename(f)
    match = re.match(r'^(\d+)_c', fname)
    if match:
        pids.add(match.group(1))

sorted_pids = sorted(list(pids))
num_test = int(len(sorted_pids) * 0.2)
test_pids = set(sorted_pids[-num_test:])

print(f"Total PIDs: {len(sorted_pids)}")
print(f"Test PIDs : {len(test_pids)} (Last {num_test})")

move_count = 0

# Move RGB
rgb_files = glob.glob(os.path.join(TRAIN_RGB, "*.jpg"))
for f in rgb_files:
    fname = os.path.basename(f)
    match = re.match(r'^(\d+)_c', fname)
    if match and match.group(1) in test_pids:
        # Move to Test Gallery (RGB)
        shutil.move(f, os.path.join(TEST_RGB, fname))
        
        # Also copy to Query? 
        # For standard ReID, Query is a subset of Gallery (or disjoint cameras).
        # Let's simple copy 1 image per PID/Cam to Query folder for standard protocols.
        # But my validation script points to *folders*.
        # Let's populate Query folder with same images for now (Self-Match is excluded usually? No).
        # Actually simplest: Use TEST_RGB as Query for RGB->IR mode.
        # So I don't necessarily need 'query' folder if script arguments are flexible.
        # But I'll copy to 'query' just in case standard tools need it.
        shutil.copy(os.path.join(TEST_RGB, fname), os.path.join(QUERY, fname))
        
        move_count += 1

# Move IR
ir_files = glob.glob(os.path.join(TRAIN_IR, "*.jpg"))
for f in ir_files:
    fname = os.path.basename(f)
    match = re.match(r'^(\d+)_c', fname)
    if match and match.group(1) in test_pids:
        # Move to Test Gallery (IR) - used for RGB->IR evaluation
        shutil.move(f, os.path.join(TEST_IR, fname))
        move_count += 1

print(f"Moved {move_count} images to Test sets.")
