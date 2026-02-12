#!/usr/bin/env python3
"""
Quick test of vehicle classification on a single sequence
"""
import sys
sys.path.insert(0, '/home/student/Toan/scripts')

from classify_vtmot_vehicles import VehicleClassifier, process_sequence
from pathlib import Path

# Test on one sequence
VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")
OUTPUT_DIR = Path("/home/student/Toan/data/VT-MOT_4cls_test")

# Find first sequence
test_seq = VT_MOT_ROOT / "images" / "test" / "photo-0310-40"

print(f"Testing on: {test_seq}")
print(f"Output: {OUTPUT_DIR}")

# Initialize classifier
classifier = VehicleClassifier()

# Process
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
processed, vehicles = process_sequence(test_seq, classifier, OUTPUT_DIR / "test")

print(f"\n✅ Test complete!")
print(f"Processed: {processed} annotations")
print(f"Vehicles classified: {vehicles}")

# Show sample output
output_labels = list((OUTPUT_DIR / "test" / "labels" / "photo-0310-40").glob("*.txt"))
print(f"\nGenerated {len(output_labels)} label files")
if output_labels:
    print(f"\nSample label ({output_labels[0].name}):")
    print(output_labels[0].read_text()[:500])
