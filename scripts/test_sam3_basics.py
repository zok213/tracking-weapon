import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import sys
import os
sys.path.append("/home/student/Toan/sam3")

# Create valid test image path
TEST_IMG = "/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/test/photo-0310-40/visible/000001.jpg"

if not os.path.exists(TEST_IMG):
    print(f"Error: Test image not found at {TEST_IMG}")
    # Try finding any jpg
    import glob
    imgs = glob.glob("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/test/*/visible/*.jpg")
    if imgs:
        TEST_IMG = imgs[0]
        print(f"Using alternative image: {TEST_IMG}")
    else:
        sys.exit(1)

print("Loading SAM3 model...")
try:
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

# Load image
image = Image.open(TEST_IMG)
print(f"Loaded image: {image.size}")

# Set image
print("Encoding image...")
inference_state = processor.set_image(image)

# Test prompts
prompts = ["car", "truck", "motorcycle", "bus"]
print(f"Testing prompts: {prompts}")

results = {}

# Try prompting sequentially (safest bet for now)
for p in prompts:
    print(f"Prompting: '{p}'")
    output = processor.set_text_prompt(state=inference_state, prompt=p)
    masks = output["masks"]
    scores = output["scores"]
    boxes = output["boxes"]
    results[p] = (boxes, scores)
    print(f"  Found {len(scores)} matches for '{p}'")

print("\nResults Summary:")
for p, (boxes, scores) in results.items():
    if len(scores) > 0:
        max_score = max(scores) if len(scores) > 0 else 0
        print(f"  {p}: {len(scores)} detected (Max score: {max_score:.4f})")
    else:
        print(f"  {p}: 0 detected")

print("\nDone.")
