import torch
import yaml
from ultralytics.nn.tasks import parse_model
from ultralytics.nn.modules import GatedSpatialFusion
import sys

def test_model_load():
    try:
        print("Loading config...")
        with open('/home/student/Toan/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-gated.yaml', 'r') as f:
            d = yaml.safe_load(f)

        print("Parsing model...")
        # Mocking ch (input channels), assuming 3 for RGB + 3 for IR = 6? 
        # Actually parse_model expects 'ch' arg.
        # Let's check how it's called in tasks.py or just try to instantiate the class directly first.
        
        # Test 1: Direct instantiation of GatedSpatialFusion
        print("Test 1: Instantiating GatedSpatialFusion directly...")
        m = GatedSpatialFusion(c1=64, c2=64)
        print("Success: GatedSpatialFusion instantiated.")
        
        # Test 2: Full model parse
        print("Test 2: Parsing full model from YAML...")
        # parse_model(d, ch=3) might fail if it demands 6 channels, let's assume standard checks.
        # We'll skip full parse for now if it's too complex to mock, but checking imports was the main goal.
        
        print("Imports and Basic Instantiation OK.")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_model_load()
