from ultralytics import YOLO
import torch

def check():
    path = "/home/student/Toan/models/yolo26x_rgbt_init.pt"
    model = YOLO(path)
    img = torch.randn(1, 4, 640, 640)
    
    print("--- Check 1: Model.eval() output ---")
    model.model.eval()
    with torch.no_grad():
        out = model.model(img)
        print(f"Type: {type(out)}")
        if isinstance(out, (list, tuple)):
             print(f"Length: {len(out)}")
             if len(out) > 0: print(f"Item 0 type: {type(out[0])}")
        elif isinstance(out, torch.Tensor):
             print(f"Shape: {out.shape}")
             
    print("\n--- Check 2: Head.training = True hack ---")
    # Identify Head
    head = model.model.model[-1]
    print(f"Head type: {type(head)}")
    
    model.model.eval() # Global eval
    head.training = True # Force head to training mode
    
    with torch.no_grad():
        out = model.model(img)
        print(f"Type: {type(out)}")
        if isinstance(out, dict):
             print(f"Keys: {out.keys()}")
        elif isinstance(out, (list, tuple)):
             print(f"Length: {len(out)}")
             print("Items:", [type(x) for x in out])

if __name__ == '__main__':
    check()
