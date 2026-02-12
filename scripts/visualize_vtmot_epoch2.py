import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
import random

# CONFIG
CHECKPOINT = Path("/home/student/Toan/checkpoints/vtmot_person_only_rgbt_1gpu/epoch2.pt")
DATASET_ROOT = Path("/home/student/Toan/data/VT-MOT_Person_Only")
OUTPUT_DIR = Path("validation_vis_epoch2")
IMG_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45

def load_rgbt_model(ckpt_path):
    print(f"Loading {ckpt_path}...")
    # Load Skeleton
    model_wrapper = YOLO("yolo26x.pt") 
    model = model_wrapper.model
    
    # Patch 4-Channel Input
    w = model.model[0].conv.weight
    new_w = torch.zeros((w.shape[0], 4, w.shape[2], w.shape[3]), device=w.device)
    model.model[0].conv.weight = torch.nn.Parameter(new_w)
    model.model[0].conv.in_channels = 4
    
    # Surgery Head (NC=1)
    head = model.model[-1]
    ch = [m[0].conv.in_channels for m in head.cv2]
    new_head = Detect(nc=1, ch=ch)
    new_head.i = head.i
    new_head.f = head.f
    new_head.type = head.type
    model.model[-1] = new_head
    model.nc = 1
    
    # Load Weights
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.to('cuda').eval()
    return model

def preprocess_rgbt(img_path, ir_dir):
    rgb = cv2.imread(str(img_path))
    ir_path = ir_dir / img_path.name
    ir = cv2.imread(str(ir_path))
    
    # Resize/Pad
    h0, w0 = rgb.shape[:2]
    r = IMG_SIZE / max(h0, w0)
    interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
    if r != 1:
        rgb = cv2.resize(rgb, (int(w0 * r), int(h0 * r)), interpolation=interp)
        ir = cv2.resize(ir, (int(w0 * r), int(h0 * r)), interpolation=interp)
    
    shape = (IMG_SIZE, IMG_SIZE)
    dw = shape[1] - rgb.shape[1]
    dh = shape[0] - rgb.shape[0]
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    
    rgb = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    ir = cv2.copyMakeBorder(ir, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    
    # 4-Channel Stack
    ir_1ch = ir[..., :1]
    input_img = np.concatenate([rgb, ir_1ch], axis=2)
    input_img = input_img.transpose((2, 0, 1))[::-1] # HWC->CHW, BGR->RGB
    input_img = np.ascontiguousarray(input_img)
    tensor = torch.from_numpy(input_img).float() / 255.0
    return tensor.unsqueeze(0).cuda(), rgb # Return original processed RGB for drawing

def postprocess(preds, conf_thresh=0.25):
    # Debug Input
    print(f"  Raw Preds Type: {type(preds)}")
    if isinstance(preds, tuple): preds = preds[0]
    if isinstance(preds, list): preds = preds[0]
    if isinstance(preds, dict): preds = list(preds.values())[0]

    if not hasattr(preds, 'shape'):
        print(f"  ERROR: Preds is still not tensor? {type(preds)}")
        return []

    print(f"  Postprocess Input Shape: {preds.shape}")
    
    # Expected: [Batch, 5, Anchors] e.g. [1, 5, 8400]
    if preds.ndim == 3:
        # [1, 5, 8400] -> [1, 8400, 5] -> [0] -> [8400, 5]
        preds = preds.permute(0, 2, 1)
        pred = preds[0] 
    elif preds.ndim == 2:
        # [5, 8400] -> [8400, 5]
        pred = preds.permute(1, 0)
    else:
        print(f"  ERROR: Unexpected shape {preds.shape}")
        return []
        
    print(f"  Processing {pred.shape[0]} anchors...")
    
    # Filter by conf
    # Col 4 is max class score? No, v8 format:
    # 0,1,2,3 = xywh (center)
    # 4... = class scores
    
    boxes = pred[:, :4]
    scores = pred[:, 4:]
    
    class_scores, class_ids = torch.max(scores, 1)
    mask = class_scores > conf_thresh
    
    boxes = boxes[mask]
    scores = class_scores[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0: return []
    
    # Convert xywh -> xyxy
    xyxy = torch.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    
    # NMS (Torchvision)
    import torchvision
    keep = torchvision.ops.nms(xyxy, scores, IOU_THRESH)
    
    return torch.cat([xyxy[keep], scores[keep].unsqueeze(1), class_ids[keep].unsqueeze(1)], dim=1)

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = load_rgbt_model(CHECKPOINT)
    
    img_dir = DATASET_ROOT / "images/train"
    ir_dir = DATASET_ROOT / "images_ir/train"
    
    # Pick 10 random images (deterministically)
    random.seed(100) # Seed 100
    all_imgs = sorted(list(img_dir.glob("*.jpg")))
    # Validate on END of list (unseen during training in my split logic?)
    # My split logic was: shuffle(train_files)[:90%].
    # But I don't have the shuffle index preserved! 
    # Ah, I set seed 42 in training script.
    
    # Let's just pick random images, statistically some will be training, some val.
    # We just want to see if it predicts ANYTHING.
    test_imgs = random.sample(all_imgs, 10)
    
    print(f"Visualizing {len(test_imgs)} images...")
    
    for img_path in test_imgs:
        tensor, rgb_draw = preprocess_rgbt(img_path, ir_dir)
        
        preds = model(tensor)
        # Handle my Dict issue from before
        if isinstance(preds, dict): preds = list(preds.values())[0] # usually [0] is boxes+scores concatenated? No..
        # Wait, inside v16h logs:
        # Key 'boxes': [1, 64, 8400] ?
        # Key 'scores': [1, 1, 8400] ? 
        # Actually YOLO head output (inference) is usually one tensor [1, 4+nc, 8400].
        # In Training mode it returns Dict.
        # In Eval mode (`model.eval()`), it should return the Tensor!
        # I set `model.eval()`.
        
        if isinstance(preds, list): preds = preds[0]
        
        detections = postprocess(preds)
        
        # Draw
        if len(detections) > 0:
            print(f"  {img_path.name}: Found {len(detections)} objects.")
            for det in detections:
                x1, y1, x2, y2, conf, cls = det.tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                original = rgb_draw.copy() # BGR
                cv2.rectangle(rgb_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(rgb_draw, f"{conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f"  {img_path.name}: No detections.")
            
        cv2.imwrite(str(OUTPUT_DIR / img_path.name), rgb_draw)
        
    print(f"Saved visualizations to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
