#!/usr/bin/env python3
"""
Test FLIR on Time 00:16 (Approx Frame 400 or 480)
Goal: Replicate User's "Person 0.28 / Bicycle 0.27" detection.
"""
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/student/Toan/YOLOv11-RGBT")
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression

SCALE = 0.415
REAL_X_OFFSET = 0
REAL_Y_OFFSET = -105

def align_frames(rgb, ir):
    h_rgb, w_rgb = rgb.shape[:2]
    h_ir, w_ir = ir.shape[:2]
    ir_aspect = w_ir / h_ir
    crop_h = int(h_rgb * SCALE)
    crop_w = int(h_rgb * ir_aspect * SCALE)
    x = (w_rgb - crop_w) // 2 + REAL_X_OFFSET
    y = (h_rgb - crop_h) // 2 + REAL_Y_OFFSET
    x = max(0, min(x, w_rgb - crop_w))
    y = max(0, min(y, h_rgb - crop_h))
    rgb_c = rgb[y:y+crop_h, x:x+crop_w]
    rgb_a = cv2.resize(rgb_c, (w_ir, h_ir))
    if len(ir.shape) == 2: ir_a = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    else: ir_a = ir
    return rgb_a, ir_a

def run_model(model, rgb, ir):
    imgsz = 640
    r_r = cv2.resize(rgb, (imgsz, imgsz))
    i_r = cv2.resize(ir, (imgsz, imgsz))
    stacked = np.concatenate([r_r, i_r], axis=2) 
    tensor = torch.from_numpy(stacked).permute(2,0,1).unsqueeze(0).float()/255.0
    tensor = tensor.to('cuda:0')
    preds = model(tensor)
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    det = non_max_suppression(pred, conf_thres=0.10, iou_thres=0.45, max_det=100)[0]
    boxes = []
    if len(det) > 0:
        det[:, :4] *= (rgb.shape[1] / imgsz)
        boxes = det.cpu().numpy()
    return boxes

def draw(rgb, ir, boxes, model_names):
    vis_rgb = rgb.copy()
    vis_ir = ir.copy()
    for *xyxy, conf, cls in boxes:
        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cls_id = int(cls)
        cls_name = model_names.get(cls_id, str(cls_id))
        
        # Highlight Person vs Bicycle
        if cls_name == 'person': color = (0, 255, 0)
        elif cls_name == 'car': color = (255, 0, 0)
        elif cls_name == 'bicycle': color = (0, 0, 255) # RED for Bicycle
        else: color = (255, 255, 0)
            
        cv2.rectangle(vis_rgb, p1, p2, color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(vis_rgb, label, (p1[0], p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return np.hstack([vis_rgb, vis_ir])

def main():
    root = Path("/home/student/Toan")
    rgb_path = root / "data/VID_EO_36_extracted/VID_EO_34.mp4"
    ir_path = root / "data/VID_EO_36_extracted/VID_IR_34.mp4"
    weight_path = root / "weights/FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.01-warmup_epochs1-SGD.pt"

    print(f"Loading Model: {weight_path}")
    model = YOLO(str(weight_path)).model.to('cuda:0')
    model.eval()

    cap = cv2.VideoCapture(str(rgb_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    
    # Target 16 seconds
    target_frame = int(16 * fps)
    print(f"Target Frame (16s): {target_frame}")
    
    # Check a range around specific time because player usage varies
    for f in range(target_frame - 5, target_frame + 5):
        fr = get_frame(rgb_path, f)
        fi = get_frame(ir_path, f)
        ra, ia = align_frames(fr, fi)
        
        with torch.no_grad():
            boxes = run_model(model, ra, ia)
            
        # Check if we found person/bicycle
        found_person = any(model.names[int(cls)] == 'person' for *_,_,cls in boxes)
        found_bike = any(model.names[int(cls)] == 'bicycle' for *_,_,cls in boxes)
        
        if found_person or found_bike:
            print(f"Generating Sample for Frame {f} (Person:{found_person}, Bike:{found_bike})")
            vis = draw(ra, ia, boxes, model.names)
            out = root / f"results/flir_frame_{f}_rep.jpg"
            cv2.imwrite(str(out), vis)
            print(f"Saved: {out}")
            # We only need one good example
            break
            
def get_frame(path, f):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    cap.release()
    return frame

if __name__ == "__main__":
    main()
