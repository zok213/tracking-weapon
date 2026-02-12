#!/usr/bin/env python3
"""
FLIR Model - Person Only Detection
Specifics:
- Model: FLIR_aligned3C...
- Class: 0 (Person) only.
- Conf: 0.15 (Low threshold to catch anything).
- Alignment: Scale 0.415, Offset (0, 60).
"""
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

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
    
    if len(ir.shape) == 2:
        ir_a = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    else:
        ir_a = ir
        
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
    # Filter for class 0 (Person)
    det = non_max_suppression(pred, conf_thres=0.15, iou_thres=0.45, classes=[0], max_det=100)[0]
    
    boxes = []
    if len(det) > 0:
        det[:, :4] *= (rgb.shape[1] / imgsz) # Scale directly to output W
        boxes = det.cpu().numpy()
        
    return boxes

def draw(rgb, ir, boxes):
    h, w = rgb.shape[:2]
    vis_rgb = rgb.copy()
    vis_ir = ir.copy()
    
    for *xyxy, conf, cls in boxes:
        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(vis_rgb, p1, p2, (0, 255, 0), 2)
        cv2.rectangle(vis_ir, p1, p2, (0, 255, 255), 2)
        label = f"Person {conf:.2f}"
        cv2.putText(vis_rgb, label, (p1[0], p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
    return np.hstack([vis_rgb, vis_ir])

def main():
    root = Path("/home/student/Toan")
    video_dir = root / "data/VID_EO_36_extracted"
    out_dir = root / "results/flir_person_only"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    weight_path = root / "weights" / "FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.01-warmup_epochs1-SGD.pt"
    
    rgb_path = video_dir / "VID_EO_34.mp4"
    ir_path = video_dir / "VID_IR_34.mp4"
    
    cap_r = cv2.VideoCapture(str(rgb_path))
    cap_i = cv2.VideoCapture(str(ir_path))
    
    w = int(cap_i.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_i.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_path = out_dir / "FLIR_person_only.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                            25, (w*2, h))
    
    print(f"Loading FLIR: {weight_path}")
    model = YOLO(str(weight_path)).model.to('cuda:0')
    model.eval()
    
    total_frames = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames...")
    
    pbar = tqdm(total=total_frames)
    total_dets = 0
    
    while True:
        ret1, fr = cap_r.read()
        ret2, fi = cap_i.read()
        if not ret1 or not ret2: break
        
        ra, ia = align_frames(fr, fi)
        
        with torch.no_grad():
            boxes = run_model(model, ra, ia)
            total_dets += len(boxes)
            
        vis = draw(ra, ia, boxes)
        writer.write(vis)
        pbar.update(1)
        
    writer.release()
    pbar.close()
    print(f"Done! Average detections: {total_dets/total_frames:.2f}")

if __name__ == "__main__":
    main()
