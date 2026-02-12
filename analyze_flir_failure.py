#!/usr/bin/env python3
"""
Analyze FLIR Model Failure on Frame 450
Goal: Determine if fragmented boxes (Legs/Feet) can be fixed with NMS or if it's a fundamental feature failure.
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

def get_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame

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

def run_test(model, rgb, ir, conf, iou, title):
    imgsz = 640
    r_r = cv2.resize(rgb, (imgsz, imgsz))
    i_r = cv2.resize(ir, (imgsz, imgsz))
    stacked = np.concatenate([r_r, i_r], axis=2) 
    tensor = torch.from_numpy(stacked).permute(2,0,1).unsqueeze(0).float()/255.0
    tensor = tensor.to('cuda:0')
    
    preds = model(tensor)
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    det = non_max_suppression(pred, conf_thres=conf, iou_thres=iou, max_det=100)[0]
    
    vis = rgb.copy()
    scale_y = rgb.shape[0] / imgsz
    scale_x = rgb.shape[1] / imgsz
    
    if len(det) > 0:
        det[:, :4] *= torch.tensor([scale_x, scale_y, scale_x, scale_y], device=det.device)
        for *xyxy, c, cls in det:
            p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
            cv2.putText(vis, f"{model.names[int(cls)]} {c:.2f}", (p1[0], p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
    cv2.putText(vis, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return vis

def main():
    root = Path("/home/student/Toan")
    rgb_path = root / "data/VID_EO_36_extracted/VID_EO_34.mp4"
    ir_path = root / "data/VID_EO_36_extracted/VID_IR_34.mp4"
    weight_path = root / "weights/FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.01-warmup_epochs1-SGD.pt"
    
    # Approx frame where problem occurred (from 3-model grid)
    # The output video frame 150 -> input frame 300+150 = 450
    frame_idx = 450
    
    fr = get_frame(rgb_path, frame_idx)
    fi = get_frame(ir_path, frame_idx)
    ra, ia = align_frames(fr, fi)
    
    model = YOLO(str(weight_path)).model.to('cuda:0')
    model.eval()
    
    # Test 1: Standard
    res1 = run_test(model, ra, ia, conf=0.15, iou=0.45, title="Std (Conf 0.15, IoU 0.45)")
    
    # Test 2: Low-Low NMS (Force merge?) -> High IoU threshold means LESS merging. 
    # Low IoU threshold means MORE merging (aggressive).
    res2 = run_test(model, ra, ia, conf=0.15, iou=0.10, title="Aggressive Merge (IoU 0.10)")
    
    # Test 3: High-Low NMS
    res3 = run_test(model, ra, ia, conf=0.15, iou=0.80, title="Loose Merge (IoU 0.80)")
    
    # Test 4: Higher Conf
    res4 = run_test(model, ra, ia, conf=0.30, iou=0.45, title="High Conf (0.30)")

    grid = np.vstack([
        np.hstack([res1, res2]),
        np.hstack([res3, res4])
    ])
    
    out_path = root / "results/flir_failure_analysis.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"Saved analysis: {out_path}")

if __name__ == "__main__":
    main()
