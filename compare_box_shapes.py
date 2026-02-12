#!/usr/bin/env python3
"""
Compare Box Shapes: FLIR vs Custom on Frame 451
Goal: Show that FLIR predicts "Tall" boxes (Street View bias) while Custom predicts "Tight" boxes (Learned Drone View).
"""
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/student/Toan/YOLOv11-RGBT")
from ultralytics import YOLO

SCALE = 0.415
REAL_X_OFFSET = 0
REAL_Y_OFFSET = -105

def get_frame(path, f):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
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
    if len(ir.shape) == 2: ir_a = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    else: ir_a = ir
    return rgb_a, ir_a

def run_prediction(model, rgb, ir, conf_thres=0.25):
    imgsz = 640
    r_r = cv2.resize(rgb, (imgsz, imgsz))
    i_r = cv2.resize(ir, (imgsz, imgsz))
    stacked = np.concatenate([r_r, i_r], axis=2) 
    tensor = torch.from_numpy(stacked).permute(2,0,1).unsqueeze(0).float()/255.0
    tensor = tensor.to('cuda:0')
    preds = model(tensor)
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    # Minimal post-process
    from ultralytics.utils.ops import non_max_suppression
    det = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.45, max_det=100)[0]
    
    vis = rgb.copy()
    if len(det) > 0:
        scale_x = rgb.shape[1] / imgsz
        det[:, :4] *= scale_x
        for *xyxy, conf, cls in det:
            p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            cname = model.names[int(cls)]
            color = (0, 255, 0) if cname == 'person' else (255, 0, 0)
            cv2.rectangle(vis, p1, p2, color, 2)
            cv2.putText(vis, cname, (p1[0], p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis

def main():
    root = Path("/home/student/Toan")
    rgb_path = root / "data/VID_EO_36_extracted/VID_EO_34.mp4"
    ir_path = root / "data/VID_EO_36_extracted/VID_IR_34.mp4"
    
    # FLIR (Street View Bias)
    flir_path = root / "weights/FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.01-warmup_epochs1-SGD.pt"
    
    # Custom (Drone View Learned) - Using the best checkpoint found previously
    custom_path = root / "runs/near_view_mcf_phase1/weights/last.pt" 
    
    frame_idx = 451
    fr = get_frame(rgb_path, frame_idx)
    fi = get_frame(ir_path, frame_idx)
    ra, ia = align_frames(fr, fi)
    
    # Run FLIR (Low Conf to see the bad box)
    model_flir = YOLO(str(flir_path)).model.to('cuda:0')
    res_flir = run_prediction(model_flir, ra, ia, conf_thres=0.10)
    cv2.putText(res_flir, "FLIR (Street View Bias)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    
    # Run Custom
    model_custom = YOLO(str(custom_path)).model.to('cuda:0')
    res_custom = run_prediction(model_custom, ra, ia, conf_thres=0.25)
    cv2.putText(res_custom, "Custom (Drone View)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    combined = np.hstack([res_flir, res_custom])
    out_path = root / "results/box_shape_comparison.jpg"
    cv2.imwrite(str(out_path), combined)
    print(f"Saved comparison: {out_path}")

if __name__ == "__main__":
    main()
