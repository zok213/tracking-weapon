#!/usr/bin/env python3
"""
3-Model Benchmark Grid (3 Rows x 2 Cols)
Models: FLIR, LLVIP, M3FD
Config: Scale 0.415, Offset (0, 60)
Start Frame: 300
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

# =============================================================================
# LOCKED CONFIGURATION
# =============================================================================
SCALE = 0.415
REAL_X_OFFSET = 0
REAL_Y_OFFSET = -105 # Corresponds to Visual Offset (0, 60) in 512p

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
    
    # Mid-Fusion Check: Stacking 6 Channels
    stacked = np.concatenate([r_r, i_r], axis=2) 
    tensor = torch.from_numpy(stacked).permute(2,0,1).unsqueeze(0).float()/255.0
    tensor = tensor.to('cuda:0')
    
    preds = model(tensor) # Standard call
    
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    det = non_max_suppression(pred, conf_thres=0.20, iou_thres=0.45, max_det=300)[0]
    
    return det

def draw_row(rgb, ir, boxes, model_name):
    """Returns a single row image [RGB | IR] with boxes."""
    h, w = rgb.shape[:2]
    vis_rgb = rgb.copy()
    vis_ir = ir.copy()
    
    scale_y = h / 640
    scale_x = w / 640
    
    # Draw boxes
    color = (0, 255, 0)
    for *xyxy, conf, cls in boxes:
        x1, y1 = int(xyxy[0] * scale_x), int(xyxy[1] * scale_y)
        x2, y2 = int(xyxy[2] * scale_x), int(xyxy[3] * scale_y)
        cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(vis_ir, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow on IR
        
    # Add Text
    cv2.putText(vis_rgb, f"{model_name} (RGB) Det:{len(boxes)}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_ir, f"{model_name} (IR)", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
               
    return np.hstack([vis_rgb, vis_ir])

def main():
    root = Path("/home/student/Toan")
    video_dir = root / "data/VID_EO_36_extracted"
    out_dir = root / "results/prediction_grid_3models"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    weights_dir = root / "weights"
    
    models_cfg = [
        ("FLIR", weights_dir / "FLIR_aligned3C-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.01-warmup_epochs1-SGD.pt"),
        ("LLVIP", weights_dir / "LLVIP-yolo11x-RGBT-midfusion-MCF.pt"),
        ("M3FD", weights_dir / "M3FD-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.001-warmup_epochs1-Adam.pt")
    ]
    
    rgb_path = video_dir / "VID_EO_34.mp4"
    ir_path = video_dir / "VID_IR_34.mp4"
    
    cap_r = cv2.VideoCapture(str(rgb_path))
    cap_i = cv2.VideoCapture(str(ir_path))
    
    # Config Output
    w = int(cap_i.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_i.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = w * 2
    out_h = h * 3 # 3 Rows
    
    output_path = out_dir / "models_comparison_grid.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                            20, (out_w, out_h))
    
    print(f"Output: {output_path}")
    
    # Pre-load all models to avoid reloading?
    # 3x YOLO11x might OOM on 24GB VRAM?
    # YOLO11x is ~120MB parameters. 3 of them is fine.
    # We will try loading all 3.
    
    loaded_models = []
    for name, path in models_cfg:
        print(f"Loading {name}...")
        try:
            m = YOLO(str(path)).model.to('cuda:0')
            m.eval()
            loaded_models.append((name, m))
        except Exception as e:
            print(f"Error loading {name}: {e}")
            return

    # Start Frame
    start_frame = 300
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_i.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    pbar = tqdm(desc="Processing Frames", unit="frame")
    
    while True:
        ret1, fr = cap_r.read()
        ret2, fi = cap_i.read()
        if not ret1 or not ret2: break
        
        ra, ia = align_frames(fr, fi)
        rows = []
        
        # Inference for each
        for name, model in loaded_models:
            with torch.no_grad():
                boxes = run_model(model, ra, ia)
                if len(boxes) > 0: boxes = boxes.cpu().numpy()
                else: boxes = []
                
            row_img = draw_row(ra, ia, boxes, name)
            rows.append(row_img)
            
        # Combine
        grid = np.vstack(rows)
        writer.write(grid)
        pbar.update(1)
        
    writer.release()
    pbar.close()
    print("Done!")

if __name__ == "__main__":
    main()
