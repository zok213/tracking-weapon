#!/usr/bin/env python3
"""
FINAL COMPARISON: Scale 0.415, Offset (0, 60)
Models:
1. Custom (Phase 1 Best)
2. LLVIP Pretrained
3. M3FD Pretrained

Engineering Logic for Offset:
- User specifies Offset (0, 60) based on visualization (512p).
- 60 pixels in 512p frame = 105 pixels in 2160p original frame.
- Direction: Visual "Down" shift = Crop "Up" shift = Negative Offset.
- Final Y_OFFSET = -105.
"""
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

sys.path.insert(0, "/home/student/Toan/YOLOv11-RGBT")
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression

# =============================================================================
# MANDATORY CONFIGURATION
# =============================================================================
SCALE = 0.415
VISUAL_Y_OFFSET = 60
# Calculation:
# Crop H = 2160 * 0.415 = 896
# Output H = 512
# Ratio = 896 / 512 = 1.75
# Real Offset = 60 * 1.75 = 105
# Crop Direction = -1 * 105 = -105
REAL_Y_OFFSET = -105
REAL_X_OFFSET = 0

print(f"CONFIGURATION LOCKED:")
print(f"Scale: {SCALE}")
print(f"Visual Offset: (0, {VISUAL_Y_OFFSET})")
print(f"Applied Crop Offset: ({REAL_X_OFFSET}, {REAL_Y_OFFSET})")

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
    
    preds = model(tensor)
    
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    det = non_max_suppression(pred, conf_thres=0.20, iou_thres=0.45, max_det=300)[0]
    
    return len(det), det

def draw(rgb, ir, boxes, label):
    h, w = rgb.shape[:2]
    vis = rgb.copy()
    
    # Scale boxes from 640 to 512/640
    scale_y = h / 640
    scale_x = w / 640
    
    for *xyxy, conf, cls in boxes:
        x1, y1 = int(xyxy[0] * scale_x), int(xyxy[1] * scale_y)
        x2, y2 = int(xyxy[2] * scale_x), int(xyxy[3] * scale_y)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    return vis

def main():
    root = Path("/home/student/Toan")
    video_dir = root / "data/VID_EO_36_extracted"
    out_dir = root / "results/final_comparison_0415"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Define Models
    models_cfg = [
        ("Custom_Best", root / "runs/near_view_mcf_phase1/weights/best.pt"),
        ("LLVIP_Pretrain", root / "weights/LLVIP-yolo11x-RGBT-midfusion-MCF.pt"),
        ("M3FD_Pretrain", root / "weights/M3FD-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.001-warmup_epochs1-Adam.pt")
    ]
    
    # 2. Setup Video
    rgb_path = video_dir / "VID_EO_34.mp4"
    ir_path = video_dir / "VID_IR_34.mp4"
    
    cap_r = cv2.VideoCapture(str(rgb_path))
    cap_i = cv2.VideoCapture(str(ir_path))
    total_frames = int(cap_i.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_r.get(cv2.CAP_PROP_FPS)
    w = int(cap_i.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_i.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output Video: Side-by-Side-by-Side? 
    # Let's do a 2x2 grid. 
    # [Custom] [LLVIP]
    # [M3FD]   [IR_Raw]
    out_w = w * 2
    out_h = h * 2
    output_path = out_dir / "comparison_grid_0415.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                            20, (out_w, out_h)) # Lower FPS for clarity
    
    print(f"Processing Video: {rgb_path}")
    print(f"Output: {output_path}")
    
    # Load all models to memory? Might OOM.
    # We should run inference frame by frame? No, switching models is slow.
    # Better: Run each model on the whole video, save detections list.
    # Then combine.
    
    detections_map = { name: [] for name, _ in models_cfg }
    
    for name, weight_path in models_cfg:
        print(f"\n>>> Running {name}...")
        if not weight_path.exists():
            print(f"CRITICAL ERROR: Weight file missing: {weight_path}")
            continue
            
        model = YOLO(str(weight_path)).model.to('cuda:0')
        model.eval()
        
        cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap_i.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_idx = 0
        pbar = tqdm(total=300) # Process 300 frames
        
        while frame_idx < 300:
            ret1, fr = cap_r.read()
            ret2, fi = cap_i.read()
            if not ret1 or not ret2: break
            
            ra, ia = align_frames(fr, fi)
            
            with torch.no_grad():
                count, boxes = run_model(model, ra, ia)
                
            # Store boxes (CPU numpy)
            if len(boxes) > 0:
                boxes = boxes.cpu().numpy()
            else:
                boxes = []
                
            detections_map[name].append(boxes)
            frame_idx += 1
            pbar.update(1)
        pbar.close()
        del model
        torch.cuda.empty_cache()
    
    # 3. Generate Grid Video
    print("\n>>> Generating Comparison Video...")
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_i.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    idx = 0
    pbar = tqdm(total=300)
    
    while idx < 300:
        ret1, fr = cap_r.read()
        ret2, fi = cap_i.read()
        if not ret1: break
        
        ra, ia = align_frames(fr, fi)
        
        # Create 4 views
        # View 1: Custom
        boxes = detections_map["Custom_Best"][idx] if idx < len(detections_map["Custom_Best"]) else []
        view1 = draw(ra.copy(), ia, boxes, "Custom")
        cv2.putText(view1, f"Custom (Best.pt): {len(boxes)} det", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # View 2: LLVIP
        boxes = detections_map["LLVIP_Pretrain"][idx] if idx < len(detections_map["LLVIP_Pretrain"]) else []
        view2 = draw(ra.copy(), ia, boxes, "LLVIP")
        cv2.putText(view2, f"LLVIP: {len(boxes)} det", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # View 3: M3FD
        boxes = detections_map["M3FD_Pretrain"][idx] if idx < len(detections_map["M3FD_Pretrain"]) else []
        view3 = draw(ra.copy(), ia, boxes, "M3FD")
        cv2.putText(view3, f"M3FD: {len(boxes)} det", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                   
        # View 4: IR Raw (Ground Truth)
        if len(ia.shape) == 2: ia = cv2.cvtColor(ia, cv2.COLOR_GRAY2BGR)
        view4 = ia.copy()
        cv2.putText(view4, "IR Thermal (Ground Truth)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Assemble Grid
        top = np.hstack([view1, view2])
        bot = np.hstack([view3, view4])
        grid = np.vstack([top, bot])
        
        writer.write(grid)
        idx += 1
        pbar.update(1)
        
    writer.release()
    pbar.close()
    print("Done!")

if __name__ == "__main__":
    main()
