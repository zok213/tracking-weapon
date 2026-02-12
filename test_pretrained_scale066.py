#!/usr/bin/env python3
"""
Pretrained Model Benchmark - Scale 0.66 (Perfect Alignment)
Goal: Isolate if failure was due to Misalignment (Scale 0.415) or Domain Shift.
Models:
1. LLVIP (Low Light Pedestrians)
2. M3FD (Multi-Target)
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

# Config - The "Engineering" Choice
OPTIMAL_SCALE = 0.66
X_OFFSET = 0
Y_OFFSET = 0 

def align_frames(rgb_frame, ir_frame, scale=OPTIMAL_SCALE, x_off=X_OFFSET, y_off=Y_OFFSET):
    h_rgb, w_rgb = rgb_frame.shape[:2]
    h_ir, w_ir = ir_frame.shape[:2]
    ir_aspect = w_ir / h_ir
    
    crop_h = int(h_rgb * scale)
    crop_w = int(h_rgb * ir_aspect * scale)
    
    x = (w_rgb - crop_w) // 2 + x_off
    y = (h_rgb - crop_h) // 2 + y_off
    
    x = max(0, min(x, w_rgb - crop_w))
    y = max(0, min(y, h_rgb - crop_h))
    
    rgb_cropped = rgb_frame[y:y+crop_h, x:x+crop_w]
    rgb_aligned = cv2.resize(rgb_cropped, (w_ir, h_ir))
    
    if len(ir_frame.shape) == 2:
        ir_aligned = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:
        ir_aligned = ir_frame
        
    return rgb_aligned, ir_aligned

def run_inference(nn_model, rgb, ir, conf=0.15, device='cuda:0'): # Lower conf to catch anything
    h, w = rgb.shape[:2]
    imgsz = 640
    
    r_r = cv2.resize(rgb, (imgsz, imgsz))
    i_r = cv2.resize(ir, (imgsz, imgsz))
    
    tensor = torch.from_numpy(np.concatenate([r_r, i_r], axis=2)).permute(2,0,1).unsqueeze(0).float()/255.0
    tensor = tensor.to(device)
    
    with torch.no_grad():
        preds = nn_model(tensor)
        
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    det = non_max_suppression(pred, conf_thres=conf, iou_thres=0.45, max_det=300)[0]
    
    boxes = []
    if len(det) > 0:
        det[:, :4] *= (w / imgsz)
        boxes = det.cpu().numpy()
        
    return boxes

def visualize(rgb, ir, boxes, title):
    h, w = rgb.shape[:2]
    vis_rgb = rgb.copy()
    vis_ir = ir.copy()
    
    for *xyxy, conf, cls in boxes:
        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(vis_rgb, p1, p2, (0, 255, 0), 2)
        cv2.rectangle(vis_ir, p1, p2, (0, 255, 255), 2)
        cv2.putText(vis_rgb, f"{conf:.2f}", (p1[0], p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
    cv2.putText(vis_rgb, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return np.hstack([vis_rgb, vis_ir])

def process_video(rgb_path, ir_path, out_path, model, title, max_frames=300):
    cap_r = cv2.VideoCapture(str(rgb_path))
    cap_i = cv2.VideoCapture(str(ir_path))
    
    fps = cap_r.get(cv2.CAP_PROP_FPS)
    w = int(cap_i.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_i.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w*2, h))
    
    frames_processed = 0
    total_dets = 0
    
    pbar = tqdm(total=max_frames, desc=title)
    
    while frames_processed < max_frames:
        r1, f_rgb = cap_r.read()
        r2, f_ir = cap_i.read()
        if not r1 or not r2: break
        
        rgb_a, ir_a = align_frames(f_rgb, f_ir)
        boxes = run_inference(model, rgb_a, ir_a, conf=0.15)
        
        total_dets += len(boxes)
        frames_processed += 1
        
        vis = visualize(rgb_a, ir_a, boxes, f"{title} | Det: {len(boxes)}")
        writer.write(vis)
        
        pbar.update(1)
        
    pbar.close()
    writer.release()
    avg = total_dets / frames_processed if frames_processed else 0
    print(f"Finished {title}: {total_dets} detections ({avg:.2f}/frame)")
    return avg

def main():
    base_dir = Path("/home/student/Toan")
    video_dir = base_dir / "data/VID_EO_36_extracted"
    out_dir = base_dir / "results/pretrained_bench_066"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    models = [
        ("LLVIP", base_dir / "weights/LLVIP-yolo11x-RGBT-midfusion-MCF.pt"),
        ("M3FD", base_dir / "weights/M3FD-yolo11x-RGBT-midfusion-MCF-e300-16-lr0=0.001-warmup_epochs1-Adam.pt")
    ]
    
    videos = ["VID_EO_34.mp4"] 
    
    device = 'cuda:0'
    
    print("-" * 60)
    print("Pretrained Benchmark (Scale 0.66 - ALIGNED)")
    print("-" * 60)
    
    for name, path in models:
        print(f"\nLoading {name}: {path}")
        if not path.exists(): continue
            
        yolo = YOLO(str(path))
        nn_model = yolo.model.to(device)
        nn_model.eval()
        
        for v_rgb in videos:
            v_ir = v_rgb.replace("EO", "IR")
            out_name = f"det_{v_rgb.replace('.mp4','')}_{name}_066.mp4"
            
            process_video(video_dir/v_rgb, video_dir/v_ir, out_dir/out_name, 
                         nn_model, f"{name} (0.66) on {v_rgb}")

if __name__ == "__main__":
    main()
