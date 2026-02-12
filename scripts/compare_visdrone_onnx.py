
import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# Config
MODEL_PATH = "/home/student/Toan/models/yolo11visdronemot.onnx"
SEQS = [
    "/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/test/photo-0319-18",
    "/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT/images/test/photo-0319-23"
]
OUTPUT_DIR = "/home/student/Toan/tracking/benchmark/vis_comparison"
CONF_THRESH = 0.25
IOU_THRESH = 0.5

def compare_seq(seq_path):
    seq_name = Path(seq_path).name
    img_dir = Path(seq_path) / "visible"
    gt_path = Path(seq_path) / "gt" / "gt.txt"
    out_video_path = os.path.join(OUTPUT_DIR, f"{seq_name}_comparison.mp4")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"🎬 Processing {seq_name}...")
    
    # Load GT
    gt_data = {} # frame_id -> list of [x1,y1,x2,y2]
    if gt_path.exists():
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                # VisDrone/MOT Format: frame, id, x, y, w, h, score, cls, vis
                # VT-MOT might be similar.
                try:
                    fid = int(parts[0])
                    # x,y,w,h
                    x1 = float(parts[2])
                    y1 = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    x2 = x1 + w
                    y2 = y1 + h
                    cls = int(parts[7])
                    vis = int(parts[8]) if len(parts) > 8 else 1
                    
                    # Store as x1,y1,x2,y2
                    if fid not in gt_data: gt_data[fid] = []
                    gt_data[fid].append([x1, y1, x2, y2, cls]) # Keep cls for debug
                except:
                    pass

    # Load Model
    # Explicitly specific task='detect' helps sometimes with generic onnx loading in Ultralytics versions
    model = YOLO(MODEL_PATH, task='detect') 
    
    # Video Writer setup
    images = sorted(list(img_dir.glob("*.jpg")))
    if not images:
        print(f"❌ No images found in {img_dir}")
        return

    first_img = cv2.imread(str(images[0]))
    h, w = first_img.shape[:2]
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
    
    # Metrics
    tp = 0
    fp = 0
    fn = 0
    
    for img_file in tqdm(images):
        # Parse Frame ID check filename format
        # usually 00001.jpg
        try:
            fid = int(img_file.stem)
        except:
            fid = -1
            
        frame = cv2.imread(str(img_file))
        
        # 1. Prediction
        results = model.predict(frame, verbose=False, conf=CONF_THRESH, device=0) # Switch to GPU
        # If onnx needs GPU, device=0 works if onnxruntime-gpu installed.
        
        preds = []
        for box in results[0].boxes:
            b = box.xyxy[0].cpu().numpy() # x1, y1, x2, y2
            conf = float(box.conf)
            cls = int(box.cls)
            preds.append(b)
            
            # Draw Pred (Green)
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"Pred {conf:.2f}", (int(b[0]), int(b[1])-5), 0, 0.5, (0, 255, 0), 1)

        # 2. GT
        gts = gt_data.get(fid, [])
        matched_gt = [False] * len(gts)
        
        for i, g in enumerate(gts):
            gx1, gy1, gx2, gy2, gcls = g
            # Draw GT (Blue)
            cv2.rectangle(frame, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (255, 0, 0), 2)
            # cv2.putText(frame, f"GT", (int(gx1), int(gy1)-5), 0, 0.5, (255, 0, 0), 1)
            
            # Metric Calculation
            # Find best IoU match
            best_iou = 0
            for p in preds:
                # Calc IoU
                px1, py1, px2, py2 = p
                
                xx1 = max(gx1, px1)
                yy1 = max(gy1, py1)
                xx2 = min(gx2, px2)
                yy2 = min(gy2, py2)
                
                ww = max(0, xx2 - xx1)
                hh = max(0, yy2 - yy1)
                inter = ww * hh
                union = (gx2-gx1)*(gy2-gy1) + (px2-px1)*(py2-py1) - inter
                iou = inter / (union + 1e-6)
                if iou > best_iou:
                    best_iou = iou
            
            if best_iou >= IOU_THRESH:
                tp += 1
                matched_gt[i] = True
            else:
                fn += 1 # GT missed
        
        # FP = Preds that didn't match any GT (Rough approx, precise matching requires hungarian or greedy assignment loop)
        # Simple count: Excess preds - Matched GTs? 
        # Actually, let's just count total matched preds.
        # This is a Rough metric for "Fast" request.
        # Approximate: FP = Total Preds - True Positives (assuming 1-to-1 match enforced, but simplified here)
        fps_in_frame = max(0, len(preds) - sum(matched_gt))
        fp += fps_in_frame

        # Overlay Metrics
        cv2.putText(frame, f"Frame: {fid} | TP: {tp} FP: {fp} FN: {fn}", (10, 30), 0, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
    out.release()
    print(f"✅ Saved video: {out_video_path}")
    
    # Calculate Final Metrics
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    print(f"📊 Metrics for {seq_name}:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")

if __name__ == "__main__":
    for seq in SEQS:
        compare_seq(seq)
