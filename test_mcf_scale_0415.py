#!/usr/bin/env python3
"""
MCF Video Inference - Scale 0.415 Test
User Request: Scale 0.415, Offset (0, 60) on output frame.
Derived Parameters:
- Scale: 0.415
- Crop Height: 2160 * 0.415 = 896 pix
- Output Height: 512 pix
- Scaling Factor: 896 / 512 = 1.75
- Y Offset in Crop: -60 * 1.75 = -105 pix (Negative = Move Crop Up = Move Image Down)
"""
import os
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
# USER CONFIGURATION
# =============================================================================
OPTIMAL_SCALE = 0.415
X_OFFSET = 0      # pixels in original image
Y_OFFSET = -105   # pixels in original image (Derived from dy=60 in 512p frame)


def align_frames(rgb_frame, ir_frame, scale=OPTIMAL_SCALE, x_off=X_OFFSET, y_off=Y_OFFSET):
    """Align RGB frame to IR using optimal scale factor."""
    h_rgb, w_rgb = rgb_frame.shape[:2]
    h_ir, w_ir = ir_frame.shape[:2]
    
    # IR aspect ratio
    ir_aspect = w_ir / h_ir  # 1.25
    
    # Calculate crop dimensions
    base_crop_h = h_rgb  # 2160
    base_crop_w = int(base_crop_h * ir_aspect)  # 2700
    
    crop_h = int(base_crop_h * scale)
    crop_w = int(base_crop_w * scale)
    
    # Center with offset
    # Note: x_off, y_off are applied to the top-left corner of the crop
    x = (w_rgb - crop_w) // 2 + x_off
    y = (h_rgb - crop_h) // 2 + y_off
    
    # Bounds check
    x = max(0, min(x, w_rgb - crop_w))
    y = max(0, min(y, h_rgb - crop_h))
    
    rgb_cropped = rgb_frame[y:y+crop_h, x:x+crop_w]
    rgb_aligned = cv2.resize(rgb_cropped, (w_ir, h_ir))
    
    if len(ir_frame.shape) == 2:
        ir_aligned = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:
        ir_aligned = ir_frame
    
    return rgb_aligned, ir_aligned, (crop_w, crop_h)


def draw_detections(frame, boxes, color=(0, 255, 0)):
    """Draw bounding boxes."""
    for box in boxes:
        x1, y1, x2, y2, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"Person {conf:.2f}"
        (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def run_mcf_inference(nn_model, rgb_aligned, ir_aligned, conf=0.25, imgsz=640, device='cuda:0'):
    """Run MCF inference on aligned frames."""
    h_orig, w_orig = rgb_aligned.shape[:2]
    
    rgb_resized = cv2.resize(rgb_aligned, (imgsz, imgsz))
    ir_resized = cv2.resize(ir_aligned, (imgsz, imgsz))
    
    stacked = np.concatenate([rgb_resized, ir_resized], axis=2)
    tensor_input = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor_input = tensor_input.to(device)
    
    with torch.no_grad():
        preds = nn_model(tensor_input)
    
    boxes = []
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    pred_nms = non_max_suppression(pred, conf_thres=conf, iou_thres=0.45, max_det=300)
    
    for det in pred_nms:
        if len(det) > 0:
            for *xyxy, conf_score, cls in det:
                x1 = float(xyxy[0]) * w_orig / imgsz
                y1 = float(xyxy[1]) * h_orig / imgsz
                x2 = float(xyxy[2]) * w_orig / imgsz
                y2 = float(xyxy[3]) * h_orig / imgsz
                boxes.append([x1, y1, x2, y2, float(conf_score)])
    
    return boxes


def create_visualization(rgb_aligned, ir_aligned, boxes, crop_dims, scale, target_height=480):
    """Create side-by-side visualization."""
    vis_scale = target_height / rgb_aligned.shape[0]
    new_w = int(rgb_aligned.shape[1] * vis_scale)
    
    rgb_vis = cv2.resize(rgb_aligned, (new_w, target_height))
    ir_vis = cv2.resize(ir_aligned, (new_w, target_height))
    
    scaled_boxes = [[b[0]*vis_scale, b[1]*vis_scale, b[2]*vis_scale, b[3]*vis_scale, b[4]] for b in boxes]
    
    rgb_det = draw_detections(rgb_vis.copy(), scaled_boxes, (0, 255, 0))
    ir_det = draw_detections(ir_vis.copy(), scaled_boxes, (0, 255, 255))
    
    cv2.putText(rgb_det, f"Scale {scale} Off(0,-105)", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(ir_det, "IR (Thermal)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return np.hstack([rgb_det, ir_det])


def process_video_pair(rgb_path, ir_path, output_path, nn_model, scale, conf=0.25, max_frames=None, device='cuda:0'):
    """Process video pair."""
    rgb_cap = cv2.VideoCapture(str(rgb_path))
    ir_cap = cv2.VideoCapture(str(ir_path))
    
    if not rgb_cap.isOpened() or not ir_cap.isOpened():
        print("ERROR: Cannot open videos")
        return None
    
    rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    rgb_total = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ir_total = int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    total_frames = min(rgb_total, ir_total)
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    out = None
    all_detections = []
    
    pbar = tqdm(total=total_frames, desc="Processing")
    
    for frame_idx in range(total_frames):
        ret_rgb, rgb_frame = rgb_cap.read()
        ret_ir, ir_frame = ir_cap.read()
        
        if not ret_rgb or not ret_ir:
            break
        
        rgb_aligned, ir_aligned, crop_dims = align_frames(rgb_frame, ir_frame, scale=scale)
        boxes = run_mcf_inference(nn_model, rgb_aligned, ir_aligned, conf=conf, device=device)
        all_detections.append(len(boxes))
        
        combined = create_visualization(rgb_aligned, ir_aligned, boxes, crop_dims, scale)
        
        if out is None:
            h, w = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, min(rgb_fps, 30), (w, h))
        
        out.write(combined)
        pbar.update(1)
    
    pbar.close()
    rgb_cap.release()
    ir_cap.release()
    if out: out.release()
    
    total_det = sum(all_detections)
    avg_det = total_det / len(all_detections) if all_detections else 0
    
    print(f"✓ Output: {output_path}")
    print(f"  Frames: {len(all_detections)}, Detections: {total_det}, Avg: {avg_det:.2f}/frame")
    
    return {'frames': len(all_detections), 'total_detections': total_det, 'avg_detections': avg_det}


def main():
    project_root = Path("/home/student/Toan")
    video_dir = project_root / "data/VID_EO_36_extracted"
    output_dir = project_root / "results/video_inference_scale0415"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Use best available weights
    weights = project_root / "runs/near_view_mcf_phase1/weights/best.pt"
    if not weights.exists():
        weights = project_root / "weights/LLVIP-yolo11x-RGBT-midfusion-MCF.pt"
    
    print("=" * 80)
    print("MCF VIDEO INFERENCE - SCALE 0.415 TEST")
    print("=" * 80)
    print(f"Model: {weights}")
    print(f"Scale: {OPTIMAL_SCALE}")
    print(f"Offset: ({X_OFFSET}, {Y_OFFSET}) [Configured]")
    print("=" * 80)
    
    model = YOLO(str(weights))
    nn_model = model.model.to(device)
    nn_model.eval()
    
    pairs = [
        ("VID_EO_34.mp4", "VID_IR_34.mp4"),
        ("VID_EO_35.mp4", "VID_IR_35.mp4"),
        ("VID_EO_36.mp4", "VID_IR_36.mp4"),
    ]
    
    results = []
    
    for rgb_name, ir_name in pairs:
        rgb_path = video_dir / rgb_name
        ir_path = video_dir / ir_name
        
        if not rgb_path.exists(): continue
        
        output_name = f"det_{rgb_name.replace('.mp4', '')}_0415.mp4"
        output_path = output_dir / output_name
        
        print(f"\nProcessing: {rgb_name}")
        result = process_video_pair(rgb_path, ir_path, output_path, nn_model,
                                   scale=OPTIMAL_SCALE, conf=0.20, max_frames=500, device=device)
        if result:
            results.append({'pair': rgb_name, **result})
            
    print("\n" + "=" * 80)
    print(f"SUMMARY (Scale {OPTIMAL_SCALE})")
    print("=" * 80)
    for r in results:
        print(f"  {r['pair']}: {r['total_detections']} det ({r['avg_detections']:.2f}/frame)")

if __name__ == "__main__":
    main()
