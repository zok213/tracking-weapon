#!/usr/bin/env python3
"""
MCF Video Inference with Side-by-Side RGB/IR Visualization
Tests YOLOv11x MCF model on paired RGB+IR videos from real-world collection.

FIXED v3: Properly handles device placement and 6-channel MCF input.
"""
import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add YOLOv11-RGBT to path
sys.path.insert(0, "/home/student/Toan/YOLOv11-RGBT")

from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression


def draw_detections(frame, boxes, color=(0, 255, 0)):
    """Draw bounding boxes on frame."""
    frame_copy = frame.copy()
    
    for box in boxes:
        x1, y1, x2, y2, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
        
        # Draw box
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"Person {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_copy, (x1, y1 - 20), (x1 + w + 4, y1), color, -1)
        cv2.putText(frame_copy, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame_copy


def run_mcf_inference(nn_model, rgb_frame, ir_frame, conf=0.25, imgsz=640, device='cuda:0'):
    """
    Run MCF inference on paired RGB+IR frames.
    Model expects 6-channel input: [B, 6, H, W]
    """
    h_orig, w_orig = rgb_frame.shape[:2]
    
    # Resize frames to model input size
    rgb_resized = cv2.resize(rgb_frame, (imgsz, imgsz))
    
    # Handle IR - ensure 3 channels
    if len(ir_frame.shape) == 2:
        ir_3ch = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:
        ir_3ch = ir_frame
    ir_resized = cv2.resize(ir_3ch, (imgsz, imgsz))
    
    # Stack RGB and IR: [H, W, 6]
    stacked = np.concatenate([rgb_resized, ir_resized], axis=2)
    
    # Convert to tensor: [1, 6, H, W], normalize to [0, 1]
    tensor_input = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor_input = tensor_input.to(device)
    
    # Run inference
    with torch.no_grad():
        preds = nn_model(tensor_input)
    
    # Post-process with NMS
    boxes = []
    if isinstance(preds, (list, tuple)):
        pred = preds[0]
    else:
        pred = preds
    
    # Apply NMS
    pred_nms = non_max_suppression(pred, conf_thres=conf, iou_thres=0.45, max_det=300)
    
    for det in pred_nms:
        if len(det) > 0:
            for *xyxy, conf_score, cls in det:
                # Scale boxes back to original size
                x1 = float(xyxy[0]) * w_orig / imgsz
                y1 = float(xyxy[1]) * h_orig / imgsz
                x2 = float(xyxy[2]) * w_orig / imgsz
                y2 = float(xyxy[3]) * h_orig / imgsz
                boxes.append([x1, y1, x2, y2, float(conf_score)])
    
    return boxes


def create_side_by_side(rgb_frame, ir_frame, boxes, target_height=540):
    """Create side-by-side visualization."""
    h, w = rgb_frame.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    
    rgb_resized = cv2.resize(rgb_frame, (new_w, target_height))
    
    if len(ir_frame.shape) == 2:
        ir_3ch = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:
        ir_3ch = ir_frame
    ir_resized = cv2.resize(ir_3ch, (new_w, target_height))
    
    # Scale boxes
    scaled_boxes = [[b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale, b[4]] for b in boxes]
    
    # Draw detections
    rgb_with_det = draw_detections(rgb_resized, scaled_boxes, color=(0, 255, 0))
    ir_with_det = draw_detections(ir_resized, scaled_boxes, color=(0, 255, 255))
    
    # Add labels
    cv2.putText(rgb_with_det, "RGB (Visible)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(ir_with_det, "IR (Thermal)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return np.hstack([rgb_with_det, ir_with_det])


def process_video_pair(rgb_path, ir_path, output_path, nn_model, conf=0.25, max_frames=None, device='cuda:0'):
    """Process a pair of RGB and IR videos."""
    rgb_cap = cv2.VideoCapture(str(rgb_path))
    ir_cap = cv2.VideoCapture(str(ir_path))
    
    if not rgb_cap.isOpened() or not ir_cap.isOpened():
        print("ERROR: Cannot open videos")
        return None
    
    rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    rgb_total = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ir_total = int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"RGB: {rgb_total} frames @ {rgb_fps:.1f} FPS")
    print(f"IR: {ir_total} frames")
    
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
        
        # Run inference
        boxes = run_mcf_inference(nn_model, rgb_frame, ir_frame, conf=conf, device=device)
        all_detections.append(len(boxes))
        
        # Create visualization
        combined = create_side_by_side(rgb_frame, ir_frame, boxes)
        
        # Add stats
        stats = f"Frame: {frame_idx+1}/{total_frames} | Detections: {len(boxes)}"
        cv2.putText(combined, stats, (10, combined.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if out is None:
            h, w = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, min(rgb_fps, 30), (w, h))
        
        out.write(combined)
        pbar.update(1)
    
    pbar.close()
    rgb_cap.release()
    ir_cap.release()
    if out:
        out.release()
    
    total_det = sum(all_detections)
    avg_det = total_det / len(all_detections) if all_detections else 0
    
    print(f"\n✓ Output: {output_path}")
    print(f"  Frames: {len(all_detections)}, Detections: {total_det}, Avg: {avg_det:.2f}/frame")
    
    return {'frames': len(all_detections), 'total_detections': total_det, 'avg_detections': avg_det}


def main():
    project_root = Path("/home/student/Toan")
    video_dir = project_root / "data/VID_EO_36_extracted"
    output_dir = project_root / "results/video_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Find best model
    model_candidates = [
        project_root / "runs/near_view_mcf_phase1/weights/best.pt",
        project_root / "runs/near_view_mcf_phase2/weights/best.pt",
        project_root / "weights/LLVIP-yolo11x-RGBT-midfusion-MCF.pt",
    ]
    
    weights = None
    for w in model_candidates:
        if w.exists():
            weights = w
            break
    
    if not weights:
        print("ERROR: No model found!")
        return
    
    print("=" * 70)
    print("MCF VIDEO INFERENCE - Side-by-Side RGB/IR Detection")
    print("=" * 70)
    print(f"Model: {weights}")
    
    # Load model and move to device
    model = YOLO(str(weights))
    nn_model = model.model.to(device)
    nn_model.eval()
    
    print(f"Model loaded on: {device}")
    
    # Video pairs
    pairs = [
        ("VID_EO_34.mp4", "VID_IR_34.mp4"),
        ("VID_EO_35.mp4", "VID_IR_35.mp4"),
        ("VID_EO_36.mp4", "VID_IR_36.mp4"),
    ]
    
    results = []
    
    for rgb_name, ir_name in pairs:
        rgb_path = video_dir / rgb_name
        ir_path = video_dir / ir_name
        
        if not rgb_path.exists() or not ir_path.exists():
            continue
        
        output_name = f"detection_{rgb_name.replace('.mp4', '')}.mp4"
        output_path = output_dir / output_name
        
        print(f"\n{'='*70}")
        print(f"Processing: {rgb_name} + {ir_name}")
        print(f"{'='*70}")
        
        result = process_video_pair(rgb_path, ir_path, output_path, nn_model, 
                                   conf=0.3, max_frames=500, device=device)
        if result:
            results.append({'pair': (rgb_name, ir_name), **result})
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  {r['pair'][0]}: {r['frames']} frames, {r['total_detections']} detections")
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
