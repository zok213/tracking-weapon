#!/usr/bin/env python3
"""
MCF Video Inference with ALIGNED RGB-IR Frames (v4)

CRITICAL FIX: Smart center-crop RGB to match IR field of view.
- RGB: 3840x2160 (16:9, aspect 1.778)
- IR:  640x512 (5:4, aspect 1.250)

The RGB camera has wider FOV than IR camera.
We center-crop RGB to match IR's narrower FOV for proper MCF fusion.
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


def calculate_rgb_crop(rgb_w, rgb_h, ir_w, ir_h):
    """
    Calculate center-crop region for RGB to match IR aspect ratio.
    
    The IR camera typically has narrower FOV, so we crop RGB center
    to align the fields of view.
    """
    rgb_aspect = rgb_w / rgb_h
    ir_aspect = ir_w / ir_h
    
    if abs(rgb_aspect - ir_aspect) < 0.01:
        # Already matched
        return 0, 0, rgb_w, rgb_h
    
    if rgb_aspect > ir_aspect:
        # RGB is wider, crop width (center crop horizontally)
        new_width = int(rgb_h * ir_aspect)
        crop_x = (rgb_w - new_width) // 2
        crop_y = 0
        crop_w = new_width
        crop_h = rgb_h
    else:
        # RGB is taller, crop height (center crop vertically)
        new_height = int(rgb_w / ir_aspect)
        crop_x = 0
        crop_y = (rgb_h - new_height) // 2
        crop_w = rgb_w
        crop_h = new_height
    
    return crop_x, crop_y, crop_w, crop_h


def align_rgb_to_ir(rgb_frame, ir_frame):
    """
    Align RGB frame to match IR frame's field of view.
    Returns cropped RGB and IR frames ready for MCF fusion.
    """
    h_rgb, w_rgb = rgb_frame.shape[:2]
    h_ir, w_ir = ir_frame.shape[:2]
    
    # Calculate crop region
    crop_x, crop_y, crop_w, crop_h = calculate_rgb_crop(w_rgb, h_rgb, w_ir, h_ir)
    
    # Apply center crop to RGB
    rgb_cropped = rgb_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    # Ensure IR is 3-channel
    if len(ir_frame.shape) == 2:
        ir_3ch = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:
        ir_3ch = ir_frame
    
    return rgb_cropped, ir_3ch, (crop_x, crop_y, crop_w, crop_h)


def draw_detections(frame, boxes, color=(0, 255, 0)):
    """Draw bounding boxes on frame."""
    for box in boxes:
        x1, y1, x2, y2, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"Person {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def run_mcf_inference(nn_model, rgb_cropped, ir_frame, conf=0.25, imgsz=640, device='cuda:0'):
    """Run MCF inference on aligned RGB+IR frames."""
    h_orig, w_orig = rgb_cropped.shape[:2]
    
    # Resize to model input size
    rgb_resized = cv2.resize(rgb_cropped, (imgsz, imgsz))
    ir_resized = cv2.resize(ir_frame, (imgsz, imgsz))
    
    # Stack RGB + IR for 6-channel input
    stacked = np.concatenate([rgb_resized, ir_resized], axis=2)
    
    # Convert to tensor
    tensor_input = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor_input = tensor_input.to(device)
    
    # Inference
    with torch.no_grad():
        preds = nn_model(tensor_input)
    
    # Post-process
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


def create_visualization(rgb_cropped, ir_frame, boxes, target_height=480):
    """Create side-by-side visualization with aligned frames."""
    h, w = rgb_cropped.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    
    rgb_vis = cv2.resize(rgb_cropped, (new_w, target_height))
    ir_vis = cv2.resize(ir_frame, (new_w, target_height))
    
    # Scale boxes
    scaled_boxes = [[b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale, b[4]] for b in boxes]
    
    # Draw detections
    rgb_det = draw_detections(rgb_vis.copy(), scaled_boxes, (0, 255, 0))
    ir_det = draw_detections(ir_vis.copy(), scaled_boxes, (0, 255, 255))
    
    # Labels
    cv2.putText(rgb_det, "RGB (Cropped & Aligned)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(ir_det, "IR (Thermal)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return np.hstack([rgb_det, ir_det])


def process_video_pair(rgb_path, ir_path, output_path, nn_model, conf=0.25, max_frames=None, device='cuda:0'):
    """Process aligned RGB+IR video pair."""
    rgb_cap = cv2.VideoCapture(str(rgb_path))
    ir_cap = cv2.VideoCapture(str(ir_path))
    
    if not rgb_cap.isOpened() or not ir_cap.isOpened():
        print("ERROR: Cannot open videos")
        return None
    
    rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    rgb_total = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ir_total = int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    rgb_w = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_h = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ir_w = int(ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ir_h = int(ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"RGB: {rgb_w}x{rgb_h} @ {rgb_fps:.1f} FPS, {rgb_total} frames")
    print(f"IR: {ir_w}x{ir_h}, {ir_total} frames")
    
    # Calculate crop
    crop_x, crop_y, crop_w, crop_h = calculate_rgb_crop(rgb_w, rgb_h, ir_w, ir_h)
    print(f"RGB CENTER CROP: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
    
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
        
        # CRITICAL: Align RGB to IR
        rgb_aligned, ir_aligned, _ = align_rgb_to_ir(rgb_frame, ir_frame)
        
        # Run inference
        boxes = run_mcf_inference(nn_model, rgb_aligned, ir_aligned, conf=conf, device=device)
        all_detections.append(len(boxes))
        
        # Create visualization
        combined = create_visualization(rgb_aligned, ir_aligned, boxes)
        
        # Stats
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
    output_dir = project_root / "results/video_inference_aligned"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Find best model
    model_candidates = [
        project_root / "runs/near_view_mcf_phase1/weights/best.pt",
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
    
    print("=" * 80)
    print("MCF VIDEO INFERENCE - ALIGNED RGB-IR (v4)")
    print("=" * 80)
    print(f"Model: {weights}")
    print(f"Device: {device}")
    print("\nKEY FIX: Center-crop RGB to match IR field of view")
    print("  RGB (16:9) → Cropped to (5:4) to match IR camera FOV")
    
    # Load model
    model = YOLO(str(weights))
    nn_model = model.model.to(device)
    nn_model.eval()
    
    # Process all video pairs
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
        
        output_name = f"aligned_{rgb_name.replace('.mp4', '')}.mp4"
        output_path = output_dir / output_name
        
        print(f"\n{'='*80}")
        print(f"Processing: {rgb_name} + {ir_name}")
        print(f"{'='*80}")
        
        result = process_video_pair(rgb_path, ir_path, output_path, nn_model, 
                                   conf=0.2, max_frames=500, device=device)
        if result:
            results.append({'pair': (rgb_name, ir_name), **result})
    
    print("\n" + "=" * 80)
    print("SUMMARY (ALIGNED)")
    print("=" * 80)
    for r in results:
        print(f"  {r['pair'][0]}: {r['frames']} frames, {r['total_detections']} detections ({r['avg_detections']:.2f}/frame)")
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
