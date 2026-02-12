#!/usr/bin/env python3
"""
MCF Video Inference with Parallax-Corrected Alignment (v5)

Understanding: Dual-lens VIO cameras have physical separation causing parallax.
- Far objects (~20m+) appear roughly aligned
- Close objects (~5m) appear offset

Solution: Use Scale 0.5 crop with adjustable X offset for best average alignment.
This script tests multiple offsets and uses the pre-determined best value.
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
# CAMERA ALIGNMENT PARAMETERS (Determined from analysis)
# =============================================================================
# Based on parallax analysis for dual-lens VIO camera:
# - RGB: 3840x2160 (16:9)
# - IR: 640x512 (5:4)
# - Scale factor: 0.5 (crop 1350x1080 from center)
# - X offset: Fine-tune based on camera baseline

ALIGNMENT_CONFIG = {
    'crop_scale': 0.5,      # Portion of IR-equivalent area to crop from RGB
    'x_offset_pixels': 0,    # Horizontal offset (positive = shift right)
    'y_offset_pixels': 0,    # Vertical offset (positive = shift down)
}


def get_aligned_crop_params(rgb_w, rgb_h, ir_w, ir_h, config):
    """
    Calculate crop parameters for parallax-corrected alignment.
    """
    # Target aspect ratio from IR
    ir_aspect = ir_w / ir_h  # 1.25
    
    # Base crop size (full height, width adjusted for IR aspect)
    base_crop_h = rgb_h  # 2160
    base_crop_w = int(base_crop_h * ir_aspect)  # 2700
    
    # Apply scale factor (zoom level)
    crop_scale = config['crop_scale']
    crop_h = int(base_crop_h * crop_scale)  # 1080
    crop_w = int(base_crop_w * crop_scale)  # 1350
    
    # Calculate center position with offset
    center_x = rgb_w // 2
    center_y = rgb_h // 2
    
    x = center_x - (crop_w // 2) + config['x_offset_pixels']
    y = center_y - (crop_h // 2) + config['y_offset_pixels']
    
    # Bounds check
    x = max(0, min(x, rgb_w - crop_w))
    y = max(0, min(y, rgb_h - crop_h))
    
    return x, y, crop_w, crop_h


def align_frames(rgb_frame, ir_frame, config):
    """
    Align RGB frame to IR using parallax-corrected crop.
    """
    h_rgb, w_rgb = rgb_frame.shape[:2]
    h_ir, w_ir = ir_frame.shape[:2]
    
    # Get crop parameters
    x, y, crop_w, crop_h = get_aligned_crop_params(w_rgb, h_rgb, w_ir, h_ir, config)
    
    # Crop RGB
    rgb_cropped = rgb_frame[y:y+crop_h, x:x+crop_w]
    
    # Resize to match IR
    rgb_aligned = cv2.resize(rgb_cropped, (w_ir, h_ir))
    
    # Ensure IR is 3-channel
    if len(ir_frame.shape) == 2:
        ir_aligned = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:
        ir_aligned = ir_frame
    
    return rgb_aligned, ir_aligned


def draw_detections(frame, boxes, color=(0, 255, 0)):
    """Draw bounding boxes."""
    for box in boxes:
        x1, y1, x2, y2, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"Person {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def run_mcf_inference(nn_model, rgb_aligned, ir_aligned, conf=0.25, imgsz=640, device='cuda:0'):
    """Run MCF inference on aligned frames."""
    h_orig, w_orig = rgb_aligned.shape[:2]
    
    # Resize to model input
    rgb_resized = cv2.resize(rgb_aligned, (imgsz, imgsz))
    ir_resized = cv2.resize(ir_aligned, (imgsz, imgsz))
    
    # Stack for 6-channel input
    stacked = np.concatenate([rgb_resized, ir_resized], axis=2)
    
    # To tensor
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


def create_visualization(rgb_aligned, ir_aligned, boxes, target_height=480):
    """Create side-by-side visualization."""
    scale = target_height / rgb_aligned.shape[0]
    new_w = int(rgb_aligned.shape[1] * scale)
    
    rgb_vis = cv2.resize(rgb_aligned, (new_w, target_height))
    ir_vis = cv2.resize(ir_aligned, (new_w, target_height))
    
    scaled_boxes = [[b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale, b[4]] for b in boxes]
    
    rgb_det = draw_detections(rgb_vis.copy(), scaled_boxes, (0, 255, 0))
    ir_det = draw_detections(ir_vis.copy(), scaled_boxes, (0, 255, 255))
    
    # Labels with alignment info
    cv2.putText(rgb_det, "RGB (Parallax Corrected)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(ir_det, "IR (Thermal)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return np.hstack([rgb_det, ir_det])


def process_video_pair(rgb_path, ir_path, output_path, nn_model, config, conf=0.25, max_frames=None, device='cuda:0'):
    """Process video pair with parallax correction."""
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
    
    print(f"RGB: {rgb_w}x{rgb_h} @ {rgb_fps:.1f} FPS")
    print(f"IR: {ir_w}x{ir_h}")
    
    # Show alignment parameters
    x, y, crop_w, crop_h = get_aligned_crop_params(rgb_w, rgb_h, ir_w, ir_h, config)
    print(f"Alignment: crop({crop_w}x{crop_h}) at ({x}, {y})")
    
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
        
        # Align frames
        rgb_aligned, ir_aligned = align_frames(rgb_frame, ir_frame, config)
        
        # Inference
        boxes = run_mcf_inference(nn_model, rgb_aligned, ir_aligned, conf=conf, device=device)
        all_detections.append(len(boxes))
        
        # Visualization
        combined = create_visualization(rgb_aligned, ir_aligned, boxes)
        
        # Stats
        stats = f"Frame: {frame_idx+1}/{total_frames} | Detections: {len(boxes)} | Scale: {config['crop_scale']} | X-off: {config['x_offset_pixels']}"
        cv2.putText(combined, stats, (10, combined.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
    output_dir = project_root / "results/video_inference_parallax"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Find model
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
    print("MCF VIDEO INFERENCE - PARALLAX CORRECTED (v5)")
    print("=" * 80)
    print(f"Model: {weights}")
    print(f"Device: {device}")
    print(f"\nAlignment Config:")
    print(f"  Crop Scale: {ALIGNMENT_CONFIG['crop_scale']}")
    print(f"  X Offset: {ALIGNMENT_CONFIG['x_offset_pixels']} pixels")
    print(f"  Y Offset: {ALIGNMENT_CONFIG['y_offset_pixels']} pixels")
    
    # Load model
    model = YOLO(str(weights))
    nn_model = model.model.to(device)
    nn_model.eval()
    
    # Process videos
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
        
        output_name = f"parallax_corrected_{rgb_name.replace('.mp4', '')}.mp4"
        output_path = output_dir / output_name
        
        print(f"\n{'='*80}")
        print(f"Processing: {rgb_name} + {ir_name}")
        print(f"{'='*80}")
        
        result = process_video_pair(rgb_path, ir_path, output_path, nn_model, 
                                   ALIGNMENT_CONFIG, conf=0.2, max_frames=500, device=device)
        if result:
            results.append({'pair': (rgb_name, ir_name), **result})
    
    print("\n" + "=" * 80)
    print("SUMMARY (PARALLAX CORRECTED)")
    print("=" * 80)
    for r in results:
        print(f"  {r['pair'][0]}: {r['frames']} frames, {r['total_detections']} detections ({r['avg_detections']:.2f}/frame)")
    
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
