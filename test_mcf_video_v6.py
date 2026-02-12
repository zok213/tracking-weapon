#!/usr/bin/env python3
"""
MCF Video Inference - Final Corrected Version (v6)

CORRECTED alignment based on visual verification:
- Scale 0.5 was WRONG (too aggressive, cuts off content)
- Aspect-ratio crop (2700x2160) correctly matches IR FOV
- Small X offset for parallax adjustment (camera baseline)
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
# CORRECTED ALIGNMENT PARAMETERS
# =============================================================================
# Based on visual verification:
# - IR camera FOV ≈ center 2700x2160 of RGB (aspect-ratio match)
# - Small X offset possible due to camera parallax baseline
# - Y offset = 0 (cameras vertically aligned)

ALIGNMENT_CONFIG = {
    'mode': 'aspect_ratio',   # 'aspect_ratio' or 'scale_factor'
    'x_offset_pixels': 0,     # Horizontal parallax offset (positive = shift right)
    'y_offset_pixels': 0,     # Vertical offset
}


def align_frames_corrected(rgb_frame, ir_frame, config):
    """
    Align RGB frame to IR using CORRECT aspect-ratio crop.
    """
    h_rgb, w_rgb = rgb_frame.shape[:2]  # 3840x2160
    h_ir, w_ir = ir_frame.shape[:2]      # 640x512
    
    # IR aspect ratio
    ir_aspect = w_ir / h_ir  # 1.25
    
    # Crop RGB to match IR aspect ratio at FULL HEIGHT
    crop_h = h_rgb  # 2160
    crop_w = int(crop_h * ir_aspect)  # 2700
    
    # Center with offset
    x = (w_rgb - crop_w) // 2 + config['x_offset_pixels']
    y = config['y_offset_pixels']
    
    # Bounds check
    x = max(0, min(x, w_rgb - crop_w))
    y = max(0, min(y, h_rgb - crop_h))
    
    # Crop RGB
    rgb_cropped = rgb_frame[y:y+crop_h, x:x+crop_w]
    
    # Resize to match IR dimensions
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
    """Create side-by-side visualization with detection overlay."""
    scale = target_height / rgb_aligned.shape[0]
    new_w = int(rgb_aligned.shape[1] * scale)
    
    rgb_vis = cv2.resize(rgb_aligned, (new_w, target_height))
    ir_vis = cv2.resize(ir_aligned, (new_w, target_height))
    
    scaled_boxes = [[b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale, b[4]] for b in boxes]
    
    rgb_det = draw_detections(rgb_vis.copy(), scaled_boxes, (0, 255, 0))
    ir_det = draw_detections(ir_vis.copy(), scaled_boxes, (0, 255, 255))
    
    # Labels
    cv2.putText(rgb_det, "RGB (Aspect-Ratio Aligned)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(ir_det, "IR (Thermal)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return np.hstack([rgb_det, ir_det])


def process_video_pair(rgb_path, ir_path, output_path, nn_model, config, conf=0.25, max_frames=None, device='cuda:0'):
    """Process video pair with correct alignment."""
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
    
    # Calculate crop dimensions
    ir_aspect = ir_w / ir_h
    crop_w = int(rgb_h * ir_aspect)
    crop_h = rgb_h
    
    print(f"RGB: {rgb_w}x{rgb_h} @ {rgb_fps:.1f} FPS")
    print(f"IR: {ir_w}x{ir_h}")
    print(f"RGB crop: {crop_w}x{crop_h} (aspect-ratio matched)")
    print(f"X offset: {config['x_offset_pixels']} pixels")
    
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
        
        # Align frames with CORRECT method
        rgb_aligned, ir_aligned = align_frames_corrected(rgb_frame, ir_frame, config)
        
        # Inference
        boxes = run_mcf_inference(nn_model, rgb_aligned, ir_aligned, conf=conf, device=device)
        all_detections.append(len(boxes))
        
        # Visualization
        combined = create_visualization(rgb_aligned, ir_aligned, boxes)
        
        # Stats
        stats = f"Frame: {frame_idx+1}/{total_frames} | Detections: {len(boxes)} | X-off: {config['x_offset_pixels']}"
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


def test_multiple_offsets(rgb_path, ir_path, output_dir, nn_model, offsets=[-100, -50, 0, 50, 100], device='cuda:0'):
    """Test multiple X offsets to find best alignment."""
    results = []
    
    for offset in offsets:
        config = {
            'mode': 'aspect_ratio',
            'x_offset_pixels': offset,
            'y_offset_pixels': 0,
        }
        output_path = output_dir / f"offset_{offset:+d}.mp4"
        
        print(f"\n--- Testing X offset: {offset} pixels ---")
        result = process_video_pair(rgb_path, ir_path, output_path, nn_model, config, 
                                   conf=0.2, max_frames=100, device=device)
        if result:
            result['offset'] = offset
            results.append(result)
    
    # Find best
    if results:
        best = max(results, key=lambda x: x['total_detections'])
        print(f"\n✓ Best offset: {best['offset']} with {best['total_detections']} detections")
        return best['offset']
    return 0


def main():
    project_root = Path("/home/student/Toan")
    video_dir = project_root / "data/VID_EO_36_extracted"
    output_dir = project_root / "results/video_inference_v6_corrected"
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
    print("MCF VIDEO INFERENCE - CORRECTED ALIGNMENT (v6)")
    print("=" * 80)
    print(f"Model: {weights}")
    print(f"Device: {device}")
    print(f"\nAlignment: Aspect-ratio crop (matches IR FOV)")
    
    # Load model
    model = YOLO(str(weights))
    nn_model = model.model.to(device)
    nn_model.eval()
    
    # First, test multiple offsets on VID_34 to find best
    print("\n--- STEP 1: Find optimal X offset ---")
    rgb_path = video_dir / "VID_EO_34.mp4"
    ir_path = video_dir / "VID_IR_34.mp4"
    
    offset_test_dir = output_dir / "offset_tests"
    offset_test_dir.mkdir(exist_ok=True)
    
    best_offset = test_multiple_offsets(rgb_path, ir_path, offset_test_dir, nn_model, 
                                        offsets=[-100, -50, 0, 50, 100], device=device)
    
    # Process all videos with best offset
    print(f"\n--- STEP 2: Process all videos with X offset = {best_offset} ---")
    
    config = {
        'mode': 'aspect_ratio',
        'x_offset_pixels': best_offset,
        'y_offset_pixels': 0,
    }
    
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
        
        output_name = f"corrected_{rgb_name.replace('.mp4', '')}_xoff{best_offset:+d}.mp4"
        output_path = output_dir / output_name
        
        print(f"\n{'='*80}")
        print(f"Processing: {rgb_name} + {ir_name}")
        print(f"{'='*80}")
        
        result = process_video_pair(rgb_path, ir_path, output_path, nn_model,
                                   config, conf=0.2, max_frames=500, device=device)
        if result:
            results.append({'pair': (rgb_name, ir_name), **result})
    
    print("\n" + "=" * 80)
    print(f"SUMMARY (CORRECTED ALIGNMENT, X offset = {best_offset})")
    print("=" * 80)
    for r in results:
        print(f"  {r['pair'][0]}: {r['frames']} frames, {r['total_detections']} detections ({r['avg_detections']:.2f}/frame)")
    
    total_all = sum(r['total_detections'] for r in results)
    print(f"\n  TOTAL: {total_all} detections")
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
