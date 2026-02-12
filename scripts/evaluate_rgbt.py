#!/usr/bin/env python3
"""
RGBT Model Evaluation Script (Fixed)
- Handles decoded model output [x1, y1, x2, y2, conf, cls]
- Computes mAP50, mAP50-95, Precision, Recall
"""

import torch
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
from collections import defaultdict
import json

def box_iou(box1, box2):
    """Calculate IoU between two boxes [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / (union + 1e-6)

def xywh2xyxy(box, img_w, img_h):
    """Convert YOLO format [cx, cy, w, h] normalized to [x1,y1,x2,y2] absolute"""
    cx, cy, w, h = box
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return [x1, y1, x2, y2]

def compute_ap(recalls, precisions):
    """Compute Average Precision using 11-point interpolation"""
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    
    if len(recalls) == 0:
        return 0
    
    sorted_idx = np.argsort(recalls)
    recalls = recalls[sorted_idx]
    precisions = precisions[sorted_idx]
    
    ap = 0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    return ap

def evaluate_predictions(all_preds, all_gts, iou_thresh=0.5):
    """Compute precision, recall, and AP at given IoU threshold"""
    pred_list = []
    for img_idx, (boxes, scores, classes) in enumerate(all_preds):
        for i in range(len(boxes)):
            pred_list.append((img_idx, scores[i], boxes[i], classes[i]))
    
    if len(pred_list) == 0:
        return 0, 0, 0
    
    pred_list.sort(key=lambda x: x[1], reverse=True)
    gt_matched = defaultdict(set)
    
    tp = []
    fp = []
    
    for img_idx, score, pred_box, pred_cls in pred_list:
        gt_boxes, gt_classes = all_gts[img_idx]
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
            if gt_idx in gt_matched[img_idx]:
                continue
            if gt_cls != pred_cls:
                continue
                
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp.append(1)
            fp.append(0)
            gt_matched[img_idx].add(best_gt_idx)
        else:
            tp.append(0)
            fp.append(1)
    
    tp = np.array(tp).cumsum()
    fp = np.array(fp).cumsum()
    
    total_gt = sum(len(boxes) for boxes, _ in all_gts)
    
    if total_gt == 0:
        return 0, 0, 0
    
    recalls = tp / total_gt
    precisions = tp / (tp + fp + 1e-6)
    
    ap = compute_ap(recalls, precisions)
    final_precision = precisions[-1] if len(precisions) > 0 else 0
    final_recall = recalls[-1] if len(recalls) > 0 else 0
    
    return ap, final_precision, final_recall

def load_labels(lbl_path, img_w, img_h):
    """Load YOLO format labels and convert to xyxy"""
    boxes = []
    classes = []
    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(float(parts[0]))
                    box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                    boxes.append(xywh2xyxy(box, img_w, img_h))
                    classes.append(cls)
    return boxes, classes

def run_inference(model, img_path, device, conf_thresh=0.25):
    """Run inference on a single 4-channel image"""
    # Load image
    img = np.load(img_path)
    orig_h, orig_w = img.shape[:2]
    
    # Preprocess
    img_resized = cv2.resize(img, (640, 640)) if img.shape[:2] != (640, 640) else img
    img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        preds = model(img_tensor)
    
    # Handle output: Model returns tuple, first element is [batch, 300, 6]
    # Format: [x1, y1, x2, y2, conf, cls] in 640x640 scale
    if isinstance(preds, tuple):
        preds = preds[0]
    
    if preds.dim() == 3:
        preds = preds[0]  # Remove batch dim -> [300, 6]
    
    # Filter by confidence
    boxes = []
    scores = []
    classes = []
    
    for det in preds:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        
        if conf < conf_thresh:
            continue
        
        # Scale from 640x640 to original image size
        scale_x = orig_w / 640
        scale_y = orig_h / 640
        
        boxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
        scores.append(float(conf))
        classes.append(int(cls))
    
    return boxes, scores, classes, (orig_w, orig_h)

def main():
    print("="*60)
    print("RGBT Model Evaluation (mAP) - Fixed Version")
    print("="*60)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model_path = "/home/student/Toan/checkpoints/rgbt_vtmot_ddp/best.pt"
    # Using KUST4K for eval (VT-MOT_RGBT missing labels)
    val_img_dir = Path("/home/student/Toan/data/KUST4K_RGBT/images/val")
    val_lbl_dir = Path("/home/student/Toan/data/KUST4K_RGBT/labels/val")
    
    print(f"Loading model from {model_path}...")
    from ultralytics import YOLO
    
    model_obj = YOLO("/home/student/Toan/models/yolo26x_rgbt_init.pt")
    state_dict = torch.load(model_path, map_location=device)
    model_obj.model.load_state_dict(state_dict)
    model_obj.model.to(device)
    model_obj.model.eval()
    
    print("✅ Model loaded successfully")
    
    val_images = sorted(list(val_img_dir.glob('*.npy')))
    print(f"Found {len(val_images)} validation images")
    
    all_preds = []
    all_gts = []
    total_preds = 0
    total_gts = 0
    
    for img_path in tqdm(val_images, desc="Evaluating"):
        pred_boxes, pred_scores, pred_classes, (img_w, img_h) = run_inference(
            model_obj.model, img_path, device, conf_thresh=0.25
        )
        all_preds.append((pred_boxes, pred_scores, pred_classes))
        total_preds += len(pred_boxes)
        
        lbl_path = val_lbl_dir / (img_path.stem + '.txt')
        gt_boxes, gt_classes = load_labels(lbl_path, img_w, img_h)
        all_gts.append((gt_boxes, gt_classes))
        total_gts += len(gt_boxes)
    
    print(f"\nTotal Predictions: {total_preds}")
    print(f"Total Ground Truth: {total_gts}")
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    ap50, prec50, rec50 = evaluate_predictions(all_preds, all_gts, iou_thresh=0.5)
    print(f"mAP@50:     {ap50*100:.2f}%")
    print(f"Precision:  {prec50*100:.2f}%")
    print(f"Recall:     {rec50*100:.2f}%")
    
    aps = []
    for iou in np.arange(0.5, 1.0, 0.05):
        ap, _, _ = evaluate_predictions(all_preds, all_gts, iou_thresh=iou)
        aps.append(ap)
    map50_95 = np.mean(aps)
    print(f"mAP@50-95:  {map50_95*100:.2f}%")
    
    results = {
        'mAP50': round(ap50 * 100, 2),
        'mAP50-95': round(map50_95 * 100, 2),
        'Precision@50': round(prec50 * 100, 2),
        'Recall@50': round(rec50 * 100, 2),
        'num_images': len(val_images),
        'total_predictions': total_preds,
        'total_ground_truth': total_gts,
        'model': model_path
    }
    
    results_path = Path("/home/student/Toan/checkpoints/rgbt_vtmot_ddp/eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {results_path}")

if __name__ == '__main__':
    main()
