#!/usr/bin/env python3
"""
Deep Dive: Car Alignment at 20s
Objective: Remove "misalignment" (offset) variable to compare PURE SIZE (Scale).
Method:
1. Extract frame at 20s.
2. Crop Car ROI.
3. For Scale 0.40 and 0.66:
    - Brute-force search X,Y shift to maximize overlap (minimize difference).
    - Show the BEST ALIGNED result.
    - If Scale 0.40 is "same size just misaligned", it will look perfect after shifting.
"""
import cv2
import numpy as np
from pathlib import Path

def crop_and_scale(rgb, w_ir, h_ir, scale):
    h_rgb, w_rgb = rgb.shape[:2]
    ir_aspect = w_ir / h_ir
    crop_h = int(h_rgb * scale)
    crop_w = int(crop_h * ir_aspect)
    x = (w_rgb - crop_w) // 2
    y = (h_rgb - crop_h) // 2
    if x<0 or y<0: return cv2.resize(rgb, (w_ir, h_ir)) # Fallback
    rgb_crop = rgb[y:y+crop_h, x:x+crop_w]
    return cv2.resize(rgb_crop, (w_ir, h_ir))

def find_best_alignment(rgb_full, ir_full, roi_box):
    """
    roi_box: (x, y, w, h) in IR coordinates (Target)
    """
    rx, ry, rw, rh = roi_box
    ir_roi = ir_full[ry:ry+rh, rx:rx+rw]
    
    best_score = float('inf')
    best_shift = (0, 0)
    best_aligned_img = rgb_full.copy()
    
    # Search range +/- 60 pixels
    search = 60
    
    # Pre-compute IR edges for robust matching (color invariant)
    ir_edges = cv2.Canny(ir_roi, 50, 150)
    
    for dy in range(-search, search, 4): # Step 4 for speed
        for dx in range(-search, search, 4):
            # Shift RGB
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            rgb_shifted_full = cv2.warpAffine(rgb_full, M, (rgb_full.shape[1], rgb_full.shape[0]))
            
            rgb_roi = rgb_shifted_full[ry:ry+rh, rx:rx+rw]
            rgb_edges = cv2.Canny(rgb_roi, 50, 150)
            
            # Metric: Edge difference? 
            # Or simple difference? RGB and IR look different.
            # Edge overlap is best.
            # Maximize Intersection of Edges
            intersection = cv2.bitwise_and(ir_edges, rgb_edges)
            score = -np.sum(intersection) # Maximize negative sum
            
            if score < best_score:
                best_score = score
                best_shift = (dx, dy)
                best_aligned_img = rgb_shifted_full
                
    return best_shift, best_aligned_img

def main():
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    output_dir = Path("/home/student/Toan/results/car_alignment_20s")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap_rgb = cv2.VideoCapture(str(video_dir / "VID_EO_34.mp4"))
    cap_ir = cv2.VideoCapture(str(video_dir / "VID_IR_34.mp4"))
    
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    target_frame = int(fps * 20)
    print(f"FPS: {fps}, Target Frame: {target_frame} (20s)")
    
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    cap_ir.set(cv2.CAP_PROP_POS_FRAMES, 550) # IR might be slightly desynced? or diff fps?
    # Let's assume sync is decent or rely on alignment to fix small temporal offsets.
    # User said "video 34 frame 20s"
    # Actually IR might have diff FPS. Let's just grab frame 500-600 range where car is visible.
    # Previous frame 250 had car. Let's use user request.
    
    ret, rgb = cap_rgb.read()
    # IR has 25fps usually, RGB 29.97?
    # Let's align by timestamp
    rgb_ts = cap_rgb.get(cv2.CAP_PROP_POS_MSEC)
    cap_ir.set(cv2.CAP_PROP_POS_MSEC, rgb_ts) 
    ret2, ir = cap_ir.read()
    
    if not ret or not ret2:
        print("Error reading frames")
        return
        
    h_ir, w_ir = ir.shape[:2]
    if len(ir.shape) == 2:
        ir_3ch = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    else:
        ir_3ch = ir
        
    # Define Car ROI in IR Frame (Approximate)
    # Based on previous grids, Car is top-left quadrant
    # x=150, y=100, w=200, h=150
    roi_box = (150, 80, 200, 150)
    
    scales_to_test = [0.40, 0.42, 0.50, 0.66]
    
    results = []
    
    for scale in scales_to_test:
        # 1. Scale
        rgb_scaled = crop_and_scale(rgb, w_ir, h_ir, scale)
        
        # 2. Align (Register)
        shift, rgb_aligned = find_best_alignment(rgb_scaled, ir_3ch, roi_box)
        
        # 3. Create Overlay
        overlay = np.zeros_like(rgb_aligned)
        overlay[:,:,0] = rgb_aligned[:,:,0] # Cyan
        overlay[:,:,1] = rgb_aligned[:,:,1] # Cyan
        overlay[:,:,2] = ir_3ch[:,:,2]      # Red (IR)
        
        # Draw ROI box
        rx, ry, rw, rh = roi_box
        cv2.rectangle(overlay, (rx, ry), (rx+rw, ry+rh), (255, 255, 0), 2)
        
        # Zoom in on ROI for final visual
        roi_vis = overlay[ry:ry+rh, rx:rx+rw]
        roi_vis = cv2.resize(roi_vis, (rw*2, rh*2), interpolation=cv2.INTER_NEAREST)
        
        label = f"Scale {scale} | Shift {shift}"
        cv2.putText(roi_vis, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        results.append(roi_vis)
        print(f"Processed Scale {scale}: Best shift {shift}")
        
    final_grid = np.hstack(results)
    cv2.imwrite(str(output_dir / "car_alignment_comparison.jpg"), final_grid)
    print(f"Saved: {output_dir}/car_alignment_comparison.jpg")

if __name__ == "__main__":
    main()
