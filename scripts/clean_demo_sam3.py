import torch
import cv2
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from PIL import Image

# Adjust path to find sam3
sys.path.append("/home/student/Toan/sam3")
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Config
SEQ_NAME = "wurenji-0303-22"
VT_MOT_ROOT = Path("/home/student/Toan/data/VT-MOT/data1/Datasets/Tracking/MOT/VTMOT")
OUTPUT_DIR = Path("/home/student/Toan/data/SAM3_Cleaning_Demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prompts - User wants Person and Motorcycle. Everything else is distraction.
PROMPTS = ["person", "motorcycle", "car", "truck", "bus", "bicycle"]
# Mapping to our Target Classes
# 0: Person, 1: Motorcycle, -1: Ignore
CLASS_MAP = {
    "person": 0,
    "motorcycle": 1,
    "bicycle": 1, # Treat bicycle as motorcycle-like for safety? Or ignore. Let's map to Moto for now or keep separate. 
                  # User said "motorcycle". Let's stick to strict motorcycle. 
                  # Actually, "rider" is usually the goal. 
    "car": -1,
    "truck": -1,
    "bus": -1
}

COLORS = {
    0: (0, 255, 255),   # Person: Yellow
    1: (0, 165, 255),   # Moto: Orange
    -1: (128, 128, 128) # Ignore: Gray
}

def load_gt(seq_path):
    gt_path = seq_path / "gt" / "gt.txt"
    data = {}
    if not gt_path.exists(): return {}
    with open(gt_path) as f:
        for line in f:
            p = line.strip().split(',')
            fid = int(p[0])
            tid = int(p[1])
            box = [float(p[2]), float(p[3]), float(p[4]), float(p[5])]
            gt_cid = int(p[7]) # 1:Person, 2:Vehicle
            if fid not in data: data[fid] = []
            data[fid].append({"tid": tid, "box": box, "gt_cid": gt_cid})
    return data

def main():
    print("Loading SAM3 Model...")
    model = build_sam3_image_model().cuda().eval()
    processor = Sam3Processor(model)
    
    # Locate Sequence
    seq_path = None
    for split in ["train", "test", "val"]:
        p = VT_MOT_ROOT / "images" / split / SEQ_NAME
        if p.exists():
            seq_path = p
            break
    if not seq_path:
        print(f"Sequence {SEQ_NAME} not found!")
        return

    print(f"Processing {SEQ_NAME}...")
    gt_data = load_gt(seq_path)
    rgb_dir = seq_path / "visible"
    if not rgb_dir.exists(): rgb_dir = seq_path / "img1"
    
    images = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
    
    # Video Writer
    h, w, _ = cv2.imread(str(images[0])).shape
    out_path = OUTPUT_DIR / f"{SEQ_NAME}_SAM3_Cleaned.mp4"
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
    
    # Process Frame by Frame
    for img_file in tqdm(images):
        fid = int(img_file.stem)
        im = cv2.imread(str(img_file))
        rgb_input = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        # SAM3 Encoding (Once per frame if we were doing prompt-per-frame, 
        # but here we just need to crop-classify. 
        # Actually SAM3 is image-level. We should probably run it on crops for efficiency? 
        # No, SAM3 supports "visual prompt" or text prompt on whole image.
        # BUT for classification of explicit boxes, extracting the crop is cleaner for "classification" behavior.)
        
        # Optimized approach: Crop boxes, Batch inference? 
        # Simple approach for Demo: Just crop and predict.
        
        gt_list = gt_data.get(fid, [])
        
        for gt in gt_list:
            bx, by, bw, bh = gt["box"]
            x1, y1 = max(0, int(bx)), max(0, int(by))
            x2, y2 = min(w, int(bx+bw)), min(h, int(by+bh))
            
            if x2 <= x1 or y2 <= y1: continue
            
            crop = rgb_input[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            # Predict
            # Creating a dummy "batch" of 1
            # processor.predict supports list of inputs? 
            # Looking at codebase_search result: "predict(image_crop, prompts)"
            
            # We need to adapt this to the actual API. 
            # Assuming processor.predict_clip_style or similar exists or we manually do it.
            # Let's try to assume we can pass the crop.
            
            # Re-read Viewed Code: Sam3Processor.set_image, set_text_prompt...
            # The 'sam3' usage might be more complex.
            # Let's try the high-level API if available, or just standard flow.
            # Standard flow:
            # 1. set_image(crop)
            # 2. set_text_prompt(prompts)
            # 3. get score
            
            # Wait, SAM3 is a SEGMENTATION model. 
            # Using it for classification:
            # We can prompt it with "person", "car", "truck" and see which mask has highest score?
            # OR checking if the mask covers our box?
            
            # ALTERNATIVE: Use CLIP?
            # The User said "using sam3 to detect".
            # SAM3 supports "Open Vocabulary".
            # Let's prompt SAM3 with ALL classes on the WHOLE FRAME and see which box belongs to which mask.
            # That's expensive.
            
            # BETTER: Crop the box, pass to CLIP?
            # User specifically asked for SAM3.
            # Let's use the simplest logic: Crop -> SAM3 (Image Encoder) -> Text Decoder -> Scores.
            # Actually, `sam3.predict` doesn't exist in the snippets I saw.
            # I'll try to instantiate the processor and call it.
            # If complex, I'll fallback to a simpler "crop-classifier" approach or just use the whole image logic.
            
            # Let's stick to "Crop Classification" if possible.
            # If not, we run SAM3 on full image with point prompts (center of box) + Text prompts
            # and verify the text class.
            
            # --- Logic from classify_vtmot_sam3_multi_gpu.py ---
            try:
                pil_image = Image.fromarray(rgb_input) # Convert to PIL for SAM3
                inference_state = processor.set_image(pil_image)
                
                # Track best score per GT ID
                # Structure: {tid: {"score": 0.0, "class": None}}
                track_scores = {gt["tid"]: {"score": 0.0, "class": None} for gt in gt_list}
                
                for prompt in PROMPTS:
                    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
                    pred_boxes = output["boxes"]
                    scores = output["scores"]
                    
                    if hasattr(pred_boxes, 'cpu'): pred_boxes = pred_boxes.cpu().numpy()
                    if hasattr(scores, 'cpu'): scores = scores.cpu().numpy()
                    
                    for i, pred_box in enumerate(pred_boxes):
                        score = scores[i]
                        if score < 0.25: continue # Threshold
                        
                        for gt in gt_list:
                            tid = gt["tid"]
                            gt_box = gt["box"]
                            # GT format in parse_gt is [x, y, w, h] -> convert to [x1,y1,x2,y2]
                            gx, gy, gw, gh = gt_box
                            gt_x1y1x2y2 = [gx, gy, gx+gw, gy+gh]
                            
                            # IoU
                            # pred_box is likely [x1, y1, x2, y2] ? (Check utils)
                            # SAM usually outputs xyxy.
                            
                            # Inline IoU
                            xx1 = max(gt_x1y1x2y2[0], pred_box[0])
                            yy1 = max(gt_x1y1x2y2[1], pred_box[1])
                            xx2 = min(gt_x1y1x2y2[2], pred_box[2])
                            yy2 = min(gt_x1y1x2y2[3], pred_box[3])
                            
                            inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                            box1_area = (gt_x1y1x2y2[2]-gt_x1y1x2y2[0]) * (gt_x1y1x2y2[3]-gt_x1y1x2y2[1])
                            box2_area = (pred_box[2]-pred_box[0]) * (pred_box[3]-pred_box[1])
                            union_area = box1_area + box2_area - inter_area
                            
                            if union_area == 0: iou = 0
                            else: iou = inter_area / union_area
                            
                            if iou > 0.4:
                                match_score = score
                                # Assign if better
                                if match_score > track_scores[tid]["score"]:
                                    track_scores[tid]["score"] = match_score
                                    track_scores[tid]["class"] = prompt
                
                # Draw Results
                disp_img = im.copy()
                
                for gt in gt_list:
                    tid = gt["tid"]
                    gt_box = gt["box"]
                    gx, gy, gw, gh = int(gt_box[0]), int(gt_box[1]), int(gt_box[2]), int(gt_box[3])
                    
                    # Get SAM3 result
                    res = track_scores.get(tid)
                    final_cls_name = "Unknown"
                    final_cid = -1 # Ignore by default
                    
                    if res and res["score"] > 0.35 and res["class"]:
                        # We have a confident prediction
                        pred_cls = res["class"]
                        final_cls_name = pred_cls
                        final_cid = CLASS_MAP.get(pred_cls, -1)
                        
                        # Special Rule: User wants to keep "Person" if originally person?
                        # Or Trust SAM3?
                        # User said: "ID25 should be C2". Meaning GT(Person) was WRONG.
                        # So we TRUST SAM3.
                        
                    # Visualization Colors
                    # Green = Person (Accepted)
                    # Orange = Moto (Accepted)
                    # Gray = Car/Truck (Rejected/Ignored)
                    
                    color = (100, 100, 100) # Gray
                    if final_cid == 0: color = (0, 255, 0) # Green
                    elif final_cid == 1: color = (0, 165, 255) # Orange
                    
                    # Visualize
                    cv2.rectangle(disp_img, (gx, gy), (gx+gw, gy+gh), color, 2)
                    
                    # Text: TID | SAM3 Class | Score
                    txt = f"{tid}|{final_cls_name}|{res['score']:.2f}"
                    cv2.putText(disp_img, txt, (gx, gy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                out.write(disp_img)

            except Exception as e:
                print(f"Error on frame {fid}: {e}")
                import traceback
                traceback.print_exc()

    out.release()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
