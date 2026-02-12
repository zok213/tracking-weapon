import sys
from pathlib import Path
import torch

# Add pipeline scripts to path
sys.path.append("/home/student/Toan/tracking/stage1/scripts")
from rgbt_pipeline_utils import RGBTDataset, SimpleRGBTValidator, RGBTTrainer
from ultralytics import YOLO

ROOT = Path("/home/student/Toan/tracking/stage1")
MODEL_PATH = ROOT / "runs/vtmot_framework/v17_rgbt_yolo26x8/weights/best.pt"
DATA_YAML = ROOT / "configs/vtmot_rgbt.yaml"
PROJECT_DIR = ROOT / "runs/vtmot_framework"
NAME = "v20_validation_manual"

def validate():
    print(f"🔍 Validating Model: {MODEL_PATH}")
    
    # 1. Load the Model (Custom logic to handle 4-channels)
    # We use the Trainer's get_model approach but simplified
    model_ckpt = torch.load(MODEL_PATH, map_location='cuda:0', weights_only=False)
    model = model_ckpt['model'].float().to('cuda:0')
    model.eval()
    
    print("✅ Model Loaded. First conv channels:", model.model[0].conv.in_channels)
    
    # 2. Instantiate Validator
    # We use SimpleRGBTValidator which wraps standard Ultralytics logic
    # but we need to ensure it uses our RGBT dataloader
    
    args = dict(model=MODEL_PATH, data=DATA_YAML, imgsz=640, batch=8, device=0, split='val', project=PROJECT_DIR, name=NAME)
    
    # HACK: Ultralytics validator instantiation is complex.
    # Easiest way "Is it Good?": Run inference on a few validation images and save them.
    # Validation loop with metrics takes time. Let's do visual check first.
    
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import colorstr
    data_cfg = check_det_dataset(DATA_YAML)
    val_path = data_cfg['val']
    
    # Load default hyps for transform building
    from types import SimpleNamespace
    hyp_dict = {'mask_ratio': 4, 'overlap_mask': True, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 
           'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 
           'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0, 'hsv_h': 0.015,
           'hsv_s': 0.7, 'hsv_v': 0.4}
    hyp = SimpleNamespace(**hyp_dict)

    dataset = RGBTDataset(
        img_path=val_path,
        data=data_cfg,
        imgsz=640,
        batch_size=4,
        augment=False,
        hyp=hyp,  # CRITICAL: Pass Wrapped hypers
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0
    )
    
    # Load 4 random images
    import cv2
    import numpy as np
    
    print(f"📸 Visualizing 4 samples from {val_path}")
    
    for i in range(4):
        # RGBTDataset returns: (img_tensor, labels_out, path, shapes)
        # But __getitem__ or get_4channel_batch logic? 
        # RGBTDataset is designed for Loader. Let's use internal load_image
        
        idx = i * 10 # Spread out
        rgb_path = dataset.im_files[idx]
        print(f"Processing: {rgb_path}")
        
        # Manually construct batch
        im_rgbt, _, _ = dataset.load_image(idx) # Returns (im, (h0,w0), shape)
        
        # Preprocess
        # Resize to 640
        h, w = im_rgbt.shape[:2]
        r = min(640/h, 640/w)
        h_new, w_new = int(h*r), int(w*r)
        im_resized = cv2.resize(im_rgbt, (w_new, h_new))
        
        # Pad to 640x640
        padded = np.zeros((640, 640, 4), dtype=np.uint8)
        padded[:h_new, :w_new, :] = im_resized
        
        # To Tensor (B, C, H, W)
        im_tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float().to('cuda:0')
        im_tensor /= 255.0  # 0-255 to 0.0-1.0
        
        # Inference
        with torch.no_grad():
            preds = model(im_tensor)
            
        # NMS
        from ultralytics.utils.nms import non_max_suppression
        preds = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)
        
        # Plot
        # Take RGB part
        vis_img = padded[:, :, :3].copy() # BGR
        
        det = preds[0]
        if len(det):
            print(f"  Found {len(det)} objects")
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_img, f"Person {conf:.2f}", (x1, y1-5), 0, 0.5, (0, 255, 0), 1)
        
        out_path = PROJECT_DIR / NAME / f"val_vis_{i}.jpg"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis_img)
        print(f"  Saved {out_path}")

if __name__ == "__main__":
    validate()
