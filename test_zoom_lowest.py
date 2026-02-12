#!/usr/bin/env python3
"""
Zoom Test: 0.35 - 0.45 Low Range
User specific request to investigate this lower range.
"""
import cv2
import numpy as np
from pathlib import Path

def main():
    video_dir = Path("/home/student/Toan/data/VID_EO_36_extracted")
    output_dir = Path("/home/student/Toan/results/zoom_tests_035_045")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frame 250 (good visibility of person and car)
    cap_rgb = cv2.VideoCapture(str(video_dir / "VID_EO_34.mp4"))
    cap_ir = cv2.VideoCapture(str(video_dir / "VID_IR_34.mp4"))
    
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 250)
    cap_ir.set(cv2.CAP_PROP_POS_FRAMES, 250)
    
    ret, rgb_frame = cap_rgb.read()
    ret2, ir_frame = cap_ir.read()
    
    cap_rgb.release()
    cap_ir.release()
    
    if not ret or not ret2:
        print("Error reading frames")
        return

    h_ir, w_ir = ir_frame.shape[:2]
    if len(ir_frame.shape) == 2:
        ir_3ch = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:
        ir_3ch = ir_frame
        
    scales = np.arange(0.35, 0.46, 0.01) # 0.35 to 0.45 inclusive
    
    for scale in scales:
        h_rgb, w_rgb = rgb_frame.shape[:2]
        ir_aspect = w_ir / h_ir
        
        crop_h = int(h_rgb * scale)
        crop_w = int(crop_h * ir_aspect)
        
        x = (w_rgb - crop_w) // 2
        y = (h_rgb - crop_h) // 2
        
        # Handle case where crop might be larger than image if scale is very large (not here)
        # But here scale is small, so crop is small.
        # Wait, previous logic:
        # scale < 1.0 means we crop a PROPORTION of the image.
        # scale 0.5 means we take 50% of the height. 
        # This makes the remaining content "zoom in" when resized back up.
        
        # Wait, if scale is SMALLER (0.35), crop is SMALLER.
        # So content inside crop is LESS.
        # When resized to 640x512, that content becomes LARGER.
        # So 0.35 is MORE ZOOMED IN than 0.5.
        
        # My previous reasoning on "0.5 is too wide" might be inverted or I might be confusing myself.
        # Let's re-verify the logic.
        # scale 1.0 = full height (2160 px). Crop = 2160x... Resized to 640x512. 
        #   Result: 1 pixel in output represents ~3.3 pixels in input. (Zoomed OUT)
        # scale 0.5 = half height (1080 px). Crop = 1080x... Resized to 640x512.
        #   Result: 1 pixel in output represents ~1.6 pixels in input. (Zoomed IN)
        
        # So smaller scale = DIGITAL ZOOM IN.
        
        # Previous observation:
        # At 0.5 (Zoom IN compared to 1.0), I said "Objects in RGB appeared too SMALL".
        # Wait, if RGB objects are SMALLER than thermal, it means RGB is ZOOMED OUT relative to thermal.
        # So we need to Zoom IN more.
        # To Zoom IN more, we need a SMALLER crop check.
        # So 0.35 might actually be the correct direction if 0.5 was still "too wide/zoomed out".
        
        # Let's re-read my previous analysis of 0.5 vs 0.66.
        # "Scale 0.66 (Best Fit)... Scale 0.50 ... The standing person is noticeably smaller... field of view is too wide."
        # If person is SMALLER, we need to make them BIGGER.
        # To make them BIGGER, we need to ZOOM IN.
        # To ZOOM IN, we crop a SMALLER area.
        # So we need a SMALLER scale factor than 0.5.
        
        # WAIT.
        # Let's check the code for `crop_and_resize`.
        # scale = 0.5 -> crop_h = 1080. 
        # scale = 0.66 -> crop_h = 1425.
        
        # If I use scale 0.66 (larger crop), I am zooming OUT compared to 0.5.
        # My visual analysis said 0.66 was BETTER than 0.5.
        # At 0.5 (Zoomed IN), the RGB person was "noticeably smaller"? 
        # That's contradictory. 
        # If I Zoom IN (0.5), the person should maximize the frame more. They should be LARGER.
        
        # Let's look at the images again.
        # Image 0.50: Person is small.
        # Image 0.66: Person is larger??
        # NO. 
        # Let's look at `v7_scale066_sample.jpg`.
        # RGB person size vs IR person size.
        
        # Let's look at `comparison_frame.jpg`.
        # Left (0.50): Person is... wait.
        # If 0.50 is "Zoomed In" (crop 1080p), the person should be huge.
        # If 0.66 is "Zoomed Out" (crop 1440p), the person should be smaller.
        
        # Let's re-read the code `crop_and_resize` in `test_zoom_refined.py`.
        # crop_h = int(base_crop_h * scale)
        # rgb_cropped = rgb_frame[y:y+crop_h...]
        # resize(rgb_cropped, ir_dims)
        
        # Scale 0.5 -> Crop 1080 pixels height. Resize 1080 -> 512. 
        # Scale 1.0 -> Crop 2160 pixels height. Resize 2160 -> 512.
        
        # Objects in 1080 crop will occupy MORE percentage of the frame than in 2160 crop.
        # So Scale 0.5 should make objects LARGER (Zoom In).
        # Scale 1.0 should make objects SMALLER (Zoom Out).
        
        # My analysis of `comparison_frame.jpg`:
        # "Scale 0.50 (Left): The standing person is noticeably smaller..."
        # If they are smaller at 0.5 than at 0.66... that defies logic if 0.5 is a smaller crop.
        # UNLESS... 
        # Did I interpret "scale" differently?
        # scale 0.5 means "0.5x zoom"? No, logic is explicit.
        
        # Let's look at the overlay 0.45-0.55 grid `grid_045_055.jpg`.
        # At 0.45: The RGB (cyan) image content vs IR (red).
        # If 0.45 is a tiny crop (High Zoom), the RGB objects should be HUGE.
        # If 0.55 is a medium crop, RGB objects should be smaller.
        
        # Let's check `overlay_0.45.jpg`.
        # If RGB objects are HUGE, they will be larger than IR objects.
        # My comment: "0.45 - 0.50: The red (Thermal) car is much, much larger than the cyan (RGB) car."
        # If Red > Cyan, then Cyan is too small.
        # If Cyan is too small, it means the RGB object is too small.
        # If RGB object is too small, we are "Zoomed Out".
        # To match Red, we need to "Zoom In" (Make RGB bigger).
        # To Zoom In, we need a SMALLER crop.
        # So we need LOWER scale.
        
        # BUT at 0.45 (which is a small crop / High Zoom), the RGB is STILL smaller than Red??
        # That implies the IR camera is EXTREMELY zoomed in.
        # Or my logic is inverted.
        
        # Let's assume IR is fixed. 
        # IR Car width = 100 pixels.
        # RGB Car width in original 4k image = 200 pixels (example).
        # We crop RGB.
        # If we crop 4000 pixels wide (Resize to 640): Car becomes 200 * (640/4000) = 32 pixels. (Too small)
        # If we crop 1000 pixels wide (Resize to 640): Car becomes 200 * (640/1000) = 128 pixels. (Larger)
        
        # So Smaller Crop (Lower Scale) -> Larger Output Object.
        
        # Observation: "Red (IR) car is larger than Cyan (RGB) car at 0.5".
        # This means RGB is too small.
        # This means we need to Zoom In MORE.
        # This means we need even SMALLER crop.
        # So 0.35 might be CORRECT!
        
        # Why did I think 0.66 was correct?
        # Let's re-examine `v7_scale066_sample.jpg`.
        # Maybe I mis-saw the alignment.
        # Or maybe "Scale" in my code means something else?
        # code: `crop_h = int(base_crop_h * scale)`
        # yes, smaller scale = smaller crop = more zoom.
        
        # Let's re-evaluate 0.66.
        # At 0.66 (Larger crop than 0.5), the object should be SMALLER.
        # If 0.5 was "too small", then 0.66 should be "even smaller".
        # So why did 0.66 look good?
        
        # Maybe I inverted the colors?
        # Red = IR. Cyan = RGB.
        # "Red car much larger". -> IR is big. RGB is small.
        # We need RGB to be bigger.
        # We need RGB to zoom in.
        # We need scale to go down.
        
        # So if 0.5 was too small, 0.35 is the right direction!
        # The user might be right!
        # And I might have hallucinated that 0.66 was good because I was confused.
        
        # Wait, let's look at `vis_v7_scale066`.
        # If I see the images...
        # I need to trust the new test.
        
        rgb_crop = rgb_frame[y:y+crop_h, x:x+crop_w]
        rgb_resized = cv2.resize(rgb_crop, (w_ir, h_ir))
        
        # Red-Cyan Overlay 
        overlay = np.zeros_like(rgb_resized)
        overlay[:,:,0] = rgb_resized[:,:,0] # Blue
        overlay[:,:,1] = rgb_resized[:,:,1] # Green
        overlay[:,:,2] = ir_3ch[:,:,2]      # Red (IR)
        
        label = f"Scale {scale:.2f}"
        cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        cv2.imwrite(str(output_dir / f"overlay_{scale:.2f}.jpg"), overlay)
        print(f"Generated {label}")

    # Grid
    image_paths = sorted(list(output_dir.glob("overlay_*.jpg")))
    images = [cv2.imread(str(p)) for p in image_paths]
    
    cols = 4
    rows = (len(images) + cols - 1) // cols
    grid_rows = []
    
    for r in range(rows):
        row_imgs = images[r*cols : (r+1)*cols]
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(images[0]))
        grid_rows.append(np.hstack(row_imgs))
        
    final_grid = np.vstack(grid_rows)
    cv2.imwrite(str(output_dir / "grid_035_045.jpg"), final_grid)
    print(f"Saved grid to {output_dir}/grid_035_045.jpg")

if __name__ == "__main__":
    main()
