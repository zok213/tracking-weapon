import torch
from ultralytics import YOLO
import torch.nn as nn

def create_4ch_checkpoint():
    src_path = '/home/student/Toan/runs/detect/checkpoints/teacher/yolo26x_kust4k/weights/best.pt'
    dst_path = '/home/student/Toan/models/yolo26x_rgbt_init.pt'
    
    print(f"Loading {src_path}...")
    model = YOLO(src_path)
    
    # 1. Access the underlying PyTorch model
    torch_model = model.model
    
    # 2. Modify the first layer
    m = torch_model.model[0] # Conv module
    old_conv = m.conv
    
    print(f"Original first layer: {old_conv.weight.shape}")
    
    # Create new 4-channel conv
    new_conv = nn.Conv2d(
        4, 
        old_conv.out_channels, 
        old_conv.kernel_size, 
        old_conv.stride, 
        old_conv.padding, 
        bias=old_conv.bias is not None
    )
    
    # Initialize weights
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias = nn.Parameter(old_conv.bias)
            
    # Replace
    m.conv = new_conv
    
    # 3. Update Model Metadata (Critical)
    # Ultralytics stores configuration in model.args (sometimes) or model.yaml
    if hasattr(torch_model, 'yaml'):
        torch_model.yaml['ch'] = 4
        print(f"Updated model.yaml['ch'] = 4")
        
    # Also update the higher level YOLO object overrides if present
    if hasattr(model, 'overrides'):
        model.overrides['ch'] = 4
    
    print(f"Modified first layer: {torch_model.model[0].conv.weight.shape}")
    
    # 4. Save the full model (weights + architecture)
    # We save the state dict + extra info to make it a loadable custom checkpoint
    # Ultralytics YOLO.save() might just save weights, let's use torch.save for full control
    # or just use model.save() if it respects the change
    
    # Let's try torch.save of the whole thing slightly adapted for Ultralytics loading
    # Ideally we just want model.save() provided it captures the change
    
    # Actually, model.save() saves 'updates' dict usually.
    # Let's simple modify `model.ckpt` dictionary and save that if we can access it, 
    # but accessing the runtime object `model.model` is verified modified.
    
    # Let's just use torch.save on the state dict and architecture?
    # No, Ultralytics expects a specific dict structure: {'epoch':..., 'best_fitness':..., 'model':...}
    
    ckpt = {
        'epoch': -1,
        'best_fitness': None,
        'model': torch_model, # This saves the modified nn.Module structure!
        'optimizer': None,
        'train_args': {}, # Reset args to empty dict (was None, causing crash)
        'date': None,
        'version': '8.x'
    }
    
    torch.save(ckpt, dst_path)
    print(f"✅ Saved 4-channel checkpoint to {dst_path}")

if __name__ == '__main__':
    create_4ch_checkpoint()
