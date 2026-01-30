#!/usr/bin/env python3
"""
Export model for GitHub (strip optimizer state, reduce size)
YOLO26x: 354MB → ~115MB after stripping optimizer
"""

import torch
from pathlib import Path

def strip_optimizer(model_path, output_path=None):
    """Remove optimizer state to reduce model size"""
    model_path = Path(model_path)
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}_stripped.pt"
    
    # Load checkpoint
    ckpt = torch.load(model_path, map_location='cpu')
    
    # Strip optimizer and training state
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'ema' in ckpt:
        # Keep EMA weights as main model
        ckpt['model'] = ckpt['ema']
        del ckpt['ema']
    if 'updates' in ckpt:
        del ckpt['updates']
    
    # Convert model to half precision for smaller size
    for key in ckpt.get('model', {}).keys() if isinstance(ckpt.get('model'), dict) else []:
        if isinstance(ckpt['model'][key], torch.Tensor):
            ckpt['model'][key] = ckpt['model'][key].half()
    
    # Save stripped model
    torch.save(ckpt, output_path)
    
    # Report size
    orig_size = model_path.stat().st_size / 1e6
    new_size = Path(output_path).stat().st_size / 1e6
    print(f"Original: {orig_size:.1f} MB")
    print(f"Stripped: {new_size:.1f} MB")
    print(f"Reduction: {100 * (1 - new_size/orig_size):.1f}%")
    print(f"Saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/home/student/Toan/runs/detect/checkpoints/teacher/yolo26x_kust4k/weights/best.pt')
    parser.add_argument('--output', default='/home/student/Toan/models/yolo26x_kust4k_best.pt')
    args = parser.parse_args()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    strip_optimizer(args.input, args.output)
