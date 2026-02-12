#!/usr/bin/env python3
"""
Model Comparison Benchmark Script
Compares YOLO models on Size, Parameters, GFLOPs, Inference Speed, and Accuracy.
"""

import sys
import time
import torch
import pandas as pd
from pathlib import Path
from tabulate import tabulate

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'tracking/stage1/scripts'))

from ultralytics import YOLO
from rgbt_pipeline_utils import RGBTDataset

ROOT = Path(__file__).parent.parent


def get_model_info(model_path):
    """Extract model statistics."""
    model = YOLO(model_path)
    info = {
        'Model': Path(model_path).stem,
        'Params (M)': f"{sum(x.numel() for x in model.model.parameters()) / 1e6:.1f}",
        'Size (MB)': f"{Path(model_path).stat().st_size / 1e6:.1f}"
    }
    return model, info


def benchmark_speed(model, device='cuda:0', img_size=640, batch=1, runs=100):
    """Benchmark inference speed."""
    model.model.to(device)
    model.model.eval()
    
    # Dummy input (4-channel)
    input_tensor = torch.zeros((batch, 4, img_size, img_size), device=device)
    
    # Warmup
    for _ in range(10):
        _ = model.model(input_tensor)
        
    # Timing
    start = time.time()
    for _ in range(runs):
        with torch.no_grad():
            _ = model.model(input_tensor)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / runs
    fps = (batch / avg_time)
    
    return fps


def compare_models(models_dict, data_yaml):
    """Compare multiple models."""
    results = []
    
    print(f"🚀 Benchmarking {len(models_dict)} models on {torch.cuda.get_device_name(0)}...")
    
    for name, path in models_dict.items():
        if not Path(path).exists():
            print(f"⚠️ Skipping {name}: Weights not found at {path}")
            continue
            
        print(f"\nProcessing {name}...")
        model, info = get_model_info(path)
        
        # Speed
        fps = benchmark_speed(model, batch=1)
        info['FPS (Batch 1)'] = f"{fps:.1f}"
        
        fps_32 = benchmark_speed(model, batch=32)
        info['FPS (Batch 32)'] = f"{fps_32:.1f}"
        
        # Accuracy (if metrics available in results.csv nearby)
        results_csv = Path(path).parent.parent / 'results.csv'
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            # Strip whitespace from columns
            df.columns = [c.strip() for c in df.columns]
            
            # Get best mAP50
            if 'metrics/mAP50(B)' in df.columns:
                best_map = df['metrics/mAP50(B)'].max()
                info['mAP50'] = f"{best_map:.4f}"
            else:
                info['mAP50'] = "N/A"
        else:
            info['mAP50'] = "Pending"
            
        results.append(info)
        
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("🏆 MODEL COMPARISON LEADERBOARD")
    print("=" * 60)
    print(tabulate(df_results, headers='keys', tablefmt='github', showindex=False))
    
    # Save to Markdown
    with open('model_comparison_report.md', 'w') as f:
        f.write("# 🏆 Model Comparison Leaderboard\n\n")
        f.write(tabulate(df_results, headers='keys', tablefmt='github', showindex=False))
        f.write(f"\n\n*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*")
        
    print("\n✅ Report saved to model_comparison_report.md")


if __name__ == "__main__":
    models = {
        'YOLOv26x (Teacher)': '/home/student/Toan/tracking/stage1/runs/vtmot_framework/v17_rgbt_yolo26x8/weights/best.pt',
        'YOLOv26m (Student 1)': '/home/student/Toan/tracking/stage1/runs/distillation/distill_26x_to_26m_timestamp/weights/best.pt',
        'YOLOv26s (Student 2a)': '/home/student/Toan/tracking/stage1/runs/distillation/distill_26m_to_26s_timestamp/weights/best.pt',
        'YOLOv26n (Student 2b)': '/home/student/Toan/tracking/stage1/runs/distillation/distill_26m_to_26n_timestamp/weights/best.pt'
    }
    
    data_yaml = '/home/student/Toan/tracking/stage1/configs/vtmot_rgbt.yaml'
    
    compare_models(models, data_yaml)
