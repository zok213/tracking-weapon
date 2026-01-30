#!/bin/bash
# Training Monitor Script
# Checks training progress every 60 seconds

echo "VT-MOT Training Monitor"
echo "======================="

while true; do
    clear
    date
    echo ""
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu --format=csv
    echo ""
    echo "=== YOLO26x Training ==="
    tail -5 /home/student/Toan/logs/yolo26x_kust4k.log 2>/dev/null || echo "Log not found"
    echo ""
    echo "=== RT-DETR-X Training ==="
    tail -5 /home/student/Toan/logs/rtdetr_x_kust4k_v3.log 2>/dev/null || echo "Log not found"
    echo ""
    echo "=== Checkpoints ==="
    ls -la /home/student/Toan/checkpoints/teacher/*/weights/*.pt 2>/dev/null | tail -4 || echo "No checkpoints yet"
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 60
done
