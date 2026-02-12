import sys
try:
    from ultralytics import YOLO
    model = YOLO('/home/student/Toan/analysis_c2kd/student_vtmot_best.pth')
    print("Success: Loaded C2KD Student Model")
    print(model.info())
except Exception as e:
    print(f"Failed to load C2KD Student: {e}")

try:
    model2 = YOLO('/home/student/Toan/analysis_c2kd/teacher_vtmot_best.pth')
    print("Success: Loaded C2KD Teacher Model")
    print(model2.info())
except Exception as e:
    print(f"Failed to load C2KD Teacher: {e}")
