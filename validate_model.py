# validate_model.py
from ultralytics import YOLO

# Load model terbaik hasil training terakhir
model = YOLO("runs/detect/train_final22/weights/best.pt")

# Evaluasi hasil training (mAP, Precision, Recall)
results = model.val()

print("\n📊 HASIL EVALUASI MODEL 📊")
print(results)