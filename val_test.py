from ultralytics import YOLO

# Load model hasil training terbaik
model = YOLO(r"runs/detect/train_final22/weights/best.pt")

# Evaluasi hasil model (akurasi, precision, recall)
results = model.val()
print(results)