# train.py
from ultralytics import YOLO
import os

# ✅ Fix OpenMP error (libiomp5md.dll duplicate)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# (optional) batasi thread biar lebih stabil
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

print("📁 Current working directory:", os.getcwd())
print("🚀 Mulai training YOLOv8...")

model = YOLO("yolov8n.pt")

model.train(
    data="yolo_data/data.yaml",
    epochs=20,
    imgsz=640,
    batch=2,
    cache='disk',
    name="train_final2",
    workers=0,
    device='cpu'
)

print("\n✅ Training selesai! Model tersimpan di:")
print("   runs/detect/train_final2/weights/best.pt")