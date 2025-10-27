# predict_video.py
from ultralytics import YOLO

# Load model terbaik hasil training (ubah path sesuai folder kamu)
model = YOLO(r"runs/detect/train_final22/weights/best.pt")

# Jalankan deteksi video
model.predict(
    source=r"videos/face_mask_detection.mp4",  # ganti dengan nama file videomu
    conf=0.5,          # confidence threshold
    show=True,         # tampilkan hasil deteksi langsung
    save=True          # simpan hasil ke folder runs/detect/predict
)

print("\n✅ Deteksi video selesai! Cek hasil di folder:")
print("   C:/SEMESTER 5/YOLO-Mask-Detection/runs/detect/predict")