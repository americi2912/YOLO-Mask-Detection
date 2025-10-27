from ultralytics import YOLO

# Load model hasil training
model = YOLO(r"runs/detect/train_final22/weights/best.pt")

# Jalankan deteksi ke semua gambar di folder test_images
model.predict(
    source=r"test_images",  # cukup tulis nama folder
    conf=0.5,               # confidence threshold
    save=True,              # simpan hasil
    show=True               # tampilkan jendela hasil
)