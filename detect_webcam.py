# detect_webcam.py
from ultralytics import YOLO
import cv2
import os

# --- 1️⃣ Atasi error OpenMP (kalau muncul duplikat libiomp5md.dll) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# --- 2️⃣ Load model hasil training kamu ---
model_path = r"runs/detect/train_final22/weights/best.pt"
model = YOLO(model_path)
print("✅ Model berhasil dimuat:", model_path)

# --- 3️⃣ Buka webcam (0 untuk kamera default laptop) ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam tidak terdeteksi! Pastikan kamera aktif.")
    exit()

print("🎥 Mulai deteksi... Tekan 'Q' untuk berhenti.\n")

# --- 4️⃣ Loop real-time deteksi ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Tidak bisa membaca frame dari webcam.")
        break

    # Jalankan deteksi YOLOv8
    results = model.predict(source=frame, conf=0.5, show=True)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Deteksi dihentikan oleh pengguna.")
        break

# --- 5️⃣ Tutup semua jendela ---
cap.release()
cv2.destroyAllWindows()