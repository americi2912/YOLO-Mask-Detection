import cv2
import os

# Path video hasil make_video_from_images.py
video_path = 'videos/mask_slideshow_200frames.mp4'
output_dir = 'work/frames'
os.makedirs(output_dir, exist_ok=True)

# Buka video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"❌ Tidak bisa membuka video di {video_path}")

print("🎞 Mulai ekstraksi frame...")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎬 Total frame di video: {frame_count}")

# Target minimal 200 frame
target_frames = 200
save_every = max(1, frame_count // target_frames)

print(f"📸 Menyimpan setiap {save_every} frame untuk mencapai minimal {target_frames} frame.")

saved = 0
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Simpan setiap "save_every" frame
    if count % save_every == 0:
        frame_name = f"{output_dir}/frame_{count:04d}.jpg"
        cv2.imwrite(frame_name, frame)
        saved += 1

    count += 1

cap.release()
print(f"✅ Sampling frame selesai! Total frame disimpan: {saved}")