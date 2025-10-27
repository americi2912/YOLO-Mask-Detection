import cv2, os, glob

# Folder gambar sumber
img_dir = 'kaggle_raw/images'
out_path = 'videos/mask_slideshow_200frames.mp4'
os.makedirs('videos', exist_ok=True)

# Ambil semua gambar (jpg dan png)
imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')) +
              glob.glob(os.path.join(img_dir, '*.jpg')))

# Batasi maksimum 200 gambar saja
imgs = imgs[:200]

# Pastikan ada gambar
if len(imgs) == 0:
    raise ValueError("❌ Tidak ada gambar di folder kaggle_raw/images")

print(f"🖼 Total gambar dipakai: {len(imgs)}")

# Ambil ukuran gambar pertama
sample = cv2.imread(imgs[0])
h, w = sample.shape[:2]

# Buat video writer (fps rendah biar video agak panjang)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 3  # 3 frame per detik → durasi sekitar 60–70 detik untuk 200 frame
vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

# Tambahkan semua gambar ke video
for i, path in enumerate(imgs):
    frame = cv2.imread(path)
    vw.write(frame)
    if (i + 1) % 20 == 0:
        print(f"📸 Menambahkan frame ke-{i + 1}")

vw.release()
print("✅ Video berhasil dibuat:", out_path)