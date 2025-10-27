import os, random, shutil

base = r"C:\SEMESTER 5\YOLO-Mask-Detection\yolo_data"
train_img = os.path.join(base, "images", "train")
train_lbl = os.path.join(base, "labels", "train")
val_img = os.path.join(base, "images", "val")
val_lbl = os.path.join(base, "labels", "val")

# buat folder val jika belum ada
os.makedirs(val_img, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)

# ambil sebagian gambar (10%)
all_imgs = [f for f in os.listdir(train_img) if f.endswith((".jpg", ".png"))]
sample_imgs = random.sample(all_imgs, max(10, len(all_imgs)//10))

for img_name in sample_imgs:
    src_img = os.path.join(train_img, img_name)
    dst_img = os.path.join(val_img, img_name)
    shutil.copy(src_img, dst_img)

    lbl_name = os.path.splitext(img_name)[0] + ".txt"
    src_lbl = os.path.join(train_lbl, lbl_name)
    dst_lbl = os.path.join(val_lbl, lbl_name)
    if os.path.exists(src_lbl):
        shutil.copy(src_lbl, dst_lbl)

print("✅ Folder val sudah dibuat dengan", len(sample_imgs), "gambar!")