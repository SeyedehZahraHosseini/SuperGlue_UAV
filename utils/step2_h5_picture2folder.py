import os
import h5py
import numpy as np
from PIL import Image

# مسیر فایل H5 و پوشه خروجی
h5_input_path = "train_database.h5"
output_dir = 'satellite_images'
os.makedirs(output_dir, exist_ok=True)

# بارگذاری فایل H5
with h5py.File(h5_input_path, 'r') as h5_file:
    if 'image_data' not in h5_file:
        print("❌ 'image_data' feature not found in the H5 file.")
        exit(1)

    image_data = h5_file['image_data']
    image_name = h5_file['image_name']
    total_images = len(image_data)
    print(f"✅ Total images: {total_images}")

    for idx in range(total_images):
        img_np = image_data[idx]
        img_name = image_name[idx].decode('utf-8')
        image = Image.fromarray(img_np)

        # ذخیره تصویر با نام شماره‌ی آن
        output_image_path = os.path.join(output_dir, f"{img_name}.png")
        image.save(output_image_path)

        print(f"✅ Saved: {output_image_path}")

print("✅ All images extracted.")
