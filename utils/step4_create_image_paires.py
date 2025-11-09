import os
import json
from PIL import Image
import numpy as np

# ---------------------------
# تنظیمات
# ---------------------------
image_paths = {
    "satellite": "satellite_reconstructed_map.png",
    "thermal": "thermal_reconstructed_map.png"
}

output_dir = "new_tiles"
os.makedirs(output_dir, exist_ok=True)

tile_width = 640    # عرض هر تایل (پیکسل)
tile_height = 480   # ارتفاع هر تایل (پیکسل)
stride_x = 448      # فاصله افقی بین تایل‌ها (پیکسل)
stride_y = 336      # فاصله عمودی بین تایل‌ها (پیکسل)
# 30 درصد اورلپ در نظر گرفتم

# دیکشنری برای ذخیره مختصات تایل‌ها
tile_coords = {
    "satellite": {},
    "thermal": {}
}

# ---------------------------
# پردازش هر تصویر
# ---------------------------
for key, path in image_paths.items():
    img = Image.open(path)
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    tile_count = 0

    for y in range(0, h, stride_y):
        for x in range(0, w, stride_x):
            y_end = y + tile_height
            x_end = x + tile_width
            
            # بررسی اندازه تایل
            if y_end > h or x_end > w:
                continue  # تایل ناقص را نادیده می‌گیریم
            
            tile = img_np[y:y_end, x:x_end]
            tile_img = Image.fromarray(tile)
            
            # نام فایل تایل
            tile_filename = f"{key}_tile_{tile_count+1}.png"
            tile_img.save(os.path.join(output_dir, tile_filename))
            
            # ذخیره مختصات چهار گوشه تایل (بر اساس تصویر اصلی)
            coords = {
                "top_left": [x, y],
                "top_right": [x_end, y],
                "bottom_left": [x, y_end],
                "bottom_right": [x_end, y_end]
            }
            tile_coords[key][tile_filename] = coords
            
            tile_count += 1

    print(f"✅ Saved {tile_count} tiles for {key} map.")

# ---------------------------
# ذخیره JSON
# ---------------------------
for key in tile_coords:
    json_path = os.path.join(output_dir, f"{key}_tile_coords.json")
    with open(json_path, 'w') as f:
        json.dump(tile_coords[key], f, indent=4)
    print(f"✅ Saved tile coordinates for {key} to {json_path}")

