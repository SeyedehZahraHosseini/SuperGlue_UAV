import os
from PIL import Image
import numpy as np
from numba import jit, prange
from tqdm import tqdm

folder_path = "F:/IDM/utils/thermal_images"
image_size = 768
sigma = 120
threshold = 15  # تنظیم کنید

# --- فایل‌ها ---
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') and f.count('@') == 2]

# --- اعتبارسنجی و مرتب‌سازی ---
files_info = []
for f in image_files:
    try:
        parts = f.split('@')
        row = int(parts[1])
        col = int(parts[2].split('.')[0])
        files_info.append((f, row, col))
    except:
        continue

files_info.sort(key=lambda x: (x[1], x[2]))
image_files = [x[0] for x in files_info]

if not image_files:
    raise ValueError("هیچ فایل معتبری پیدا نشد!")

# --- ابعاد ---
all_rows = [x[1] for x in files_info]
all_cols = [x[2] for x in files_info]
min_row, min_col = min(all_rows), min(all_cols)

final_h = max(all_rows) - min_row + image_size
final_w = max(all_cols) - min_col + image_size

print(f"تصویر نهایی: {final_w}×{final_h} | کاشی: {len(image_files)}")

# --- آرایه نهایی ---
final_image = np.zeros((final_h, final_w, 3), dtype=np.float32)
weight_map = np.zeros((final_h, final_w, 3), dtype=np.float32)

# --- ماسک گاوسی ---
def gaussian_mask(size, sigma):
    ax = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(ax, ax)
    mask = np.exp(-(xx**2 + yy**2) / (2 * (sigma/10)**2))
    return (mask / mask.max()).astype(np.float32)

gauss_base = gaussian_mask(image_size, sigma)
gauss_3c = np.repeat(gauss_base[:, :, np.newaxis], 3, axis=2)

# --- تابع numba با final_weight ---
@jit(nopython=True, parallel=True, cache=True)
def blend_with_final_weight(final_img, weight_img, tile, y0, x0, final_weight, h, w):
    for dy in prange(h):
        for dx in range(w):
            gy = y0 + dy
            gx = x0 + dx
            if gy < final_img.shape[0] and gx < final_img.shape[1]:
                for c in range(3):
                    g = final_weight[dy, dx, c]
                    final_img[gy, gx, c] += tile[dy, dx, c] * g
                    weight_img[gy, gx, c] += g

# --- حلقه اصلی ---
for f in tqdm(image_files, desc="دوخت با ماسک داده واقعی"):
    path = os.path.join(folder_path, f)
    parts = f.split('@')
    row = int(parts[1])
    col = int(parts[2].split('.')[0])
    
    y0 = row - min_row
    x0 = col - min_col
    
    # --- خواندن کاشی ---
    tile = np.array(Image.open(path), dtype=np.float32)
    
    # --- ماسک داده واقعی ---
    gray = np.mean(tile, axis=2)
    data_mask = (gray > threshold).astype(np.float32)
    data_mask_3c = np.repeat(data_mask[:, :, np.newaxis], 3, axis=2)
    
    # --- وزن نهایی ---
    final_weight = gauss_3c * data_mask_3c
    
    # --- ترکیب ---
    blend_with_final_weight(final_image, weight_map, tile, y0, x0, final_weight, image_size, image_size)

# --- نرمال‌سازی ---
final_image = final_image / (weight_map + 1e-6)
final_image = np.clip(final_image, 0, 255).astype(np.uint8)

# --- برش ---
mask = weight_map.sum(axis=2) > 0
rows = np.any(mask, axis=1)
cols = np.any(mask, axis=0)
ymin, ymax = np.where(rows)[0][[0, -1]]
xmin, xmax = np.where(cols)[0][[0, -1]]

cropped = final_image[ymin:ymax+1, xmin:xmax+1]
output_path = "F:/IDM/utils/thermal_data_mask.png"
Image.fromarray(cropped).save(output_path)

print(f"ذخیره شد: {output_path}")
