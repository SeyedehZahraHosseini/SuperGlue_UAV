
# # ======================== تنظیمات ========================
# video_path = "../Thermal/Encode_1080P_83.mp4"        # مسیر ویدیو
# output_h5 = "thermal_dataset.h5"    # مسیر ذخیره دیتاست نهایی
# resize_target = 512                 # اندازه تصویر ریسایز شده
# start_from_second = 926               # از چه ثانیه‌ای استخراج شروع شود
# fps_sample = 10                      # تعداد فریم در ثانیه برای استخراج
# chunk_size = 500                     # هر chunk چند فریم باشد
# # ========================================================
import cv2
import h5py
import numpy as np
from tqdm import tqdm

# ======================== تنظیمات ========================
video_path = "../Thermal/Encode_1080P_83.mp4"       # مسیر ویدیو
output_h5 = "thermal_dataset.h5"    # مسیر ذخیره دیتاست
resize_target = 512                 # اندازه تصویر ریسایز شده
start_from_second = 926              # از چه ثانیه‌ای استخراج شروع شود
fps_sample = 5                       # تعداد فریم در ثانیه برای استخراج
chunk_size = 100                      # تعداد فریم برای ذخیره هر بار
# ========================================================

# باز کردن ویدیو
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS ویدیو:", fps)
print("کل فریم‌ها:", total_frames)

# محاسبه فریم شروع
start_frame = int(start_from_second * fps)
if start_frame >= total_frames:
    raise ValueError("زمان شروع بیشتر از طول ویدیو است.")
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_interval = int(fps / fps_sample)
frame_id = start_frame
saved_count = 0

# ایجاد فایل HDF5 فقط برای نسخه resize
h5 = h5py.File(output_h5, "w")

resized_dataset = h5.create_dataset(
    "resized_image",
    shape=(0, resize_target, resize_target, 3),
    maxshape=(None, resize_target, resize_target, 3),
    dtype=np.uint8,
    chunks=True
)

name_dtype = h5py.string_dtype(encoding='utf-8')
name_dataset = h5.create_dataset(
    "image_name",
    shape=(0,),
    maxshape=(None,),
    dtype=name_dtype,
    chunks=True
)

buffer_resized = []
buffer_names = []

pbar = tqdm(total=total_frames - start_frame, desc="Processing frames")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if (frame_id - start_frame) % frame_interval != 0:
        frame_id += 1
        pbar.update(1)
        continue

    # BGR -> RGB و resize
    resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                         (resize_target, resize_target),
                         interpolation=cv2.INTER_AREA)

    buffer_resized.append(resized)
    buffer_names.append(f"frame_{frame_id:06d}.png")

    frame_id += 1
    saved_count += 1
    pbar.update(1)

    # ذخیره chunk در HDF5
    if len(buffer_resized) >= chunk_size:
        start_idx = resized_dataset.shape[0]
        new_size = start_idx + len(buffer_resized)

        resized_dataset.resize(new_size, axis=0)
        resized_dataset[start_idx:new_size] = buffer_resized

        name_dataset.resize(new_size, axis=0)
        name_dataset[start_idx:new_size] = np.array(buffer_names, dtype='S')

        buffer_resized, buffer_names = [], []

# ذخیره باقی مانده buffer
if len(buffer_resized) > 0:
    start_idx = resized_dataset.shape[0]
    new_size = start_idx + len(buffer_resized)

    resized_dataset.resize(new_size, axis=0)
    resized_dataset[start_idx:new_size] = buffer_resized

    name_dataset.resize(new_size, axis=0)
    name_dataset[start_idx:new_size] = np.array(buffer_names, dtype='S')

pbar.close()
cap.release()
h5.close()

print("HDF5 dataset saved as:", output_h5)
print("Total frames saved:", saved_count)

