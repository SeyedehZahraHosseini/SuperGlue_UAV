import os
import glob
import tarfile

folder = "."  
output_file = os.path.join(folder, "satellite_thermal_dataset_v3.tar.gz")

parts = sorted(glob.glob(os.path.join(folder, "satellite_thermal_dataset_v3.tar.gz.part*")))

print(f"Found {len(parts)} parts:")
for p in parts:
    print("  ", p)

with open(output_file, "wb") as outfile:
    for part in parts:
        print(f"Merging {os.path.basename(part)} ...")
        with open(part, "rb") as infile:
            while True:
                chunk = infile.read(1024 * 1024 * 200)  # 100MB در هر بار خواندن
                if not chunk:
                    break
                outfile.write(chunk)

print("\n✅ All parts merged successfully!")

extract_path = os.path.join(folder, "extracted_dataset")
os.makedirs(extract_path, exist_ok=True)

print("Extracting the archive... (this may take a while)")
with tarfile.open(output_file, "r:gz") as tar:
    tar.extractall(path=extract_path)

print(f"\n✅ Done! Files extracted in: {extract_path}")
