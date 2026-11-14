from pathlib import Path
from PIL import Image

root = Path("data/rice/raw/Rice_Leaf_AUG")

total = 0
corrupted = []

for class_folder in root.iterdir():
    if not class_folder.is_dir():
        continue

    for image_path in class_folder.iterdir():
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        total += 1

        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as error:
            corrupted.append((image_path, str(error)))

print("\n--- RICE DATASET INSPECTION ---")
print("Total images checked:", total)
print("Corrupted images:", len(corrupted))

for path, error in corrupted:
    print(path, error)