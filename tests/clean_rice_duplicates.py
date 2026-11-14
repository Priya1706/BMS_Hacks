from pathlib import Path
import hashlib
import shutil

source = Path("data/rice/raw/Rice_Leaf_AUG")
destination = Path("data/rice/processed/unique")

seen = set()
copied = 0
duplicates = 0

for class_folder in source.iterdir():
    if not class_folder.is_dir():
        continue

    output_folder = destination / class_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)

    for image_path in class_folder.iterdir():
        if not image_path.is_file():
            continue

        file_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()

        if file_hash in seen:
            duplicates += 1
            continue

        seen.add(file_hash)
        shutil.copy2(image_path, output_folder / image_path.name)
        copied += 1

print("Unique images copied:", copied)
print("Duplicate copies skipped:", duplicates)
