
from pathlib import Path
from collections import defaultdict
import hashlib
import shutil

source = Path("data/rice/raw/Rice_Leaf_AUG")
destination = Path("data/rice/processed/pristine")

seen_hashes = set()
class_counts = defaultdict(int)
duplicates = 0
augmented_skipped = 0

for class_folder in source.iterdir():
    if not class_folder.is_dir():
        continue

    output_folder = destination / class_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)

    for image_path in class_folder.iterdir():
        if not image_path.is_file():
            continue

        # Exclude previously augmented images
        if image_path.name.startswith("aug_0_"):
            augmented_skipped += 1
            continue

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        file_hash = hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest()

        if file_hash in seen_hashes:
            duplicates += 1
            continue

        seen_hashes.add(file_hash)

        shutil.copy2(
            image_path,
            output_folder / image_path.name
        )

        class_counts[class_folder.name] += 1

print("\n--- PRISTINE RICE DATASET ---")

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")

print("\nTotal unique original images:", sum(class_counts.values()))
print("Augmented images excluded:", augmented_skipped)
print("Exact duplicates excluded:", duplicates)
