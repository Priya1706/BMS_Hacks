from pathlib import Path
from collections import defaultdict
import hashlib

root = Path("data/rice/raw/Rice_Leaf_AUG")

hashes = defaultdict(list)

for image_path in root.rglob("*"):
    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        continue

    file_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    hashes[file_hash].append(str(image_path))

duplicates = {
    h: paths for h, paths in hashes.items()
    if len(paths) > 1
}

print("Total images:", sum(len(v) for v in hashes.values()))
print("Duplicate groups:", len(duplicates))
print("Extra duplicate copies:", sum(len(v) - 1 for v in duplicates.values()))

for paths in duplicates.values():
    print("\nDuplicate files:")
    for path in paths:
        print(path)