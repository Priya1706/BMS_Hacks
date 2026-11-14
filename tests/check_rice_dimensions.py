from pathlib import Path
from PIL import Image
from collections import Counter

root = Path("data/rice/processed/unique")

dimensions = Counter()
formats = Counter()

for image_path in root.rglob("*"):
    if not image_path.is_file():
        continue

    try:
        with Image.open(image_path) as img:
            dimensions[img.size] += 1
            formats[img.format] += 1
    except Exception:
        continue

print("\nIMAGE DIMENSIONS:")
for size, count in dimensions.most_common(15):
    print(size, ":", count)

print("\nIMAGE FORMATS:")
for image_format, count in formats.items():
    print(image_format, ":", count)