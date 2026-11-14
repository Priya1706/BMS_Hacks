from pathlib import Path
from PIL import Image

root = Path("data/rice/processed/unique")

for class_folder in root.iterdir():
    if not class_folder.is_dir():
        continue

    print(f"\nCLASS: {class_folder.name}")

    for image_path in list(class_folder.iterdir())[:3]:
        with Image.open(image_path) as img:
            print(
                image_path.name,
                "Size:", img.size,
                "Mode:", img.mode
            )