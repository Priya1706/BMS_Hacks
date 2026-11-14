
from pathlib import Path
from collections import Counter

root = Path("data/rice/processed/pristine")

for folder in sorted(root.iterdir()):
    if not folder.is_dir():
        continue

    files = list(folder.glob("*.jpg"))

    timestamped = [
        p for p in files
        if p.name.startswith(("IMG_", "2023"))
    ]

    numbered = [
        p for p in files
        if p not in timestamped
    ]

    print(f"\n{folder.name}")
    print("Total:", len(files))
    print("Timestamped:", len(timestamped))
    print("Numbered/other:", len(numbered))
    print("Examples:", [p.name for p in files[:5]])
