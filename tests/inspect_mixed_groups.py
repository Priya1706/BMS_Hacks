
import csv
import re
from pathlib import Path
from collections import defaultdict

manifest = Path("data/rice/processed/rice_groups.csv")
groups = defaultdict(list)

with manifest.open(encoding="utf-8") as file:
    for row in csv.DictReader(file):
        if row["source_type"] == "timestamped":
            groups[row["group_id"]].append(row)

print("MIXED-CLASS GROUPS")

for group_id, images in groups.items():
    classes = sorted(set(row["class"] for row in images))

    if len(classes) > 1:
        print(f"\n{group_id} | {len(images)} images")
        print("Classes:", classes)

        for row in images:
            print(f"  {row['class']} | {row['filename']}")
