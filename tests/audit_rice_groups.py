
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import re

manifest = Path("data/rice/processed/rice_groups.csv")

pattern = re.compile(r"(?:IMG_)?(202310\d{2})_(\d{6})")

groups = defaultdict(list)

with manifest.open(encoding="utf-8") as file:
    for row in csv.DictReader(file):
        if row["source_type"] != "timestamped":
            continue

        match = pattern.search(row["filename"])

        if not match:
            continue

        timestamp = datetime.strptime(
            match.group(1) + match.group(2),
            "%Y%m%d%H%M%S"
        )

        groups[row["group_id"]].append({
            "class": row["class"],
            "filename": row["filename"],
            "time": timestamp
        })

print("TOTAL TIMESTAMP GROUPS:", len(groups))

print("\nGROUP AUDIT:")

for group_id, images in sorted(
    groups.items(),
    key=lambda item: len(item[1]),
    reverse=True
):
    times = [image["time"] for image in images]
    classes = defaultdict(int)

    for image in images:
        classes[image["class"]] += 1

    duration = (max(times) - min(times)).total_seconds()

    print(f"\n{group_id}")
    print("Images:", len(images))
    print("Time span:", duration, "seconds")
    print("Classes:", dict(classes))
