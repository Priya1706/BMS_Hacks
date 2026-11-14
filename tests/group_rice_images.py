
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re
import csv

root = Path("data/rice/processed/pristine")
output = Path("data/rice/processed/rice_groups.csv")

timestamp_pattern = re.compile(
    r"(?:IMG_)?(202310\d{2})_(\d{6})"
)

records = []
timestamped = []

# First collect image information
for class_folder in sorted(root.iterdir()):
    if not class_folder.is_dir():
        continue

    for image_path in sorted(class_folder.glob("*.jpg")):
        match = timestamp_pattern.search(image_path.name)

        if match:
            timestamp = datetime.strptime(
                match.group(1) + match.group(2),
                "%Y%m%d%H%M%S"
            )

            timestamped.append({
                "path": image_path,
                "class": class_folder.name,
                "timestamp": timestamp
            })
        else:
            records.append({
                "class": class_folder.name,
                "filename": image_path.name,
                "source_type": "numbered_or_other",
                "group_id": f"individual_{len(records):04d}"
            })

# Group timestamped photos taken within 10 seconds
timestamped.sort(key=lambda x: x["timestamp"])

groups = []
current_group = []
previous_time = None

for item in timestamped:
    current_time = item["timestamp"]

    if (
        previous_time is not None
        and (current_time - previous_time).total_seconds() > 10
    ):
        groups.append(current_group)
        current_group = []

    current_group.append(item)
    previous_time = current_time

if current_group:
    groups.append(current_group)

for index, group in enumerate(groups):
    group_id = f"timestamp_group_{index:04d}"

    for item in group:
        records.append({
            "class": item["class"],
            "filename": item["path"].name,
            "source_type": "timestamped",
            "group_id": group_id
        })

# Save manifest
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["class", "filename", "source_type", "group_id"]
    )
    writer.writeheader()
    writer.writerows(records)

print("Total images:", len(records))
print("Timestamp groups:", len(groups))
print("Timestamped images:", len(timestamped))
print("Numbered/other images:", len(records) - len(timestamped))
print("Manifest saved:", output)

print("\nLargest timestamp groups:")
for group in sorted(groups, key=len, reverse=True)[:10]:
    print(len(group), "images:", group[0]["timestamp"])
