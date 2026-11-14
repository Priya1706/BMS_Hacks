
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from app.services.predictor import predict_crop_disease

SPLIT_FILE = Path("data/rice/processed/rice_splits.csv")
IMAGE_DIR = Path("data/rice/processed/pristine")

with open(SPLIT_FILE, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

test_rows = [row for row in rows if row["split"] == "test"]

y_true = []
y_pred = []
failed = []

for i, row in enumerate(test_rows, start=1):
    image_path = IMAGE_DIR / row["class"] / row["filename"]

    if not image_path.exists():
        failed.append(str(image_path))
        continue

    result = predict_crop_disease("rice", image_path)

    if result["status"] != "success":
        failed.append(str(image_path))
        continue

    y_true.append(row["class"])
    y_pred.append(result["prediction"])

    print(f"{i}/{len(test_rows)} | Actual: {row['class']} | Predicted: {result['prediction']}")

labels = sorted(set(y_true) | set(y_pred))

print("\n========== RICE MODEL EVALUATION ==========")
print(f"Evaluated images: {len(y_true)}")
print(f"Failed images: {len(failed)}")

if y_true:
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))

if failed:
    print("\nFailed image paths:")
    for path in failed:
        print(path)
