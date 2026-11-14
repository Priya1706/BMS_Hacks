
"""Fine-tune the existing RootCause rice MobileNetV2 candidate.

Uses the existing fixed train/validation/test split.
Does not overwrite the baseline model or split manifests.
"""

import csv
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization

ROOT = Path(__file__).resolve().parent

SPLITS_CSV = ROOT / "data/rice/processed/rice_splits.csv"
GROUPS_CSV = ROOT / "data/rice/processed/rice_groups.csv"
BASELINE = ROOT / "app/models/rice_mobilenet_candidate.keras"
OUTPUT = ROOT / "app/models/rice_mobilenet_finetuned.keras"
ARTIFACTS = ROOT / "app/models/rice_mobilenet_finetuned_evaluation"

SEED = 42
BATCH_SIZE = 16
EPOCHS = 8
PATIENCE = 2
LEARNING_RATE = 1e-5
UNFREEZE_LAST = 30


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest():
    with SPLITS_CSV.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    required = {"class", "filename", "group_id", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Invalid rice split manifest")

    if {r["split"] for r in rows} - {"train", "val", "test"}:
        raise ValueError("Unknown split found")

    if GROUPS_CSV.exists():
        with GROUPS_CSV.open(newline="", encoding="utf-8-sig") as f:
            groups = {
                (r["class"], r["filename"]): r["group_id"]
                for r in csv.DictReader(f)
            }

        for row in rows:
            key = (row["class"], row["filename"])
            if groups.get(key) != row["group_id"]:
                raise ValueError(f"Group mismatch: {key}")

        assignments = {}
        for row in rows:
            assignments.setdefault(row["group_id"], set()).add(row["split"])

        if any(len(splits) != 1 for splits in assignments.values()):
            raise ValueError("A group crosses split boundaries")

    classes = sorted({r["class"] for r in rows})

    class_file = ROOT / "app/models/rice_classes.json"
    if class_file.exists():
        saved = json.loads(class_file.read_text(encoding="utf-8"))
        if saved != classes:
            raise ValueError("Class order differs from saved model labels")

    return rows, classes


def load_images(rows, classes):
    from PIL import Image

    class_to_idx = {name: i for i, name in enumerate(classes)}
    data = {s: ([], []) for s in ("train", "val", "test")}

    for row in rows:
        path = ROOT / "data/rice/processed/pristine" / row["class"] / row["filename"]

        if not path.is_file():
            raise FileNotFoundError(path)

        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (224, 224), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 255.0

        images, labels = data[row["split"]]
        images.append(array)
        labels.append(class_to_idx[row["class"]])

    result = {}
    for split, (images, labels) in data.items():
        result[split] = (
            np.stack(images),
            np.asarray(labels, dtype=np.int32)
        )

    return result, class_to_idx


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    tf.config.set_visible_devices([], "GPU")

    if not BASELINE.is_file():
        raise FileNotFoundError(BASELINE)

    manifest_hash_before = sha256(SPLITS_CSV)
    baseline_hash_before = sha256(BASELINE)

    rows, classes = load_manifest()
    datasets, class_to_idx = load_images(rows, classes)

    x_train, y_train = datasets["train"]
    x_val, y_val = datasets["val"]
    x_test, y_test = datasets["test"]

    print("Classes:", classes)
    print("Train:", len(y_train))
    print("Validation:", len(y_val))
    print("Test:", len(y_test))

    counts = Counter(y_train.tolist())
    class_weights = {
        i: len(y_train) / (len(classes) * counts[i])
        for i in range(len(classes))
    }

    print("\nLoading existing baseline model...")
    model = keras.models.load_model(BASELINE)

    backbone = model.get_layer("mobilenetv2_1.00_224")
    backbone.trainable = True

    # Freeze all backbone layers first.
    for layer in backbone.layers:
        layer.trainable = False

    # Unfreeze only the final 30 layers, excluding BatchNorm.
    for layer in backbone.layers[-UNFREEZE_LAST:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    # Keep the original augmentation, preprocessing and classification head.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    trainable_count = sum(
        int(tf.size(variable).numpy())
        for variable in model.trainable_variables
    )

    print("Trainable parameters:", trainable_count)
    print("Fine-tuning final", UNFREEZE_LAST, "backbone layers")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(OUTPUT),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    start = time.time()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop],
        verbose=2
    )

    duration = time.time() - start

    if sha256(SPLITS_CSV) != manifest_hash_before:
        raise RuntimeError("Split manifest changed during training")

    if sha256(BASELINE) != baseline_hash_before:
        raise RuntimeError("Baseline model changed during training")

    # Evaluate the selected checkpoint on the fixed test set once.
    best_model = keras.models.load_model(OUTPUT)

    test_loss, test_accuracy = best_model.evaluate(
        x_test, y_test, batch_size=BATCH_SIZE, verbose=0
    )

    probabilities = best_model.predict(
        x_test, batch_size=BATCH_SIZE, verbose=0
    )
    predictions = np.argmax(probabilities, axis=1)

    matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for truth, prediction in zip(y_test, predictions):
        matrix[int(truth), int(prediction)] += 1

    report = {}

    for i, name in enumerate(classes):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        report[name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": int(matrix[i].sum())
        }

    macro_f1 = float(np.mean([
        report[name]["f1_score"] for name in classes
    ]))

    metrics = {
        "experiment": "rice_mobilenet_finetune_last_30",
        "test_samples": len(y_test),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "macro_f1": macro_f1,
        "training_seconds": duration,
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "class_weights": {
            classes[i]: float(w) for i, w in class_weights.items()
        },
        "manifest_sha256": manifest_hash_before,
        "baseline_sha256": baseline_hash_before
    }

    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    (ARTIFACTS / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    (ARTIFACTS / "training_history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )

    with (ARTIFACTS / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["true/predicted", *classes])
        for name, values in zip(classes, matrix.tolist()):
            writer.writerow([name, *values])

    print("\nEXPERIMENT COMPLETE")
    print(json.dumps(metrics, indent=2))
    print("Classification report:", ARTIFACTS / "classification_report.json")
    print("Confusion matrix:", ARTIFACTS / "confusion_matrix.csv")
    print("Fine-tuned model:", OUTPUT)


if __name__ == "__main__":
    main()
