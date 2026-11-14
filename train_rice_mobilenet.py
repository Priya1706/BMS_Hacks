"""Train/evaluate a MobileNetV2 rice classifier using the fixed group split.

This is deliberately separate from train_rice_model.py: it never writes the
split manifest or the currently deployed model.
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
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
PRISTINE_DIR = ROOT / "data/rice/processed/pristine"
SPLITS_CSV = ROOT / "data/rice/processed/rice_splits.csv"
GROUPS_CSV = ROOT / "data/rice/processed/rice_groups.csv"
MODELS_DIR = ROOT / "app/models"
WEIGHTS = MODELS_DIR / "pretrained/mobilenet_v2_weights.h5"
MODEL_OUT = MODELS_DIR / "rice_mobilenet_candidate.keras"
ARTIFACTS_DIR = MODELS_DIR / "rice_mobilenet_candidate_evaluation"
SIZE = (224, 224)
SEED = 42
BATCH_SIZE = 16
EPOCHS = 30
BASELINE_ACCURACY = 46 / 124  # Existing 37.10% all-Sheath-Blight baseline.


def load_manifest(path=SPLITS_CSV):
    """Read and validate the fixed split and group manifests; never rewrite them."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"class", "filename", "group_id", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Split manifest must contain {sorted(required)}")
    if {r["split"] for r in rows} - {"train", "val", "test"}:
        raise ValueError("Manifest contains an unknown split name")
    if len({(r["class"], r["filename"]) for r in rows}) != len(rows):
        raise ValueError("Split manifest contains duplicate class/filename rows")
    classes = sorted({r["class"] for r in rows})
    saved_classes_path = MODELS_DIR / "rice_classes.json"
    if saved_classes_path.exists():
        saved_classes = json.loads(saved_classes_path.read_text(encoding="utf-8"))
        if saved_classes != classes:
            raise ValueError(f"Manifest class order {classes} differs from saved labels {saved_classes}")
    saved_mapping_path = MODELS_DIR / "rice_class_mapping.json"
    if saved_mapping_path.exists():
        expected_mapping = {str(index): name for index, name in enumerate(classes)}
        saved_mapping = json.loads(saved_mapping_path.read_text(encoding="utf-8"))
        if saved_mapping != expected_mapping:
            raise ValueError("Manifest class order differs from rice_class_mapping.json")
    if GROUPS_CSV.exists():
        with GROUPS_CSV.open(newline="", encoding="utf-8-sig") as f:
            groups = {(r["class"], r["filename"]): r["group_id"] for r in csv.DictReader(f)}
        if set(groups) != {(row["class"], row["filename"]) for row in rows}:
            raise ValueError("Split manifest rows do not exactly match rice_groups.csv")
        assignments = {}
        for row in rows:
            key = (row["class"], row["filename"])
            if groups.get(key) != row["group_id"]:
                raise ValueError(f"Group manifest mismatch for {key}")
            assignments.setdefault(row["group_id"], set()).add(row["split"])
        crossed = [gid for gid, splits in assignments.items() if len(splits) != 1]
        if crossed:
            raise ValueError(f"Groups cross split boundaries: {crossed[:5]}")
    return rows, classes


def load_image(path):
    """Shared image resize and [0,1] representation used at train and inference."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize(SIZE, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def mobilenet_v2_preprocess(images):
    """Map RGB float images in [0,1] to MobileNetV2's expected [-1,1]."""
    return images * 2.0 - 1.0


def load_split_data(rows, classes):
    class_to_idx = {name: index for index, name in enumerate(classes)}
    data = {name: ([], []) for name in ("train", "val", "test")}
    for row in rows:
        path = PRISTINE_DIR / row["class"] / row["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {path}")
        try:
            image = load_image(path)
        except Exception as exc:
            raise ValueError(f"Cannot decode manifest image {path}: {exc}") from exc
        images, labels = data[row["split"]]
        images.append(image)
        labels.append(class_to_idx[row["class"]])
    return {
        split: (np.stack(images), np.asarray(labels, dtype=np.int32))
        for split, (images, labels) in data.items()
    }, class_to_idx


def build_model(num_classes):
    """Frozen ImageNet backbone with training-only augmentation and a small head."""
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.06, fill_mode="reflect", seed=SEED + 1),
        layers.RandomZoom(0.08, seed=SEED + 2),
    ], name="training_augmentation")
    backbone = keras.applications.MobileNetV2(
        input_shape=(*SIZE, 3), include_top=False, weights=None
    )
    backbone.load_weights(WEIGHTS)
    backbone.trainable = False

    inputs = keras.Input(shape=(*SIZE, 3), name="rgb_0_1")
    x = augmentation(inputs)
    # keras.applications.mobilenet_v2.preprocess_input performs x * 127.5 - 1.
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_v2_preprocessing")(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25, seed=SEED)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="rice_classes")(x)
    model = keras.Model(inputs, outputs, name="RootCause_Rice_MobileNetV2")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def compute_metrics(y_true, y_pred, classes):
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    report = {}
    for idx, name in enumerate(classes):
        tp = int(matrix[idx, idx]); fp = int(matrix[:, idx].sum() - tp)
        fn = int(matrix[idx, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[name] = {"precision": precision, "recall": recall, "f1-score": f1,
                        "support": int(matrix[idx].sum()),
                        "predicted": int(matrix[:, idx].sum())}
    accuracy = float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0
    macro = {
        metric: float(np.mean([report[name][metric] for name in classes]))
        for metric in ("precision", "recall", "f1-score")
    }
    total = int(matrix.sum())
    weighted = {
        metric: float(sum(report[name][metric] * report[name]["support"] for name in classes) / total)
        if total else 0.0
        for metric in ("precision", "recall", "f1-score")
    }
    report["accuracy"] = accuracy
    report["macro avg"] = {**macro, "support": total}
    report["weighted avg"] = {**weighted, "support": total}
    return accuracy, matrix, report


def main():
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    tf.config.set_visible_devices([], "GPU")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Local MobileNetV2 weights not found: {WEIGHTS}")
    manifest_hash = hashlib.sha256(SPLITS_CSV.read_bytes()).hexdigest()
    rows, classes = load_manifest()
    datasets, class_to_idx = load_split_data(rows, classes)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = (
        datasets["train"], datasets["val"], datasets["test"])
    counts = Counter(y_train.tolist())
    class_weights = {idx: len(y_train) / (len(classes) * counts[idx]) for idx in range(len(classes))}
    model = build_model(len(classes))
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                restore_best_weights=True, verbose=1)
    start = time.time()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        class_weight=class_weights, callbacks=[early_stop], verbose=2)
    duration = time.time() - start
    test_loss, _ = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    probabilities = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy, matrix, report = compute_metrics(y_test, predictions, classes)
    if hashlib.sha256(SPLITS_CSV.read_bytes()).hexdigest() != manifest_hash:
        raise RuntimeError("The fixed split manifest changed during training")
    model.save(MODEL_OUT)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "test_samples": int(len(y_test)), "test_loss": float(test_loss),
        "test_accuracy": accuracy, "baseline_accuracy": BASELINE_ACCURACY,
        "accuracy_delta_percentage_points": (accuracy - BASELINE_ACCURACY) * 100,
        "predicted_class_counts": {c: int((predictions == i).sum()) for i, c in enumerate(classes)},
        "class_weights": {classes[i]: float(w) for i, w in class_weights.items()},
        "class_mapping": class_to_idx, "training_seconds": duration,
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "manifest_sha256_before_training": manifest_hash,
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "training_history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    with (ARTIFACTS_DIR / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["true\\pred", *classes])
        for label, values in zip(classes, matrix.tolist()): writer.writerow([label, *values])
    print(json.dumps(metrics, indent=2))
    print(f"Classification report: {ARTIFACTS_DIR / 'classification_report.json'}")
    print(f"Confusion matrix: {ARTIFACTS_DIR / 'confusion_matrix.csv'}")
    print(f"Candidate model: {MODEL_OUT}")


if __name__ == "__main__":
    main()
