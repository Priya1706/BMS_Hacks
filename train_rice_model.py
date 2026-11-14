"""Training pipeline for RootCause Rice Leaf Disease Classifier.

This script:
1. Loads pristine rice images from data/rice/processed/pristine/ and grouping info from data/rice/processed/rice_groups.csv.
2. Creates a reproducible, leakage-free group-aware split (Train 70%, Val 15%, Test 15%).
3. Preprocesses images to 224x224.
4. Applies data augmentation ONLY to training images.
5. Trains a lightweight CNN on CPU with early stopping.
6. Evaluates on the held-out test set (Accuracy, Precision, Recall, F1, Confusion Matrix).
7. Saves the trained model to app/models/rice_model.keras and class mapping to app/models/rice_classes.json.
"""

import csv
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# Suppress verbose TensorFlow / oneDNN logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Paths
PRISTINE_DIR = Path("data/rice/processed/pristine")
GROUPS_CSV = Path("data/rice/processed/rice_groups.csv")
SPLITS_CSV = Path("data/rice/processed/rice_splits.csv")
MODELS_DIR = Path("app/models")
MODEL_OUT_PATH = MODELS_DIR / "rice_model.keras"
CLASSES_JSON_PATH = MODELS_DIR / "rice_classes.json"
MAPPING_JSON_PATH = MODELS_DIR / "rice_class_mapping.json"

TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_EPOCHS = 15


def create_reproducible_split(groups_csv_path: Path, output_csv_path: Path):
    """Create a leakage-safe split where capture sessions (groups) are never separated."""
    with open(groups_csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    class_totals = Counter(r["class"] for r in rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)

    # Deterministic shuffle then sort by group size
    group_ids = list(groups.keys())
    rnd = random.Random(RANDOM_SEED)
    rnd.shuffle(group_ids)
    group_ids.sort(key=lambda g: len(groups[g]), reverse=True)

    target_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_class_counts = {"train": Counter(), "val": Counter(), "test": Counter()}
    group_assignment = {}

    for gid in group_ids:
        items = groups[gid]
        g_classes = [it["class"] for it in items]

        best_split = None
        best_score = -float("inf")

        for s in ["train", "val", "test"]:
            score = 0.0
            # Class-deficit balancing
            for c in g_classes:
                target_class_count = target_ratios[s] * class_totals[c]
                score += (target_class_count - split_class_counts[s][c])
            # Overall capacity balance
            target_total = target_ratios[s] * len(rows)
            score += 0.5 * (target_total - split_counts[s])

            if score > best_score:
                best_score = score
                best_split = s

        group_assignment[gid] = best_split
        split_counts[best_split] += len(items)
        for c in g_classes:
            split_class_counts[best_split][c] += 1

    # Write split manifest
    split_rows = []
    for r in rows:
        r_copy = dict(r)
        r_copy["split"] = group_assignment[r["group_id"]]
        split_rows.append(r_copy)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class", "filename", "source_type", "group_id", "split"]
        )
        writer.writeheader()
        writer.writerows(split_rows)

    print("=== REPRODUCIBLE SPLIT CREATED ===")
    print(f"Manifest saved to: {output_csv_path}")
    for s in ["train", "val", "test"]:
        print(f"  {s.upper()}: {split_counts[s]} images ({split_counts[s]/len(rows)*100:.1f}%)")
        print(f"    Classes: {dict(sorted(split_class_counts[s].items()))}")

    return split_rows


def load_dataset_from_manifest(split_rows, pristine_dir: Path):
    """Load images from disk using OpenCV into numpy arrays for fast CPU training."""
    # Discover class names
    classes = sorted(list(set(r["class"] for r in split_rows)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}

    print(f"\nDiscovered {len(classes)} classes: {classes}")

    splits_data = {
        "train": {"images": [], "labels": []},
        "val": {"images": [], "labels": []},
        "test": {"images": [], "labels": []},
    }

    t0 = time.time()
    for row in split_rows:
        img_path = pristine_dir / row["class"] / row["filename"]
        if not img_path.exists():
            continue

        # Fast read with OpenCV and resize to 224x224
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        # Normalize to [0.0, 1.0] float32
        norm_img = resized.astype(np.float32) / 255.0

        target_split = row["split"]
        splits_data[target_split]["images"].append(norm_img)
        splits_data[target_split]["labels"].append(class_to_idx[row["class"]])

    print(f"Images loaded & preprocessed in {time.time() - t0:.2f}s")

    X_train = np.array(splits_data["train"]["images"], dtype=np.float32)
    y_train = np.array(splits_data["train"]["labels"], dtype=np.int32)
    X_val = np.array(splits_data["val"]["images"], dtype=np.float32)
    y_val = np.array(splits_data["val"]["labels"], dtype=np.int32)
    X_test = np.array(splits_data["test"]["images"], dtype=np.float32)
    y_test = np.array(splits_data["test"]["labels"], dtype=np.int32)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), classes, class_to_idx, idx_to_class


def build_lightweight_model(num_classes: int):
    """Construct a lightweight CNN image classifier optimized for CPU inference."""
    # Augmentation layers (active ONLY during training)
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.08),
            layers.RandomBrightness(0.08),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 4
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Classifier Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="RootCause_Rice_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model(model, X_test, y_test, classes):
    """Compute test accuracy, per-class metrics, and confusion matrix."""
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    preds = model.predict(X_test, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    num_classes = len(classes)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_lbl, pred_lbl in zip(y_test, y_pred):
        conf_matrix[true_lbl, pred_lbl] += 1

    per_class_results = []
    for i, c in enumerate(classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        support = conf_matrix[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class_results.append(
            {
                "class": c,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": int(support),
            }
        )

    return test_acc, test_loss, per_class_results, conf_matrix


def main():
    print("=" * 60)
    print("ROOTCAUSE: RICE DISEASE DETECTION MODEL TRAINING")
    print("=" * 60)

    # 1. Reproducible split
    split_rows = create_reproducible_split(GROUPS_CSV, SPLITS_CSV)

    # 2. Load dataset
    (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
        classes,
        class_to_idx,
        idx_to_class,
    ) = load_dataset_from_manifest(split_rows, PRISTINE_DIR)

    print(f"\nTraining set size:   {len(X_train)} samples")
    print(f"Validation set size: {len(X_val)} samples")
    print(f"Test set size:       {len(X_test)} samples")

    # 3. Build model
    model = build_lightweight_model(num_classes=len(classes))
    model.summary()

    # 4. Train with EarlyStopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nStarting CPU model training...")
    start_train_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
    )

    training_duration = time.time() - start_train_time
    print(f"\nTraining completed in {training_duration:.2f} seconds ({training_duration/60:.2f} mins)")

    # 5. Save model and class mapping
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_OUT_PATH))
    print(f"Model successfully saved to: {MODEL_OUT_PATH}")

    # Save class lists
    with open(CLASSES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    with open(MAPPING_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({str(i): c for i, c in enumerate(classes)}, f, indent=2)

    print(f"Class labels saved to: {CLASSES_JSON_PATH} and {MAPPING_JSON_PATH}")

    # 6. Evaluate on Test set
    test_acc, test_loss, per_class, conf_matrix = evaluate_model(
        model, X_test, y_test, classes
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS ON HELD-OUT TEST SET")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss:     {test_loss:.4f}\n")

    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 69)
    for p in per_class:
        print(
            f"{p['class']:<25} {p['precision']:<12.4f} {p['recall']:<12.4f} {p['f1']:<12.4f} {p['support']:<8}"
        )

    print("\nConfusion Matrix (Rows = Ground Truth, Columns = Predicted):")
    print(f"{'':<25}" + "".join([f"{c[:10]:>12}" for c in classes]))
    for i, c in enumerate(classes):
        row_str = "".join([f"{conf_matrix[i, j]:>12}" for j in range(len(classes))])
        print(f"{c:<25}" + row_str)

    print("=" * 60)


if __name__ == "__main__":
    main()
