"""Train and evaluate an isolated MobileNetV2 wheat candidate.

The fixed test folders are read only. Exact train/test content duplicates are
removed from the training pool, and remaining exact duplicates stay together
in the stratified train/validation split.
"""

import csv
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/wheat/selected/data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = ROOT / "app/models"
WEIGHTS = MODELS_DIR / "pretrained/mobilenet_v2_weights.h5"
MODEL_OUT = MODELS_DIR / "wheat_mobilenet_candidate.keras"
CLASSES_OUT = MODELS_DIR / "wheat_mobilenet_candidate_classes.json"
ARTIFACTS_DIR = MODELS_DIR / "wheat_mobilenet_candidate_evaluation"

# Preserve the dataset's explicit class order while exposing a clear disease name.
FOLDER_TO_LABEL = {
    "Black Rust": "Black Rust",
    "Brown Rust": "Brown Rust",
    "Yellow Rust": "Yellow Rust",
    "Mildew": "Powdery Mildew",
}
CLASSES = list(FOLDER_TO_LABEL.values())
TEST_FOLDER_TO_CLASS = {
    "black_rust_test": "Black Rust",
    "brown_rust_test": "Brown Rust",
    "yellow_rust_test": "Yellow Rust",
    "mildew_test": "Powdery Mildew",
}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGE_SIZE = (224, 224)
SEED = 42
VALIDATION_FRACTION = 0.20
BATCH_SIZE = 16
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5


def image_paths(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_image(path):
    """PIL resize and [0,1] conversion exactly match predictor.preprocess_image."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def validate_dataset():
    if not TRAIN_DIR.is_dir() or not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Expected training and test directories under {DATA_DIR}")
    train_dirs = {p.name for p in TRAIN_DIR.iterdir() if p.is_dir()}
    test_dirs = {p.name for p in TEST_DIR.iterdir() if p.is_dir()}
    if train_dirs != set(FOLDER_TO_LABEL):
        raise ValueError(f"Training class folders differ from expected set: {train_dirs}")
    if test_dirs != set(TEST_FOLDER_TO_CLASS):
        raise ValueError(f"Test class folders differ from expected set: {test_dirs}")

    train_files, test_files = [], []
    for folder, label in FOLDER_TO_LABEL.items():
        files = image_paths(TRAIN_DIR / folder)
        if not files:
            raise ValueError(f"No readable image files found in training class {folder}")
        train_files.extend((p, label) for p in files)
    for folder, label in TEST_FOLDER_TO_CLASS.items():
        files = image_paths(TEST_DIR / folder)
        if len(files) != 50:
            raise ValueError(f"Expected exactly 50 test images in {folder}, found {len(files)}")
        test_files.extend((p, label) for p in files)

    # Validate file structure without performing the expensive train-time resize.
    for path, _ in train_files + test_files:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"Cannot decode dataset image {path}: {exc}") from exc
    return train_files, test_files


def prepare_splits(train_files, test_files):
    """Remove exact test duplicates and stratify exact-image groups by class."""
    test_hash_to_labels = defaultdict(set)
    for path, label in test_files:
        test_hash_to_labels[file_sha256(path)].add(label)
    for digest, labels in test_hash_to_labels.items():
        if len(labels) > 1:
            raise ValueError(f"A test image has conflicting class labels: {labels} ({digest})")

    train_groups = {label: defaultdict(list) for label in CLASSES}
    excluded = Counter()
    for path, label in train_files:
        digest = file_sha256(path)
        if digest in test_hash_to_labels:
            excluded[label] += 1
            continue
        train_groups[label][digest].append(path)

    train_rows, val_rows = [], []
    unique_group_counts = {}
    for class_index, label in enumerate(CLASSES):
        groups = list(train_groups[label].items())
        unique_group_counts[label] = len(groups)
        if len(groups) < 2:
            raise ValueError(f"Too few unique non-test images for stratifying class {label}")
        random.Random(SEED + class_index).shuffle(groups)
        validation_count = max(1, round(len(groups) * VALIDATION_FRACTION))
        val_hashes = {digest for digest, _ in groups[:validation_count]}
        for digest, paths in groups:
            # One file per exact image content; hashes define the split grouping.
            row = (str(paths[0]), class_index)
            (val_rows if digest in val_hashes else train_rows).append(row)

    test_rows = [(str(path), CLASSES.index(label)) for path, label in test_files]
    return train_rows, val_rows, test_rows, dict(excluded), unique_group_counts


def _pil_read(path_tensor):
    raw_path = path_tensor.numpy() if hasattr(path_tensor, "numpy") else path_tensor
    if isinstance(raw_path, np.ndarray):
        raw_path = raw_path.item()
    raw_path = raw_path.decode("utf-8") if isinstance(raw_path, bytes) else str(raw_path)
    return load_image(raw_path)


def make_dataset(rows, training=False):
    paths = np.asarray([row[0] for row in rows], dtype=str)
    labels = np.asarray([row[1] for row in rows], dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(rows), seed=SEED, reshuffle_each_iteration=True)

    def decode(path, label):
        image = tf.numpy_function(_pil_read, [path], Tout=tf.float32)
        image.set_shape((*IMAGE_SIZE, 3))
        label.set_shape(())
        return image, label

    dataset = dataset.map(decode, num_parallel_calls=2, deterministic=True)
    dataset = dataset.batch(BATCH_SIZE).prefetch(1)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options)


def build_model(num_classes):
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.06, fill_mode="reflect", seed=SEED + 1),
        layers.RandomZoom(0.08, seed=SEED + 2),
    ], name="training_augmentation")
    backbone = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=None
    )
    backbone.load_weights(WEIGHTS)
    backbone.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="rgb_0_1")
    x = augmentation(inputs)
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_v2_preprocessing")(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25, seed=SEED)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="wheat_classes")(x)
    model = keras.Model(inputs, outputs, name="RootCause_Wheat_MobileNetV2")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def metrics_for(y_true, y_pred):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    report = {}
    for idx, label in enumerate(CLASSES):
        tp = int(matrix[idx, idx])
        fp = int(matrix[:, idx].sum() - tp)
        fn = int(matrix[idx, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {"precision": precision, "recall": recall, "f1-score": f1,
                         "support": int(matrix[idx].sum()),
                         "predicted": int(matrix[:, idx].sum())}
    accuracy = float(np.trace(matrix) / matrix.sum())
    total = int(matrix.sum())
    for title, weights in (("macro avg", None), ("weighted avg", "support")):
        report[title] = {
            metric: float(
                np.mean([report[label][metric] for label in CLASSES])
                if weights is None else
                sum(report[label][metric] * report[label][weights] for label in CLASSES) / total
            )
            for metric in ("precision", "recall", "f1-score")
        }
        report[title]["support"] = total
    report["accuracy"] = accuracy
    return accuracy, matrix, report


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.set_visible_devices([], "GPU")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Local MobileNetV2 weights not found: {WEIGHTS}")

    train_files, test_files = validate_dataset()
    train_rows, val_rows, test_rows, excluded, unique_groups = prepare_splits(train_files, test_files)
    print("Dataset split summary:", json.dumps({
        "source_train_files": len(train_files), "test_files": len(test_files),
        "excluded_train_test_duplicates": excluded, "unique_train_groups": unique_groups,
        "train_samples": len(train_rows), "validation_samples": len(val_rows),
        "test_samples": len(test_rows),
    }, indent=2))

    train_counts = Counter(label for _, label in train_rows)
    class_weights = {idx: len(train_rows) / (len(CLASSES) * train_counts[idx])
                     for idx in range(len(CLASSES))}
    train_ds = make_dataset(train_rows, training=True)
    val_ds = make_dataset(val_rows)
    test_ds = make_dataset(test_rows)

    model = build_model(len(CLASSES))
    callbacks = [keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True, verbose=1,
    )]
    started = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=MAX_EPOCHS,
        class_weight=class_weights, callbacks=callbacks, verbose=2, shuffle=False,
    )
    training_seconds = time.time() - started
    test_loss, _ = model.evaluate(test_ds, verbose=0)
    probabilities = model.predict(test_ds, verbose=0)
    y_true = np.asarray([label for _, label in test_rows], dtype=np.int32)
    y_pred = np.argmax(probabilities, axis=1)
    accuracy, matrix, report = metrics_for(y_true, y_pred)
    model.save(MODEL_OUT)
    CLASSES_OUT.write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "test_samples": len(test_rows), "test_loss": float(test_loss),
        "test_accuracy": accuracy, "balanced_test_chance_accuracy": 0.25,
        "predicted_class_counts": {label: int((y_pred == idx).sum()) for idx, label in enumerate(CLASSES)},
        "training_seconds": training_seconds,
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "class_weights": {CLASSES[idx]: float(weight) for idx, weight in class_weights.items()},
        "source_train_file_counts": dict(Counter(label for _, label in train_files)),
        "excluded_train_test_duplicate_files": excluded,
        "unique_train_image_counts_after_leakage_filter": unique_groups,
        "train_samples": len(train_rows), "validation_samples": len(val_rows),
        "class_mapping": {str(idx): label for idx, label in enumerate(CLASSES)},
        "folder_to_label": FOLDER_TO_LABEL,
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "training_history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    with (ARTIFACTS_DIR / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *CLASSES])
        for label, row in zip(CLASSES, matrix.tolist()):
            writer.writerow([label, *row])

    print(json.dumps(metrics, indent=2))
    print(f"Classification report: {ARTIFACTS_DIR / 'classification_report.json'}")
    print(f"Confusion matrix: {ARTIFACTS_DIR / 'confusion_matrix.csv'}")
    print(f"Candidate model: {MODEL_OUT}")


if __name__ == "__main__":
    main()
