"""Shared utilities for isolated rice accuracy experiments.

This package never edits the original manifests or existing model artifacts.
The source test assignment is kept fixed and excluded from training/selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/rice/processed/pristine"
SPLITS = ROOT / "data/rice/processed/rice_splits.csv"
GROUPS = ROOT / "data/rice/processed/rice_groups.csv"
WEIGHTS = ROOT / "app/models/pretrained/mobilenet_v2_weights.h5"
BASELINE = ROOT / "app/models/rice_mobilenet_5class.keras"
OUT_ROOT = ROOT / "app/models/rice_accuracy_experiments"
CLASSES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf scald",
    "Sheath Blight",
]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
SIZE = (224, 224)
SEED = 42
BATCH_SIZE = 16
MAX_FINE_TUNE_EPOCHS = 6
WARMUP_EPOCHS = 2
VAL_FRACTION = 0.20

VARIANTS: dict[str, dict[str, Any]] = {
    "augmented_ft30": {
        "description": "Moderate augmentation, 30 eligible MobileNetV2 layers fine-tuned",
        "unfreeze_eligible": 30,
        "label_smoothing": 0.0,
        "rotation": 0.08,
        "zoom": 0.10,
        "contrast": 0.08,
        "dropout": 0.35,
    },
    "shallow_ft10": {
        "description": "Same training recipe as augmented_ft30, only 10 eligible layers fine-tuned",
        "unfreeze_eligible": 10,
        "label_smoothing": 0.0,
        "rotation": 0.08,
        "zoom": 0.10,
        "contrast": 0.08,
        "dropout": 0.35,
    },
    "label_smoothing_ft30": {
        "description": "Same as augmented_ft30 with 0.05 label smoothing",
        "unfreeze_eligible": 30,
        "label_smoothing": 0.05,
        "rotation": 0.08,
        "zoom": 0.10,
        "contrast": 0.08,
        "dropout": 0.35,
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_source_rows() -> list[dict[str, str]]:
    with SPLITS.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"class", "filename", "group_id", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid source split manifest; required {sorted(required)}")
    if len({(r["class"], r["filename"]) for r in rows}) != len(rows):
        raise ValueError("Source split manifest contains duplicate image rows")
    if {r["split"] for r in rows} - {"train", "val", "test"}:
        raise ValueError("Source split manifest contains unknown split names")
    if GROUPS.is_file():
        with GROUPS.open(newline="", encoding="utf-8-sig") as stream:
            group_rows = list(csv.DictReader(stream))
        group_map = {(r["class"], r["filename"]): r["group_id"] for r in group_rows}
        for row in rows:
            if group_map.get((row["class"], row["filename"])) != row["group_id"]:
                raise ValueError(f"Source split/group mismatch: {row['class']}/{row['filename']}")
    return rows


def make_candidate_split(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Stratify source train+val per class; copy source test rows unchanged."""
    rng = random.Random(SEED)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["class"] in CLASS_TO_INDEX and row["split"] in {"train", "val"}:
            by_class[row["class"]].append(row)

    candidate: list[dict[str, str]] = []
    for label in CLASSES:
        samples = sorted(by_class[label], key=lambda r: r["filename"])
        if len(samples) < 5:
            raise ValueError(f"Too few non-test samples for stratified split: {label}")
        rng.shuffle(samples)
        n_val = max(1, round(len(samples) * VAL_FRACTION))
        val_names = {row["filename"] for row in samples[:n_val]}
        for row in samples:
            candidate.append({
                "class": row["class"],
                "filename": row["filename"],
                "group_id": row["group_id"],
                "split": "validation" if row["filename"] in val_names else "train",
                "source_split": row["split"],
            })

    # Keep the existing test rows exactly in the fixed test assignment. They are
    # recorded for audit only, never used by training or validation selection.
    for row in rows:
        if row["class"] in CLASS_TO_INDEX and row["split"] == "test":
            candidate.append({
                "class": row["class"],
                "filename": row["filename"],
                "group_id": row["group_id"],
                "split": "fixed_test",
                "source_split": "test",
            })
    return sorted(candidate, key=lambda r: (r["split"], r["class"], r["filename"]))


def load_images(rows: list[dict[str, str]], split_name: str) -> tuple[np.ndarray, np.ndarray]:
    x: list[np.ndarray] = []
    y: list[int] = []
    for row in rows:
        if row["split"] != split_name:
            continue
        path = DATA_ROOT / row["class"] / row["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            resized = image.convert("RGB").resize(SIZE, Image.Resampling.BILINEAR)
            x.append(np.asarray(resized, dtype=np.float32) / 255.0)
        y.append(CLASS_TO_INDEX[row["class"]])
    if not x:
        raise ValueError(f"No selected rice images in split {split_name!r}")
    return np.stack(x), np.asarray(y, dtype=np.int32)


def group_overlap(rows: list[dict[str, str]]) -> list[str]:
    assignments: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["split"] in {"train", "validation"}:
            assignments[row["group_id"]].add(row["split"])
    return sorted(group for group, splits in assignments.items() if len(splits) > 1)


def make_model(config: dict[str, Any]) -> tuple[keras.Model, list[str]]:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Local ImageNet MobileNetV2 weights missing: {WEIGHTS}")
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(config["rotation"], fill_mode="reflect", seed=SEED + 1),
        layers.RandomZoom(config["zoom"], fill_mode="reflect", seed=SEED + 2),
        layers.RandomContrast(config["contrast"], seed=SEED + 3),
    ], name="training_augmentation")
    backbone = keras.applications.MobileNetV2(
        input_shape=(*SIZE, 3), include_top=False, weights=None,
    )
    backbone.load_weights(str(WEIGHTS))
    backbone.trainable = False

    inputs = keras.Input(shape=(*SIZE, 3), name="rgb_0_1")
    x = augmentation(inputs)
    # Input data and predictor use RGB [0,1]. MobileNetV2 requires [-1,1].
    x = layers.Rescaling(2.0, offset=-1.0, name="mobilenet_v2_preprocessing")(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(config["dropout"], seed=SEED + 4, name="head_dropout")(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="rice_classes")(x)
    model = keras.Model(inputs, outputs, name=f"RootCause_{config['name']}")
    if model.output_shape[-1] != len(CLASSES):
        raise RuntimeError(f"Invalid candidate output shape {model.output_shape}")
    return model, backbone.layers


def compile_model(model: keras.Model, learning_rate: float, smoothing: float) -> None:
    loss: Any = (keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)
                 if smoothing else "sparse_categorical_crossentropy")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy") if smoothing
                 else keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


class BestValLossCheckpoint(keras.callbacks.Callback):
    """Save only when val_loss beats the best warmup or fine-tuning value."""
    def __init__(self, path: Path, initial_best: float):
        super().__init__()
        self.path = path
        self.best = initial_best

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get("val_loss")
        if value is not None and np.isfinite(value) and value < self.best:
            self.best = float(value)
            self.model.save(self.path)
            print(f"Saved validation-best checkpoint: val_loss={self.best:.5f}")


def train_variant(name: str, x_train: np.ndarray, y_train: np.ndarray,
                  x_val: np.ndarray, y_val: np.ndarray, out_dir: Path) -> dict[str, Any]:
    if name not in VARIANTS:
        raise ValueError(f"Unknown experiment {name}; choose one of {sorted(VARIANTS)}")
    config = {**VARIANTS[name], "name": name}
    model_path = out_dir / f"rice_5class_{name}.keras"
    if out_dir.exists() or model_path.exists():
        raise FileExistsError(f"Refusing to overwrite experiment output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.set_visible_devices([], "GPU")
    model, backbone_layers = make_model(config)
    eligible = [layer for layer in backbone_layers if not isinstance(layer, BatchNormalization)]
    selected = eligible[-config["unfreeze_eligible"]:]
    if len(selected) != config["unfreeze_eligible"]:
        raise RuntimeError("Backbone did not provide requested eligible fine-tune depth")
    counts = Counter(y_train.tolist())
    if set(counts) != set(range(len(CLASSES))):
        raise ValueError("All five classes must be represented in candidate training data")
    weights = {index: len(y_train) / (len(CLASSES) * counts[index])
               for index in range(len(CLASSES))}
    if config["label_smoothing"]:
        y_train_fit = keras.utils.to_categorical(y_train, num_classes=len(CLASSES))
        y_val_fit = keras.utils.to_categorical(y_val, num_classes=len(CLASSES))
    else:
        y_train_fit, y_val_fit = y_train, y_val
    history_parts: list[dict[str, list[float]]] = []
    start = time.perf_counter()

    # Phase 1: train a fresh softmax head while keeping the pretrained backbone frozen.
    compile_model(model, learning_rate=5e-4, smoothing=config["label_smoothing"])
    warm = model.fit(
        x_train, y_train_fit, validation_data=(x_val, y_val_fit),
        epochs=WARMUP_EPOCHS, batch_size=BATCH_SIZE, class_weight=weights,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True,
        )], verbose=2, shuffle=True,
    )
    history_parts.append(warm.history)
    best_val_loss = min(float(v) for v in warm.history["val_loss"])

    # Phase 2: unfreeze only the requested tail; keep every BN layer frozen.
    backbone = model.get_layer("mobilenetv2_1.00_224")
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in selected:
        layer.trainable = True
    if any(layer.trainable for layer in backbone.layers if isinstance(layer, BatchNormalization)):
        raise RuntimeError("BatchNormalization unexpectedly trainable")
    compile_model(model, learning_rate=1e-5, smoothing=config["label_smoothing"])
    model.save(model_path)
    checkpoint = BestValLossCheckpoint(model_path, best_val_loss)
    fine = model.fit(
        x_train, y_train_fit, validation_data=(x_val, y_val_fit),
        epochs=MAX_FINE_TUNE_EPOCHS, batch_size=BATCH_SIZE, class_weight=weights,
        callbacks=[
            checkpoint,
            keras.callbacks.EarlyStopping(
                monitor="val_loss", baseline=best_val_loss, patience=2,
                restore_best_weights=True, verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=1,
                min_lr=1e-7, verbose=1,
            ),
        ], verbose=2, shuffle=True,
    )
    history_parts.append(fine.history)
    duration = time.perf_counter() - start
    selected_model = keras.models.load_model(model_path)
    val_probabilities = selected_model.predict(x_val, batch_size=BATCH_SIZE, verbose=0)
    val_predictions = val_probabilities.argmax(axis=1)
    _, report, matrix = classification_metrics(y_val, val_predictions)
    history = merge_history(history_parts)
    metrics = {
        "experiment": name,
        "description": config["description"],
        "split": "reproducible stratified 80/20 split of source train+validation only",
        "validation_samples": int(len(y_val)),
        "validation_accuracy": report["accuracy"],
        "validation_macro_f1": report["macro avg"]["f1-score"],
        "validation_weighted_f1": report["weighted avg"]["f1-score"],
        "validation_class_prediction_counts": {
            label: int((val_predictions == index).sum()) for index, label in enumerate(CLASSES)
        },
        "class_weights": {CLASSES[index]: float(value) for index, value in weights.items()},
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "warmup_epochs_requested": WARMUP_EPOCHS,
        "fine_tune_epochs_max": MAX_FINE_TUNE_EPOCHS,
        "fine_tune_eligible_layers": config["unfreeze_eligible"],
        "label_smoothing": config["label_smoothing"],
        "augmentation": {key: config[key] for key in ("rotation", "zoom", "contrast")},
        "dropout": config["dropout"],
        "training_seconds": duration,
        "model_bytes": model_path.stat().st_size,
        "class_to_index": CLASS_TO_INDEX,
        "val_classification_report": report,
        "val_confusion_matrix": matrix.tolist(),
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "training_history.json", history)
    write_json(out_dir / "classification_report_validation.json", report)
    write_csv_matrix(out_dir / "confusion_matrix_validation.csv", matrix)
    write_confusion_svg(out_dir / "confusion_matrix_validation.svg", matrix)
    write_history_svg(out_dir / "training_curves.svg", history)
    write_loss_svg(out_dir / "loss_curves.svg", history)
    del selected_model, model
    keras.backend.clear_session()
    return metrics


def merge_history(parts: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = defaultdict(list)
    for part in parts:
        for key, values in part.items():
            merged[key].extend(float(value) for value in values)
    return dict(merged)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    report: dict[str, Any] = {}
    for index, label in enumerate(CLASSES):
        tp = int(matrix[index, index]); fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {"precision": precision, "recall": recall,
                         "f1-score": f1, "support": int(matrix[index].sum())}
    total = int(matrix.sum())
    report["accuracy"] = float(np.trace(matrix) / total) if total else 0.0
    for name, weighted in (("macro avg", False), ("weighted avg", True)):
        denom = total if weighted else len(CLASSES)
        report[name] = {
            metric: float(sum(report[label][metric] *
                              (report[label]["support"] if weighted else 1)
                              for label in CLASSES) / denom) if denom else 0.0
            for metric in ("precision", "recall", "f1-score")
        }
        report[name]["support"] = total
    return float(report["accuracy"]), report, matrix


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\predicted", *CLASSES])
        for label, row in zip(CLASSES, matrix.tolist()):
            writer.writerow([label, *row])


def write_confusion_svg(path: Path, matrix: np.ndarray) -> None:
    cell = 72; left = 190; top = 65
    width = left + cell * len(CLASSES) + 20
    height = top + cell * len(CLASSES) + 20
    maximum = max(1, int(matrix.max()))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font:12px Arial,sans-serif;fill:#14212b}.title{font-size:16px;font-weight:bold}</style>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="12" y="24" class="title">Confusion matrix (rows=true, columns=predicted)</text>']
    for j, label in enumerate(CLASSES):
        parts.append(f'<text transform="translate({left+j*cell+cell/2} {top-5}) rotate(-40)" text-anchor="start">{label}</text>')
    for i, label in enumerate(CLASSES):
        parts.append(f'<text x="{left-8}" y="{top+i*cell+cell/2+4}" text-anchor="end">{label}</text>')
        for j in range(len(CLASSES)):
            value = int(matrix[i, j]); strength = int(235 - 130 * value / maximum)
            color = f'rgb({strength},{min(255,strength+12)},{255})'
            x = left + j * cell; y = top + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#fff"/>')
            parts.append(f'<text x="{x+cell/2}" y="{y+cell/2+5}" text-anchor="middle">{value}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def write_history_svg(path: Path, history: dict[str, list[float]]) -> None:
    width, height = 760, 360
    x0, y0, plot_w, plot_h = 64, 34, 650, 275
    keys = [("accuracy", "train accuracy", "#1677b8"),
            ("val_accuracy", "validation accuracy", "#ed7d31")]
    n = max((len(history.get(k, [])) for k, _, _ in keys), default=1)
    vals = [v for k, _, _ in keys for v in history.get(k, [])]
    lo, hi = (min(vals), max(vals)) if vals else (0, 1)
    if hi-lo < 0.05: lo=max(0,lo-0.05); hi=min(1,hi+0.05)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font:12px Arial,sans-serif;fill:#14212b}.title{font-size:16px;font-weight:bold}</style>',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="16" y="22" class="title">Training/validation accuracy ({n} recorded phase epochs)</text>']
    for tick in range(6):
        y = y0 + plot_h * tick / 5; value = hi - (hi-lo)*tick/5
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" stroke="#d9e0e6"/>')
        parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end">{value:.2f}</text>')
    for key, label, color in keys:
        points=[]
        for i, value in enumerate(history.get(key, [])):
            x=x0+(plot_w*(i/(max(1,n-1))))
            y=y0+plot_h*(hi-float(value))/(hi-lo if hi>lo else 1)
            points.append(f'{x:.1f},{y:.1f}')
        if points:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>')
        parts.append(f'<text x="{x0+8+(0 if key=="accuracy" else 180)}" y="{height-16}" fill="{color}">{label}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def write_loss_svg(path: Path, history: dict[str, list[float]]) -> None:
    width, height = 760, 360
    x0, y0, plot_w, plot_h = 64, 34, 650, 275
    keys = [("loss", "train loss", "#1677b8"),
            ("val_loss", "validation loss", "#ed7d31")]
    n = max((len(history.get(k, [])) for k, _, _ in keys), default=1)
    vals = [float(v) for k, _, _ in keys for v in history.get(k, []) if np.isfinite(v)]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    if hi - lo < 0.05:
        lo = max(0.0, lo - 0.05)
        hi += 0.05
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font:12px Arial,sans-serif;fill:#14212b}.title{font-size:16px;font-weight:bold}</style>',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="16" y="22" class="title">Training/validation loss ({n} recorded phase epochs)</text>']
    for tick in range(6):
        y = y0 + plot_h * tick / 5
        value = hi - (hi - lo) * tick / 5
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" stroke="#d9e0e6"/>')
        parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end">{value:.2f}</text>')
    for key, label, color in keys:
        points = []
        for i, value in enumerate(history.get(key, [])):
            x = x0 + plot_w * (i / max(1, n - 1))
            y = y0 + plot_h * (hi - float(value)) / (hi - lo if hi > lo else 1)
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>')
        parts.append(f'<text x="{x0+8+(0 if key=="loss" else 180)}" y="{height-16}" fill="{color}">{label}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")
