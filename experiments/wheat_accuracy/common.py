"""Utilities for reproducible, isolated wheat MobileNetV2 experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization

import train_wheat_mobilenet as source

ROOT = source.ROOT
DATA_ROOT = source.DATA_DIR
TRAIN_DIR = source.TRAIN_DIR
TEST_DIR = source.TEST_DIR
WEIGHTS = source.WEIGHTS
BASELINE = source.MODEL_OUT
BASELINE_CLASSES = source.CLASSES_OUT
BASELINE_EVALUATION = source.ARTIFACTS_DIR
OUT_ROOT = source.MODELS_DIR / "wheat_accuracy_experiments"
CLASSES = list(source.CLASSES)
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
FOLDER_TO_LABEL = dict(source.FOLDER_TO_LABEL)
TEST_FOLDER_TO_LABEL = dict(source.TEST_FOLDER_TO_CLASS)
IMAGE_SIZE = tuple(source.IMAGE_SIZE)
SEED = 42
BATCH_SIZE = 16
VAL_FRACTION = 0.20
WARMUP_EPOCHS = 2
MAX_FINETUNE_EPOCHS = 4

VARIANTS: dict[str, dict[str, Any]] = {
    "augmented_frozen": {
        "description": "Frozen MobileNetV2, moderate train-only augmentation, dropout and LR scheduling",
        "unfreeze_eligible": 0,
        "label_smoothing": 0.0,
        "max_epochs": 7,
        "rotation": 0.08,
        "zoom": 0.10,
        "contrast": 0.08,
        "dropout": 0.35,
    },
    "finetune_last20": {
        "description": "Same augmentation with final 20 eligible non-BatchNorm layers fine-tuned",
        "unfreeze_eligible": 20,
        "label_smoothing": 0.0,
        "max_epochs": MAX_FINETUNE_EPOCHS,
        "rotation": 0.08,
        "zoom": 0.10,
        "contrast": 0.08,
        "dropout": 0.35,
    },
    "finetune_last20_smoothing": {
        "description": "Same 20-layer fine-tune with 0.05 label smoothing",
        "unfreeze_eligible": 20,
        "label_smoothing": 0.05,
        "max_epochs": MAX_FINETUNE_EPOCHS,
        "rotation": 0.08,
        "zoom": 0.10,
        "contrast": 0.08,
        "dropout": 0.35,
    },
}

SUPPORTED_EXTENSIONS = set(source.SUPPORTED_EXTENSIONS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def memory_status() -> dict[str, float]:
    """Query physical RAM with the Windows API (the project target platform)."""
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return {
        "total_gb": status.ullTotalPhys / (1024 ** 3),
        "available_gb": status.ullAvailPhys / (1024 ** 3),
        "memory_load_percent": float(status.dwMemoryLoad),
    }


def discover_files() -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    train = []
    test = []
    for folder, label in FOLDER_TO_LABEL.items():
        train.extend((path, label) for path in source.image_paths(TRAIN_DIR / folder))
    for folder, label in TEST_FOLDER_TO_LABEL.items():
        test.extend((path, label) for path in source.image_paths(TEST_DIR / folder))
    train.sort(key=lambda row: (row[1], str(row[0]).lower()))
    test.sort(key=lambda row: (row[1], str(row[0]).lower()))
    expected_train = {"Black Rust", "Brown Rust", "Yellow Rust", "Mildew"}
    expected_test = set(TEST_FOLDER_TO_LABEL)
    actual_train = {p.parent.name for p, _ in train}
    actual_test = {p.parent.name for p, _ in test}
    if actual_train != expected_train or actual_test != expected_test:
        raise ValueError(f"Unexpected wheat folders: train={actual_train}, test={actual_test}")
    if any(sum(label == c for _, label in test) != 50 for c in CLASSES):
        raise ValueError("Fixed wheat test set must contain 50 images per class")
    return train, test


def _inspect_one(record: tuple[Path, str, str]) -> dict[str, Any]:
    path, label, split = record
    digest = sha256_file(path)
    with Image.open(path) as image:
        dimensions = image.size
        image_format = image.format
        grayscale = np.asarray(
            ImageOps.grayscale(image).resize((8, 8), Image.Resampling.BILINEAR),
            dtype=np.int16,
        )
    # A deliberately coarse lead-generation fingerprint, not duplicate proof.
    bits = grayscale > grayscale.mean()
    average_hash = sum(int(bit) << index for index, bit in enumerate(bits.flatten()))
    return {
        "path": path,
        "label": label,
        "source_split": split,
        "sha256": digest,
        "dimensions": dimensions,
        "format": image_format,
        "average_hash": average_hash,
    }


def audit_and_make_split(
    train_files: list[tuple[Path, str]],
    test_files: list[tuple[Path, str]],
    out_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    records = [(p, label, "train") for p, label in train_files]
    records.extend((p, label, "test") for p, label in test_files)
    with ThreadPoolExecutor(max_workers=8) as pool:
        audited = list(pool.map(_inspect_one, records))

    hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in audited:
        hash_groups[item["sha256"]].append(item)
    test_hashes = {item["sha256"] for item in audited if item["source_split"] == "test"}
    test_label_conflicts = [
        group for group in hash_groups.values()
        if len({x["label"] for x in group if x["source_split"] == "test"}) > 1
    ]
    if test_label_conflicts:
        raise ValueError("Fixed test contains exact identical content with conflicting labels")

    conflicting_hashes = {
        digest for digest, group in hash_groups.items() if len({x["label"] for x in group}) > 1
    }
    train_test_groups = [
        group for group in hash_groups.values()
        if any(x["source_split"] == "test" for x in group)
        and any(x["source_split"] == "train" for x in group)
    ]
    test_duplicate_groups = [
        group for group in hash_groups.values()
        if sum(x["source_split"] == "test" for x in group) > 1
    ]

    usable_by_class: dict[str, dict[str, dict[str, Any]]] = {
        label: {} for label in CLASSES
    }
    excluded_by_reason: Counter = Counter()
    excluded_by_class: dict[str, Counter] = {label: Counter() for label in CLASSES}
    for item in audited:
        if item["source_split"] != "train":
            continue
        digest, label = item["sha256"], item["label"]
        if digest in test_hashes:
            excluded_by_reason["exact_test_copy"] += 1
            excluded_by_class[label]["exact_test_copy"] += 1
        elif digest in conflicting_hashes:
            excluded_by_reason["cross_class_exact_content_conflict"] += 1
            excluded_by_class[label]["cross_class_exact_content_conflict"] += 1
        else:
            current = usable_by_class[label].get(digest)
            if current is None or str(item["path"]).lower() < str(current["path"]).lower():
                usable_by_class[label][digest] = item

    candidates: list[dict[str, Any]] = []
    for class_index, label in enumerate(CLASSES):
        group_items = sorted(
            usable_by_class[label].items(),
            key=lambda pair: (pair[0], str(pair[1]["path"]).lower()),
        )
        if len(group_items) < 2:
            raise ValueError(f"Not enough usable training images for {label}")
        random.Random(SEED + class_index).shuffle(group_items)
        val_count = max(1, round(len(group_items) * VAL_FRACTION))
        val_hashes = {digest for digest, _ in group_items[:val_count]}
        for digest, representative in group_items:
            candidates.append({
                "path": representative["path"],
                "label": label,
                "class_index": class_index,
                "sha256": digest,
                "average_hash": representative["average_hash"],
                "split": "validation" if digest in val_hashes else "train",
                "source_split": "train",
            })

    used_paths = {str(row["path"]) for row in candidates}
    for item in audited:
        if item["source_split"] != "train" or str(item["path"]) in used_paths:
            continue
        digest = item["sha256"]
        if digest in test_hashes:
            reason = "excluded_train_exact_duplicate_of_fixed_test"
        elif digest in conflicting_hashes:
            reason = "excluded_exact_content_with_conflicting_class_labels"
        else:
            reason = "excluded_same_class_exact_duplicate_representative"
        candidates.append({
            "path": item["path"], "label": item["label"],
            "class_index": CLASS_TO_INDEX[item["label"]],
            "sha256": digest, "split": reason, "source_split": "train",
        })
    for item in audited:
        if item["source_split"] == "test":
            candidates.append({
                "path": item["path"], "label": item["label"],
                "class_index": CLASS_TO_INDEX[item["label"]],
                "sha256": item["sha256"], "split": "fixed_test",
                "source_split": "test", "average_hash": item["average_hash"],
            })

    train_hashes_by_class = {
        label: {row["sha256"] for row in candidates if row["label"] == label and row["split"] == "train"}
        for label in CLASSES
    }
    val_hashes_by_class = {
        label: {row["sha256"] for row in candidates if row["label"] == label and row["split"] == "validation"}
        for label in CLASSES
    }
    if any(train_hashes_by_class[c] & val_hashes_by_class[c] for c in CLASSES):
        raise RuntimeError("An exact-content group crosses candidate train/validation")
    if any(row["split"] in {"train", "validation"} and row["sha256"] in test_hashes
           for row in candidates):
        raise RuntimeError("An exact test image leaked into candidate training or validation")

    # Cross-split near-image screening using the coarse aHash. Results are only
    # leads: visual review confirmed both false positives and same-scene examples.
    train_candidates = [r for r in candidates if r["split"] == "train"]
    val_candidates = [r for r in candidates if r["split"] == "validation"]
    test_candidates = [r for r in candidates if r["split"] == "fixed_test"]
    near_train_val, near_examples_val = count_near_pairs(train_candidates, val_candidates)
    near_train_test, near_examples_test = count_near_pairs(train_candidates, test_candidates)

    exact_cross_groups = [g for g in hash_groups.values() if len({x["label"] for x in g}) > 1]
    exact_train_only_conflicts = [
        g for g in exact_cross_groups
        if any(x["source_split"] == "train" for x in g)
        and not any(x["source_split"] == "test" for x in g)
    ]
    dims_by_class: dict[str, Counter] = {label: Counter() for label in CLASSES}
    formats_by_class: dict[str, Counter] = {label: Counter() for label in CLASSES}
    for item in audited:
        dims_by_class[item["label"]][item["dimensions"]] += 1
        formats_by_class[item["label"]][item["format"]] += 1
    split_counts = {
        split: dict(Counter(row["label"] for row in candidates if row["split"] == split))
        for split in ("train", "validation", "fixed_test")
    }
    summary = {
        "class_order": CLASSES,
        "folder_to_label": FOLDER_TO_LABEL,
        "test_folder_to_label": TEST_FOLDER_TO_LABEL,
        "source_train_counts": dict(Counter(label for _, label in train_files)),
        "fixed_test_counts": dict(Counter(label for _, label in test_files)),
        "source_image_count": len(audited),
        "all_images_decoded": True,
        "dimensions_top": [
            {"dimensions": f"{w}x{h}", "count": n}
            for (w, h), n in Counter(item["dimensions"] for item in audited).most_common(20)
        ],
        "dimensions_by_class": {
            label: {f"{w}x{h}": n for (w, h), n in dims.items()}
            for label, dims in dims_by_class.items()
        },
        "formats_by_class": {label: dict(values) for label, values in formats_by_class.items()},
        "exact_duplicate_group_count": sum(len(group) > 1 for group in hash_groups.values()),
        "exact_train_test_group_count": len(train_test_groups),
        "train_files_exactly_matching_test_by_class": {
            label: sum(1 for group in train_test_groups for item in group
                       if item["source_split"] == "train" and item["label"] == label)
            for label in CLASSES
        },
        "exact_duplicate_groups_within_fixed_test": len(test_duplicate_groups),
        "fixed_test_duplicate_group_examples": [
            [str(item["path"].relative_to(ROOT)) for item in group]
            for group in test_duplicate_groups
        ],
        "cross_class_exact_content_group_count": len(exact_cross_groups),
        "cross_class_conflict_groups_exclusively_in_training": len(exact_train_only_conflicts),
        "conflicting_train_files_excluded_by_class": {
            label: excluded_by_class[label]["cross_class_exact_content_conflict"] for label in CLASSES
        },
        "training_test_exact_duplicate_files_excluded_by_class": {
            label: excluded_by_class[label]["exact_test_copy"] for label in CLASSES
        },
        "candidate_unique_exact_hash_groups_by_class": {
            label: len(usable_by_class[label]) for label in CLASSES
        },
        "candidate_split_counts": split_counts,
        "candidate_split_seed": SEED,
        "validation_fraction": VAL_FRACTION,
        "candidate_split_policy": (
            "Deterministic per-class stratification of exact-content groups from training folders only; "
            "exclude training files identical to any fixed test image and exact-content groups appearing in multiple classes. "
            "Fixed test files remain exactly as supplied."
        ),
        "group_metadata": (
            "No capture-session/group manifest exists for wheat. Exact file-content hashes are the only trusted groups; "
            "filename numbering is not interpreted as capture-session metadata."
        ),
        "near_duplicate_screen": {
            "method": "64-bit 8x8 grayscale average hash, Hamming distance <= 4, same-class pairs",
            "train_validation_pairs": near_train_val,
            "train_test_pairs": near_train_test,
            "interpretation": "Coarse audit leads only, not confirmed near duplicates; manual examples include false positives.",
            "train_validation_examples": near_examples_val,
            "train_test_examples": near_examples_test,
        },
        "mil_disease_label_check": (
            "Folder Mildew maps to the existing Powdery Mildew class. Reviewed training/test samples show white superficial "
            "powdery growth consistent with powdery mildew. This was a sample review, not exhaustive botanical verification; "
            "some training images contain visible source watermarks."
        ),
    }
    write_json(out_root / "dataset_audit.json", summary)
    manifest_path = out_root / "candidate_split_manifest.csv"
    with manifest_path.open("x", newline="", encoding="utf-8") as stream:
        fields = ["relative_path", "class", "class_index", "sha256", "split", "source_split"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(candidates, key=lambda r: (r["split"], r["label"], str(r["path"]))):
            writer.writerow({
                "relative_path": str(row["path"].relative_to(ROOT)).replace("\\", "/"),
                "class": row["label"], "class_index": row["class_index"],
                "sha256": row["sha256"], "split": row["split"],
                "source_split": row["source_split"],
            })
    make_mildew_montage(train_files, test_files, out_root / "mildew_label_visual_audit.jpg")
    source_hashes = {
        str(item["path"].relative_to(ROOT)).replace("\\", "/"): item["sha256"]
        for item in audited
    }
    return candidates, summary, source_hashes


def count_near_pairs(left: list[dict[str, Any]], right: list[dict[str, Any]],
                     threshold: int = 4, max_examples: int = 25):
    if not left or not right:
        return 0, []
    lookup = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)
    left_hash = np.asarray([row["average_hash"] for row in left], dtype=np.uint64)
    right_hash = np.asarray([row["average_hash"] for row in right], dtype=np.uint64)
    left_labels = np.asarray([row["label"] for row in left])
    right_labels = np.asarray([row["label"] for row in right])
    total = 0
    examples = []
    for start in range(0, len(left), 64):
        block = left_hash[start:start + 64, None] ^ right_hash[None, :]
        distances = lookup[block.view(np.uint8).reshape(block.shape + (8,))].sum(axis=2)
        hits = (distances <= threshold) & (
            left_labels[start:start + 64, None] == right_labels[None, :]
        )
        total += int(np.count_nonzero(hits))
        if len(examples) < max_examples:
            for i, j in np.argwhere(hits)[:max_examples - len(examples)]:
                examples.append({
                    "hamming_distance": int(distances[i, j]),
                    "first": str(left[start + i]["path"].relative_to(ROOT)),
                    "second": str(right[j]["path"].relative_to(ROOT)),
                })
    return total, examples


def make_mildew_montage(train_files, test_files, path: Path) -> None:
    rng = random.Random(SEED)
    samples = []
    for title, files in (
        ("Mildew training", [p for p, label in train_files if label == "Powdery Mildew"]),
        ("Mildew fixed test", [p for p, label in test_files if label == "Powdery Mildew"]),
    ):
        chosen = sorted(files)
        rng.shuffle(chosen)
        samples.extend((title, p) for p in chosen[:8])
    width, height, columns = 240, 180, 4
    label_height, margin = 30, 8
    canvas = Image.new(
        "RGB",
        (columns * (width + margin) + margin,
         ((len(samples) + columns - 1) // columns) * (height + label_height + margin) + margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, path_item) in enumerate(samples):
        x = margin + (index % columns) * (width + margin)
        y = margin + (index // columns) * (height + label_height + margin)
        with Image.open(path_item) as image:
            thumbnail = ImageOps.fit(image.convert("RGB"), (width, height))
        canvas.paste(thumbnail, (x, y))
        draw.text((x, y + height + 2), f"{title}: {path_item.name[:24]}", fill="black")
    canvas.save(path, quality=88)


def load_image(path: Path) -> np.ndarray:
    # Identical to predictor.preprocess_image: RGB, bilinear 224px, float [0,1].
    return source.load_image(path)


def rows_for_split(candidates: list[dict[str, Any]], split: str):
    return [(Path(row["path"]), int(row["class_index"]))
            for row in candidates if row["split"] == split]


def make_dataset(rows: list[tuple[Path, int]], training: bool = False) -> tf.data.Dataset:
    paths = np.asarray([str(path) for path, _ in rows], dtype=str)
    labels = np.asarray([label for _, label in rows], dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(rows), seed=SEED, reshuffle_each_iteration=True)

    def decode(path, label):
        def pil_decode(value):
            if isinstance(value, np.ndarray):
                value = value.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return load_image(Path(str(value)))

        image = tf.numpy_function(
            pil_decode,
            [path], Tout=tf.float32,
        )
        image.set_shape((*IMAGE_SIZE, 3))
        label.set_shape(())
        return image, label

    dataset = dataset.map(decode, num_parallel_calls=4, deterministic=True)
    dataset = dataset.batch(BATCH_SIZE).prefetch(2)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options)


def build_model(config: dict[str, Any]) -> tuple[keras.Model, keras.Model]:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Local MobileNetV2 ImageNet weights missing: {WEIGHTS}")
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(config["rotation"], fill_mode="reflect", seed=SEED + 1),
        layers.RandomZoom(config["zoom"], fill_mode="reflect", seed=SEED + 2),
        layers.RandomContrast(config["contrast"], seed=SEED + 3),
    ], name="training_augmentation")
    backbone = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=None
    )
    backbone.load_weights(str(WEIGHTS))
    backbone.trainable = False
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="rgb_0_1")
    x = augmentation(inputs)
    x = layers.Rescaling(2.0, offset=-1.0, name="mobilenet_v2_preprocessing")(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(config["dropout"], seed=SEED + 4, name="wheat_dropout")(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="wheat_classes")(x)
    model = keras.Model(inputs, outputs, name=f"RootCause_Wheat_{config['name']}")
    if tuple(model.input_shape[1:]) != (*IMAGE_SIZE, 3) or model.output_shape[-1] != 4:
        raise RuntimeError(f"Unexpected wheat candidate shape {model.input_shape}->{model.output_shape}")
    return model, backbone


def compile_model(model: keras.Model, learning_rate: float, smoothing: float) -> None:
    loss = (keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)
            if smoothing else "sparse_categorical_crossentropy")
    metric = (keras.metrics.CategoricalAccuracy(name="accuracy") if smoothing
              else keras.metrics.SparseCategoricalAccuracy(name="accuracy"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss, metrics=[metric],
    )


class BestValCheckpoint(keras.callbacks.Callback):
    def __init__(self, path: Path, starting_loss: float = float("inf")):
        super().__init__()
        self.path = path
        self.best = float(starting_loss)

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get("val_loss")
        if value is not None and np.isfinite(value) and value < self.best:
            self.best = float(value)
            self.model.save(self.path)
            print(f"Saved validation-best checkpoint (val_loss={self.best:.5f})")


def merge_history(parts: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = defaultdict(list)
    for part in parts:
        for key, values in part.items():
            merged[key].extend(float(value) for value in values)
    return dict(merged)


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for truth, predicted in zip(y_true, y_pred):
        matrix[int(truth), int(predicted)] += 1
    report: dict[str, Any] = {}
    for index, label in enumerate(CLASSES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {
            "precision": precision, "recall": recall, "f1-score": f1,
            "support": int(matrix[index].sum()),
            "predicted": int(matrix[:, index].sum()),
        }
    total = int(matrix.sum())
    report["accuracy"] = float(np.trace(matrix) / total) if total else 0.0
    for title, weighted in (("macro avg", False), ("weighted avg", True)):
        denom = total if weighted else len(CLASSES)
        report[title] = {
            metric: float(sum(
                report[label][metric] * (report[label]["support"] if weighted else 1)
                for label in CLASSES
            ) / denom) if denom else 0.0
            for metric in ("precision", "recall", "f1-score")
        }
        report[title]["support"] = total
    return report["accuracy"], matrix, report


def train_variant(name: str, train_rows, val_rows, out_dir: Path) -> dict[str, Any]:
    if name not in VARIANTS:
        raise ValueError(f"Unknown experiment {name}")
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite variant output: {out_dir}")
    config = {**VARIANTS[name], "name": name}
    out_dir.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(SEED)
    model, backbone = build_model(config)
    y_train = np.asarray([label for _, label in train_rows], dtype=np.int32)
    counts = Counter(y_train.tolist())
    if set(counts) != set(range(len(CLASSES))):
        raise ValueError("Candidate training split does not contain all four wheat classes")
    class_weights = {
        index: len(y_train) / (len(CLASSES) * counts[index])
        for index in range(len(CLASSES))
    }
    train_dataset = make_dataset(train_rows, training=True)
    val_dataset = make_dataset(val_rows)
    model_path = out_dir / f"wheat_{name}.keras"
    history_parts: list[dict[str, list[float]]] = []
    started = time.perf_counter()

    if config["unfreeze_eligible"] == 0:
        compile_model(model, learning_rate=5e-4, smoothing=config["label_smoothing"])
        history = model.fit(
            train_dataset, validation_data=val_dataset,
            epochs=config["max_epochs"], batch_size=BATCH_SIZE, class_weight=class_weights,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    model_path, monitor="val_loss", save_best_only=True, verbose=1,
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.3, patience=1, min_lr=1e-6, verbose=1,
                ),
            ], verbose=2,
        )
        history_parts.append(history.history)
    else:
        if config["label_smoothing"]:
            y_train_fit = keras.utils.to_categorical(y_train, num_classes=len(CLASSES))
        else:
            y_train_fit = y_train
        if config["label_smoothing"]:
            train_dataset = train_dataset.map(
                lambda image, label: (image, tf.one_hot(label, len(CLASSES))),
                num_parallel_calls=1,
            )
            val_dataset = val_dataset.map(
                lambda image, label: (image, tf.one_hot(label, len(CLASSES))),
                num_parallel_calls=1,
            )
        # Head-only warm-up before low-rate fine-tuning.
        compile_model(model, learning_rate=5e-4, smoothing=config["label_smoothing"])
        warm = model.fit(
            train_dataset, validation_data=val_dataset,
            epochs=WARMUP_EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weights,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True, verbose=1,
            )], verbose=2,
        )
        history_parts.append(warm.history)
        best_loss = min(float(value) for value in warm.history["val_loss"])
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = False
        eligible = [layer for layer in backbone.layers if not isinstance(layer, BatchNormalization)]
        selected = eligible[-config["unfreeze_eligible"]:]
        if len(selected) != config["unfreeze_eligible"]:
            raise RuntimeError("MobileNetV2 has fewer eligible layers than requested")
        for layer in selected:
            layer.trainable = True
        if any(layer.trainable for layer in backbone.layers if isinstance(layer, BatchNormalization)):
            raise RuntimeError("BatchNormalization layer unexpectedly unfrozen")
        compile_model(model, learning_rate=1e-5, smoothing=config["label_smoothing"])
        model.save(model_path)
        fine = model.fit(
            train_dataset, validation_data=val_dataset,
            epochs=config["max_epochs"], batch_size=BATCH_SIZE, class_weight=class_weights,
            callbacks=[
                BestValCheckpoint(model_path, best_loss),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", baseline=best_loss, patience=2,
                    restore_best_weights=True, verbose=1,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.3, patience=1, min_lr=1e-7, verbose=1,
                ),
            ], verbose=2,
        )
        history_parts.append(fine.history)

    if not model_path.is_file():
        raise RuntimeError(f"No validation-selected model checkpoint was saved: {model_path}")
    seconds = time.perf_counter() - started
    selected = keras.models.load_model(str(model_path))
    val_prob = selected.predict(val_dataset, verbose=0)
    y_val = np.asarray([label for _, label in val_rows], dtype=np.int32)
    y_pred = val_prob.argmax(axis=1)
    _, matrix, report = metrics_for(y_val, y_pred)
    history_data = merge_history(history_parts)
    metrics = {
        "experiment": name,
        "description": config["description"],
        "validation_samples": len(y_val),
        "validation_accuracy": report["accuracy"],
        "validation_macro_f1": report["macro avg"]["f1-score"],
        "validation_weighted_f1": report["weighted avg"]["f1-score"],
        "validation_predicted_counts": {label: int((y_pred == i).sum())
                                        for i, label in enumerate(CLASSES)},
        "class_weights": {CLASSES[i]: float(w) for i, w in class_weights.items()},
        "batch_size": BATCH_SIZE,
        "warmup_epochs": WARMUP_EPOCHS if config["unfreeze_eligible"] else 0,
        "fine_tune_max_epochs": config["max_epochs"],
        "fine_tune_eligible_layers": config["unfreeze_eligible"],
        "label_smoothing": config["label_smoothing"],
        "augmentation": {
            "horizontal_flip": True, "rotation": config["rotation"],
            "zoom": config["zoom"], "contrast": config["contrast"],
            "training_only": True,
        },
        "dropout": config["dropout"],
        "seed": SEED,
        "training_seconds": seconds,
        "model_bytes": model_path.stat().st_size,
        "class_to_index": CLASS_TO_INDEX,
        "validation_classification_report": report,
        "validation_confusion_matrix": matrix.tolist(),
        "model_path": str(model_path),
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "classification_report_validation.json", report)
    write_json(out_dir / "training_history.json", history_data)
    write_csv_matrix(out_dir / "confusion_matrix_validation.csv", matrix)
    write_confusion_svg(out_dir / "confusion_matrix_validation.svg", matrix)
    write_curves_svg(out_dir / "training_validation_curves.svg", history_data)
    write_curves_svg(out_dir / "training_validation_loss.svg", history_data, loss=True)
    del selected, model
    keras.backend.clear_session()
    return metrics


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\predicted", *CLASSES])
        for label, row in zip(CLASSES, matrix.tolist()):
            writer.writerow([label, *row])


def write_confusion_svg(path: Path, matrix: np.ndarray) -> None:
    cell, left, top = 90, 190, 72
    width, height = left + cell * len(CLASSES) + 20, top + cell * len(CLASSES) + 30
    maximum = max(1, int(matrix.max()))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font:13px Arial,sans-serif;fill:#14212b}.title{font-size:16px;font-weight:bold}</style>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="12" y="24" class="title">Confusion matrix (rows=true, columns=predicted)</text>']
    for j, label in enumerate(CLASSES):
        parts.append(f'<text transform="translate({left+j*cell+cell/2} {top-5}) rotate(-35)" text-anchor="start">{label}</text>')
    for i, label in enumerate(CLASSES):
        parts.append(f'<text x="{left-8}" y="{top+i*cell+cell/2+4}" text-anchor="end">{label}</text>')
        for j in range(len(CLASSES)):
            value = int(matrix[i, j])
            blue = int(255 - 125 * value / maximum)
            x, y = left + j * cell, top + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="rgb({blue},{blue},255)" stroke="white"/>')
            parts.append(f'<text x="{x+cell/2}" y="{y+cell/2+5}" text-anchor="middle">{value}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_curves_svg(path: Path, history: dict[str, list[float]], loss: bool = False) -> None:
    width, height = 760, 350
    x0, y0, plot_w, plot_h = 65, 38, 650, 255
    keys = ([("loss", "train loss", "#1677b8"), ("val_loss", "validation loss", "#ed7d31")]
            if loss else [("accuracy", "train accuracy", "#1677b8"),
                          ("val_accuracy", "validation accuracy", "#ed7d31")])
    values = [float(v) for key, _, _ in keys for v in history.get(key, []) if np.isfinite(v)]
    low, high = (min(values), max(values)) if values else (0.0, 1.0)
    if high - low < 0.05:
        low = max(0.0, low - 0.05)
        high += 0.05
    n = max((len(history.get(key, [])) for key, _, _ in keys), default=1)
    title = "Training/validation loss" if loss else "Training/validation accuracy"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font:12px Arial,sans-serif;fill:#14212b}.title{font-size:16px;font-weight:bold}</style>',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="16" y="23" class="title">{title} ({n} epochs)</text>']
    for tick in range(6):
        y = y0 + plot_h * tick / 5
        value = high - (high - low) * tick / 5
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" stroke="#d9e0e6"/>')
        parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end">{value:.2f}</text>')
    for key, label, color in keys:
        points = []
        for i, value in enumerate(history.get(key, [])):
            x = x0 + plot_w * i / max(1, n - 1)
            y = y0 + plot_h * (high - float(value)) / (high - low if high > low else 1)
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            joined = " ".join(points)
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{joined}"/>')
        parts.append(f'<text x="{x0+10+(0 if key in {"accuracy","loss"} else 180)}" y="{height-14}" fill="{color}">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
