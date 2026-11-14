"""Five-class rice fine-tuning experiment based on the six-class candidate.

The existing rice_mobilenet_candidate.keras supplies the augmentation,
preprocessing, MobileNetV2 backbone, and pooling/dropout features. Its six-class
Dense output is intentionally discarded and replaced with a fresh five-class
head. The original split assignments and test set are preserved.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense

ROOT = Path(__file__).resolve().parent
PRISTINE_DIR = ROOT / "data/rice/processed/pristine"
SPLITS_CSV = ROOT / "data/rice/processed/rice_splits.csv"
GROUPS_CSV = ROOT / "data/rice/processed/rice_groups.csv"
BASELINE = ROOT / "app/models/rice_mobilenet_candidate.keras"
FINETUNED_MODEL = ROOT / "app/models/rice_mobilenet_finetuned.keras"
OUTPUT = ROOT / "app/models/rice_mobilenet_5class.keras"
ARTIFACTS = ROOT / "app/models/rice_mobilenet_5class_evaluation"

# Deliberate, stable output order. Leaf Blast is not part of this experiment.
CLASSES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf scald",
    "Sheath Blight",
]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
MAX_EPOCHS = 8
PATIENCE = 2
LEARNING_RATE = 1e-5
UNFREEZE_LAST_ELIGIBLE = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    """Hash all protected source assets that this experiment must preserve."""
    protected = {
        "baseline_model": BASELINE,
        "fine_tuned_model": FINETUNED_MODEL,
        "split_manifest": SPLITS_CSV,
        "group_manifest": GROUPS_CSV,
    }
    missing = [str(path) for path in protected.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Protected inputs are missing: " + ", ".join(missing))
    return {name: sha256(path) for name, path in protected.items()}


def verify_protected_hashes(before: dict[str, str]) -> None:
    paths = {
        "baseline_model": BASELINE,
        "fine_tuned_model": FINETUNED_MODEL,
        "split_manifest": SPLITS_CSV,
        "group_manifest": GROUPS_CSV,
    }
    changed = [name for name, path in paths.items() if sha256(path) != before[name]]
    if changed:
        raise RuntimeError("Protected input hashes changed: " + ", ".join(changed))


def check_output_paths(overwrite: bool) -> None:
    existing = [path for path in (OUTPUT, ARTIFACTS) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing five-class outputs: " + names
            + ". Re-run with --overwrite only if replacement is explicitly intended."
        )
    if overwrite:
        if OUTPUT.is_dir():
            raise IsADirectoryError(f"Model output path is a directory: {OUTPUT}")
        if OUTPUT.exists():
            OUTPUT.unlink()
        if ARTIFACTS.exists():
            if not ARTIFACTS.is_dir():
                raise NotADirectoryError(f"Evaluation output path is not a directory: {ARTIFACTS}")
            shutil.rmtree(ARTIFACTS)


def load_manifest():
    """Load fixed assignments and verify the group manifest without editing either."""
    with SPLITS_CSV.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"class", "filename", "group_id", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Split manifest must contain {sorted(required)}")
    if {row["split"] for row in rows} - {"train", "val", "test"}:
        raise ValueError("Split manifest contains an unknown split")
    keys = [(row["class"], row["filename"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Split manifest contains duplicate class/filename rows")
    manifest_keys = set(keys)

    with GROUPS_CSV.open(newline="", encoding="utf-8-sig") as f:
        group_rows = list(csv.DictReader(f))
    group_map = {(row["class"], row["filename"]): row["group_id"] for row in group_rows}
    if set(group_map) != manifest_keys:
        raise ValueError("rice_splits.csv and rice_groups.csv do not contain the same images")
    assignments = defaultdict(set)
    for row in rows:
        key = (row["class"], row["filename"])
        if group_map[key] != row["group_id"]:
            raise ValueError(f"Group ID mismatch for {key}")
        assignments[row["group_id"]].add(row["split"])
    crossed_groups = [group for group, splits in assignments.items() if len(splits) != 1]
    if crossed_groups:
        raise ValueError(f"Capture groups cross split boundaries: {crossed_groups[:10]}")

    present = {row["class"] for row in rows}
    if not set(CLASSES).issubset(present):
        raise ValueError(f"Required five-class labels are missing: {set(CLASSES) - present}")
    return rows


def verify_no_cross_split_duplicates(rows) -> None:
    """Reject exact duplicate files assigned to different fixed splits."""
    seen = {}
    for row in rows:
        if row["class"] not in CLASS_TO_INDEX:
            continue
        path = PRISTINE_DIR / row["class"] / row["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {path}")
        digest = sha256(path)
        if digest in seen:
            prior_split, prior_class, prior_path = seen[digest]
            if prior_split != row["split"]:
                raise ValueError(
                    "Exact duplicate image crosses fixed split assignments: "
                    f"{prior_path} ({prior_split}) and {path} ({row['split']})"
                )
            if prior_class != row["class"]:
                raise ValueError(
                    f"Exact duplicate content has conflicting labels: {prior_path} and {path}"
                )
        seen[digest] = (row["split"], row["class"], path)


def load_split(rows, split):
    """Read only the requested five-class split using the predictor's PIL pipeline."""
    images, labels = [], []
    for row in rows:
        if row["split"] != split or row["class"] not in CLASS_TO_INDEX:
            continue
        path = PRISTINE_DIR / row["class"] / row["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {path}")
        try:
            with Image.open(path) as image:
                image = image.convert("RGB").resize(
                    IMAGE_SIZE, Image.Resampling.BILINEAR
                )
                images.append(np.asarray(image, dtype=np.float32) / 255.0)
        except Exception as exc:
            raise ValueError(f"Cannot decode {path}: {exc}") from exc
        labels.append(CLASS_TO_INDEX[row["class"]])
    if not images:
        raise ValueError(f"No five-class images found in split {split!r}")
    return np.stack(images), np.asarray(labels, dtype=np.int32)


def build_five_class_model():
    """Reuse candidate feature layers and replace its original six-class head."""
    source = keras.models.load_model(BASELINE)
    if tuple(source.input_shape[1:]) != (*IMAGE_SIZE, 3) or source.output_shape[-1] != 6:
        raise ValueError(
            f"Expected the six-class 224x224 candidate, got {source.input_shape} -> {source.output_shape}"
        )

    augmentation = source.get_layer("training_augmentation")
    preprocessing = source.get_layer("mobilenet_v2_preprocessing")
    backbone = source.get_layer("mobilenetv2_1.00_224")
    pooling = source.get_layer("global_average_pooling2d")
    dropout = source.get_layer("dropout")
    original_head = source.get_layer("rice_classes")
    if not isinstance(original_head, Dense) or original_head.units != 6:
        raise ValueError("The source model's six-class Dense head was not found as expected")

    # Make every backbone layer non-trainable first, then unfreeze the final 30
    # eligible (non-BatchNormalization) layers. BN remains frozen even though the
    # nested backbone itself must be marked trainable for selected layers to train.
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    eligible = [layer for layer in backbone.layers if not isinstance(layer, BatchNormalization)]
    if len(eligible) < UNFREEZE_LAST_ELIGIBLE:
        raise ValueError(f"Backbone has only {len(eligible)} eligible layers")
    selected = eligible[-UNFREEZE_LAST_ELIGIBLE:]
    for layer in selected:
        layer.trainable = True
    if any(layer.trainable for layer in backbone.layers if isinstance(layer, BatchNormalization)):
        raise RuntimeError("A BatchNormalization layer was unexpectedly unfrozen")

    inputs = keras.Input(shape=source.input_shape[1:], name="rgb_0_1")
    x = augmentation(inputs)
    x = preprocessing(x)
    x = backbone(x, training=False)
    x = pooling(x)
    x = dropout(x)
    outputs = Dense(len(CLASSES), activation="softmax", name="rice_5class_output")(x)
    model = keras.Model(inputs, outputs, name="RootCause_Rice_MobileNetV2_5Class")
    if model.get_layer("rice_5class_output") is original_head:
        raise RuntimeError("The six-class head was reused unexpectedly")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, [layer.name for layer in selected]


def evaluation_report(y_true, y_pred):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth), int(prediction)] += 1
    report = {}
    for index, label in enumerate(CLASSES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": int(matrix[index, :].sum()),
        }
    total = int(matrix.sum())
    for name, weighted in (("macro avg", False), ("weighted avg", True)):
        report[name] = {
            metric: float(
                sum(report[label][metric] * (report[label]["support"] if weighted else 1)
                    for label in CLASSES)
                / (total if weighted else len(CLASSES))
            )
            for metric in ("precision", "recall", "f1_score")
        }
        report[name]["support"] = total
    report["accuracy"] = float(np.trace(matrix) / total) if total else 0.0
    return matrix, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Explicitly replace existing five-class model/evaluation outputs.",
    )
    args = parser.parse_args()

    check_output_paths(args.overwrite)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.set_visible_devices([], "GPU")

    protected_before = protected_hashes()
    rows = load_manifest()
    verify_no_cross_split_duplicates(rows)
    selected_rows = [row for row in rows if row["class"] in CLASS_TO_INDEX]
    split_counts = {
        split: Counter(row["class"] for row in selected_rows if row["split"] == split)
        for split in ("train", "val", "test")
    }
    if any(not split_counts[split] for split in split_counts):
        raise ValueError("One of the fixed splits has no five-class samples")

    print("Class-to-index mapping:", json.dumps(CLASS_TO_INDEX, indent=2))
    print("Fixed split counts:", json.dumps({k: dict(v) for k, v in split_counts.items()}, indent=2))
    print("Leaf Blast rows excluded:", sum(row["class"] == "Leaf Blast" for row in rows))
    print("Loading train and validation data; test data remains unopened until model selection.")
    x_train, y_train = load_split(rows, "train")
    x_val, y_val = load_split(rows, "val")

    train_counts = Counter(y_train.tolist())
    if set(train_counts) != set(range(len(CLASSES))):
        raise ValueError("Every selected class must have at least one fixed training sample")
    class_weights = {
        index: len(y_train) / (len(CLASSES) * train_counts[index])
        for index in range(len(CLASSES))
    }
    model, unfrozen_layer_names = build_five_class_model()
    if model.output_shape[-1] != 5:
        raise RuntimeError(f"Five-class model has invalid output shape {model.output_shape}")
    print("Output shape:", model.output_shape)
    print("Unfrozen eligible backbone layers:", len(unfrozen_layer_names))
    print("BatchNormalization layers remain frozen.")

    ARTIFACTS.mkdir(parents=True, exist_ok=False)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(OUTPUT), monitor="val_loss", save_best_only=True, verbose=1
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1
    )
    started = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stopping],
        verbose=2,
        shuffle=True,
    )
    training_seconds = time.time() - started
    if not OUTPUT.is_file():
        raise RuntimeError("ModelCheckpoint did not save a selected validation checkpoint")

    verify_protected_hashes(protected_before)
    selected_model = keras.models.load_model(OUTPUT)
    if selected_model.output_shape[-1] != len(CLASSES):
        raise RuntimeError(f"Saved model output shape is {selected_model.output_shape}, expected five classes")

    # Test split is loaded only after validation-based checkpoint selection.
    x_test, y_test = load_split(rows, "test")
    test_loss, _ = selected_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    probabilities = selected_model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    if probabilities.shape != (len(y_test), len(CLASSES)):
        raise RuntimeError(f"Inference returned unexpected shape {probabilities.shape}")
    predictions = np.argmax(probabilities, axis=1)
    matrix, report = evaluation_report(y_test, predictions)
    metrics = {
        "experiment": "rice_mobilenet_5class_finetune_last_30_eligible",
        "class_to_index": CLASS_TO_INDEX,
        "classes": CLASSES,
        "test_samples": int(len(y_test)),
        "test_loss": float(test_loss),
        "test_accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1_score"],
        "training_seconds": training_seconds,
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "class_weights": {CLASSES[index]: float(weight) for index, weight in class_weights.items()},
        "unfrozen_backbone_layer_names": unfrozen_layer_names,
        "protected_sha256": protected_before,
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ARTIFACTS / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ARTIFACTS / "training_history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    with (ARTIFACTS / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\predicted", *CLASSES])
        for label, values in zip(CLASSES, matrix.tolist()):
            writer.writerow([label, *values])

    verify_protected_hashes(protected_before)
    print("Five-class experiment complete.")
    print(json.dumps(metrics, indent=2))
    print("Model:", OUTPUT)
    print("Evaluation artifacts:", ARTIFACTS)


if __name__ == "__main__":
    main()
