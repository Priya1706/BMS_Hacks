"""Run validation-selected rice experiments, then evaluate only the winner on test.

Usage:
    .venv/Scripts/python.exe -m experiments.rice_accuracy.run_controlled_experiments

All output is isolated under app/models/rice_accuracy_experiments/. Existing
artifacts are never overwritten.
"""

from __future__ import annotations

import csv
import json
import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from experiments.rice_accuracy import common as exp


def existing_protected_hashes() -> dict[str, str]:
    protected: dict[str, Path] = {
        "fixed_splits": exp.SPLITS,
        "group_manifest": exp.GROUPS,
        "imagenet_weights": exp.WEIGHTS,
        "six_class_rice_candidate": exp.ROOT / "app/models/rice_mobilenet_candidate.keras",
        "six_class_finetuned": exp.ROOT / "app/models/rice_mobilenet_finetuned.keras",
        "five_class_baseline": exp.BASELINE,
    }
    for artifact_dir in (
        exp.ROOT / "app/models/rice_mobilenet_5class_evaluation",
        exp.ROOT / "app/models/rice_mobilenet_finetuned_evaluation",
    ):
        if artifact_dir.is_dir():
            for path in artifact_dir.rglob("*"):
                if path.is_file():
                    protected[str(path.relative_to(exp.ROOT))] = path
    missing = [str(p) for p in protected.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Protected project inputs missing: " + ", ".join(missing))
    return {name: exp.sha256(path) for name, path in protected.items()}


def verify_hashes(expected: dict[str, str]) -> None:
    paths = {
        "fixed_splits": exp.SPLITS,
        "group_manifest": exp.GROUPS,
        "imagenet_weights": exp.WEIGHTS,
        "six_class_rice_candidate": exp.ROOT / "app/models/rice_mobilenet_candidate.keras",
        "six_class_finetuned": exp.ROOT / "app/models/rice_mobilenet_finetuned.keras",
        "five_class_baseline": exp.BASELINE,
    }
    for name in expected:
        if (name.startswith("app/models/rice_mobilenet_5class_evaluation/")
                or name.startswith("app/models/rice_mobilenet_finetuned_evaluation/")):
            paths[name] = exp.ROOT / name
    changed = [name for name, path in paths.items()
               if name in expected and (not path.is_file() or exp.sha256(path) != expected[name])]
    if changed:
        raise RuntimeError("Protected source hashes changed: " + ", ".join(changed))


def save_candidate_split(rows: list[dict[str, str]], out_dir: Path):
    candidate = exp.make_candidate_split(rows)
    source_test = {(r["class"], r["filename"]) for r in rows
                   if r["class"] in exp.CLASS_TO_INDEX and r["split"] == "test"}
    candidate_test = {(r["class"], r["filename"]) for r in candidate if r["split"] == "fixed_test"}
    if source_test != candidate_test:
        raise RuntimeError("Candidate manifest changed the fixed test assignment")
    train_names = {(r["class"], r["filename"]) for r in candidate if r["split"] == "train"}
    val_names = {(r["class"], r["filename"]) for r in candidate if r["split"] == "validation"}
    if train_names & val_names or (train_names | val_names) & source_test:
        raise RuntimeError("Candidate split overlaps train/validation/test rows")
    path = out_dir / "candidate_split.csv"
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class", "filename", "group_id", "split", "source_split"])
        writer.writeheader()
        writer.writerows(candidate)
    overlap = exp.group_overlap(candidate)
    counts = {split: dict(Counter(r["class"] for r in candidate if r["split"] == split))
              for split in ("train", "validation", "fixed_test")}
    audit = {
        "seed": exp.SEED,
        "validation_fraction_requested": exp.VAL_FRACTION,
        "candidate_split_method": "class-stratified randomized split of original train+validation images; original test copied unchanged",
        "candidate_counts_by_class": counts,
        "source_manifest_sha256": exp.sha256(exp.SPLITS),
        "group_manifest_sha256": exp.sha256(exp.GROUPS),
        "fixed_test_rows_unchanged": True,
        "source_test_count": len(source_test),
        "timestamp_group_cross_train_validation_count": len(overlap),
        "timestamp_groups_are_not_trusted_capture_sessions": True,
        "group_metadata_limitations": (
            "The existing group script chains all timestamped files globally when consecutive names are within 10 seconds. "
            "The source is Kaggle data and no capture-session metadata verifies that this indicates related images; "
            "mixed-class timestamp groups demonstrate that this is only a heuristic. Candidate split is stratified by image, "
            "so remaining correlated/near-duplicate imagery cannot be ruled out."
        ),
    }
    exp.write_json(out_dir / "split_audit.json", audit)
    return candidate, audit


def data_quality_audit(rows: list[dict[str, str]], out_dir: Path) -> dict[str, Any]:
    """Check decodability and exact/low-resolution perceptual duplicate candidates."""
    import hashlib
    from PIL import Image, ImageOps

    selected = [r for r in rows if r["class"] in exp.CLASS_TO_INDEX]
    hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    perceptual: list[tuple[int, dict[str, str]]] = []
    missing_or_bad: list[str] = []
    for row in selected:
        path = exp.DATA_ROOT / row["class"] / row["filename"]
        if not path.is_file():
            missing_or_bad.append(f"missing:{path}")
            continue
        raw = path.read_bytes()
        hashes[hashlib.sha256(raw).hexdigest()].append(row)
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source)
                gray = np.asarray(ImageOps.grayscale(image).resize((8, 8)), dtype=np.int16)
            bits = gray > gray.mean()
            value = sum(int(bit) << index for index, bit in enumerate(bits.flatten()))
            perceptual.append((value, row))
        except Exception as exc:
            missing_or_bad.append(f"decode:{path}:{exc}")
    exact_groups = [items for items in hashes.values() if len(items) > 1]
    exact_cross_split = [items for items in exact_groups if len({r["split"] for r in items}) > 1]
    # This intentionally broad 64-bit average-hash screen is a lead list only;
    # similar scenes/leaf shapes can collide, so it is not treated as leakage.
    near_pairs = []
    for idx, (left_hash, left) in enumerate(perceptual):
        for right_hash, right in perceptual[idx + 1:]:
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= 4 and left["split"] != right["split"]:
                near_pairs.append({
                    "hamming_distance": distance,
                    "left": f"{left['class']}/{left['filename']} [{left['split']}]",
                    "right": f"{right['class']}/{right['filename']} [{right['split']}]",
                })
    report = {
        "selected_source_image_count": len(selected),
        "missing_or_decode_errors": missing_or_bad,
        "exact_duplicate_group_count": len(exact_groups),
        "exact_cross_split_duplicate_group_count": len(exact_cross_split),
        "cross_split_average_hash_pairs_hamming_le_4": len(near_pairs),
        "average_hash_is_coarse_screen_not_duplicate_confirmation": True,
        "near_pair_examples": near_pairs[:40],
    }
    exp.write_json(out_dir / "image_duplicate_audit.json", report)
    if missing_or_bad:
        raise ValueError("Image audit found missing or undecodable selected images; see audit")
    if exact_cross_split:
        raise ValueError("Exact byte duplicates cross original split boundaries; refusing experiments")
    return report


def softmax_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-8, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def calibration_scores(probabilities: np.ndarray, y: np.ndarray, bins: int = 10) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predicted == y).astype(float)
    one_hot = np.eye(len(exp.CLASSES))[y]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    ece = 0.0
    for low in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        high = low + 1.0 / bins
        mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {"expected_calibration_error_10_bins": ece, "multiclass_brier_score": brier}


def calibration_report(probabilities: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    candidates = np.arange(0.50, 3.001, 0.05)
    nlls = []
    for temperature in candidates:
        p = softmax_temperature(probabilities, float(temperature))
        nlls.append(float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1)).mean()))
    temperature = float(candidates[int(np.argmin(nlls))])
    before = calibration_scores(probabilities, y)
    scaled = softmax_temperature(probabilities, temperature)
    after = calibration_scores(scaled, y)
    confidence = scaled.max(axis=1)
    predicted = scaled.argmax(axis=1)
    abstain_threshold = None
    for threshold in sorted(set(float(v) for v in confidence)):
        mask = confidence >= threshold
        if mask.mean() >= 0.20 and float((predicted[mask] == y[mask]).mean()) >= 0.90:
            abstain_threshold = threshold
            break
    return {
        "temperature_selected_on_validation_only": temperature,
        "validation_scores_uncalibrated": before,
        "validation_scores_temperature_scaled": after,
        "validation_confidence_threshold_for_90pct_selective_accuracy_min_20pct_coverage": abstain_threshold,
        "threshold_policy_note": "Threshold is a validation-derived abstention candidate, not a medical treatment rule.",
    }


def bootstrap_delta(y: np.ndarray, pred_base: np.ndarray, pred_candidate: np.ndarray,
                    replicates: int = 3000) -> dict[str, Any]:
    rng = np.random.default_rng(exp.SEED)
    n = len(y)
    accuracy_deltas = np.empty(replicates, dtype=np.float64)
    macro_deltas = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        ix = rng.integers(0, n, n)
        _, base_report, _ = exp.classification_metrics(y[ix], pred_base[ix])
        _, candidate_report, _ = exp.classification_metrics(y[ix], pred_candidate[ix])
        accuracy_deltas[i] = candidate_report["accuracy"] - base_report["accuracy"]
        macro_deltas[i] = candidate_report["macro avg"]["f1-score"] - base_report["macro avg"]["f1-score"]
    return {
        "method": "paired bootstrap over the single fixed test sample; descriptive only",
        "replicates": replicates,
        "accuracy_difference_95pct_interval": [float(v) for v in np.quantile(accuracy_deltas, [0.025, 0.975])],
        "macro_f1_difference_95pct_interval": [float(v) for v in np.quantile(macro_deltas, [0.025, 0.975])],
    }


def evaluation_artifacts(model_path: Path, x: np.ndarray, y: np.ndarray, out_dir: Path,
                         split_label: str):
    model = tf.keras.models.load_model(str(model_path))
    if model.output_shape[-1] != len(exp.CLASSES):
        raise RuntimeError(f"Unexpected output dimensions in {model_path}: {model.output_shape}")
    # One unmeasured warm-up batch, then one pass for approximate CPU latency.
    model.predict(x[:min(len(x), exp.BATCH_SIZE)], batch_size=exp.BATCH_SIZE, verbose=0)
    started = time.perf_counter()
    probabilities = model.predict(x, batch_size=exp.BATCH_SIZE, verbose=0)
    latency_ms = (time.perf_counter() - started) * 1000.0 / len(x)
    if probabilities.shape != (len(y), len(exp.CLASSES)):
        raise RuntimeError(f"Unexpected prediction shape {probabilities.shape}")
    predictions = probabilities.argmax(axis=1)
    _, report, matrix = exp.classification_metrics(y, predictions)
    result = {
        "split": split_label,
        "samples": len(y),
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "predicted_class_counts": {label: int((predictions == i).sum())
                                   for i, label in enumerate(exp.CLASSES)},
        "class_collapsed": len(set(predictions.tolist())) <= 1,
        "model_bytes": model_path.stat().st_size,
        "mean_inference_ms_per_image_cpu": latency_ms,
        "class_to_index": exp.CLASS_TO_INDEX,
    }
    exp.write_json(out_dir / "metrics.json", result)
    exp.write_json(out_dir / "classification_report.json", report)
    exp.write_csv_matrix(out_dir / "confusion_matrix.csv", matrix)
    exp.write_confusion_svg(out_dir / "confusion_matrix.svg", matrix)
    exp.write_json(out_dir / "predictions.json", {
        "y_true": y.tolist(), "y_pred": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    })
    del model
    tf.keras.backend.clear_session()
    return result, probabilities


def write_comparison(path: Path, baseline: dict[str, Any], experiments: list[dict[str, Any]],
                     winner: dict[str, Any], candidate: dict[str, Any], uncertainty: dict[str, Any],
                     split_audit: dict[str, Any], duplicate_audit: dict[str, Any]) -> None:
    lines = [
        "# Rice five-class controlled experiment comparison", "",
        "## Evaluation protocol", "",
        "The existing fixed test assignments were excluded from candidate training, validation, and model selection. "
        "The winner was selected by validation macro F1; only then was the fixed test set used for the final comparison. "
        "Non-winning variants have validation results only.", "",
        f"- Fixed test samples: {baseline['samples']} (identical examples for baseline and selected candidate).",
        f"- Candidate validation split: seed {exp.SEED}, stratified 80/20 from source train+validation only.",
        f"- Original split manifest SHA-256: {exp.sha256(exp.SPLITS)}.",
        f"- Bacterial Leaf Blight test support: {baseline['classification_report']['Bacterial Leaf Blight']['support']} images.",
        f"- Timestamp-group overlap across candidate train/validation: {split_audit['timestamp_group_cross_train_validation_count']}; "
        "unverified timestamps are not reliable capture-session identifiers.",
        f"- Exact duplicate groups: {duplicate_audit['exact_duplicate_group_count']}; "
        f"coarse average-hash near-pair candidates across source splits: {duplicate_audit['cross_split_average_hash_pairs_hamming_le_4']} "
        "(not confirmed duplicates).", "",
        "## Results", "",
        "| Model/experiment | Validation macro F1 | Test accuracy | Test macro F1 | Weighted F1 | Model MB | CPU ms/image |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Existing five-class baseline | n/a (historical fixed split) | {baseline['accuracy']:.4f} | {baseline['macro_f1']:.4f} | {baseline['weighted_f1']:.4f} | {baseline['model_bytes']/1e6:.2f} | {baseline['mean_inference_ms_per_image_cpu']:.2f} |",
    ]
    for item in experiments:
        selected = item["experiment"] == winner["experiment"]
        test_acc = f"{candidate['accuracy']:.4f}" if selected else "not evaluated"
        test_macro = f"{candidate['macro_f1']:.4f}" if selected else "not evaluated"
        weighted = f"{candidate['weighted_f1']:.4f}" if selected else "—"
        latency = f"{candidate['mean_inference_ms_per_image_cpu']:.2f}" if selected else "—"
        lines.append(f"| {item['experiment']} ({'selected' if selected else 'validation only'}) | "
                     f"{item['validation_macro_f1']:.4f} | {test_acc} | {test_macro} | {weighted} | "
                     f"{item['model_bytes']/1e6:.2f} | {latency} |")
    delta_acc = (candidate["accuracy"] - baseline["accuracy"]) * 100
    delta_f1 = candidate["macro_f1"] - baseline["macro_f1"]
    lines += [
        "", "## Selected model and trade-offs", "",
        f"Selected by highest validation macro F1: {winner['experiment']} ({winner['validation_macro_f1']:.4f}).",
        f"Candidate model: {winner['model_path']}.",
        f"Test accuracy delta vs baseline: {delta_acc:+.2f} percentage points; macro-F1 delta: {delta_f1:+.4f}.",
        f"Paired bootstrap intervals: {json.dumps(uncertainty)}.",
        "Only four Bacterial Leaf Blight test images limit disease-specific reliability and make modest overall changes statistically uncertain. "
        "A held-out score alone does not establish production readiness.", "",
        "## Class-level test metrics", "",
        "See baseline/classification_report.json and selected_candidate/classification_report.json; "
        "confusion matrices are available as CSV and SVG.", "",
        "## Calibration and uncertainty", "",
        f"Validation-only temperature scaling and abstention analysis: {json.dumps(candidate.get('calibration', {}))}.",
        "The abstention threshold is a validation-derived research result, not a treatment recommendation. "
        "Low-confidence images should be routed for human review.", "",
        "## Reproducibility and limitations", "",
        "- CPU-only native Windows TensorFlow; batch size 16; 224x224 RGB [0,1], model-embedded augmentation active only during training, "
        "then MobileNetV2 rescaling to [-1,1].",
        "- Local ImageNet MobileNetV2 weights; fresh head warm-up for up to 2 epochs, then up to 6 fine-tuning epochs; "
        "class weighting, early stopping, validation-loss checkpointing, and ReduceLROnPlateau.",
        "- Source group timestamps do not verify capture sessions. Exact duplicates were absent; coarse near-image candidates need human review "
        "and correlated imagery cannot be ruled out.",
        "- Selected model remains an experimental artifact and was not integrated into the predictor/API.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if exp.OUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment artifacts: {exp.OUT_ROOT}")
    for path in (exp.SPLITS, exp.GROUPS, exp.WEIGHTS, exp.BASELINE):
        if not path.is_file():
            raise FileNotFoundError(path)
    before = existing_protected_hashes()
    tf.config.set_visible_devices([], "GPU")
    tf.keras.utils.set_random_seed(exp.SEED)
    exp.OUT_ROOT.mkdir(parents=True, exist_ok=False)
    rows = exp.read_source_rows()
    candidate_rows, split_audit = save_candidate_split(rows, exp.OUT_ROOT)
    duplicate_audit = data_quality_audit(rows, exp.OUT_ROOT)
    x_train, y_train = exp.load_images(candidate_rows, "train")
    x_val, y_val = exp.load_images(candidate_rows, "validation")
    exp.write_json(exp.OUT_ROOT / "dataset_card.json", {
        "selected_classes": exp.CLASSES,
        "class_to_index": exp.CLASS_TO_INDEX,
        "image_size": exp.SIZE,
        "source_manifest_sha256": exp.sha256(exp.SPLITS),
        "source_group_manifest_sha256": exp.sha256(exp.GROUPS),
        "source_rows_by_class_and_split": {
            split: dict(Counter(r["class"] for r in rows
                                if r["split"] == split and r["class"] in exp.CLASS_TO_INDEX))
            for split in ("train", "val", "test")
        },
        "candidate_split_counts": split_audit["candidate_counts_by_class"],
        "image_quality_and_duplicate_audit": duplicate_audit,
        "gpu_disabled": True,
        "cpu_count": __import__("os").cpu_count(),
    })

    # The fixed test rows are not loaded during candidate training/selection.
    experiment_metrics = []
    for name in exp.VARIANTS:
        print(f"\n=== Experiment: {name} ===", flush=True)
        metrics = exp.train_variant(name, x_train, y_train, x_val, y_val, exp.OUT_ROOT / name)
        metrics["model_path"] = str(exp.OUT_ROOT / name / f"rice_5class_{name}.keras")
        experiment_metrics.append(metrics)
        verify_hashes(before)

    winner = max(experiment_metrics,
                 key=lambda item: (item["validation_macro_f1"], item["validation_accuracy"]))
    selected_model_path = Path(winner["model_path"])
    selection = {
        "selection_metric": "validation macro F1; validation accuracy is tie-breaker",
        "selected_experiment": winner["experiment"],
        "selected_model_path": str(selected_model_path),
        "all_variants_tested_on_validation_only": True,
        "selected_validation_macro_f1": winner["validation_macro_f1"],
        "selection_locked_before_test_evaluation": True,
    }
    exp.write_json(exp.OUT_ROOT / "selection.json", selection)

    # Only after the validation decision is frozen, open the fixed test set.
    x_test, y_test = exp.load_images(candidate_rows, "fixed_test")
    baseline_dir = exp.OUT_ROOT / "baseline"
    baseline_dir.mkdir()
    baseline_metrics, baseline_probs = evaluation_artifacts(
        exp.BASELINE, x_test, y_test, baseline_dir, "existing_fixed_test")
    baseline_metrics["experiment"] = "existing_five_class_baseline"
    history_path = exp.ROOT / "app/models/rice_mobilenet_5class_evaluation/training_history.json"
    if history_path.is_file():
        baseline_history = json.loads(history_path.read_text(encoding="utf-8"))
        exp.write_history_svg(baseline_dir / "training_curves.svg", baseline_history)
        exp.write_loss_svg(baseline_dir / "loss_curves.svg", baseline_history)

    final_dir = exp.OUT_ROOT / "selected_candidate"
    final_dir.mkdir()
    candidate_metrics, candidate_probs = evaluation_artifacts(
        selected_model_path, x_test, y_test, final_dir, "existing_fixed_test")
    candidate_metrics["experiment"] = winner["experiment"]

    # Temperature and abstention threshold are selected using validation only.
    selected_model = tf.keras.models.load_model(str(selected_model_path))
    val_probs = selected_model.predict(x_val, batch_size=exp.BATCH_SIZE, verbose=0)
    candidate_metrics["calibration"] = calibration_report(val_probs, y_val)
    temperature = candidate_metrics["calibration"]["temperature_selected_on_validation_only"]
    calibrated_test_probs = softmax_temperature(candidate_probs, temperature)
    candidate_metrics["test_calibration_scores_uncalibrated"] = calibration_scores(candidate_probs, y_test)
    candidate_metrics["test_calibration_scores_temperature_scaled"] = calibration_scores(calibrated_test_probs, y_test)
    threshold = candidate_metrics["calibration"]["validation_confidence_threshold_for_90pct_selective_accuracy_min_20pct_coverage"]
    if threshold is not None:
        confidence = calibrated_test_probs.max(axis=1)
        accepted = confidence >= threshold
        candidate_metrics["test_selective_result_at_validation_threshold"] = {
            "threshold": threshold,
            "coverage": float(accepted.mean()),
            "accuracy_on_accepted_images": float((calibrated_test_probs.argmax(axis=1)[accepted] == y_test[accepted]).mean()) if accepted.any() else None,
            "accepted_images": int(accepted.sum()),
            "note": "Threshold selected using validation only.",
        }
    exp.write_json(final_dir / "metrics.json", candidate_metrics)
    del selected_model
    tf.keras.backend.clear_session()

    base_predictions = baseline_probs.argmax(axis=1)
    selected_predictions = candidate_probs.argmax(axis=1)
    uncertainty = bootstrap_delta(y_test, base_predictions, selected_predictions)
    baseline_metrics["classification_report"] = json.loads(
        (baseline_dir / "classification_report.json").read_text(encoding="utf-8"))
    candidate_metrics["classification_report"] = json.loads(
        (final_dir / "classification_report.json").read_text(encoding="utf-8"))
    exp.write_json(exp.OUT_ROOT / "test_comparison.json", {
        "selection": selection,
        "baseline": baseline_metrics,
        "selected_candidate": candidate_metrics,
        "paired_bootstrap_difference": uncertainty,
        "baseline_model_sha256": before["five_class_baseline"],
        "split_manifest_sha256": before["fixed_splits"],
        "protected_source_hashes_unchanged": True,
    })
    write_comparison(exp.OUT_ROOT / "comparison_report.md", baseline_metrics,
                     experiment_metrics, winner, candidate_metrics, uncertainty,
                     split_audit, duplicate_audit)
    verify_hashes(before)
    print("\nExperiments complete. Test was evaluated only after validation selection.")
    print(json.dumps({
        "selection": selection,
        "baseline_test": baseline_metrics,
        "selected_candidate_test": candidate_metrics,
        "paired_bootstrap_difference": uncertainty,
        "outputs": str(exp.OUT_ROOT),
    }, indent=2))


def train_variant_standalone(name: str) -> None:
    """Run one configuration for reproducibility without reading/evaluating test images."""
    if name not in exp.VARIANTS:
        raise ValueError(f"Unknown variant {name}")
    output_root = exp.OUT_ROOT.parent / f"rice_accuracy_{name}_standalone"
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    protected = existing_protected_hashes()
    output_root.mkdir(parents=True, exist_ok=False)
    rows = exp.read_source_rows()
    candidate, split_audit = save_candidate_split(rows, output_root)
    duplicate_audit = data_quality_audit(rows, output_root)
    x_train, y_train = exp.load_images(candidate, "train")
    x_val, y_val = exp.load_images(candidate, "validation")
    metrics = exp.train_variant(name, x_train, y_train, x_val, y_val, output_root / name)
    exp.write_json(output_root / "run_manifest.json", {
        "variant": name,
        "test_images_opened": False,
        "source_hashes_before": protected,
        "source_split_audit": split_audit,
        "duplicate_audit": duplicate_audit,
    })
    verify_hashes(protected)
    print(json.dumps({"variant": name, "validation_metrics": metrics,
                      "output_root": str(output_root), "test_evaluated": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(exp.VARIANTS),
                        help="Train only this configuration; no test images are opened.")
    args = parser.parse_args()
    if args.variant:
        train_variant_standalone(args.variant)
    else:
        main()
