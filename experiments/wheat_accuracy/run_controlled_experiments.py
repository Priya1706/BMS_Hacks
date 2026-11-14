"""Run three wheat transfer-learning experiments and test only the val-selected winner.

From the project root:
  .venv/Scripts/python.exe -m experiments.wheat_accuracy.run_controlled_experiments

Writes only to app/models/wheat_accuracy_experiments/ and refuses to overwrite.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from app.services.predictor import get_class_labels_for_crop, get_model_path
from experiments.wheat_accuracy import common as exp


def protected_hashes() -> dict[str, str]:
    paths = {
        "production_wheat_model": exp.BASELINE,
        "production_wheat_labels": exp.BASELINE_CLASSES,
        "mobilenetv2_imagenet_weights": exp.WEIGHTS,
        "existing_training_script": exp.ROOT / "train_wheat_mobilenet.py",
    }
    for path in exp.BASELINE_EVALUATION.rglob("*"):
        if path.is_file():
            paths[str(path.relative_to(exp.ROOT)).replace("\\", "/")] = path
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Protected baseline files missing: " + ", ".join(missing))
    return {name: exp.sha256_file(path) for name, path in paths.items()}


def verify_protected_hashes(expected: dict[str, str]) -> None:
    paths = {
        "production_wheat_model": exp.BASELINE,
        "production_wheat_labels": exp.BASELINE_CLASSES,
        "mobilenetv2_imagenet_weights": exp.WEIGHTS,
        "existing_training_script": exp.ROOT / "train_wheat_mobilenet.py",
    }
    for name in expected:
        if name.startswith("app/models/wheat_mobilenet_candidate_evaluation/"):
            paths[name] = exp.ROOT / name
    changed = [
        name for name, path in paths.items()
        if name in expected and (not path.is_file() or exp.sha256_file(path) != expected[name])
    ]
    if changed:
        raise RuntimeError("Protected production wheat artifacts changed: " + ", ".join(changed))


def metric_artifacts(model_path: Path, x: np.ndarray, y: np.ndarray, out_dir: Path,
                     split_name: str) -> tuple[dict[str, Any], np.ndarray]:
    model = tf.keras.models.load_model(str(model_path))
    if tuple(model.input_shape[1:]) != (*exp.IMAGE_SIZE, 3) or model.output_shape[-1] != len(exp.CLASSES):
        raise RuntimeError(f"Invalid model dimensions for {model_path}: {model.input_shape}->{model.output_shape}")
    model.predict(x[:min(exp.BATCH_SIZE, len(x))], batch_size=exp.BATCH_SIZE, verbose=0)
    started = time.perf_counter()
    probabilities = model.predict(x, batch_size=exp.BATCH_SIZE, verbose=0)
    elapsed_ms = (time.perf_counter() - started) * 1000 / len(x)
    if probabilities.shape != (len(y), len(exp.CLASSES)):
        raise RuntimeError(f"Expected 4 probabilities per image, got {probabilities.shape}")
    y_pred = probabilities.argmax(axis=1)
    _, matrix, report = exp.metrics_for(y, y_pred)
    result = {
        "split": split_name,
        "samples": int(len(y)),
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "predicted_class_counts": {
            label: int((y_pred == index).sum()) for index, label in enumerate(exp.CLASSES)
        },
        "class_collapsed": len(set(y_pred.tolist())) <= 1,
        "model_bytes": model_path.stat().st_size,
        "mean_cpu_inference_ms_per_image": elapsed_ms,
        "class_to_index": exp.CLASS_TO_INDEX,
        "model_path": str(model_path),
    }
    exp.write_json(out_dir / "metrics.json", result)
    exp.write_json(out_dir / "classification_report.json", report)
    exp.write_csv_matrix(out_dir / "confusion_matrix.csv", matrix)
    exp.write_confusion_svg(out_dir / "confusion_matrix.svg", matrix)
    exp.write_json(out_dir / "predictions.json", {
        "y_true": y.tolist(), "y_pred": y_pred.tolist(),
        "probabilities": probabilities.tolist(),
    })
    del model
    tf.keras.backend.clear_session()
    return result, probabilities


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-8, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def calibration_scores(probabilities: np.ndarray, y_true: np.ndarray, bins: int = 10):
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predicted == y_true).astype(float)
    brier = float(np.mean(np.sum((probabilities - np.eye(len(exp.CLASSES))[y_true]) ** 2, axis=1)))
    ece = 0.0
    for low in np.linspace(0, 1, bins + 1)[:-1]:
        high = low + 1 / bins
        mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    nll = float(-np.log(np.clip(probabilities[np.arange(len(y_true)), y_true], 1e-8, 1)).mean())
    return {"ece_10_bins": ece, "multiclass_brier": brier, "nll": nll}


def validation_calibration(probabilities: np.ndarray, y_true: np.ndarray):
    temperatures = np.arange(0.50, 3.001, 0.05)
    scores = [
        calibration_scores(temperature_scale(probabilities, float(t)), y_true)["nll"]
        for t in temperatures
    ]
    temperature = float(temperatures[int(np.argmin(scores))])
    calibrated = temperature_scale(probabilities, temperature)
    confidence = calibrated.max(axis=1)
    predictions = calibrated.argmax(axis=1)
    abstain = None
    for threshold in sorted(set(float(value) for value in confidence)):
        accepted = confidence >= threshold
        if accepted.mean() >= 0.20 and (predictions[accepted] == y_true[accepted]).mean() >= 0.90:
            abstain = threshold
            break
    return {
        "temperature_selected_on_validation_only": temperature,
        "validation_uncalibrated": calibration_scores(probabilities, y_true),
        "validation_temperature_scaled": calibration_scores(calibrated, y_true),
        "abstention_threshold_for_90pct_accuracy_min_20pct_coverage": abstain,
        "note": "Confidence is not proof of disease; abstain and seek human review when uncertain.",
    }


def paired_cluster_bootstrap(y_true: np.ndarray, base_pred: np.ndarray,
                             candidate_pred: np.ndarray, hashes: list[str],
                             replicates: int = 3000):
    by_hash: dict[str, list[int]] = defaultdict(list)
    for index, digest in enumerate(hashes):
        by_hash[digest].append(index)
    clusters = list(by_hash.values())
    rng = np.random.default_rng(exp.SEED)
    accuracy_delta = np.empty(replicates, dtype=np.float64)
    macro_delta = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        chosen = rng.integers(0, len(clusters), len(clusters))
        indices = np.asarray([index for cluster_index in chosen
                              for index in clusters[int(cluster_index)]], dtype=np.int32)
        _, _, base_report = exp.metrics_for(y_true[indices], base_pred[indices])
        _, _, candidate_report = exp.metrics_for(y_true[indices], candidate_pred[indices])
        accuracy_delta[i] = candidate_report["accuracy"] - base_report["accuracy"]
        macro_delta[i] = candidate_report["macro avg"]["f1-score"] - base_report["macro avg"]["f1-score"]
    return {
        "method": "paired bootstrap over exact-content test clusters; descriptive only",
        "replicates": replicates,
        "test_rows": int(len(y_true)),
        "independent_exact_content_clusters": len(clusters),
        "accuracy_difference_95pct_interval": [float(x) for x in np.quantile(accuracy_delta, [0.025, 0.975])],
        "macro_f1_difference_95pct_interval": [float(x) for x in np.quantile(macro_delta, [0.025, 0.975])],
    }


def write_comparison(path: Path, baseline: dict[str, Any], variants: list[dict[str, Any]],
                     winner: dict[str, Any], candidate: dict[str, Any],
                     bootstrap: dict[str, Any], audit: dict[str, Any]):
    lines = [
        "# RootCause wheat model experiment comparison", "",
        "## Evaluation protocol", "",
        "The fixed 200-image test folders were preserved. Candidate training and selection used only source training "
        "images; model selection used validation macro F1. The selected candidate alone was evaluated on the fixed test. "
        "The existing production baseline was evaluated for comparison, not used to choose the candidate.", "",
        f"- Class order: {', '.join(exp.CLASSES)}.",
        f"- Candidate validation split seed: {exp.SEED}; per-class 80/20 stratification of exact-content training groups.",
        f"- Source/fixed-test files: {audit['source_image_count']} total; no unreadable files.",
        f"- Exact train/test hash overlaps excluded from candidate training: {audit['exact_train_test_group_count']} groups.",
        f"- Cross-class exact-content groups excluded from candidate train/validation: {audit['cross_class_conflict_groups_exclusively_in_training']} training-only groups; labels were not changed.",
        f"- Candidate fixed-test support: 50 per class, including an exact duplicate group noted in the dataset audit.",
        "- Near-duplicate counts are coarse average-hash screening leads, not confirmed duplicates.", "",
        "## Baseline and selected candidate", "",
        "| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Size MB | CPU ms/image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Production baseline | {baseline['accuracy']:.4f} | {baseline['macro_precision']:.4f} | {baseline['macro_recall']:.4f} | {baseline['macro_f1']:.4f} | {baseline['weighted_f1']:.4f} | {baseline['model_bytes']/1e6:.2f} | {baseline['mean_cpu_inference_ms_per_image']:.2f} |",
        f"| Validation-selected: {winner['experiment']} | {candidate['accuracy']:.4f} | {candidate['macro_precision']:.4f} | {candidate['macro_recall']:.4f} | {candidate['macro_f1']:.4f} | {candidate['weighted_f1']:.4f} | {candidate['model_bytes']/1e6:.2f} | {candidate['mean_cpu_inference_ms_per_image']:.2f} |",
        "",
        f"Accuracy delta: {(candidate['accuracy']-baseline['accuracy'])*100:+.2f} percentage points. "
        f"Macro F1 delta: {candidate['macro_f1']-baseline['macro_f1']:+.4f}.",
        f"Paired bootstrap: {json.dumps(bootstrap)}. The interval is descriptive; it is not a significance claim.", "",
        "## Validation experiment comparison", "",
        "| Experiment | Validation accuracy | Validation macro F1 | Fine-tuned eligible layers | Label smoothing |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in variants:
        lines.append(
            f"| {item['experiment']}{' (selected)' if item['experiment']==winner['experiment'] else ''} "
            f"| {item['validation_accuracy']:.4f} | {item['validation_macro_f1']:.4f} "
            f"| {item['fine_tune_eligible_layers']} | {item['label_smoothing']:.2f} |"
        )
    lines += ["", "## Per-class test results", ""]
    report = candidate["classification_report"]
    lines += [
        "| Class | Baseline precision / recall / F1 | Candidate precision / recall / F1 | Support |",
        "|---|---:|---:|---:|",
    ]
    base_report = baseline["classification_report"]
    for label in exp.CLASSES:
        b, c = base_report[label], report[label]
        lines.append(
            f"| {label} | {b['precision']:.3f} / {b['recall']:.3f} / {b['f1-score']:.3f} "
            f"| {c['precision']:.3f} / {c['recall']:.3f} / {c['f1-score']:.3f} | {c['support']} |"
        )
    lines += [
        "", "## Data and deployment considerations", "",
        "Exact-content duplicate groups under both Black Rust and Brown Rust indicate unresolved labels. The experiment "
        "excluded ambiguous training groups rather than relabeling them. The fixed test still reflects its original folders, "
        "including one repeated Black Rust test image pair; its raw 200-image score is reported as requested.",
        "The fixed test has 50 rows per class, but independent capture-session metadata is absent. The supplied dataset "
        "contains heterogeneous image dimensions and watermarked Mildew samples. The checked-in wheat split is generated "
        "at training time rather than stored as a manifest; this experiment saves its own reproducible assignment.",
        f"Validation-only calibration: {json.dumps(candidate.get('calibration', {}))}.",
        "Confidence should support abstention and human review, not stand alone as disease confirmation or treatment advice.",
        f"Candidate checkpoint: {winner['model_path']}. It remains separate from production and was not integrated.",
        "The model remains a research candidate until label conflicts and likely near-duplicate/capture-session issues are resolved.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiments(only_variant: str | None = None, output_root: Path | None = None) -> None:
    out_root = output_root or exp.OUT_ROOT
    if out_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing wheat experiment output: {out_root}")
    labels = json.loads(exp.BASELINE_CLASSES.read_text(encoding="utf-8"))
    if labels != exp.CLASSES or get_class_labels_for_crop("wheat") != exp.CLASSES:
        raise RuntimeError(f"Production label mapping is not the expected four-class order: {labels}")
    if get_model_path("wheat") != exp.BASELINE:
        raise RuntimeError("The active wheat production model is not the expected baseline")
    # Strictly validate the four expected source folders and every image before use.
    train_files, test_files = exp.source.validate_dataset()
    before = protected_hashes()
    before_memory = exp.memory_status()
    if before_memory["available_gb"] < 1.5:
        raise MemoryError(f"Only {before_memory['available_gb']:.2f} GB RAM is available; refusing training")
    tf.config.set_visible_devices([], "GPU")
    tf.keras.utils.set_random_seed(exp.SEED)
    out_root.mkdir(parents=True, exist_ok=False)

    candidates, audit, source_hashes = exp.audit_and_make_split(train_files, test_files, out_root)
    exp.write_json(out_root / "hardware_and_reproducibility.json", {
        "platform": "Windows CPU-only TensorFlow",
        "logical_cpu_count": __import__("os").cpu_count(),
        "memory_before_training": before_memory,
        "memory_rechecked_per_variant": True,
        "batch_size": exp.BATCH_SIZE,
        "seed": exp.SEED,
        "protected_hashes": before,
        "dataset_sha256_by_relative_path": source_hashes,
    })
    exp.write_json(out_root / "class_mapping.json", {
        "folder_to_label": exp.FOLDER_TO_LABEL,
        "test_folder_to_label": exp.TEST_FOLDER_TO_LABEL,
        "class_to_index": exp.CLASS_TO_INDEX,
    })

    # Baseline evaluation establishes actual fixed-test reference metrics.
    test_rows = exp.rows_for_split(candidates, "fixed_test")
    x_test = np.stack([exp.load_image(path) for path, _ in test_rows])
    y_test = np.asarray([label for _, label in test_rows], dtype=np.int32)
    baseline_dir = out_root / "baseline"
    baseline_dir.mkdir()
    baseline, baseline_prob = metric_artifacts(
        exp.BASELINE, x_test, y_test, baseline_dir, "fixed_test_baseline",
    )
    baseline["experiment"] = "production_wheat_mobilenet_candidate"
    baseline["classification_report"] = json.loads(
        (baseline_dir / "classification_report.json").read_text(encoding="utf-8"))
    old_history = exp.BASELINE_EVALUATION / "training_history.json"
    if old_history.is_file():
        history = json.loads(old_history.read_text(encoding="utf-8"))
        exp.write_curves_svg(baseline_dir / "training_validation_curves.svg", history)
        exp.write_curves_svg(baseline_dir / "training_validation_loss.svg", history, loss=True)
    del x_test, y_test

    train_rows = exp.rows_for_split(candidates, "train")
    val_rows = exp.rows_for_split(candidates, "validation")
    names = [only_variant] if only_variant else list(exp.VARIANTS)
    variant_metrics = []
    for name in names:
        memory = exp.memory_status()
        if memory["available_gb"] < 1.5:
            raise MemoryError(f"Only {memory['available_gb']:.2f} GB RAM available before {name}")
        print(f"\n=== Wheat experiment: {name} ===", flush=True)
        metrics = exp.train_variant(
            name, train_rows, val_rows, out_root / name,
        )
        variant_metrics.append(metrics)
        verify_protected_hashes(before)

    winner = max(
        variant_metrics,
        key=lambda item: (item["validation_macro_f1"], item["validation_accuracy"]),
    )
    selection = {
        "selected_experiment": winner["experiment"],
        "selected_model_path": winner["model_path"],
        "selection_rule": "highest validation macro F1; validation accuracy as tie-breaker",
        "fixed_test_used_for_selection": False,
        "locked_before_selected_candidate_test_evaluation": True,
    }
    exp.write_json(out_root / "selection.json", selection)

    # Only the selected candidate is evaluated on test after validation selection.
    x_test = np.stack([exp.load_image(path) for path, _ in test_rows])
    y_test = np.asarray([label for _, label in test_rows], dtype=np.int32)
    final_dir = out_root / "selected_candidate"
    final_dir.mkdir()
    candidate, candidate_prob = metric_artifacts(
        Path(winner["model_path"]), x_test, y_test, final_dir, "fixed_test_selected_candidate",
    )
    candidate["experiment"] = winner["experiment"]
    candidate["classification_report"] = json.loads(
        (final_dir / "classification_report.json").read_text(encoding="utf-8"))

    # Calibration temperature and abstention are estimated on candidate validation only.
    model = tf.keras.models.load_model(winner["model_path"])
    val_dataset = exp.make_dataset(val_rows)
    val_prob = model.predict(val_dataset, verbose=0)
    y_val = np.asarray([label for _, label in val_rows], dtype=np.int32)
    candidate["calibration"] = validation_calibration(val_prob, y_val)
    temperature = candidate["calibration"]["temperature_selected_on_validation_only"]
    calibrated_test = temperature_scale(candidate_prob, temperature)
    candidate["test_calibration_uncalibrated"] = calibration_scores(candidate_prob, y_test)
    candidate["test_calibration_temperature_scaled"] = calibration_scores(calibrated_test, y_test)
    threshold = candidate["calibration"]["abstention_threshold_for_90pct_accuracy_min_20pct_coverage"]
    if threshold is not None:
        confidence = calibrated_test.max(axis=1)
        accepted = confidence >= threshold
        candidate["test_selective_result"] = {
            "threshold_selected_on_validation": threshold,
            "coverage": float(accepted.mean()),
            "accuracy_accepted": float(
                (calibrated_test.argmax(axis=1)[accepted] == y_test[accepted]).mean()
            ) if accepted.any() else None,
            "accepted_count": int(accepted.sum()),
        }
    exp.write_json(final_dir / "metrics.json", candidate)

    test_hashes = [row["sha256"] for row in candidates if row["split"] == "fixed_test"]
    bootstrap = paired_cluster_bootstrap(
        y_test, np.asarray(baseline_prob).argmax(axis=1),
        np.asarray(candidate_prob).argmax(axis=1), test_hashes,
    )
    exp.write_json(out_root / "test_comparison.json", {
        "baseline": baseline,
        "selected_candidate": candidate,
        "selection": selection,
        "paired_cluster_bootstrap": bootstrap,
        "test_rows_unchanged": len(test_rows) == 200,
    })
    write_comparison(
        out_root / "comparison_report.md", baseline, variant_metrics,
        winner, candidate, bootstrap, audit,
    )
    exp.write_json(out_root / "selection.json", selection)

    # Verify no source dataset, production model/mapping, or existing report changed.
    verify_protected_hashes(before)
    with (out_root / "candidate_split_manifest.csv").open(
            newline="", encoding="utf-8") as stream:
        current = {
            row["relative_path"]: exp.sha256_file(exp.ROOT / row["relative_path"])
            for row in csv.DictReader(stream)
        }
    changed_dataset = [path for path, digest in source_hashes.items()
                       if current.get(path) != digest]
    if changed_dataset:
        raise RuntimeError("Source wheat dataset changed during experiment: " + ", ".join(changed_dataset[:10]))
    if get_model_path("wheat") != exp.BASELINE:
        raise RuntimeError("Production wheat model path changed unexpectedly")
    print("\nWheat experiments complete. Production model was not changed.")
    print(json.dumps({
        "selection": selection,
        "baseline": baseline,
        "selected_candidate": candidate,
        "bootstrap": bootstrap,
        "output_directory": str(out_root),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(exp.VARIANTS),
                        help="Run one variant (then evaluate it); default runs all three.")
    args = parser.parse_args()
    standalone_root = (exp.OUT_ROOT / f"standalone_{args.variant}") if args.variant else exp.OUT_ROOT
    run_experiments(args.variant, standalone_root)


if __name__ == "__main__":
    main()
