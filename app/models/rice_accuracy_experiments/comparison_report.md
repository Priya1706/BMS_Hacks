# Rice five-class controlled experiment comparison

## Evaluation protocol

The existing fixed test assignments were excluded from candidate training, validation, and model selection. The winner was selected by validation macro F1; only then was the fixed test set used for the final comparison. Non-winning variants have validation results only.

- Fixed test samples: 120 (identical examples for baseline and selected candidate).
- Candidate validation split: seed 42, stratified 80/20 from source train+validation only.
- Original split manifest SHA-256: daf746d2f45afb72d82a3087474220312022fe9a149522846e8f5c66136c6abb.
- Bacterial Leaf Blight test support: 4 images.
- Timestamp-group overlap across candidate train/validation: 29; unverified timestamps are not reliable capture-session identifiers.
- Exact duplicate groups: 0; coarse average-hash near-pair candidates across source splits: 107 (not confirmed duplicates).

## Results

| Model/experiment | Validation macro F1 | Test accuracy | Test macro F1 | Weighted F1 | Model MB | CPU ms/image |
|---|---:|---:|---:|---:|---:|---:|
| Existing five-class baseline | n/a (historical fixed split) | 0.8000 | 0.7551 | 0.8007 | 23.51 | 35.65 |
| augmented_ft30 (validation only) | 0.7429 | not evaluated | not evaluated | — | 23.52 | — |
| shallow_ft10 (validation only) | 0.7105 | not evaluated | not evaluated | — | 17.98 | — |
| label_smoothing_ft30 (selected) | 0.7509 | 0.8250 | 0.8191 | 0.8235 | 23.52 | 32.61 |

## Selected model and trade-offs

Selected by highest validation macro F1: label_smoothing_ft30 (0.7509).
Candidate model: C:\Semister 3 notes\Hackathon\BMS_Hacks\BMS_Hacks\app\models\rice_accuracy_experiments\label_smoothing_ft30\rice_5class_label_smoothing_ft30.keras.
Test accuracy delta vs baseline: +2.50 percentage points; macro-F1 delta: +0.0640.
Paired bootstrap intervals: {"method": "paired bootstrap over the single fixed test sample; descriptive only", "replicates": 3000, "accuracy_difference_95pct_interval": [-0.05833333333333335, 0.09999999999999998], "macro_f1_difference_95pct_interval": [-0.08004781849772191, 0.23272138023105496]}.
Only four Bacterial Leaf Blight test images limit disease-specific reliability and make modest overall changes statistically uncertain. A held-out score alone does not establish production readiness.

## Class-level test metrics

See baseline/classification_report.json and selected_candidate/classification_report.json; confusion matrices are available as CSV and SVG.

## Calibration and uncertainty

Validation-only temperature scaling and abstention analysis: {"temperature_selected_on_validation_only": 0.8000000000000003, "validation_scores_uncalibrated": {"expected_calibration_error_10_bins": 0.09036335462832268, "multiclass_brier_score": 0.33115605485338406}, "validation_scores_temperature_scaled": {"expected_calibration_error_10_bins": 0.06334014612299795, "multiclass_brier_score": 0.32045920574356634}, "validation_confidence_threshold_for_90pct_selective_accuracy_min_20pct_coverage": 0.6519932150840759, "threshold_policy_note": "Threshold is a validation-derived abstention candidate, not a medical treatment rule."}.
The abstention threshold is a validation-derived research result, not a treatment recommendation. Low-confidence images should be routed for human review.

## Reproducibility and limitations

- CPU-only native Windows TensorFlow; batch size 16; 224x224 RGB [0,1], model-embedded augmentation active only during training, then MobileNetV2 rescaling to [-1,1].
- Local ImageNet MobileNetV2 weights; fresh head warm-up for up to 2 epochs, then up to 6 fine-tuning epochs; class weighting, early stopping, validation-loss checkpointing, and ReduceLROnPlateau.
- Source group timestamps do not verify capture sessions. Exact duplicates were absent; coarse near-image candidates need human review and correlated imagery cannot be ruled out.
- Selected model remains an experimental artifact and was not integrated into the predictor/API.
