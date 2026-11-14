# RootCause wheat model experiment comparison

## Evaluation protocol

The fixed 200-image test folders were preserved. Candidate training and selection used only source training images; model selection used validation macro F1. The selected candidate alone was evaluated on the fixed test. The existing production baseline was evaluated for comparison, not used to choose the candidate.

- Class order: Black Rust, Brown Rust, Yellow Rust, Powdery Mildew.
- Candidate validation split seed: 42; per-class 80/20 stratification of exact-content training groups.
- Source/fixed-test files: 4429 total; no unreadable files.
- Exact train/test hash overlaps excluded from candidate training: 49 groups.
- Cross-class exact-content groups excluded from candidate train/validation: 91 training-only groups; labels were not changed.
- Candidate fixed-test support: 50 per class, including an exact duplicate group noted in the dataset audit.
- Near-duplicate counts are coarse average-hash screening leads, not confirmed duplicates.

## Baseline and selected candidate

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Size MB | CPU ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Production baseline | 0.7100 | 0.7116 | 0.7100 | 0.7020 | 0.7020 | 9.70 | 25.38 |
| Validation-selected: finetune_last20_smoothing | 0.7600 | 0.7854 | 0.7600 | 0.7547 | 0.7547 | 22.53 | 25.72 |

Accuracy delta: +5.00 percentage points. Macro F1 delta: +0.0528.
Paired bootstrap: {"method": "paired bootstrap over exact-content test clusters; descriptive only", "replicates": 3000, "test_rows": 200, "independent_exact_content_clusters": 199, "accuracy_difference_95pct_interval": [0.0, 0.10050251256281406], "macro_f1_difference_95pct_interval": [-0.00019204377624885272, 0.10580701171472795]}. The interval is descriptive; it is not a significance claim.

## Validation experiment comparison

| Experiment | Validation accuracy | Validation macro F1 | Fine-tuned eligible layers | Label smoothing |
|---|---:|---:|---:|---:|
| augmented_frozen | 0.8635 | 0.8137 | 0 | 0.00 |
| finetune_last20 | 0.8726 | 0.8380 | 20 | 0.00 |
| finetune_last20_smoothing (selected) | 0.8804 | 0.8470 | 20 | 0.05 |

## Per-class test results

| Class | Baseline precision / recall / F1 | Candidate precision / recall / F1 | Support |
|---|---:|---:|---:|
| Black Rust | 0.794 / 0.540 / 0.643 | 0.733 / 0.440 / 0.550 | 50 |
| Brown Rust | 0.510 / 0.520 / 0.515 | 0.560 / 0.840 / 0.672 | 50 |
| Yellow Rust | 0.806 / 1.000 / 0.893 | 0.943 / 1.000 / 0.971 | 50 |
| Powdery Mildew | 0.736 / 0.780 / 0.757 | 0.905 / 0.760 / 0.826 | 50 |

## Data and deployment considerations

Exact-content duplicate groups under both Black Rust and Brown Rust indicate unresolved labels. The experiment excluded ambiguous training groups rather than relabeling them. The fixed test still reflects its original folders, including one repeated Black Rust test image pair; its raw 200-image score is reported as requested.
The fixed test has 50 rows per class, but independent capture-session metadata is absent. The supplied dataset contains heterogeneous image dimensions and watermarked Mildew samples. The checked-in wheat split is generated at training time rather than stored as a manifest; this experiment saves its own reproducible assignment.
Validation-only calibration: {"temperature_selected_on_validation_only": 0.7500000000000002, "validation_uncalibrated": {"ece_10_bins": 0.05181619924128598, "multiclass_brier": 0.1727508741705219, "nll": 0.3418453335762024}, "validation_temperature_scaled": {"ece_10_bins": 0.025138604276667028, "multiclass_brier": 0.16756035551834253, "nll": 0.32439011335372925}, "abstention_threshold_for_90pct_accuracy_min_20pct_coverage": 0.47344037890434265, "note": "Confidence is not proof of disease; abstain and seek human review when uncertain."}.
Confidence should support abstention and human review, not stand alone as disease confirmation or treatment advice.
Candidate checkpoint: C:\Semister 3 notes\Hackathon\BMS_Hacks\BMS_Hacks\app\models\wheat_accuracy_experiments\finetune_last20_smoothing\wheat_finetune_last20_smoothing.keras. It remains separate from production and was not integrated.
The model remains a research candidate until label conflicts and likely near-duplicate/capture-session issues are resolved.
