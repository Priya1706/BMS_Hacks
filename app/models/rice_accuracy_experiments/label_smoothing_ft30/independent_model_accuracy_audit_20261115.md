# Independent selected-model test audit

Evaluation date: 2026-11-15. Exact selected checkpoints were evaluated in inference mode on fixed held-out assignments. No weights, datasets, thresholds, or split assignments were changed. Images were decoded as RGB, resized to 224x224 with bilinear interpolation, converted to float32 in [0,1]; the checkpoint graph contains MobileNetV2 rescaling. Training-only augmentation was inactive. Every assigned test image was included; decode/inference failures: 0.

| Metric | Rice | Wheat |
|---|---:|---:|
| Test dataset size | 120 | 200 |
| Correct predictions | 99 | 152 |
| Incorrect predictions | 21 | 48 |
| Accuracy | 0.8250 | 0.7600 |
| Macro precision | 0.8032 | 0.7854 |
| Macro recall | 0.8456 | 0.7600 |
| Macro F1 | 0.8191 | 0.7547 |
| Weighted precision | 0.8298 | 0.7854 |
| Weighted recall | 0.8250 | 0.7600 |
| Weighted F1 | 0.8235 | 0.7547 |

## Rice

Checkpoint: `app\models\rice_accuracy_experiments\label_smoothing_ft30\rice_5class_label_smoothing_ft30.keras`
SHA-256: `92e98fe6d9c4bd27429fe887d4bc3380d34ee354a69bb810b9fd40699288c528`
Input/output: `[None, 224, 224, 3] -> [None, 5]`
Classes in output order: Bacterial Leaf Blight, Brown Spot, Healthy Rice Leaf, Leaf scald, Sheath Blight
Previously reported accuracy/macro F1: 82.5% / 0.8191; independently recalculated values match.

### Per-class results

| Class | Support | Precision | Recall | F1 | Correct | Predicted count |
|---|---:|---:|---:|---:|---:|---:|
| Bacterial Leaf Blight | 4 | 0.8000 | 1.0000 | 0.8889 | 4 | 5 |
| Brown Spot | 24 | 0.8000 | 0.6667 | 0.7273 | 16 | 20 |
| Healthy Rice Leaf | 31 | 0.8056 | 0.9355 | 0.8657 | 29 | 36 |
| Leaf scald | 15 | 0.7059 | 0.8000 | 0.7500 | 12 | 17 |
| Sheath Blight | 46 | 0.9048 | 0.8261 | 0.8636 | 38 | 42 |

### Confusion matrix (rows actual, columns predicted)

| Actual / Predicted | Bacterial Leaf Blight | Brown Spot | Healthy Rice Leaf | Leaf scald | Sheath Blight |
|---:|---:|---:|---:|---:|---:|
| Bacterial Leaf Blight | 4 | 0 | 0 | 0 | 0 |
| Brown Spot | 1 | 16 | 3 | 3 | 1 |
| Healthy Rice Leaf | 0 | 0 | 29 | 0 | 2 |
| Leaf scald | 0 | 1 | 1 | 12 | 1 |
| Sheath Blight | 0 | 3 | 3 | 2 | 38 |

Most frequent error pairs: Brown Spot -> Leaf scald: 3; Brown Spot -> Healthy Rice Leaf: 3; Sheath Blight -> Brown Spot: 3; Sheath Blight -> Healthy Rice Leaf: 3; Healthy Rice Leaf -> Sheath Blight: 2; Sheath Blight -> Leaf scald: 2; Brown Spot -> Sheath Blight: 1; Brown Spot -> Bacterial Leaf Blight: 1.

Rice integrity: fixed `rice_splits.csv` test rows were used for retained classes; no exact-content overlap with non-test was found. Coarse perceptual-hash audit flagged possible near-duplicates, not confirmed duplicates. Group IDs are disjoint, although timestamp-derived groups do not prove capture-session independence.

## Wheat

Checkpoint: `app\models\wheat_accuracy_experiments\finetune_last20_smoothing\wheat_finetune_last20_smoothing.keras`
SHA-256: `b3e0e0fb0aa07a7321a2b63566458a0a2f4326570ab856e2e700b7a648f868f1`
Input/output: `[None, 224, 224, 3] -> [None, 4]`
Classes in output order: Black Rust, Brown Rust, Yellow Rust, Powdery Mildew
Previously reported accuracy/macro F1: 76.0% / 0.7547; independently recalculated values match.

### Per-class results

| Class | Support | Precision | Recall | F1 | Correct | Predicted count |
|---|---:|---:|---:|---:|---:|---:|
| Black Rust | 50 | 0.7333 | 0.4400 | 0.5500 | 22 | 30 |
| Brown Rust | 50 | 0.5600 | 0.8400 | 0.6720 | 42 | 75 |
| Yellow Rust | 50 | 0.9434 | 1.0000 | 0.9709 | 50 | 53 |
| Powdery Mildew | 50 | 0.9048 | 0.7600 | 0.8261 | 38 | 42 |

### Confusion matrix (rows actual, columns predicted)

| Actual / Predicted | Black Rust | Brown Rust | Yellow Rust | Powdery Mildew |
|---:|---:|---:|---:|---:|
| Black Rust | 22 | 28 | 0 | 0 |
| Brown Rust | 2 | 42 | 2 | 4 |
| Yellow Rust | 0 | 0 | 50 | 0 |
| Powdery Mildew | 6 | 5 | 1 | 38 |

Most frequent error pairs: Black Rust -> Brown Rust: 28; Powdery Mildew -> Black Rust: 6; Powdery Mildew -> Brown Rust: 5; Brown Rust -> Powdery Mildew: 4; Brown Rust -> Yellow Rust: 2; Brown Rust -> Black Rust: 2; Powdery Mildew -> Yellow Rust: 1.

Black Rust -> Brown Rust: **28**; Brown Rust -> Black Rust: **2**. These are model confusions, not output-index mapping errors. Historical audit found cross-split exact copies; selected wheat manifest excluded detected exact test copies from train/validation. Near-duplicate candidates remain and limit independence.

## Interpretation and limitations

Rice: 99/120 (82.5%), macro F1 0.8191. Wheat: 152/200 (76.0%), macro F1 0.7547. These scores describe these datasets, not field performance. Rice has only four Bacterial Leaf Blight test examples. Wheat Black Rust recall is 44% (22/50), mainly confused with Brown Rust. Input shape, preprocessing, class order, and finite output checks passed, so these are model/data errors rather than API mapping failures. Historical split audits note exact-duplicate concerns in wheat and coarse near-duplicate candidates in both; fixed test assignments were preserved. No bootstrap interval was calculated.

Adjacent files contain per-crop audit JSON with test-image SHA-256 hashes, confusion-matrix CSVs, and complete misclassification lists.
