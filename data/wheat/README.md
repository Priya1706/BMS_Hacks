# Wheat image dataset

This folder documents the wheat leaf image datasets being evaluated for RootCause.

## Intended use

Use clear images of wheat leaves to train and evaluate an offline crop-disease classifier. Keep the original class folder names from each source dataset. Do not merge datasets or rename classes until the image contents and label meanings have been checked.

## Source datasets

### 1. Kushagra3204 — Wheat Plant Diseases
- Source: https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases
- Use as the primary candidate dataset.
- The associated dataset documentation describes multiple wheat diseases and pests. The actual downloaded archive's directory names and image contents still need local inspection.
- Verify the Kaggle dataset's current licence and redistribution terms before committing images to this public repository.

### 2. Olyadgetch — Wheat Leaf Dataset
- Source: https://www.kaggle.com/datasets/olyadgetch/wheat-leaf-dataset
- Treat as a supplementary dataset, not an automatic merge.
- The Open Plant Image Archive metadata describes 407 JPEG images in three categories: Healthy, Septoria, and Stripe Rust.
- The archive is marked for academic use; do not copy its image files into this public repository unless redistribution permission is confirmed.
- Metadata: https://ngdc.cncb.ac.cn/opia/dataset/datasets?dataId=16

## Candidate leaf classes

After inspecting the actual archive folders and sample images, consider only these leaf-focused categories where present:

- Healthy
- Brown Rust / Leaf Rust
- Yellow Rust / Stripe Rust
- Black Rust / Stem Rust
- Septoria Leaf Blotch
- Tan Spot
- Powdery Mildew
- Leaf Blight / Helminthosporium Leaf Blight
- Spot Blotch

These are candidate mappings, not confirmed folder names. Preserve the source's exact folder names and document any mapping in the model's label map.

## Exclude from a leaf-image classifier

Unless image inspection proves otherwise, exclude pest categories (Aphid, Mite, Stem Fly), root diseases (Common Root Rot), and head/grain-focused diseases (Fusarium Head Blight / Scab and Smut). Exclude unrelated, ambiguous, corrupt, and duplicate images.

## Required local layout

Create one source-specific directory per dataset. Keep the original class folder names below each source directory:

```
data/
└── wheat/
    ├── kushagra/
    │   └── <original class folders>/
    ├── olyadgetch/
    │   └── <original class folders>/
    └── README.md
```

Do not create guessed class folders before inspecting the archive.

## Validation checklist

Before using or publishing any images:

1. Record the exact archive folder names and selected class folders.
2. Check that every selected file is a readable image; record unreadable files.
3. Record image counts and image dimensions per class.
4. Check exact duplicates (for example, by SHA-256 hash) and investigate near-duplicates.
5. Keep source and class provenance; do not silently combine classes with different label meanings.
6. Confirm each source's licence permits the intended use and public redistribution before committing images.

## Status

- [ ] Download and inspect the Kushagra archive.
- [ ] Confirm the Kushagra licence before redistributing images.
- [ ] Inspect and validate selected leaf images.
- [ ] Confirm permission before redistributing Olyadgetch images.
- [ ] Add verified class folders and image-count/validation results.
- [ ] Keep large datasets out of ordinary Git history; use an approved data host or Git LFS if needed.

No image files have been added yet. This README is a planning and provenance record, not a claim that the datasets have already been downloaded or validated.
