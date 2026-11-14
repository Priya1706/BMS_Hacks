"""Integration checks for the evaluated wheat MobileNetV2 candidate."""

import io
import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import train_wheat_mobilenet as training
from app.services.advisory import get_disease_advisory
from app.services.predictor import (
    get_class_labels_for_crop,
    get_model_path,
    predict_crop_disease,
    preprocess_image,
)


class TestWheatMobileNet(unittest.TestCase):
    def test_class_mapping_matches_dataset_and_predictor(self):
        saved = json.loads(training.CLASSES_OUT.read_text(encoding="utf-8"))
        self.assertEqual(saved, training.CLASSES)
        self.assertEqual(saved, ["Black Rust", "Brown Rust", "Yellow Rust", "Powdery Mildew"])
        self.assertEqual(get_class_labels_for_crop("wheat"), saved)
        self.assertEqual(training.FOLDER_TO_LABEL["Mildew"], "Powdery Mildew")
        self.assertEqual(get_disease_advisory("wheat", "Brown Rust")["disease_id"], "leaf_rust")
        self.assertEqual(get_disease_advisory("wheat", "Yellow Rust")["disease_id"], "stripe_rust")
        self.assertEqual(get_disease_advisory("wheat", "Powdery Mildew")["disease_id"], "powdery_mildew")

    def test_predictor_selects_candidate(self):
        self.assertEqual(get_model_path("wheat"), training.MODEL_OUT)

    def test_candidate_load_and_predictor_inference(self):
        model = training.keras.models.load_model(training.MODEL_OUT)
        sample = np.zeros((1, *training.IMAGE_SIZE, 3), dtype=np.float32)
        scores = model.predict(sample, verbose=0)
        self.assertEqual(scores.shape, (1, 4))
        self.assertTrue(np.isfinite(scores).all())
        self.assertAlmostEqual(float(scores[0].sum()), 1.0, places=5)

        mildew_paths = training.image_paths(training.TEST_DIR / "mildew_test")
        if mildew_paths:
            test_image = mildew_paths[0]
            inference = predict_crop_disease("wheat", test_image)
        else:
            temp_img = Image.new("RGB", (224, 224), color=(100, 150, 80))
            buf = io.BytesIO()
            temp_img.save(buf, format="JPEG")
            inference = predict_crop_disease("wheat", buf.getvalue())
        self.assertEqual(inference["status"], "success")
        self.assertIn(inference["prediction"], get_class_labels_for_crop("wheat"))
        self.assertEqual(len(inference["raw_scores"]), 4)

    def test_training_preprocessing_matches_predictor(self):
        mildew_paths = training.image_paths(training.TEST_DIR / "mildew_test")
        if not mildew_paths:
            self.skipTest("Wheat dataset test images excluded from git")
        sample = mildew_paths[0]
        trained = training.load_image(sample)
        inferred = preprocess_image(sample)[0]
        np.testing.assert_allclose(trained, inferred, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
