"""Regression checks for the separate MobileNetV2 rice candidate pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import train_rice_mobilenet as training
from app.services import predictor


class TestRiceMobileNetPipeline(unittest.TestCase):
    def test_training_image_preprocessing_matches_predictor(self):
        image = Image.new("RGB", (301, 177), color=(45, 130, 220))
        expected = np.asarray(image.resize(training.SIZE, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.png"
            image.save(image_path)
            train_pixels = training.load_image(image_path)
        inference_pixels = predictor.preprocess_image(image)[0]
        np.testing.assert_allclose(train_pixels, expected, rtol=0, atol=0)
        np.testing.assert_allclose(inference_pixels, train_pixels, rtol=0, atol=1e-6)
        np.testing.assert_allclose(training.mobilenet_v2_preprocess(train_pixels), train_pixels * 2 - 1)

    def test_fixed_manifest_mapping_matches_saved_class_list(self):
        rows, classes = training.load_manifest()
        saved = json.loads((training.MODELS_DIR / "rice_classes.json").read_text(encoding="utf-8"))
        self.assertEqual(classes, sorted({row["class"] for row in rows}))
        self.assertEqual(classes, saved)
        self.assertEqual({name: index for index, name in enumerate(classes)},
                         {name: index for index, name in enumerate(saved)})

    def test_candidate_model_load_and_inference(self):
        candidate = training.MODEL_OUT
        if not candidate.is_file():
            self.skipTest("Run train_rice_mobilenet.py before checking saved candidate inference")
        model = training.keras.models.load_model(candidate)
        sample = np.zeros((1, *training.SIZE, 3), dtype=np.float32)
        scores = model.predict(sample, verbose=0)
        self.assertEqual(scores.shape, (1, 6))
        self.assertTrue(np.isfinite(scores).all())
        self.assertAlmostEqual(float(scores[0].sum()), 1.0, places=5)

    def test_predictor_prefers_the_promoted_candidate(self):
        self.assertEqual(predictor.get_model_path("rice"), training.MODEL_OUT)


if __name__ == "__main__":
    unittest.main()
