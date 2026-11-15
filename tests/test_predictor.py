"""Unit tests for Predictor service.

Covers:
- Image preprocessing for CNN inference (shape, normalization, dtype)
- Model-not-available response when weights are not yet deployed
- Model loader architecture for separate rice and wheat models
- Unsupported crop handling
- System model status inspection
"""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image

from app.services.predictor import (
    CROP_CLASS_LABELS,
    SUPPORTED_CROPS,
    TARGET_IMAGE_SIZE,
    get_model_path,
    get_model_status,
    is_model_available,
    predict_crop_disease,
    preprocess_image,
)


class TestPredictorService(unittest.TestCase):
    """Test suite for the crop disease predictor service."""

    def setUp(self):
        # Create a sample test PIL image
        self.test_img = Image.new("RGB", (320, 240), color=(80, 140, 70))
        buf = io.BytesIO()
        self.test_img.save(buf, format="JPEG")
        self.test_bytes = buf.getvalue()

    def test_preprocess_from_pil(self):
        """Test preprocessing directly from a PIL Image instance."""
        batch = preprocess_image(self.test_img)
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        self.assertGreaterEqual(batch.min(), 0.0)
        self.assertLessEqual(batch.max(), 1.0)

    def test_preprocess_from_bytes(self):
        """Test preprocessing from raw image bytes."""
        batch = preprocess_image(self.test_bytes)
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)

    def test_preprocess_from_buffer(self):
        """Test preprocessing from a BytesIO buffer."""
        buf = io.BytesIO(self.test_bytes)
        batch = preprocess_image(buf)
        self.assertEqual(batch.shape, (1, 224, 224, 3))

    def test_model_not_available_rice(self):
        """When model files are absent, return an informative response."""
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch("app.services.predictor.MODELS_DIR", Path(temp_dir)), \
             patch("app.services.predictor._MODEL_CACHE", {}):
            result = predict_crop_disease("rice", self.test_bytes)
        self.assertEqual(result["status"], "model_not_available")
        self.assertEqual(result["crop_type"], "rice")
        self.assertIsNone(result["prediction"])
        self.assertIsNone(result["confidence"])
        self.assertFalse(result["is_model_loaded"])
        self.assertIn("rice_accuracy_experiments", result["message"])
        self.assertIn("model_info", result)

    def test_wheat_selected_model_inference(self):
        """The selected wheat model returns a label from the canonical output order."""
        result = predict_crop_disease("wheat", self.test_bytes)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["crop_type"], "wheat")
        self.assertIn(result["prediction"], CROP_CLASS_LABELS["wheat"])
        self.assertEqual(len(result["raw_scores"]), 4)

    def test_selected_class_mappings_and_paths(self):
        self.assertEqual(CROP_CLASS_LABELS["rice"], [
            "Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf",
            "Leaf scald", "Sheath Blight",
        ])
        self.assertEqual(CROP_CLASS_LABELS["wheat"], [
            "Black Rust", "Brown Rust", "Yellow Rust", "Powdery Mildew",
        ])
        self.assertTrue(get_model_path("rice").as_posix().endswith(
            "rice_accuracy_experiments/label_smoothing_ft30/rice_5class_label_smoothing_ft30.keras"))
        self.assertTrue(get_model_path("wheat").as_posix().endswith(
            "wheat_accuracy_experiments/finetune_last20_smoothing/wheat_finetune_last20_smoothing.keras"))

    def test_unsupported_crop_type(self):
        """Unsupported crop type should return an error dict with supported list."""
        result = predict_crop_disease("cotton", self.test_bytes)
        self.assertEqual(result["status"], "error")
        self.assertIn("cotton", result["message"])
        self.assertIn("rice", result["message"])

    def test_case_insensitivity(self):
        """Crop type should be case-insensitive ('Rice', 'WHEAT')."""
        result_upper = predict_crop_disease("RICE", self.test_bytes)
        self.assertEqual(result_upper["crop_type"], "rice")
        self.assertEqual(result_upper["status"], "success")
        self.assertIn(result_upper["prediction"], CROP_CLASS_LABELS["rice"])

    def test_model_status_inspection(self):
        """Status report should list all supported crops and their availability."""
        status = get_model_status()
        self.assertIn("crops", status)
        self.assertIn("rice", status["crops"])
        self.assertIn("wheat", status["crops"])
        self.assertTrue(status["crops"]["rice"]["available"])
        self.assertTrue(status["crops"]["wheat"]["available"])
        self.assertEqual(status["crops"]["rice"]["output_classes"], 5)
        self.assertEqual(status["crops"]["wheat"]["output_classes"], 4)
        self.assertTrue(Path(status["crops"]["rice"]["path"]).as_posix().endswith(
            "rice_5class_label_smoothing_ft30.keras"))
        self.assertTrue(Path(status["crops"]["wheat"]["path"]).as_posix().endswith(
            "wheat_finetune_last20_smoothing.keras"))

    def test_is_model_available(self):
        """Promoted rice and wheat candidates exist."""
        self.assertTrue(is_model_available("rice"))
        self.assertTrue(is_model_available("wheat"))


if __name__ == "__main__":
    unittest.main()
