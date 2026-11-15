"""Checks for the isolated wheat accuracy experiment model and mapping."""

import json
import unittest

import numpy as np
import tensorflow as tf

import train_wheat_mobilenet as production_training
from app.services.predictor import (
    MODELS_DIR,
    SELECTED_MODEL_RELATIVE_PATHS,
    get_class_labels_for_crop,
    get_model_path,
    preprocess_image,
)
from experiments.wheat_accuracy import common as experiment


class WheatAccuracyExperimentTests(unittest.TestCase):
    def test_existing_four_class_mapping_is_preserved(self):
        self.assertEqual(experiment.CLASSES, [
            "Black Rust", "Brown Rust", "Yellow Rust", "Powdery Mildew",
        ])
        self.assertEqual(
            json.loads(experiment.BASELINE_CLASSES.read_text(encoding="utf-8")),
            experiment.CLASSES,
        )
        self.assertEqual(get_class_labels_for_crop("wheat"), experiment.CLASSES)
        self.assertEqual(experiment.FOLDER_TO_LABEL["Mildew"], "Powdery Mildew")

    def test_experiment_model_shape_and_embedded_preprocessing(self):
        config = {**experiment.VARIANTS["augmented_frozen"], "name": "test"}
        model, backbone = experiment.build_model(config)
        self.assertEqual(tuple(model.input_shape[1:]), (224, 224, 3))
        self.assertEqual(model.output_shape[-1], 4)
        preprocessing = model.get_layer("mobilenet_v2_preprocessing")
        np.testing.assert_allclose(
            preprocessing(tf.constant([0.0, 0.5, 1.0])).numpy(),
            [-1.0, 0.0, 1.0], atol=1e-6,
        )
        self.assertIsNotNone(model.get_layer("training_augmentation"))
        self.assertFalse(backbone.trainable)

    def test_predictor_preprocessing_and_selected_production_path(self):
        samples = production_training.image_paths(
            production_training.TEST_DIR / "mildew_test"
        )
        if not samples:
            self.skipTest("Wheat test images are not available")
        expected = experiment.load_image(samples[0])
        actual = preprocess_image(samples[0])[0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        self.assertEqual(
            get_model_path("wheat"),
            MODELS_DIR / SELECTED_MODEL_RELATIVE_PATHS["wheat"],
        )


if __name__ == "__main__":
    unittest.main()
