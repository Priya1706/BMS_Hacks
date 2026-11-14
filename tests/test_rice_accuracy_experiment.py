"""Safety and mapping checks for the isolated rice experiment pipeline."""

import unittest

import numpy as np
import tensorflow as tf

from experiments.rice_accuracy import common


class RiceAccuracyExperimentTests(unittest.TestCase):
    def test_class_mapping_is_stable_and_excludes_leaf_blast(self):
        self.assertEqual(common.CLASS_TO_INDEX, {
            "Bacterial Leaf Blight": 0,
            "Brown Spot": 1,
            "Healthy Rice Leaf": 2,
            "Leaf scald": 3,
            "Sheath Blight": 4,
        })
        self.assertNotIn("Leaf Blast", common.CLASS_TO_INDEX)

    def test_stratified_candidate_split_is_reproducible_and_keeps_test_fixed(self):
        source = common.read_source_rows()
        first = common.make_candidate_split(source)
        second = common.make_candidate_split(source)
        self.assertEqual(first, second)
        expected = {(row["class"], row["filename"]) for row in source
                    if row["class"] in common.CLASS_TO_INDEX and row["split"] == "test"}
        actual = {(row["class"], row["filename"]) for row in first
                  if row["split"] == "fixed_test"}
        self.assertEqual(actual, expected)
        training = {(row["class"], row["filename"]) for row in first if row["split"] == "train"}
        validation = {(row["class"], row["filename"]) for row in first if row["split"] == "validation"}
        self.assertFalse(training & validation)
        self.assertFalse((training | validation) & expected)
        self.assertEqual({row["class"] for row in first if row["split"] == "validation"},
                         set(common.CLASSES))

    def test_mobilenet_preprocessing_maps_rgb_unit_interval_to_expected_range(self):
        preprocessing = tf.keras.layers.Rescaling(2.0, offset=-1.0)
        values = preprocessing(tf.constant([0.0, 0.5, 1.0])).numpy()
        np.testing.assert_allclose(values, [-1.0, 0.0, 1.0], atol=1e-6)

    def test_experiment_model_has_five_outputs_and_embedded_preprocessing(self):
        config = {**common.VARIANTS["augmented_ft30"], "name": "test_model"}
        model, _ = common.make_model(config)
        self.assertEqual(model.output_shape[-1], 5)
        self.assertEqual(tuple(model.input_shape[1:]), (224, 224, 3))
        self.assertEqual(model.get_layer("mobilenet_v2_preprocessing").scale, 2.0)
        self.assertEqual(model.get_layer("mobilenet_v2_preprocessing").offset, -1.0)
        self.assertIsNotNone(model.get_layer("training_augmentation"))


if __name__ == "__main__":
    unittest.main()
