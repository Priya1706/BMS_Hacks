"""Standalone validation-only runner for the stronger augmentation, 30-layer variant."""
from experiments.rice_accuracy.run_controlled_experiments import train_variant_standalone


if __name__ == "__main__":
    train_variant_standalone("augmented_ft30")
