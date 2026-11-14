"""Standalone validation-only runner for the shallow 10-layer fine-tune."""
from experiments.rice_accuracy.run_controlled_experiments import train_variant_standalone


if __name__ == "__main__":
    train_variant_standalone("shallow_ft10")
