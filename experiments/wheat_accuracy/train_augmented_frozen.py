"""Run only the frozen-backbone augmentation wheat experiment."""
from experiments.wheat_accuracy import common
from experiments.wheat_accuracy.run_controlled_experiments import run_experiments


if __name__ == "__main__":
    run_experiments("augmented_frozen", common.OUT_ROOT / "standalone_augmented_frozen")
