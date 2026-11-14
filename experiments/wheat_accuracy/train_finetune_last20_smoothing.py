"""Run only the label-smoothed 20-layer fine-tuning wheat experiment."""
from experiments.wheat_accuracy import common
from experiments.wheat_accuracy.run_controlled_experiments import run_experiments


if __name__ == "__main__":
    run_experiments("finetune_last20_smoothing", common.OUT_ROOT / "standalone_finetune_last20_smoothing")
