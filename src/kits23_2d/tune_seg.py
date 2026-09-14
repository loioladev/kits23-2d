"""Search segmentation hyperparameters with Optuna.

The search space lives in the YAML config so it can be edited without touching
code. Trials should run on a subset of the data for a few epochs; the winning
configuration is then retrained at full scale with kits23-train-seg.

Usage:
    uv run kits23-tune-seg --config configs/tune_seg.yaml --n-trials 20
"""

from kits23_2d.config import resolve_config
from kits23_2d.train_seg import build_parser, run_training
from kits23_2d.tuning import add_tuning_args, run_study


def main():
    """Run the segmentation hyperparameter study."""
    parser = build_parser()
    add_tuning_args(parser)
    cfg = resolve_config(parser)
    run_study(cfg, run_training, default_study_name="kits23-seg")


if __name__ == "__main__":
    main()
