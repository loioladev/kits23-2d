"""Shared Optuna driver for the two tuning entry points.

Both studies do the same thing: sample a point from the YAML search space,
apply it on top of the base config, and run a real (but shortened) training,
reporting the monitored metric to Optuna after every epoch so unpromising
trials get pruned early.

Each trial is an MLflow run nested under a parent run representing the study,
so the whole search shows up as a single tree in the MLflow UI.
"""

import argparse

import mlflow
import optuna
import torch

from kits23_2d.config import apply_overrides, suggest_from_space
from kits23_2d.tracking import start_run


def add_tuning_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the arguments controlling the Optuna study itself.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to extend.

    Returns
    -------
    argparse.ArgumentParser
        The same parser, for chaining.
    """
    group = parser.add_argument_group("optuna")
    group.add_argument("--n-trials", type=int, default=20)
    group.add_argument("--timeout", type=float, help="Stop the study after N seconds.")
    group.add_argument("--study-name", default=None)
    group.add_argument(
        "--storage",
        default="optuna.db",
        help="SQLite file backing the study, so it can be resumed and inspected.",
    )
    group.add_argument("--n-startup-trials", type=int, default=5,
                       help="Random trials before the TPE sampler takes over.")
    group.add_argument("--n-warmup-steps", type=int, default=2,
                       help="Epochs before a trial becomes eligible for pruning.")
    return parser


def run_study(cfg, run_training, default_study_name: str) -> optuna.Study:
    """Create or resume a study and optimize it.

    Parameters
    ----------
    cfg : argparse.Namespace
        The base configuration; each trial gets a copy with overrides applied.
    run_training : callable
        ``run_training(cfg, trial)`` returning the metric to maximize.
    default_study_name : str
        Study name used when --study-name is not given.

    Returns
    -------
    optuna.Study
        The completed study.
    """
    space = cfg.config_values.get("search_space")
    if not space:
        raise SystemExit(
            "no 'search_space' block in the config; pass --config configs/tune_*.yaml"
        )

    study = optuna.create_study(
        study_name=cfg.study_name or default_study_name,
        storage=f"sqlite:///{cfg.storage}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=cfg.seed, n_startup_trials=cfg.n_startup_trials
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=cfg.n_startup_trials, n_warmup_steps=cfg.n_warmup_steps
        ),
    )

    with start_run(cfg, run_name=study.study_name):
        mlflow.set_tag("optuna_study", study.study_name)
        study.optimize(
            _make_objective(cfg, run_training, space),
            n_trials=cfg.n_trials,
            timeout=cfg.timeout,
        )
        _log_study_summary(study)

    _print_summary(study)
    return study


def _make_objective(cfg, run_training, space: dict):
    """Build the Optuna objective closure.

    Parameters
    ----------
    cfg : argparse.Namespace
        The base configuration.
    run_training : callable
        The training entry point.
    space : dict
        The search-space mapping from the YAML config.

    Returns
    -------
    callable
        A function suitable for study.optimize.
    """

    def objective(trial: optuna.Trial) -> float:
        params = suggest_from_space(trial, space)
        trial_cfg = apply_overrides(cfg, params)
        # Each trial writes its own checkpoints, or they would overwrite
        # each other in the shared output directory.
        trial_cfg.output_dir = trial_cfg.output_dir / f"trial_{trial.number:03d}"

        with start_run(trial_cfg, run_name=f"trial_{trial.number:03d}", nested=True):
            mlflow.set_tag("optuna_trial", trial.number)
            try:
                return run_training(trial_cfg, trial=trial)
            except optuna.TrialPruned:
                mlflow.set_tag("pruned", "true")
                raise
            except torch_oom_errors() as error:
                # A batch size the sampler tried simply does not fit in 8 GB;
                # that is a property of the search space, not a crash.
                mlflow.set_tag("failed", str(error)[:250])
                print(f"trial {trial.number} ran out of memory; pruning it")
                raise optuna.TrialPruned() from error

    return objective


def torch_oom_errors() -> tuple:
    """Return the exception types that mean "the GPU ran out of memory".

    Returns
    -------
    tuple
        Exception classes to catch.
    """
    return (torch.cuda.OutOfMemoryError,)


def _log_study_summary(study: optuna.Study) -> None:
    """Log the best trial's params and value to the parent MLflow run.

    Parameters
    ----------
    study : optuna.Study
        The study that has finished optimizing.
    """
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        return
    mlflow.log_metric("best_value", study.best_value)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_dict(
        {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "n_pruned": sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
            ),
        },
        "optuna_summary.json",
    )


def _print_summary(study: optuna.Study) -> None:
    """Print the best trial and how to retrain it at full scale.

    Parameters
    ----------
    study : optuna.Study
        The study that has finished optimizing.
    """
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        print("no trial completed")
        return
    print(f"\nbest value: {study.best_value:.4f}")
    for name, value in study.best_params.items():
        print(f"  {name}: {value}")
    flags = " ".join(
        f"--{name.replace('_', '-')} {value}"
        for name, value in study.best_params.items()
    )
    print(f"\nretrain at full scale with:\n  {flags}")
