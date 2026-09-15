"""Shared configuration plumbing for the training, tuning, and evaluation CLIs.

Every script builds an argparse parser, optionally overlays a YAML file given
with --config, and ends up with a plain argparse.Namespace used as the config
object. Precedence is: command line > YAML file > parser default.

The module also knows how to turn a YAML "search_space" block into Optuna
suggestions, so hyperparameter ranges live in configs/ instead of in code.
"""

import argparse
import copy
import json
from pathlib import Path

import optuna
import yaml

# Category ids come from convert.py: 1=kidney, 2=tumor, 3=cyst, plus background.
CLASS_NAMES = ("background", "kidney", "tumor", "cyst")
NUM_CLASSES = len(CLASS_NAMES)

# KiTS "hierarchical evaluation classes": each one is a union of label ids.
HEC_GROUPS = {
    "kidney_and_masses": (1, 2, 3),
    "masses": (2, 3),
    "tumor": (2,),
}


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the arguments shared by every training/tuning entry point.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to extend.

    Returns
    -------
    argparse.ArgumentParser
        The same parser, for chaining.
    """
    data = parser.add_argument_group("data")
    data.add_argument("--config", type=Path, help="YAML file with default values.")
    data.add_argument("--dataset-dir", type=Path, default=Path("kits32-2d"))
    data.add_argument("--train-split", default="train")
    data.add_argument("--val-split", default="val")
    data.add_argument("--size", type=int, default=512, help="Square input size.")
    data.add_argument(
        "--empty-ratio",
        type=float,
        default=1.0,
        help="Fraction of annotation-free training slices to keep (1.0 = all).",
    )
    data.add_argument("--max-train-images", type=int, help="Subsample the train split.")
    data.add_argument("--max-val-images", type=int, help="Subsample the val split.")
    data.add_argument("--num-workers", type=int, default=8)
    data.add_argument(
        "--aug-strength",
        choices=("none", "light", "medium", "heavy"),
        default="medium",
    )

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--epochs", type=int, default=20)
    optim.add_argument("--batch-size", type=int, default=8)
    optim.add_argument("--lr", type=float, default=3e-4)
    optim.add_argument("--weight-decay", type=float, default=1e-4)
    optim.add_argument("--optimizer", choices=("adamw", "adam", "sgd"), default="adamw")
    optim.add_argument("--momentum", type=float, default=0.9, help="SGD only.")
    optim.add_argument(
        "--scheduler", choices=("none", "cosine", "plateau", "onecycle"), default="cosine"
    )
    optim.add_argument("--warmup-iters", type=int, default=250)
    optim.add_argument("--accum-steps", type=int, default=1)
    optim.add_argument("--clip-grad", type=float, default=0.0, help="0 disables it.")
    optim.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mixed precision. On Pascal this mainly saves memory, not time.",
    )
    optim.add_argument("--patience", type=int, default=5, help="0 disables early stop.")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--device", default="cuda")
    runtime.add_argument("--output-dir", type=Path, default=Path("runs"))
    runtime.add_argument("--run-name", help="MLflow run name; defaults to a timestamp.")
    runtime.add_argument("--experiment", default="kits23-2d")
    runtime.add_argument(
        "--mlflow-uri", help="MLflow tracking URI; defaults to sqlite:///mlflow.db."
    )
    runtime.add_argument(
        "--log-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload the best checkpoint to MLflow (~100s of MB).",
    )
    runtime.add_argument("--log-images", type=int, default=8, help="0 disables.")
    return parser


def resolve_config(parser: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    """Parse the CLI with an optional YAML file supplying the defaults.

    The YAML keys use the same names as the arguments, with dashes or
    underscores (``batch_size`` and ``batch-size`` both work).

    Parameters
    ----------
    parser : argparse.ArgumentParser
        A parser already populated with arguments.
    argv : list | None
        The argument list to parse; defaults to sys.argv[1:].

    Returns
    -------
    argparse.Namespace
        The resolved configuration.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args(argv)

    raw = {}
    if known.config is not None:
        raw = load_yaml(known.config)
        known_dests = {action.dest for action in parser._actions}
        defaults = {}
        for key, value in raw.items():
            if key == "search_space":
                continue
            dest = key.replace("-", "_")
            if dest not in known_dests:
                raise SystemExit(f"{known.config}: unknown config key {key!r}")
            defaults[dest] = _coerce(parser, dest, value)
        parser.set_defaults(**defaults)

    args = parser.parse_args(argv)
    args.config_values = raw
    return args


def _coerce(parser: argparse.ArgumentParser, dest: str, value):
    """Apply the argument's own ``type`` callable to a YAML-provided value.

    YAML gives us strings/ints/floats, but an argument declared with
    ``type=Path`` expects a Path. Without this, ``--dataset-dir`` coming from a
    file would stay a str and break path arithmetic downstream.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser owning the argument.
    dest : str
        The argument's destination name.
    value : object
        The raw value read from YAML.

    Returns
    -------
    object
        The value converted to the argument's declared type.
    """
    if value is None:
        return None
    action = next(a for a in parser._actions if a.dest == dest)
    if action.type is None or isinstance(value, (list, dict)):
        return value
    return action.type(value) if not isinstance(value, action.type) else value


def load_yaml(path: Path) -> dict:
    """Read a YAML file into a dict.

    Parameters
    ----------
    path : Path
        The file to read.

    Returns
    -------
    dict
        The parsed mapping, or an empty dict for an empty file.
    """
    with open(path) as f:
        return yaml.safe_load(f) or {}


def suggest_from_space(trial: optuna.Trial, space: dict) -> dict:
    """Turn a YAML search-space block into concrete Optuna suggestions.

    Each entry is ``name: {type: float|int|categorical, ...}``::

        lr:          {type: float, low: 1.0e-5, high: 1.0e-2, log: true}
        batch_size:  {type: categorical, choices: [4, 8, 16]}
        depth:       {type: int, low: 3, high: 5}

    Parameters
    ----------
    trial : optuna.Trial
        The trial doing the sampling.
    space : dict
        The search-space mapping described above.

    Returns
    -------
    dict
        Mapping of config attribute name -> sampled value.
    """
    params = {}
    for name, spec in space.items():
        dest = name.replace("-", "_")
        kind = spec["type"]
        if kind == "categorical":
            params[dest] = trial.suggest_categorical(dest, spec["choices"])
        elif kind == "int":
            params[dest] = trial.suggest_int(
                dest, spec["low"], spec["high"], step=spec.get("step", 1),
                log=spec.get("log", False),
            )
        elif kind == "float":
            params[dest] = trial.suggest_float(
                dest, spec["low"], spec["high"], log=spec.get("log", False),
                step=spec.get("step"),
            )
        else:
            raise ValueError(f"{name}: unsupported search-space type {kind!r}")
    return params


def apply_overrides(cfg: argparse.Namespace, overrides: dict) -> argparse.Namespace:
    """Copy a config and set the given attributes on the copy.

    Parameters
    ----------
    cfg : argparse.Namespace
        The base configuration; it is not modified.
    overrides : dict
        Attribute name -> value.

    Returns
    -------
    argparse.Namespace
        The updated copy.
    """
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if not hasattr(out, key):
            raise ValueError(f"unknown config attribute {key!r}")
        setattr(out, key, value)
    return out


def config_to_dict(cfg: argparse.Namespace) -> dict:
    """Render a config as JSON-friendly scalars, for logging.

    Parameters
    ----------
    cfg : argparse.Namespace
        The configuration to render.

    Returns
    -------
    dict
        Mapping of name -> str/int/float/bool/None.
    """
    out = {}
    for key, value in sorted(vars(cfg).items()):
        if key == "config_values":
            continue
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def save_config(cfg: argparse.Namespace, path: Path) -> None:
    """Write the resolved config next to the run's other artifacts.

    Parameters
    ----------
    cfg : argparse.Namespace
        The configuration to serialize.
    path : Path
        Destination file (.json).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config_to_dict(cfg), f, indent=2, default=str)
