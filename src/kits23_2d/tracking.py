"""MLflow helpers and prediction visualizations."""

import json
from datetime import UTC, datetime

import cv2
import mlflow
import numpy as np
import torch

from kits23_2d.config import CLASS_NAMES, config_to_dict
from kits23_2d.stats import CATEGORY_COLORS, draw_annotations

#     uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def start_run(cfg, run_name: str | None = None, nested: bool = False):
    """Open an MLflow run and log the resolved configuration as parameters.

    Parameters
    ----------
    cfg : argparse.Namespace
        The run configuration.
    run_name : str | None
        Overrides cfg.run_name; used to name Optuna trial runs.
    nested : bool
        True for an Optuna trial nested under the study's parent run.

    Returns
    -------
    mlflow.ActiveRun
        The started run, to be used as a context manager.
    """
    mlflow.set_tracking_uri(cfg.mlflow_uri or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(cfg.experiment)
    name = run_name or cfg.run_name or datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    run = mlflow.start_run(run_name=name, nested=nested)
    mlflow.log_params(config_to_dict(cfg))
    return run


def log_metrics(metrics: dict, step: int, prefix: str = "") -> None:
    """Log a dict of metrics, skipping any non-finite entries.

    A class that is absent from a split yields a NaN Dice by design; MLflow
    stores NaN but it renders as a broken curve, so those are dropped instead.

    Parameters
    ----------
    metrics : dict
        Mapping of metric name -> value.
    step : int
        The step (epoch) to record them at.
    prefix : str
        Optional prefix prepended to each metric name.
    """
    clean = {
        f"{prefix}{name}": float(value)
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    }
    if clean:
        mlflow.log_metrics(clean, step=step)


def log_json(payload: dict, artifact_name: str) -> None:
    """Log a dict as a JSON artifact of the active run.

    Parameters
    ----------
    payload : dict
        The JSON-serializable content.
    artifact_name : str
        The artifact's file name, e.g. "metrics.json".
    """
    mlflow.log_dict(json.loads(json.dumps(payload, default=str)), artifact_name)


def overlay_label_map(image: np.ndarray, label: np.ndarray, alpha: float = 0.5):
    """Blend a semantic label map over a slice using the stats.py palette.

    Parameters
    ----------
    image : np.ndarray
        The BGR image to draw on (not modified in place).
    label : np.ndarray
        The label map as [H, W] with values in {0, 1, 2, 3}.
    alpha : float
        Blending weight of the colour over the image.

    Returns
    -------
    np.ndarray
        A copy of the image with the label map painted on top.
    """
    overlay = image.copy()
    for category_id, color in CATEGORY_COLORS.items():
        mask = label == category_id
        if not mask.any():
            continue
        overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(color) * alpha).astype(
            np.uint8
        )
    return overlay


def denormalize(image: torch.Tensor, mean, std) -> np.ndarray:
    """Turn a normalized CHW tensor back into a BGR uint8 image.

    Parameters
    ----------
    image : torch.Tensor
        The tensor as [C, H, W].
    mean : Sequence[float]
        The per-channel mean used to normalize it.
    std : Sequence[float]
        The per-channel standard deviation used to normalize it.

    Returns
    -------
    np.ndarray
        The image as [H, W, 3] uint8 BGR.
    """
    array = image.detach().cpu().float().numpy()
    mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
    array = np.clip(array * std + mean, 0.0, 1.0) * 255.0
    array = array.astype(np.uint8).transpose(1, 2, 0)
    if array.shape[2] == 1:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return array


def seg_prediction_grid(images, targets, predictions, mean, std) -> np.ndarray:
    """Build a ground-truth vs prediction strip for a few segmentation samples.

    Parameters
    ----------
    images : torch.Tensor
        A batch of normalized images as [B, C, H, W].
    targets : torch.Tensor
        The ground-truth label maps as [B, H, W].
    predictions : torch.Tensor
        The predicted label maps as [B, H, W].
    mean : Sequence[float]
        Normalization mean, to undo it.
    std : Sequence[float]
        Normalization standard deviation, to undo it.

    Returns
    -------
    np.ndarray
        A BGR image with one row per sample: input, ground truth, prediction.
    """
    rows = []
    for image, target, prediction in zip(images, targets, predictions, strict=True):
        base = denormalize(image, mean, std)
        truth = overlay_label_map(base, target.cpu().numpy())
        guess = overlay_label_map(base, prediction.cpu().numpy())
        rows.append(np.concatenate([base, truth, guess], axis=1))
    grid = np.concatenate(rows, axis=0)
    return _annotate_columns(grid, ("input", "ground truth", "prediction"))


def det_prediction_grid(images, targets, outputs, score_threshold=0.5) -> np.ndarray:
    """Build a ground-truth vs prediction strip for a few detection samples.

    Parameters
    ----------
    images : list
        A batch of [0, 1] image tensors as [C, H, W].
    targets : list
        The target dicts with boxes (xyxy) and labels.
    outputs : list
        The model outputs with boxes, labels, and scores.
    score_threshold : float
        Minimum score for a prediction to be drawn.

    Returns
    -------
    np.ndarray
        A BGR image with one row per sample: ground truth, prediction.
    """
    categories = {i: name for i, name in enumerate(CLASS_NAMES)}
    rows = []
    for image, target, output in zip(images, targets, outputs, strict=True):
        base = denormalize(image, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
        truth = draw_annotations(base, _as_anns(target), categories)
        guess = draw_annotations(base, _as_anns(output, score_threshold), categories)
        rows.append(np.concatenate([truth, guess], axis=1))
    grid = np.concatenate(rows, axis=0)
    return _annotate_columns(grid, ("ground truth", "prediction"))


def _as_anns(record: dict, score_threshold: float | None = None) -> list:
    """Convert a torchvision target/output dict into stats.py annotation dicts.

    draw_annotations expects COCO-style records, so xyxy boxes are converted
    back to xywh and no segmentation key is emitted (boxes only).

    Parameters
    ----------
    record : dict
        A dict with boxes (xyxy), labels, and optionally scores.
    score_threshold : float | None
        When given, drop entries scoring below it.

    Returns
    -------
    list
        Annotation dicts with bbox and category_id.
    """
    boxes = record["boxes"].detach().cpu().numpy()
    labels = record["labels"].detach().cpu().numpy()
    scores = record.get("scores")
    keep = np.ones(len(boxes), dtype=bool)
    if scores is not None and score_threshold is not None:
        keep = scores.detach().cpu().numpy() >= score_threshold

    anns = []
    for box, label, flag in zip(boxes, labels, keep, strict=True):
        if not flag:
            continue
        x1, y1, x2, y2 = box
        anns.append({"bbox": [x1, y1, x2 - x1, y2 - y1], "category_id": int(label)})
    return anns


def _annotate_columns(grid: np.ndarray, titles) -> np.ndarray:
    """Add a header band naming each column of a comparison grid.

    Parameters
    ----------
    grid : np.ndarray
        The concatenated BGR grid.
    titles : Sequence[str]
        One title per column.

    Returns
    -------
    np.ndarray
        The grid with a 24 px header on top.
    """
    width = grid.shape[1] // len(titles)
    header = np.zeros((24, grid.shape[1], 3), dtype=np.uint8)
    for i, title in enumerate(titles):
        cv2.putText(
            header,
            title,
            (i * width + 8, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return np.concatenate([header, grid], axis=0)


def log_image(image: np.ndarray, artifact_name: str) -> None:
    """Log a BGR image as an MLflow artifact.

    Parameters
    ----------
    image : np.ndarray
        The BGR image to log.
    artifact_name : str
        Artifact path, e.g. "predictions/epoch_003.png".
    """
    mlflow.log_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), artifact_name)
