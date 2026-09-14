"""Train a Faster R-CNN to detect kidneys, tumors, and cysts in 2D slices.

Boxes come straight from the COCO export. Annotations can be as small as a
single pixel, so degenerate boxes are filtered out by the dataset; and roughly
half the slices carry no object at all, which the model handles as an empty
target but which can be subsampled with --empty-ratio to speed up an epoch.

Usage:
    uv run kits23-train-det --config configs/det_frcnn.yaml
    uv run kits23-train-det --arch mobilenet --batch-size 4
"""

import argparse
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from kits23_2d.config import add_common_args, resolve_config, save_config
from kits23_2d.datasets import (
    CocoIndex,
    KiTSDetDataset,
    detection_collate,
    make_box_rescaler,
)
from kits23_2d.engine import Warmup, train_one_epoch_det, validate_det
from kits23_2d.models import (
    DET_ARCHS,
    build_detection_model,
    build_optimizer,
    build_scheduler,
)
from kits23_2d.tracking import det_prediction_grid, log_image, log_metrics, start_run
from kits23_2d.transforms import build_det_transforms
from kits23_2d.utils import (
    EarlyStopping,
    resolve_device,
    save_checkpoint,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the detection training CLI.

    Returns
    -------
    argparse.ArgumentParser
        The parser, with the common and detection-specific arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    model = parser.add_argument_group("model")
    model.add_argument("--arch", choices=DET_ARCHS, default="resnet50")
    model.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    model.add_argument(
        "--trainable-backbone-layers",
        type=int,
        default=3,
        help="0 freezes the backbone entirely, 5 trains all of it.",
    )
    model.add_argument(
        "--min-box-size",
        type=float,
        default=4.0,
        help="Drop boxes with a side shorter than this, in pixels.",
    )
    model.add_argument("--score-threshold", type=float, default=0.5,
                       help="Only used when drawing predictions.")
    model.add_argument("--monitor", default="mAP", help="Metric to select on.")
    return parser


def build_loaders(cfg) -> tuple:
    """Build the training and validation data loaders.

    Parameters
    ----------
    cfg : argparse.Namespace
        The run configuration.

    Returns
    -------
    train_loader, val_loader, val_index : tuple
        The two DataLoaders plus the validation index, which COCOeval needs for
        the matching ground truth.
    """
    train_index = CocoIndex(
        cfg.dataset_dir, cfg.train_split, empty_ratio=cfg.empty_ratio,
        max_images=cfg.max_train_images, seed=cfg.seed,
    )
    val_index = CocoIndex(
        cfg.dataset_dir, cfg.val_split, max_images=cfg.max_val_images, seed=cfg.seed
    )

    train_set = KiTSDetDataset(
        train_index,
        build_det_transforms(True, cfg.size, cfg.aug_strength, cfg.min_box_size),
        cfg.min_box_size,
    )
    val_set = KiTSDetDataset(
        val_index,
        build_det_transforms(False, cfg.size, "none", cfg.min_box_size),
        cfg.min_box_size,
    )

    common = {
        "num_workers": cfg.num_workers,
        "pin_memory": True,
        "persistent_workers": cfg.num_workers > 0,
        "collate_fn": detection_collate,
    }
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, **common)
    print(
        f"train: {len(train_set)}/{train_index.num_images_total} images | "
        f"val: {len(val_set)}/{val_index.num_images_total} images"
    )
    return train_loader, val_loader, val_index


def run_training(cfg, trial=None) -> float:
    """Train a detection model and return the best monitored metric.

    Parameters
    ----------
    cfg : argparse.Namespace
        The run configuration.
    trial : optuna.Trial | None
        When given, intermediate values are reported to it for pruning.

    Returns
    -------
    float
        The best value of cfg.monitor observed during the run.
    """
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    train_loader, val_loader, val_index = build_loaders(cfg)
    coco_gt = val_index.coco_subset()
    scale_back = make_box_rescaler(val_index, cfg.size)

    model = build_detection_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler, per_iteration = build_scheduler(cfg, optimizer, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    warmup = Warmup(optimizer, cfg.warmup_iters, enabled=not per_iteration)

    run_dir = Path(cfg.output_dir) / "det"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.json")

    stopper = EarlyStopping(cfg.patience)
    best, global_step = float("-inf"), 0

    for epoch in range(cfg.epochs):
        losses, global_step = train_one_epoch_det(
            cfg, model, train_loader, optimizer, scaler, scheduler, per_iteration,
            warmup, device, epoch, global_step,
        )
        metrics = validate_det(cfg, model, val_loader, device, coco_gt, scale_back)
        metrics.update(losses)
        metrics["lr"] = optimizer.param_groups[0]["lr"]

        if scheduler is not None and not per_iteration:
            if cfg.scheduler == "plateau":
                scheduler.step(metrics[cfg.monitor])
            else:
                scheduler.step()

        score = metrics[cfg.monitor]
        print(
            f"epoch {epoch}: train_loss={losses['train_loss']:.4f} "
            f"mAP={metrics['mAP']:.4f} mAP50={metrics['mAP_50']:.4f} "
            f"AP(kidney/tumor/cyst)={metrics['AP_kidney']:.3f}/"
            f"{metrics['AP_tumor']:.3f}/{metrics['AP_cyst']:.3f}"
        )
        log_metrics(metrics, step=epoch)

        if score > best:
            best = score
            save_checkpoint(run_dir / "best.pt", model, cfg, epoch, metrics, "det")
            if cfg.log_images:
                _log_predictions(cfg, model, val_loader, device, epoch)

        if trial is not None:
            trial.report(score, epoch)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()

        if stopper.step(score):
            print(f"early stopping at epoch {epoch}")
            break

    save_checkpoint(run_dir / "last.pt", model, cfg, cfg.epochs - 1, metrics, "det")
    mlflow.log_metric("best_" + cfg.monitor, best)
    if cfg.log_model:
        mlflow.log_artifact(str(run_dir / "best.pt"))
    return best


@torch.no_grad()
def _log_predictions(cfg, model, loader, device, epoch: int) -> None:
    """Log a ground-truth vs prediction grid for one validation batch.

    Parameters
    ----------
    cfg : argparse.Namespace
        The run configuration.
    model : torch.nn.Module
        The model to run.
    loader : torch.utils.data.DataLoader
        The validation loader to draw a batch from.
    device : torch.device
        The device to run on.
    epoch : int
        The epoch, used to name the artifact.
    """
    model.eval()
    images, targets = next(iter(loader))
    images, targets = images[: cfg.log_images], targets[: cfg.log_images]
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
        outputs = model([image.to(device) for image in images])
    outputs = [{k: v.float() for k, v in o.items()} for o in outputs]
    grid = det_prediction_grid(images, targets, outputs, cfg.score_threshold)
    log_image(grid, f"predictions/epoch_{epoch:03d}.png")


def main():
    """Run a single detection training from the command line."""
    cfg = resolve_config(build_parser())
    with start_run(cfg):
        best = run_training(cfg)
    print(f"best {cfg.monitor}: {best:.4f}")


if __name__ == "__main__":
    main()
