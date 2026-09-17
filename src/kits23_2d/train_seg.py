"""Train a UNet for semantic segmentation of the KiTS23 2D slices.

The target is a 4-class label map (background, kidney, tumor, cyst) built by
compositing the overlapping COCO instance masks, with lesions painted over the
kidney they sit inside.

Usage:
    uv run kits23-train-seg --config configs/seg_unet.yaml
    uv run kits23-train-seg --epochs 30 --encoder resnet50 --batch-size 4
"""

import argparse
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from kits23_2d.config import (
    add_common_args,
    resolve_config,
    save_config,
)
from kits23_2d.datasets import CocoIndex, KiTSSegDataset
from kits23_2d.engine import Warmup, train_one_epoch_seg, validate_seg
from kits23_2d.losses import LOSS_NAMES, build_loss
from kits23_2d.models import (
    SEG_ARCHS,
    build_optimizer,
    build_scheduler,
    build_segmentation_model,
)
from kits23_2d.tracking import log_image, log_metrics, seg_prediction_grid, start_run
from kits23_2d.transforms import IMAGENET_MEAN, IMAGENET_STD, build_seg_transforms
from kits23_2d.utils import (
    EarlyStopping,
    resolve_device,
    save_checkpoint,
    seed_everything,
    shutdown_loaders,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the segmentation training CLI.

    Returns
    -------
    argparse.ArgumentParser
        The parser, with the common and segmentation-specific arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    model = parser.add_argument_group("model")
    model.add_argument("--arch", choices=SEG_ARCHS, default="unet")
    model.add_argument("--encoder", default="resnet34")
    model.add_argument("--encoder-weights", default="imagenet")
    model.add_argument("--in-channels", type=int, choices=(1, 3), default=3)
    model.add_argument(
        "--attention", default="", help="smp decoder attention, e.g. scse."
    )

    loss = parser.add_argument_group("loss")
    loss.add_argument("--loss", choices=LOSS_NAMES, default="dicece")
    loss.add_argument("--dice-weight", type=float, default=0.5)
    loss.add_argument("--ce-weight", type=float, default=0.5)
    loss.add_argument(
        "--class-weights",
        type=float,
        nargs="+",
        help="One weight per class (background kidney tumor cyst).",
    )
    loss.add_argument("--monitor", default="mean_dice", help="Metric to select on.")
    return parser


def build_loaders(cfg) -> tuple:
    """Build the training and validation data loaders.

    Parameters
    ----------
    cfg : argparse.Namespace
        The run configuration.

    Returns
    -------
    train_loader, val_loader : tuple
        The two DataLoaders.
    """
    train_index = CocoIndex(
        cfg.dataset_dir,
        cfg.train_split,
        empty_ratio=cfg.empty_ratio,
        max_images=cfg.max_train_images,
        seed=cfg.seed,
    )
    val_index = CocoIndex(
        cfg.dataset_dir, cfg.val_split, max_images=cfg.max_val_images, seed=cfg.seed
    )

    train_set = KiTSSegDataset(
        train_index,
        build_seg_transforms(True, cfg.size, cfg.aug_strength, cfg.in_channels),
        cfg.in_channels,
    )
    val_set = KiTSSegDataset(
        val_index,
        build_seg_transforms(False, cfg.size, "none", cfg.in_channels),
        cfg.in_channels,
    )

    common = {
        "num_workers": cfg.num_workers,
        "pin_memory": True,
        "persistent_workers": cfg.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, **common)
    print(
        f"train: {len(train_set)}/{train_index.num_images_total} images | "
        f"val: {len(val_set)}/{val_index.num_images_total} images"
    )
    return train_loader, val_loader


def run_training(cfg, trial=None) -> float:
    """Train a segmentation model and return the best monitored metric.

    Kept separate from main() so Optuna can drive a full training run in-process
    and prune it between epochs.

    Parameters
    ----------
    cfg : argparse.Namespace
        The run configuration.
    trial : optuna.Trial | None
        When given, intermediate values are reported to it and the run is
        pruned as soon as Optuna says it is not competitive.

    Returns
    -------
    float
        The best value of cfg.monitor observed during the run.
    """
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    train_loader, val_loader = build_loaders(cfg)

    model = build_segmentation_model(cfg).to(device)
    criterion = build_loss(cfg, device)
    optimizer = build_optimizer(cfg, model)
    scheduler, per_iteration = build_scheduler(cfg, optimizer, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    warmup = Warmup(optimizer, cfg.warmup_iters, enabled=not per_iteration)

    run_dir = Path(cfg.output_dir) / "seg"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.json")

    stopper = EarlyStopping(cfg.patience)
    best, global_step = float("-inf"), 0

    for epoch in range(cfg.epochs):
        train_loss, global_step = train_one_epoch_seg(
            cfg,
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            scheduler,
            per_iteration,
            warmup,
            device,
            epoch,
            global_step,
        )
        metrics = validate_seg(cfg, model, val_loader, criterion, device)
        metrics["train_loss"] = train_loss
        metrics["lr"] = optimizer.param_groups[0]["lr"]

        if scheduler is not None and not per_iteration:
            if cfg.scheduler == "plateau":
                scheduler.step(metrics[cfg.monitor])
            else:
                scheduler.step()

        score = metrics[cfg.monitor]
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"dice(kidney/tumor/cyst)="
            f"{metrics['dice_kidney']:.3f}/{metrics['dice_tumor']:.3f}/"
            f"{metrics['dice_cyst']:.3f} {cfg.monitor}={score:.4f}"
        )
        log_metrics(metrics, step=epoch)

        if score > best:
            best = score
            save_checkpoint(run_dir / "best.pt", model, cfg, epoch, metrics, task="seg")
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

    save_checkpoint(run_dir / "last.pt", model, cfg, cfg.epochs - 1, metrics, "seg")
    mlflow.log_metric("best_" + cfg.monitor, best)
    if cfg.log_model:
        mlflow.log_artifact(str(run_dir / "best.pt"))
    shutdown_loaders(train_loader, val_loader)
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
        predictions = model(images.to(device)).argmax(dim=1)
    mean = IMAGENET_MEAN if cfg.in_channels == 3 else (IMAGENET_MEAN[0],)
    std = IMAGENET_STD if cfg.in_channels == 3 else (IMAGENET_STD[0],)
    grid = seg_prediction_grid(images, targets, predictions, mean, std)
    log_image(grid, f"predictions/epoch_{epoch:03d}.png")


def main():
    """Run a single segmentation training from the command line."""
    cfg = resolve_config(build_parser())
    with start_run(cfg):
        best = run_training(cfg)
    print(f"best {cfg.monitor}: {best:.4f}")


if __name__ == "__main__":
    main()
