"""Score a trained checkpoint on a held-out split and save example overlays.

Works for both tasks: the checkpoint records which one it came from and the
configuration it was trained with, so the model is rebuilt exactly as it was.

Usage:
    uv run kits23-evaluate --task seg --checkpoint runs/seg/best.pt --split test
    uv run kits23-evaluate --task det --checkpoint runs/det/best.pt --split test
"""

import argparse
import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kits23_2d.config import CLASS_NAMES
from kits23_2d.datasets import (
    CocoIndex,
    KiTSDetDataset,
    KiTSSegDataset,
    detection_collate,
    make_box_rescaler,
)
from kits23_2d.metrics import DetectionMetrics, SegmentationMetrics
from kits23_2d.models import build_detection_model, build_segmentation_model
from kits23_2d.tracking import det_prediction_grid, seg_prediction_grid
from kits23_2d.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_det_transforms,
    build_seg_transforms,
)
from kits23_2d.utils import load_checkpoint, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    """Create the evaluation CLI.

    Returns
    -------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("seg", "det"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("kits32-2d"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_output"))
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class CheckpointConfig:
    """Adapts a checkpoint's stored config dict back into attribute access.

    Values were serialized with str() for logging, so the few the model
    builders need are coerced back to their real types here.
    """

    def __init__(self, stored: dict, overrides: dict):
        """Rebuild a config object from a checkpoint.

        Parameters
        ----------
        stored : dict
            The "config" entry written by save_checkpoint.
        overrides : dict
            Evaluation-time values that replace the training ones.
        """
        self._values = dict(stored)
        self._values.update(overrides)
        for key in ("size", "in_channels", "trainable_backbone_layers"):
            if key in self._values and self._values[key] is not None:
                self._values[key] = int(self._values[key])
        self._values["min_box_size"] = float(self._values.get("min_box_size", 4.0))
        # The weights come from the checkpoint, so never re-download pretrained
        # ones just to overwrite them a moment later.
        self._values["encoder_weights"] = None
        self._values["pretrained"] = False

    def __getattr__(self, name: str):
        """Expose config entries as attributes.

        Parameters
        ----------
        name : str
            The config key.

        Returns
        -------
        object
            The stored value.
        """
        try:
            return self._values[name]
        except KeyError as error:
            raise AttributeError(name) from error


def load_model(args, device: torch.device) -> tuple:
    """Rebuild the model described by a checkpoint and load its weights.

    Parameters
    ----------
    args : argparse.Namespace
        The evaluation arguments.
    device : torch.device
        Device to move the model to.

    Returns
    -------
    model, cfg : tuple
        The ready-to-run model and the reconstructed training config.
    """
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint["task"] != args.task:
        raise SystemExit(
            f"checkpoint was trained for task {checkpoint['task']!r}, not {args.task!r}"
        )
    cfg = CheckpointConfig(checkpoint["config"], {"device": args.device})
    build = build_segmentation_model if args.task == "seg" else build_detection_model
    model = build(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    print(
        f"loaded {args.checkpoint} (epoch {checkpoint['epoch']}, "
        f"val {checkpoint['metrics'].get('mean_dice', checkpoint['metrics'].get('mAP'))})"
    )
    return model.to(device).eval(), cfg


@torch.no_grad()
def evaluate_seg(args, model, cfg, index: CocoIndex, device) -> dict:
    """Score a segmentation checkpoint over a split.

    Parameters
    ----------
    args : argparse.Namespace
        The evaluation arguments.
    model : torch.nn.Module
        The loaded model.
    cfg : CheckpointConfig
        The training configuration from the checkpoint.
    index : CocoIndex
        The split to evaluate.
    device : torch.device
        Device to run on.

    Returns
    -------
    dict
        The segmentation metrics.
    """
    dataset = KiTSSegDataset(
        index,
        build_seg_transforms(False, cfg.size, "none", cfg.in_channels),
        cfg.in_channels,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    metrics = SegmentationMetrics(device=device)
    saved = False

    for images, targets in tqdm(loader, desc=f"eval seg [{args.split}]"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            logits = model(images)
        predictions = logits.argmax(dim=1)
        metrics.update(predictions, targets)

        if not saved and args.num_examples:
            mean = IMAGENET_MEAN if cfg.in_channels == 3 else (IMAGENET_MEAN[0],)
            std = IMAGENET_STD if cfg.in_channels == 3 else (IMAGENET_STD[0],)
            grid = seg_prediction_grid(
                images[: args.num_examples].cpu(),
                targets[: args.num_examples].cpu(),
                predictions[: args.num_examples].cpu(),
                mean,
                std,
            )
            _save_image(grid, args.output_dir / f"seg_{args.split}_examples.png")
            saved = True

    return metrics.compute()


@torch.no_grad()
def evaluate_det(args, model, cfg, index: CocoIndex, device) -> dict:
    """Score a detection checkpoint over a split.

    Parameters
    ----------
    args : argparse.Namespace
        The evaluation arguments.
    model : torch.nn.Module
        The loaded model.
    cfg : CheckpointConfig
        The training configuration from the checkpoint.
    index : CocoIndex
        The split to evaluate.
    device : torch.device
        Device to run on.

    Returns
    -------
    dict
        The COCO detection metrics.
    """
    dataset = KiTSDetDataset(
        index,
        build_det_transforms(False, cfg.size, "none", cfg.min_box_size),
        cfg.min_box_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=detection_collate,
    )
    metrics = DetectionMetrics(index.coco_subset())
    scale_back = make_box_rescaler(index, cfg.size)
    saved = False

    for images, targets in tqdm(loader, desc=f"eval det [{args.split}]"):
        device_images = [image.to(device, non_blocking=True) for image in images]
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            outputs = model(device_images)
        outputs = [{k: v.float() for k, v in o.items()} for o in outputs]
        metrics.update(targets, outputs, scale_back=scale_back)

        if not saved and args.num_examples:
            grid = det_prediction_grid(
                images[: args.num_examples],
                targets[: args.num_examples],
                outputs[: args.num_examples],
                args.score_threshold,
            )
            _save_image(grid, args.output_dir / f"det_{args.split}_examples.png")
            saved = True

    return metrics.compute()


def _save_image(image, path: Path) -> None:
    """Write a BGR image to disk, creating the directory if needed.

    Parameters
    ----------
    image : np.ndarray
        The BGR image.
    path : Path
        Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def print_report(task: str, metrics: dict) -> None:
    """Print the metrics in the same style as the dataset report.

    Parameters
    ----------
    task : str
        Either "seg" or "det".
    metrics : dict
        The computed metrics.
    """
    print(f"\n=== {task} ===")
    if task == "seg":
        for name in CLASS_NAMES[1:]:
            print(
                f"  {name:>8}: dice={metrics[f'dice_{name}']:.4f} "
                f"iou={metrics[f'iou_{name}']:.4f}"
            )
        print("  KiTS hierarchical evaluation classes:")
        for key, value in metrics.items():
            if key.startswith("dice_hec_"):
                print(f"    {key.removeprefix('dice_hec_'):>18}: {value:.4f}")
        print(f"  mean dice: {metrics['mean_dice']:.4f}")
        print(f"  pixel accuracy: {metrics['pixel_accuracy']:.4f}")
    else:
        print(f"  mAP:    {metrics['mAP']:.4f}")
        print(f"  mAP@50: {metrics['mAP_50']:.4f}")
        print(f"  mAP@75: {metrics['mAP_75']:.4f}")
        print(f"  mAR@100:{metrics['mAR_100']:.4f}")
        for name in CLASS_NAMES[1:]:
            print(f"  {name:>8}: AP={metrics[f'AP_{name}']:.4f}")


def main():
    """Evaluate a checkpoint and write its metrics to disk."""
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, cfg = load_model(args, device)

    index = CocoIndex(
        args.dataset_dir, args.split, max_images=args.max_images, seed=args.seed
    )
    print(f"{args.split}: {len(index)} images")

    evaluator = evaluate_seg if args.task == "seg" else evaluate_det
    metrics = evaluator(args, model, cfg, index, device)
    print_report(args.task, metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.task}_{args.split}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
