"""Model builders for the segmentation and detection tasks."""

import segmentation_models_pytorch as smp
import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from kits23_2d.config import NUM_CLASSES

SEG_ARCHS = ("unet", "unetplusplus")
DET_ARCHS = ("resnet50", "mobilenet")


def build_segmentation_model(cfg) -> torch.nn.Module:
    """Build the UNet (or UNet++) used for semantic segmentation.

    Parameters
    ----------
    cfg : argparse.Namespace
        Needs arch, encoder, encoder_weights, in_channels, and attention.

    Returns
    -------
    torch.nn.Module
        A model mapping [B, C, H, W] to [B, 4, H, W] logits.
    """
    if cfg.arch not in SEG_ARCHS:
        raise ValueError(f"unknown segmentation arch {cfg.arch!r}")
    return smp.create_model(
        arch=cfg.arch,
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights or None,
        in_channels=cfg.in_channels,
        classes=NUM_CLASSES,
        decoder_attention_type=cfg.attention or None,
    )


def build_detection_model(cfg) -> torch.nn.Module:
    """Build the Faster R-CNN used for lesion detection.

    The internal GeneralizedRCNNTransform is pinned to the dataset's square
    input size so it does not re-resize images the pipeline already shaped, and
    it keeps doing the ImageNet normalization (the detection augmentation
    pipeline deliberately leaves images in [0, 1]).

    Parameters
    ----------
    cfg : argparse.Namespace
        Needs arch, size, trainable_backbone_layers, and pretrained.

    Returns
    -------
    torch.nn.Module
        A Faster R-CNN predicting background plus the three KiTS classes.
    """
    if cfg.arch not in DET_ARCHS:
        raise ValueError(f"unknown detection arch {cfg.arch!r}")
    builder = (
        fasterrcnn_resnet50_fpn_v2
        if cfg.arch == "resnet50"
        else fasterrcnn_mobilenet_v3_large_fpn
    )
    model = builder(
        weights="DEFAULT" if cfg.pretrained else None,
        weights_backbone="DEFAULT" if cfg.pretrained else None,
        trainable_backbone_layers=(
            cfg.trainable_backbone_layers if cfg.pretrained else None
        ),
        min_size=cfg.size,
        max_size=cfg.size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build the optimizer named by the config, over the trainable parameters.

    Parameters
    ----------
    cfg : argparse.Namespace
        Needs optimizer, lr, weight_decay, and momentum.
    model : torch.nn.Module
        The model whose parameters are optimized.

    Returns
    -------
    torch.optim.Optimizer
        The configured optimizer.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(
        params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )


def build_scheduler(cfg, optimizer: torch.optim.Optimizer, steps_per_epoch: int):
    """Build the learning-rate scheduler named by the config.

    Parameters
    ----------
    cfg : argparse.Namespace
        Needs scheduler, epochs, and lr.
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    steps_per_epoch : int
        Optimizer steps per epoch, needed by the per-iteration schedulers.

    Returns
    -------
    scheduler, per_iteration : tuple
        The scheduler (or None) and whether it must be stepped every iteration
        rather than every epoch.
    """
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cfg.epochs, 1)
        ), False
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        ), False
    if cfg.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            epochs=max(cfg.epochs, 1),
            steps_per_epoch=max(steps_per_epoch, 1),
        ), True
    return None, False
