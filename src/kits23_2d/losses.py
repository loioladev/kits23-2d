"""Segmentation losses, composed from the ones shipped with smp."""

import torch
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, TverskyLoss

from kits23_2d.config import NUM_CLASSES

LOSS_NAMES = ("ce", "dice", "dicece", "focaldice", "tversky")


class CombinedLoss(torch.nn.Module):
    """A weighted sum of a region loss and a pixel-wise loss."""

    def __init__(self, region, pixel, region_weight: float, pixel_weight: float):
        """Store the two terms and their weights.

        Parameters
        ----------
        region : torch.nn.Module | None
            A region-overlap loss such as Dice or Tversky.
        pixel : torch.nn.Module | None
            A per-pixel loss such as cross-entropy or focal.
        region_weight : float
            Weight applied to the region term.
        pixel_weight : float
            Weight applied to the pixel term.
        """
        super().__init__()
        self.region = region
        self.pixel = pixel
        self.region_weight = region_weight
        self.pixel_weight = pixel_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the weighted loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model outputs as [B, C, H, W].
        target : torch.Tensor
            Ground-truth label map as [B, H, W] int64.

        Returns
        -------
        torch.Tensor
            The scalar loss.
        """
        total = logits.new_zeros(())
        if self.region is not None and self.region_weight:
            total = total + self.region_weight * self.region(logits, target)
        if self.pixel is not None and self.pixel_weight:
            total = total + self.pixel_weight * self.pixel(logits, target)
        return total


def build_loss(cfg, device: torch.device) -> torch.nn.Module:
    """Build the segmentation criterion named by the config.

    Parameters
    ----------
    cfg : argparse.Namespace
        Needs loss, dice_weight, ce_weight, and class_weights.
    device : torch.device
        Device the class-weight tensor must live on.

    Returns
    -------
    torch.nn.Module
        The criterion, called as ``loss(logits, target)``.
    """
    if cfg.loss not in LOSS_NAMES:
        raise ValueError(f"unknown loss {cfg.loss!r}")

    weights = None
    if cfg.class_weights:
        if len(cfg.class_weights) != NUM_CLASSES:
            raise ValueError(f"--class-weights needs {NUM_CLASSES} values")
        weights = torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)

    cross_entropy = torch.nn.CrossEntropyLoss(weight=weights)
    dice = DiceLoss(mode="multiclass", from_logits=True)

    if cfg.loss == "ce":
        return CombinedLoss(None, cross_entropy, 0.0, 1.0)
    if cfg.loss == "dice":
        return CombinedLoss(dice, None, 1.0, 0.0)
    if cfg.loss == "dicece":
        return CombinedLoss(dice, cross_entropy, cfg.dice_weight, cfg.ce_weight)
    if cfg.loss == "focaldice":
        focal = FocalLoss(mode="multiclass", gamma=2.0)
        return CombinedLoss(dice, focal, cfg.dice_weight, cfg.ce_weight)
    tversky = TverskyLoss(mode="multiclass", from_logits=True, alpha=0.3, beta=0.7)
    return CombinedLoss(tversky, cross_entropy, cfg.dice_weight, cfg.ce_weight)
