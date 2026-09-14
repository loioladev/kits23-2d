"""Training and validation loops shared by the segmentation and detection runs.

Both tasks use the same skeleton: an optional linear warmup, gradient
accumulation, optional AMP, and an epoch-level metric object. Keeping the loops
here means train_seg.py and train_det.py only have to wire up a model, a loader,
and a criterion.

A note on AMP: the target GPU is a GTX 1080 (Pascal), which has no tensor cores,
so float16 autocast buys memory headroom rather than throughput. It is still on
by default because the memory is what limits the batch size at 512x512.
"""

import math

import torch
from tqdm import tqdm

from kits23_2d.metrics import DetectionMetrics, SegmentationMetrics


class Warmup:
    """Linear learning-rate warmup over the first iterations of training.

    Faster R-CNN in particular diverges easily in the first few hundred
    iterations when the box head is freshly initialized.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, iters: int, enabled: bool):
        """Capture the base learning rates the warmup ramps up to.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose param groups are scaled.
        iters : int
            Number of iterations to ramp over; 0 disables the warmup.
        enabled : bool
            False disables the warmup (used when the scheduler already ramps,
            as OneCycleLR does).
        """
        self.optimizer = optimizer
        self.iters = iters if enabled else 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.done = self.iters <= 0

    def step(self, global_step: int) -> None:
        """Set the learning rate for the upcoming iteration.

        Parameters
        ----------
        global_step : int
            The number of optimizer steps taken so far in the whole run.
        """
        if self.done:
            return
        if global_step >= self.iters:
            for group, base in zip(self.optimizer.param_groups, self.base_lrs,
                                   strict=True):
                group["lr"] = base
            self.done = True
            return
        factor = (global_step + 1) / self.iters
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * factor


def _optimizer_step(cfg, optimizer, scaler, model, scheduler, per_iteration) -> None:
    """Unscale, clip, step the optimizer, and advance a per-iteration scheduler.

    Parameters
    ----------
    cfg : argparse.Namespace
        Needs clip_grad.
    optimizer : torch.optim.Optimizer
        The optimizer to step.
    scaler : torch.amp.GradScaler
        The AMP scaler (a disabled one is a no-op).
    model : torch.nn.Module
        The model whose gradients are clipped.
    scheduler : object | None
        The learning-rate scheduler.
    per_iteration : bool
        Whether the scheduler advances every optimizer step.
    """
    if cfg.clip_grad:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if scheduler is not None and per_iteration:
        scheduler.step()


def train_one_epoch_seg(
    cfg, model, loader, criterion, optimizer, scaler, scheduler, per_iteration,
    warmup, device, epoch, global_step,
) -> tuple:
    """Run one training epoch of the segmentation model.

    Parameters
    ----------
    cfg : argparse.Namespace
        The run configuration.
    model : torch.nn.Module
        The segmentation model.
    loader : torch.utils.data.DataLoader
        The training loader.
    criterion : torch.nn.Module
        The loss, called as ``criterion(logits, target)``.
    optimizer : torch.optim.Optimizer
        The optimizer.
    scaler : torch.amp.GradScaler
        The AMP scaler.
    scheduler : object | None
        The learning-rate scheduler.
    per_iteration : bool
        Whether the scheduler advances every optimizer step.
    warmup : Warmup
        The warmup helper.
    device : torch.device
        The device to run on.
    epoch : int
        The current epoch, for the progress bar.
    global_step : int
        Optimizer steps taken so far.

    Returns
    -------
    mean_loss, global_step : tuple
        The mean training loss and the updated step counter.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, count = 0.0, 0
    bar = tqdm(loader, desc=f"train seg [{epoch}]", leave=False)

    for i, (images, targets) in enumerate(bar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss at epoch {epoch} step {i}")

        scaler.scale(loss / cfg.accum_steps).backward()
        if (i + 1) % cfg.accum_steps == 0:
            warmup.step(global_step)
            _optimizer_step(cfg, optimizer, scaler, model, scheduler, per_iteration)
            global_step += 1

        total += loss.item() * images.size(0)
        count += images.size(0)
        bar.set_postfix(loss=f"{total / count:.4f}")

    return total / max(count, 1), global_step


@torch.no_grad()
def validate_seg(cfg, model, loader, criterion, device) -> dict:
    """Evaluate the segmentation model over a whole split.

    Parameters
    ----------
    cfg : argparse.Namespace
        The run configuration.
    model : torch.nn.Module
        The segmentation model.
    loader : torch.utils.data.DataLoader
        The validation loader.
    criterion : torch.nn.Module
        The loss, for reporting a validation loss alongside the metrics.
    device : torch.device
        The device to run on.

    Returns
    -------
    dict
        The metrics from SegmentationMetrics plus val_loss.
    """
    model.eval()
    metrics = SegmentationMetrics(device=device)
    total, count = 0.0, 0

    for images, targets in tqdm(loader, desc="val seg", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            logits = model(images)
            loss = criterion(logits, targets)
        metrics.update(logits.argmax(dim=1), targets)
        total += loss.item() * images.size(0)
        count += images.size(0)

    out = metrics.compute()
    out["val_loss"] = total / max(count, 1)
    return out


def train_one_epoch_det(
    cfg, model, loader, optimizer, scaler, scheduler, per_iteration,
    warmup, device, epoch, global_step,
) -> tuple:
    """Run one training epoch of the detection model.

    torchvision's detection models return a dict of losses when called in train
    mode with targets, so there is no external criterion here.

    Parameters
    ----------
    cfg : argparse.Namespace
        The run configuration.
    model : torch.nn.Module
        The Faster R-CNN.
    loader : torch.utils.data.DataLoader
        The training loader.
    optimizer : torch.optim.Optimizer
        The optimizer.
    scaler : torch.amp.GradScaler
        The AMP scaler.
    scheduler : object | None
        The learning-rate scheduler.
    per_iteration : bool
        Whether the scheduler advances every optimizer step.
    warmup : Warmup
        The warmup helper.
    device : torch.device
        The device to run on.
    epoch : int
        The current epoch, for the progress bar.
    global_step : int
        Optimizer steps taken so far.

    Returns
    -------
    losses, global_step : tuple
        A dict with the mean total loss and each individual loss term, and the
        updated step counter.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums, count = {}, 0
    bar = tqdm(loader, desc=f"train det [{epoch}]", leave=False)

    for i, (images, targets) in enumerate(bar):
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [
            {k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets
        ]

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss at epoch {epoch} step {i}")

        scaler.scale(loss / cfg.accum_steps).backward()
        if (i + 1) % cfg.accum_steps == 0:
            warmup.step(global_step)
            _optimizer_step(cfg, optimizer, scaler, model, scheduler, per_iteration)
            global_step += 1

        count += 1
        sums["train_loss"] = sums.get("train_loss", 0.0) + loss.item()
        for name, value in loss_dict.items():
            sums[name] = sums.get(name, 0.0) + value.item()
        bar.set_postfix(loss=f"{sums['train_loss'] / count:.4f}")

    return {k: v / max(count, 1) for k, v in sums.items()}, global_step


@torch.no_grad()
def validate_det(cfg, model, loader, device, coco_gt, scale_back=None) -> dict:
    """Evaluate the detection model over a whole split with COCOeval.

    Parameters
    ----------
    cfg : argparse.Namespace
        The run configuration.
    model : torch.nn.Module
        The Faster R-CNN.
    loader : torch.utils.data.DataLoader
        The validation loader.
    device : torch.device
        The device to run on.
    coco_gt : dict
        Ground truth in COCO format, covering exactly the evaluated images.
    scale_back : callable | None
        Maps predicted boxes back to original image coordinates.

    Returns
    -------
    dict
        The metrics from DetectionMetrics.
    """
    model.eval()
    metrics = DetectionMetrics(coco_gt)

    for images, targets in tqdm(loader, desc="val det", leave=False):
        images = [image.to(device, non_blocking=True) for image in images]
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            outputs = model(images)
        outputs = [{k: v.float() for k, v in o.items()} for o in outputs]
        metrics.update(targets, outputs, scale_back=scale_back)

    return metrics.compute()
