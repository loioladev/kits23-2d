"""Small helpers shared by the training and evaluation entry points."""

import random
from pathlib import Path

import numpy as np
import torch

from kits23_2d.config import config_to_dict


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch, and make cuDNN pick fast algorithms.

    cudnn.benchmark is enabled rather than deterministic mode: every batch has
    the same shape here, so autotuning pays off, and exact bitwise
    reproducibility is not worth the slowdown on a single consumer GPU.

    Parameters
    ----------
    seed : int
        The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(name: str) -> torch.device:
    """Turn a device name into a torch.device, falling back to CPU.

    Parameters
    ----------
    name : str
        The requested device, e.g. "cuda" or "cpu".

    Returns
    -------
    torch.device
        The device to run on.
    """
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


class EarlyStopping:
    """Stops a run once the monitored metric has not improved for a while."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        """Configure the stopper.

        Parameters
        ----------
        patience : int
            Epochs without improvement before stopping; 0 disables it.
        min_delta : float
            Minimum increase that counts as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        """Record one epoch's score.

        Parameters
        ----------
        score : float
            The monitored metric, where higher is better.

        Returns
        -------
        bool
            True when training should stop.
        """
        if self.patience <= 0:
            return False
        if score > self.best + self.min_delta:
            self.best = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def save_checkpoint(
    path: Path, model: torch.nn.Module, cfg, epoch: int, metrics: dict, task: str
) -> None:
    """Write a checkpoint carrying everything needed to rebuild the model.

    Parameters
    ----------
    path : Path
        Destination file.
    model : torch.nn.Module
        The model whose weights are saved.
    cfg : argparse.Namespace
        The run configuration, stored so evaluate.py can rebuild the model.
    epoch : int
        The epoch the weights come from.
    metrics : dict
        The validation metrics at that epoch.
    task : str
        Either "seg" or "det".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": task,
            "epoch": epoch,
            "metrics": metrics,
            "config": config_to_dict(cfg),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path) -> dict:
    """Read a checkpoint written by save_checkpoint.

    Parameters
    ----------
    path : Path
        The checkpoint file.

    Returns
    -------
    dict
        The checkpoint contents.
    """
    return torch.load(path, map_location="cpu", weights_only=False)
