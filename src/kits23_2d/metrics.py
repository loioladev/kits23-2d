"""Evaluation metrics for both tasks.

Segmentation metrics are accumulated into a single confusion matrix over the
whole split and reduced only at the end. Averaging a per-image Dice would be
misleading here: about half of the exported slices contain no annotation at
all, and Dice on an empty image is either undefined or trivially 1.0, so a
per-image mean mostly measures how many empty slices are in the split.

The same confusion matrix also yields the KiTS hierarchical evaluation classes
(kidney and masses / masses / tumor), because merging labels into a group is
just summing the corresponding block of the matrix.
"""

import contextlib
import io

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from kits23_2d.config import CLASS_NAMES, HEC_GROUPS, NUM_CLASSES


class SegmentationMetrics:
    """Confusion-matrix accumulator for multiclass segmentation."""

    def __init__(self, num_classes: int = NUM_CLASSES, device: str = "cpu"):
        """Allocate the confusion matrix.

        Parameters
        ----------
        num_classes : int
            Number of classes including background.
        device : str
            Device the matrix is accumulated on; keeping it on the GPU avoids
            a synchronization per batch.
        """
        self.num_classes = num_classes
        self.matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64, device=device
        )

    def reset(self) -> None:
        """Zero the accumulated counts."""
        self.matrix.zero_()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted labels as [B, H, W] int64.
        target : torch.Tensor
            Ground-truth labels as [B, H, W] int64.
        """
        pred = pred.reshape(-1).to(self.matrix.device)
        target = target.reshape(-1).to(self.matrix.device)
        indices = target * self.num_classes + pred
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def _group_counts(self, group) -> tuple:
        """Reduce the matrix to binary tp/fp/fn for a group of labels.

        Parameters
        ----------
        group : Sequence[int]
            The label ids that form the positive class.

        Returns
        -------
        tp, fp, fn : tuple
            The three counts as floats.
        """
        matrix = self.matrix.double()
        mask = torch.zeros(self.num_classes, dtype=torch.bool, device=matrix.device)
        mask[list(group)] = True
        tp = matrix[mask][:, mask].sum()
        fp = matrix[~mask][:, mask].sum()
        fn = matrix[mask][:, ~mask].sum()
        return tp.item(), fp.item(), fn.item()

    def compute(self) -> dict:
        """Reduce the accumulated matrix into the reported metrics.

        A group that is absent from both prediction and ground truth has no
        defined Dice; it is reported as NaN rather than as a perfect 1.0, and
        is skipped by the mean.

        Returns
        -------
        dict
            Dice and IoU per class, the three KiTS HEC Dice scores, the mean
            Dice over the foreground classes, and the pixel accuracy.
        """
        out = {}
        foreground = []
        for label in range(1, self.num_classes):
            tp, fp, fn = self._group_counts((label,))
            name = CLASS_NAMES[label]
            dice = _safe_dice(tp, fp, fn)
            out[f"dice_{name}"] = dice
            out[f"iou_{name}"] = _safe_iou(tp, fp, fn)
            foreground.append(dice)

        for name, group in HEC_GROUPS.items():
            tp, fp, fn = self._group_counts(group)
            out[f"dice_hec_{name}"] = _safe_dice(tp, fp, fn)

        present = [d for d in foreground if not np.isnan(d)]
        out["mean_dice"] = float(np.mean(present)) if present else float("nan")
        total = self.matrix.sum().item()
        correct = self.matrix.diagonal().sum().item()
        out["pixel_accuracy"] = correct / total if total else float("nan")
        return out


def _safe_dice(tp: float, fp: float, fn: float) -> float:
    """Compute Dice, returning NaN when the class is absent everywhere.

    Parameters
    ----------
    tp : float
        True positive count.
    fp : float
        False positive count.
    fn : float
        False negative count.

    Returns
    -------
    float
        The Dice coefficient, or NaN if undefined.
    """
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else float("nan")


def _safe_iou(tp: float, fp: float, fn: float) -> float:
    """Compute IoU, returning NaN when the class is absent everywhere.

    Parameters
    ----------
    tp : float
        True positive count.
    fp : float
        False positive count.
    fn : float
        False negative count.

    Returns
    -------
    float
        The Jaccard index, or NaN if undefined.
    """
    denom = tp + fp + fn
    return tp / denom if denom else float("nan")


class DetectionMetrics:
    """Collects Faster R-CNN predictions and scores them with COCOeval."""

    def __init__(self, coco_gt: dict):
        """Wrap the ground truth the predictions will be scored against.

        Parameters
        ----------
        coco_gt : dict
            A COCO-format dict covering exactly the images being evaluated,
            as produced by CocoIndex.coco_subset.
        """
        self.coco_gt = COCO()
        self.coco_gt.dataset = coco_gt
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco_gt.createIndex()
        self.results = []

    def reset(self) -> None:
        """Drop the accumulated predictions."""
        self.results = []

    @torch.no_grad()
    def update(self, targets: list, outputs: list, scale_back=None) -> None:
        """Accumulate one batch of predictions in COCO result format.

        Parameters
        ----------
        targets : list
            The target dicts of the batch, used for their image_id.
        outputs : list
            The model outputs, each with boxes (xyxy), scores, and labels.
        scale_back : callable | None
            Optional mapping from network-space xyxy boxes back to the original
            image coordinates the ground truth is expressed in.
        """
        for target, output in zip(targets, outputs, strict=True):
            image_id = int(target["image_id"])
            boxes = output["boxes"].detach().cpu()
            if scale_back is not None and len(boxes):
                boxes = scale_back(image_id, boxes)
            scores = output["scores"].detach().cpu().tolist()
            labels = output["labels"].detach().cpu().tolist()
            for box, score, label in zip(boxes.tolist(), scores, labels, strict=True):
                x1, y1, x2, y2 = box
                self.results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )

    def compute(self) -> dict:
        """Run COCOeval over everything accumulated so far.

        Returns
        -------
        dict
            mAP, mAP@50, mAP@75, the size-stratified APs, mean recall, and the
            per-class AP. All zeros when the model predicted nothing.
        """
        if not self.results:
            out = {
                "mAP": 0.0,
                "mAP_50": 0.0,
                "mAP_75": 0.0,
                "mAP_small": 0.0,
                "mAP_medium": 0.0,
                "mAP_large": 0.0,
                "mAR_100": 0.0,
            }
            for label in range(1, NUM_CLASSES):
                out[f"AP_{CLASS_NAMES[label]}"] = 0.0
            return out

        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = self.coco_gt.loadRes(list(self.results))
            evaluator = COCOeval(self.coco_gt, coco_dt, "bbox")
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()

        stats = evaluator.stats
        out = {
            "mAP": float(stats[0]),
            "mAP_50": float(stats[1]),
            "mAP_75": float(stats[2]),
            "mAP_small": float(stats[3]),
            "mAP_medium": float(stats[4]),
            "mAP_large": float(stats[5]),
            "mAR_100": float(stats[8]),
        }
        out.update(self._per_class_ap(evaluator))
        return out

    def _per_class_ap(self, evaluator: COCOeval) -> dict:
        """Extract AP@[.5:.95] for each category from an evaluated COCOeval.

        Parameters
        ----------
        evaluator : COCOeval
            An evaluator that has already run accumulate().

        Returns
        -------
        dict
            Mapping of "AP_<class name>" -> average precision.
        """
        precision = evaluator.eval["precision"]  # [iou, recall, cat, area, maxdet]
        out = {}
        for index, category_id in enumerate(evaluator.params.catIds):
            values = precision[:, :, index, 0, 2]
            values = values[values > -1]
            name = CLASS_NAMES[category_id]
            out[f"AP_{name}"] = float(np.mean(values)) if values.size else float("nan")
        return out
