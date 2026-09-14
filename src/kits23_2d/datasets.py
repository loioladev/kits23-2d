"""PyTorch datasets over the 2D COCO export produced by convert.py.

Two views of the same files are provided:

- KiTSSegDataset yields (image, label_map) for semantic segmentation. The COCO
  RLE masks are decoded on the fly and composited into a single label map.
  KiTS instance masks overlap (a kidney annotation covers the tumor growing
  inside it), so masks are painted in ascending category order, which leaves
  tumor (2) and cyst (3) on top of kidney (1).
- KiTSDetDataset yields (image, target) in the format torchvision's detection
  models expect: boxes in xyxy, labels as int64, empty images allowed.

Roughly half the exported slices carry no annotation at all, so both datasets
can subsample the empty ones through ``empty_ratio``.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

# Reading PNGs in worker processes is already parallel; OpenCV's own thread pool
# only adds contention on top of it.
cv2.setNumThreads(0)


class CocoIndex:
    """An in-memory index of one split's COCO annotation file.

    Attributes
    ----------
    images : list
        The (possibly subsampled) image records, sorted by file name.
    image_id_to_anns : dict
        Mapping of image id -> list of annotation records.
    categories : dict
        Mapping of category id -> category name.
    images_dir : Path
        Directory holding this split's PNG files.
    """

    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        empty_ratio: float = 1.0,
        max_images: int | None = None,
        seed: int = 42,
    ):
        """Load and optionally subsample one split.

        Parameters
        ----------
        dataset_dir : Path
            The dataset root produced by convert.py.
        split : str
            The split name (train/val/test).
        empty_ratio : float
            Fraction of annotation-free images to keep. 1.0 keeps all of them.
        max_images : int | None
            Cap on the number of images, applied after the empty subsampling.
            Useful for smoke tests and for Optuna trials.
        seed : int
            Seed for the subsampling RNG, so a run is reproducible.
        """
        path = dataset_dir / "annotations" / f"instances_{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing annotation file: {path}")
        with open(path) as f:
            coco = json.load(f)

        self.images_dir = dataset_dir / "images" / split
        self.categories = {c["id"]: c["name"] for c in coco["categories"]}
        self.raw_categories = coco["categories"]

        self.image_id_to_anns = {}
        for ann in coco["annotations"]:
            self.image_id_to_anns.setdefault(ann["image_id"], []).append(ann)

        images = sorted(coco["images"], key=lambda r: r["file_name"])
        self.num_images_total = len(images)
        images = self._subsample_empty(images, empty_ratio, seed)
        if max_images is not None and max_images < len(images):
            rng = random.Random(seed)
            images = sorted(rng.sample(images, max_images), key=lambda r: r["file_name"])
        self.images = images

    def _subsample_empty(self, images: list, empty_ratio: float, seed: int) -> list:
        """Drop a fraction of the images that have no annotation.

        Parameters
        ----------
        images : list
            All image records of the split.
        empty_ratio : float
            Fraction of empty images to keep.
        seed : int
            Seed for the sampling RNG.

        Returns
        -------
        list
            The kept image records, still sorted by file name.
        """
        if empty_ratio >= 1.0:
            return images
        annotated = [r for r in images if self.image_id_to_anns.get(r["id"])]
        empty = [r for r in images if not self.image_id_to_anns.get(r["id"])]
        keep = round(len(empty) * max(empty_ratio, 0.0))
        rng = random.Random(seed)
        kept_empty = rng.sample(empty, keep) if keep else []
        return sorted(annotated + kept_empty, key=lambda r: r["file_name"])

    def coco_subset(self) -> dict:
        """Build a COCO dict restricted to the indexed images.

        COCOeval must score exactly the images that were actually run, so the
        ground truth handed to it has to follow the same subsampling.

        Returns
        -------
        dict
            A COCO-format dictionary with images, annotations, and categories.
        """
        image_ids = {r["id"] for r in self.images}
        annotations = [
            ann
            for image_id in image_ids
            for ann in self.image_id_to_anns.get(image_id, [])
        ]
        return {
            "images": self.images,
            "annotations": annotations,
            "categories": self.raw_categories,
        }

    def __len__(self) -> int:
        """Return the number of indexed images."""
        return len(self.images)


def read_image(path: Path) -> np.ndarray:
    """Read one exported slice as a single-channel uint8 array.

    Parameters
    ----------
    path : Path
        The PNG file to read.

    Returns
    -------
    np.ndarray
        The image as [H, W] uint8.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"cv2.imread failed for {path}")
    return image


def build_label_map(anns: list, height: int, width: int) -> np.ndarray:
    """Composite COCO RLE instance masks into a single semantic label map.

    Annotations are painted in ascending category order so that tumor (2) and
    cyst (3) overwrite the kidney (1) pixels they overlap, which is the label
    convention KiTS scores against.

    Parameters
    ----------
    anns : list
        The COCO annotations of one image.
    height : int
        Image height.
    width : int
        Image width.

    Returns
    -------
    np.ndarray
        The label map as [H, W] uint8 with values in {0, 1, 2, 3}.
    """
    label = np.zeros((height, width), dtype=np.uint8)
    for ann in sorted(anns, key=lambda a: a["category_id"]):
        mask = mask_utils.decode(ann["segmentation"]).astype(bool)
        label[mask] = ann["category_id"]
    return label


class KiTSSegDataset(Dataset):
    """Semantic segmentation view of the 2D KiTS export."""

    def __init__(self, index: CocoIndex, transform, in_channels: int = 3):
        """Wrap a CocoIndex with an Albumentations pipeline.

        Parameters
        ----------
        index : CocoIndex
            The loaded split.
        transform : albumentations.Compose
            A pipeline ending in ToTensorV2, applied to image and mask.
        in_channels : int
            1 to keep the slice grayscale, 3 to replicate it (which matches the
            ImageNet statistics the smp encoders were pretrained with).
        """
        self.index = index
        self.transform = transform
        self.in_channels = in_channels

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.index.images)

    def __getitem__(self, i: int):
        """Load one slice and its label map.

        Parameters
        ----------
        i : int
            The index into the split.

        Returns
        -------
        image, target : tuple
            The image as [C, H, W] float32 and the label map as [H, W] int64.
        """
        record = self.index.images[i]
        image = read_image(self.index.images_dir / record["file_name"])
        anns = self.index.image_id_to_anns.get(record["id"], [])
        label = build_label_map(anns, image.shape[0], image.shape[1])

        image = np.repeat(image[:, :, None], self.in_channels, axis=2)
        out = self.transform(image=image, mask=label)
        return out["image"], out["mask"].long()


class KiTSDetDataset(Dataset):
    """Detection view of the 2D KiTS export, in torchvision target format."""

    def __init__(self, index: CocoIndex, transform, min_box_size: float = 4.0):
        """Wrap a CocoIndex with an Albumentations bbox pipeline.

        Parameters
        ----------
        index : CocoIndex
            The loaded split.
        transform : albumentations.Compose
            A pipeline with bbox_params in "coco" format, ending in ToTensorV2.
        min_box_size : float
            Boxes with a side shorter than this are dropped. The export contains
            annotations as small as one pixel, and a degenerate box makes the
            torchvision loss produce NaNs.
        """
        self.index = index
        self.transform = transform
        self.min_box_size = min_box_size

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.index.images)

    def __getitem__(self, i: int):
        """Load one slice and its boxes.

        Parameters
        ----------
        i : int
            The index into the split.

        Returns
        -------
        image, target : tuple
            The image as [3, H, W] float32 in [0, 1], and a dict with boxes
            (xyxy float32), labels (int64), image_id, area, and iscrowd.
        """
        record = self.index.images[i]
        image = read_image(self.index.images_dir / record["file_name"])
        image = np.repeat(image[:, :, None], 3, axis=2)

        boxes, labels = [], []
        for ann in self.index.image_id_to_anns.get(record["id"], []):
            x, y, w, h = ann["bbox"]
            if w < self.min_box_size or h < self.min_box_size:
                continue
            boxes.append([x, y, w, h])
            labels.append(ann["category_id"])

        out = self.transform(image=image, bboxes=boxes, labels=labels)
        image = out["image"]
        boxes = self._to_xyxy(out["bboxes"])
        labels = torch.as_tensor(list(out["labels"]), dtype=torch.int64)

        keep = self._keep_mask(boxes)
        boxes, labels = boxes[keep], labels[keep]

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(record["id"], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return image, target

    def _to_xyxy(self, bboxes) -> torch.Tensor:
        """Convert Albumentations' COCO-format output to torchvision's xyxy.

        Parameters
        ----------
        bboxes : Sequence
            Boxes as (x, y, w, h), possibly empty.

        Returns
        -------
        torch.Tensor
            Boxes as [N, 4] float32 in (x1, y1, x2, y2).
        """
        if len(bboxes) == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        arr = np.asarray([b[:4] for b in bboxes], dtype=np.float32)
        arr[:, 2] += arr[:, 0]
        arr[:, 3] += arr[:, 1]
        return torch.from_numpy(arr)

    def _keep_mask(self, boxes: torch.Tensor) -> torch.Tensor:
        """Flag the boxes that are still large enough after augmentation.

        Parameters
        ----------
        boxes : torch.Tensor
            Boxes as [N, 4] in xyxy.

        Returns
        -------
        torch.Tensor
            A boolean mask of the boxes to keep.
        """
        if len(boxes) == 0:
            return torch.zeros((0,), dtype=torch.bool)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        return (widths >= self.min_box_size) & (heights >= self.min_box_size)


def detection_collate(batch):
    """Collate detection samples without stacking the variable-length targets.

    Parameters
    ----------
    batch : list
        A list of (image, target) pairs.

    Returns
    -------
    images, targets : tuple
        A list of image tensors and a list of target dicts.
    """
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def undo_resize_pad(
    boxes: torch.Tensor, orig_height: int, orig_width: int, size: int
) -> torch.Tensor:
    """Map boxes from the padded network input back to original coordinates.

    The transforms resize by the longest side and centre-pad to a square, so
    predictions come out in that square's frame while the COCO ground truth is
    in the original frame. Most slices are already 512x512 (making this the
    identity), but the export also contains 512x796, 512x651, and 512x632
    images, and COCOeval would score those against shifted boxes.

    Parameters
    ----------
    boxes : torch.Tensor
        Boxes as [N, 4] xyxy in the padded square.
    orig_height : int
        Height of the original image.
    orig_width : int
        Width of the original image.
    size : int
        Side of the padded square the model saw.

    Returns
    -------
    torch.Tensor
        Boxes as [N, 4] xyxy in the original image's coordinates.
    """
    scale = size / max(orig_height, orig_width)
    new_height = round(orig_height * scale)
    new_width = round(orig_width * scale)
    pad_top = (size - new_height) // 2
    pad_left = (size - new_width) // 2

    out = boxes.clone().float()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_left) * (orig_width / new_width)
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_top) * (orig_height / new_height)
    return out


def make_box_rescaler(index: CocoIndex, size: int):
    """Build the scale_back callable DetectionMetrics expects.

    Parameters
    ----------
    index : CocoIndex
        The split being evaluated, used for each image's original size.
    size : int
        Side of the padded square the model saw.

    Returns
    -------
    callable
        A function mapping (image_id, boxes) to boxes in original coordinates.
    """
    shapes = {r["id"]: (r["height"], r["width"]) for r in index.images}

    def scale_back(image_id: int, boxes: torch.Tensor) -> torch.Tensor:
        height, width = shapes[image_id]
        return undo_resize_pad(boxes, height, width, size)

    return scale_back
