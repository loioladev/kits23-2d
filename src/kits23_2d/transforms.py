"""Data augmentation pipelines for the KiTS23 2D slices.

The slices are windowed CT (level 40 / width 400), so the augmentations are
chosen for that domain rather than for natural images:

- Vertical flips are excluded: the anterior/posterior orientation of an axial
  abdominal slice is anatomically fixed, and flipping it produces images the
  model will never see at test time. Horizontal flips only swap the left and
  right kidney, which is fine.
- Hue/saturation jitter is meaningless on a grayscale modality; intensity
  variation is modelled with brightness/contrast, gamma, noise, and blur, which
  stand in for reconstruction kernel and dose differences between scanners.
- Elastic and grid distortion are applied to segmentation only. They move
  pixels non-affinely, which keeps masks correct but makes bounding boxes drift
  away from the object they are supposed to enclose.

The export is not uniformly sized (most slices are 512x512 but some are
512x796), so every pipeline resizes by the longest side and pads to a square
instead of stretching the odd ones.
"""

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Intensity augmentations: (rotation degrees, scale jitter, translation 
# fraction, brightness/contrast limit, probability of each intensity op).
STRENGTHS = {
    "light": (7.0, 0.05, 0.03, 0.10, 0.2),
    "medium": (15.0, 0.10, 0.06, 0.20, 0.3),
    "heavy": (25.0, 0.20, 0.10, 0.35, 0.5),
}


def _resize_pad(size: int) -> list:
    """Build the geometry-normalizing head shared by every pipeline.

    Parameters
    ----------
    size : int
        The square side the network expects.

    Returns
    -------
    list
        The Albumentations transforms that make an image exactly size x size.
    """
    return [
        A.LongestMaxSize(max_size=size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=size,
            min_width=size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            position="center",
        ),
    ]


def _intensity(strength: str) -> list:
    """Build the grayscale intensity augmentations for a strength level.

    Parameters
    ----------
    strength : str
        One of light/medium/heavy.

    Returns
    -------
    list
        The Albumentations intensity transforms.
    """
    _, _, _, bc_limit, p = STRENGTHS[strength]
    ops = [
        A.RandomBrightnessContrast(brightness_limit=bc_limit, contrast_limit=bc_limit, p=p),
        A.RandomGamma(gamma_limit=(80, 120), p=p),
        A.GaussNoise(std_range=(0.01, 0.05), per_channel=False, p=p),
        A.GaussianBlur(blur_limit=(3, 5), p=p / 2),
    ]
    if strength == "heavy":
        ops.append(
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.12),
                hole_width_range=(0.05, 0.12),
                fill=0,
                p=0.2,
            )
        )
    return ops


def _affine(strength: str) -> A.Affine:
    """Build the shared affine transform for a strength level.

    Parameters
    ----------
    strength : str
        One of light/medium/heavy.

    Returns
    -------
    A.Affine
        The configured affine augmentation.
    """
    rotate, scale, translate, _, _ = STRENGTHS[strength]
    return A.Affine(
        rotate=(-rotate, rotate),
        scale=(1.0 - scale, 1.0 + scale),
        translate_percent=(-translate, translate),
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fill_mask=0,
        p=0.7,
    )


def build_seg_transforms(
    train: bool,
    size: int = 512,
    strength: str = "medium",
    in_channels: int = 3,
) -> A.Compose:
    """Build the segmentation pipeline for one stage.

    Parameters
    ----------
    train : bool
        True for the training pipeline (augmented), False for validation/test.
    size : int
        The square input size.
    strength : str
        One of none/light/medium/heavy. "none" disables augmentation entirely.
    in_channels : int
        Number of image channels, so the normalization statistics match.

    Returns
    -------
    A.Compose
        A pipeline producing a float tensor image and an integer mask.
    """
    mean = IMAGENET_MEAN[:in_channels] if in_channels == 3 else (IMAGENET_MEAN[0],)
    std = IMAGENET_STD[:in_channels] if in_channels == 3 else (IMAGENET_STD[0],)

    ops = _resize_pad(size)
    if train and strength != "none":
        ops += [
            A.HorizontalFlip(p=0.5),
            _affine(strength),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=40, sigma=8, approximate=True),
                    A.GridDistortion(num_steps=5, distort_limit=0.2),
                ],
                p=0.2 if strength == "light" else 0.3,
            ),
            *_intensity(strength),
        ]
    ops += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(ops)


def build_det_transforms(
    train: bool,
    size: int = 512,
    strength: str = "medium",
    min_box_size: float = 4.0,
) -> A.Compose:
    """Build the detection pipeline for one stage.

    No normalization is applied here on purpose: torchvision's detection models
    wrap a GeneralizedRCNNTransform that normalizes internally, so the model is
    fed raw [0, 1] images and would otherwise be normalized twice.

    Parameters
    ----------
    train : bool
        True for the training pipeline (augmented), False for validation/test.
    size : int
        The square input size.
    strength : str
        One of none/light/medium/heavy.
    min_box_size : float
        Boxes whose area falls below this squared, or that end up mostly
        cropped away, are dropped by Albumentations itself.

    Returns
    -------
    A.Compose
        A pipeline producing a float tensor image in [0, 1] plus its boxes.
    """
    ops = _resize_pad(size)
    if train and strength != "none":
        ops += [A.HorizontalFlip(p=0.5), _affine(strength), *_intensity(strength)]
    ops += [A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0), ToTensorV2()]
    return A.Compose(
        ops,
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            min_area=min_box_size * min_box_size,
            min_visibility=0.2,
            filter_invalid_bboxes=True,
            clip=True,
        ),
    )
