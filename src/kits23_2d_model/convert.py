"""Convert the KiTS23 dataset (3D CT volumes + per-lesion, per-annotator masks)
into a 2D COCO-format instance detection/segmentation dataset.

Each case's imaging.nii.gz is sliced along the axial axis. For every lesion
instance under instances/, the up-to-3 annotator masks are fused via majority
vote, then any axial slice touching that instance becomes a COCO annotation
(bbox + RLE segmentation). A configurable ratio of slices with no annotated
instance is also included so downstream models see negative examples.

KiTS instance masks overlap across classes: a kidney instance covers the whole
kidney *including* any tumor or cyst growing inside it, and the official
segmentation.nii.gz resolves that by painting labels in LABEL_AGGREGATION_ORDER.
The export preserves the raw instance masks, so a tumor annotation and its
parent kidney annotation cover the same pixels. Painting the exported
annotations in that same order (as coco_dataset.rasterize_semantic_mask does)
reproduces segmentation.nii.gz pixel for pixel.

Usage:
    python -m kits23.export.coco_convert --output-dir coco_dataset
"""

import argparse
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from pycocotools import mask as mask_utils
from tqdm import tqdm

KITS_LABEL_NAMES = {1: "kidney", 2: "tumor", 3: "cyst"}
INSTANCE_FNAME_RE = re.compile(
    r"^(?P<cls>kidney|tumor|cyst)_instance-(?P<instance>\d+)_annotation-(?P<annotator>\d+)\.nii\.gz$"
)

# Prefix of the hidden per-annotation correction files.
# They are not annotators in their own right.
SUPPLEMENT_PREFIX = ".suppl."

CLASS_NAME_TO_LABEL = {name: label for label, name in KITS_LABEL_NAMES.items()}

COCO_INFO = {
    "description": "KiTS23 axial slices as 2D COCO instance annotations",
    "url": "https://github.com/neheller/kits23",
}


_SITK_PIXEL_IDS = {
    np.float32: sitk.sitkFloat32,
    np.int16: sitk.sitkInt16,
    np.uint8: sitk.sitkUInt8,
}


def read_volume(path: Path, dtype=None) -> np.ndarray:
    """Read a NIfTI volume as a numpy array with axial slices as axis 0.

    KiTS23 files are not stored in a consistent orientation, so the array
    axis that is actually superior-inferior (axial) is not always axis 0.
    Reorienting to LPS first guarantees axis 0 is always the axial slice
    axis, consistently across imaging and every instance mask.

    Every KiTS23 volume is stored as float64, which is 611x512x512. Pass
    dtype to have the reader narrow it on the way out.

    Parameters
    ----------
    path : Path
        The path to the NIfTI file to read.
    dtype : type, optional
        The desired numpy dtype for the output array. If None, the original
        dtype is preserved.
    
    Returns
    -------
    np.ndarray
        A 3D numpy array with shape (num_slices, height, width) and the specified dtype.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    if dtype is not None:
        reader.SetOutputPixelType(_SITK_PIXEL_IDS[dtype])
    return sitk.GetArrayFromImage(sitk.DICOMOrient(reader.Execute(), "LPS"))


def window_slice(slice_2d: np.ndarray, level: float, width: float) -> np.ndarray:
    """Apply an HU window to one slice and rescale to uint8.

    Parameters
    ----------
    slice_2d : np.ndarray
        A 2D numpy array representing a single axial slice in Hounsfield Units (HU).
    level : float
        The HU window level to apply to the slice.
    width : float
        The HU window width to apply to the slice.

    Returns
    -------
    np.ndarray
        A 2D numpy array of the same shape as slice_2d, with values
        rescaled to the range [0, 255] and cast to uint8.
    """
    lo, hi = level - width / 2, level + width / 2
    scaled = (np.clip(slice_2d, lo, hi) - lo) / max(hi - lo, 1e-6) * 255.0
    return scaled.astype(np.uint8)


def fill_slice_interiors(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes in every axial slice of a boolean mask, in place.
    
    Parameters
    ----------
    mask : np.ndarray
        A 3D boolean numpy array representing a volumetric mask.

    Returns
    -------
    np.ndarray
        The mask with enclosed holes filled.
    """
    for z in range(mask.shape[0]):
        if not mask[z].any():
            continue
        flooded = mask[z].astype(np.uint8)
        ff_mask = np.zeros((flooded.shape[0] + 2, flooded.shape[1] + 2), np.uint8)
        cv2.floodFill(flooded, ff_mask, (0, 0), 1)
        mask[z][flooded == 0] = True
    return mask


def read_annotator_mask(path: Path) -> np.ndarray:
    """Read one annotator's instance mask as bool, applying its supplement.

    Parameters
    ----------
    path : Path
        The path to the annotator's instance mask file.

    Returns
    -------
    np.ndarray
        The annotator's instance mask as a boolean array.
    """
    mask = read_volume(path, np.uint8) > 0
    suppl_path = path.parent / (SUPPLEMENT_PREFIX + path.name)
    if not suppl_path.exists():
        return mask
    mask |= read_volume(suppl_path, np.uint8) == 2
    return fill_slice_interiors(mask)


def group_instance_files(case_dir: Path) -> dict:
    """Map (class_name, instance_id) -> list of annotator mask file paths."""
    groups = defaultdict(list)
    instances_dir = case_dir / "instances"
    if not instances_dir.is_dir():
        return groups
    for f in instances_dir.iterdir():
        m = INSTANCE_FNAME_RE.match(f.name)
        if not m:
            continue
        key = (m.group("cls"), int(m.group("instance")))
        groups[key].append(f)
    return groups


def majority_vote_instance(files: list) -> np.ndarray:
    """Create the majority-vote mask for one instance from all its annotators.

    Strictly more than half of the annotators must agree, matching the behavior of
    KiTS32 official segmentation pattern.

    Parameters
    ----------
    files : list
        A list of file paths to the annotator mask files for the instance.

    Returns
    -------
    np.ndarray
        The majority-vote mask for the instance.
    """
    votes = None
    for f in sorted(files):
        mask = read_annotator_mask(f)
        votes = mask.astype(np.uint8) if votes is None else votes + mask
    return votes >= (len(files) // 2) + 1


def build_case_instances(case_dir: Path) -> list:
    """Create the list of instances for one case.

    Only the slices an instance actually touches are kept.

    Parameters
    ----------
    case_dir : Path
        The case directory containing imaging.nii.gz and instances/.

    Returns
    -------
    list
        A list of dictionaries, each representing an instance with its class name,
        instance ID, and a mapping of slice indices to 2D masks.
    """
    instances = []
    for (cls_name, instance_id), files in sorted(
        group_instance_files(case_dir).items()
    ):
        volume = majority_vote_instance(files)
        present_z = np.where(volume.any(axis=(1, 2)))[0]
        if len(present_z) == 0:
            continue
        instances.append(
            {
                "class_name": cls_name,
                "instance_id": instance_id,
                "slices": {int(z): volume[z].copy() for z in present_z},
            }
        )
    return instances


def process_case(
    case_dir: Path,
    split_name: str,
    images_dir: Path,
    window_level: float,
    window_width: float,
    empty_ratio: float,
    seed: int,
) -> tuple:
    """Slice one case and write PNGs.

    Records reference each other by file_name (not a global id) so cases can be
    processed independently in parallel; global ids are assigned by the caller.

    Parameters
    ----------
    case_dir : Path
        The case directory containing imaging.nii.gz and instances/.
    split_name : str
        The split name (train/val/test) for this case.
    images_dir : Path
        The output directory for the PNG images.
    window_level : float
        The HU window level to apply to the imaging slices.
    window_width : float
        The HU window width to apply to the imaging slices.
    empty_ratio : float
        The number of empty images (slices) without annotations to include per annotated slice.
    seed : int
        A seed for the random number generator to ensure reproducibility.

    Returns
    -------
    case_id, image_records, annotation_records : tuple
        Returns the case directory name, a list of image records, and a list of annotation records.
    """
    case_id = case_dir.name

    # Build the list of instances
    instances = build_case_instances(case_dir)

    # Read the imaging volume and determine which slices to include
    volume = read_volume(case_dir / "imaging.nii.gz", np.float32)

    # Determine which slices to include in the output
    num_slices, height, width = volume.shape
    slice_to_instances = defaultdict(list)
    for inst in instances:
        for z in sorted(inst["slices"]):
            slice_to_instances[z].append(inst)

    # Sample empty slices based on the empty_ratio and random seed
    annotated_slices = sorted(slice_to_instances.keys())
    empty_slices = [z for z in range(num_slices) if z not in slice_to_instances]

    # Use a deterministic random generator to sample empty slices for reproducibility
    rng = random.Random(f"{seed}-{case_id}")
    n_empty = min(int(round(len(annotated_slices) * empty_ratio)), len(empty_slices))  # noqa: RUF046
    sampled_empty = set(rng.sample(empty_slices, n_empty)) if n_empty > 0 else set()

    # Combine annotated and sampled empty slices, ensuring uniqueness and sorting
    included_slices = sorted(set(annotated_slices) | sampled_empty)

    # Create variables
    out_dir = images_dir / split_name
    image_records = []
    annotation_records = []

    for z in included_slices:
        # Process image slice
        file_name = f"{case_id}_z{z:04d}.png"
        windowed = window_slice(volume[z], window_level, window_width)
        if not cv2.imwrite(str(out_dir / file_name), windowed):
            raise RuntimeError(f"cv2.imwrite failed for {out_dir / file_name}")

        image_records.append(
            {
                "file_name": file_name,
                "height": int(height),
                "width": int(width),
                "case_id": case_id,
                "slice_index": int(z),
            }
        )

        # Apply annotation to the image slice
        for inst in slice_to_instances.get(z, []):
            mask_2d = inst["slices"][z].astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(mask_2d))
            bbox = mask_utils.toBbox(rle).tolist()
            area = float(mask_utils.area(rle))
            annotation_records.append(
                {
                    "file_name": file_name,
                    "category_id": CLASS_NAME_TO_LABEL[inst["class_name"]],
                    "segmentation": {
                        "size": rle["size"],
                        "counts": rle["counts"].decode("ascii"),
                    },
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )

    return case_id, image_records, annotation_records


def discover_cases(dataset_dir: Path, wanted_cases: list, max_cases: int) -> list:
    """Get the list of case directories to process, optionally filtered by name and/or count.

    Parameters
    ----------
    dataset_dir : Path
        The root directory of the dataset containing case subdirectories.
    wanted_cases : list
        A list of case names to include. If None, all cases are included.
    max_cases : int
        The maximum number of cases to include. If None, all cases are included.

    Returns
    -------
    list
        A list of case directories to process.
    """
    all_cases = sorted(
        p
        for p in dataset_dir.iterdir()
        if p.is_dir() and p.name.startswith("case_") and (p / "imaging.nii.gz").exists()
    )
    if wanted_cases:
        wanted = set(wanted_cases)
        all_cases = [c for c in all_cases if c.name in wanted]
    if max_cases:
        all_cases = all_cases[:max_cases]
    return all_cases


def split_cases(case_ids: list, split_ratios: tuple, seed: int) -> dict:
    """Create the split of train/val/test case ids, shuffled by seed.

    Parameters
    ----------
    case_ids : list
        A list of case identifiers to split.
    split_ratios : tuple
        A tuple of three floats representing the ratios for train, val, and test splits.
    seed : int
        A seed for the random number generator to ensure reproducibility.

    Parameters
    ----------
    dict
        A dictionary with keys "train", "val", and "test" mapping to lists of case identifiers.
    """
    shuffled = case_ids[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * split_ratios[0]))  # noqa: RUF046
    n_val = int(round(n * split_ratios[1]))  # noqa: RUF046
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def build_categories() -> list:
    """Build the COCO categories list from KITS_LABEL_NAMES."""
    return [
        {"id": label, "name": name, "supercategory": "kits23"}
        for label, name in sorted(KITS_LABEL_NAMES.items())
    ]


def assemble_coco(image_records: list, annotation_records: list) -> dict:
    """Generate the COCO-format dictionary from the image and annotation records.

    Parameters
    ----------
    image_records : list
        A list of dictionaries representing image records.
    annotation_records : list
        A list of dictionaries representing annotation records.

    Returns
    -------
    dict
        A dictionary in COCO format containing info, licenses, images, annotations, and categories.
    """
    images = []
    file_name_to_id = {}
    for next_id, rec in enumerate(image_records, start=1):
        rec = dict(rec)
        rec["id"] = next_id
        file_name_to_id[rec["file_name"]] = next_id
        images.append(rec)

    annotations = []
    for next_id, rec in enumerate(annotation_records, start=1):
        rec = dict(rec)
        rec["id"] = next_id
        rec["image_id"] = file_name_to_id[rec.pop("file_name")]
        annotations.append(rec)

    return {
        "info": dict(COCO_INFO),
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": build_categories(),
    }


def parse_args() -> argparse.Namespace:
    """Create the CLI.

    Returns
    -------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # Define the arguments
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("kits32-2d"))
    parser.add_argument("--window-level", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument(
        "--empty-ratio",
        type=float,
        default=1.0,
        help="Empty slices sampled per case, as a multiple of that case's annotated slice count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel cases. Each worker holds one case in memory (~1-2GB, more for cases with many lesions).",
    )
    parser.add_argument(
        "--cases",
        type=str,
        nargs="*",
        default=None,
        help="Restrict to these case ids (e.g. case_00000 case_00001).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Restrict to the first N discovered cases (smoke testing).",
    )
    args = parser.parse_args()

    # Validate arguments
    if abs(sum(args.split) - 1.0) > 1e-6:
        parser.error("--split ratios must sum to 1.0")
    case_dirs = discover_cases(args.dataset_dir, args.cases, args.max_cases)
    if not case_dirs:
        parser.error(f"No cases with imaging.nii.gz found under {args.dataset_dir}")

    return args


def main():
    """Start the conversion process."""
    args = parse_args()

    # Get the paths of files to process
    case_dirs = discover_cases(args.dataset_dir, args.cases, args.max_cases)
    case_dir_by_id = {c.name: c for c in case_dirs}

    # Get train/val/test splits
    splits = split_cases(list(case_dir_by_id.keys()), tuple(args.split), args.seed)

    # Create the output directory in COCO format
    images_dir = args.output_dir / "images"
    annotations_dir = args.output_dir / "annotations"
    for split_name in splits:
        (images_dir / split_name).mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Build the list of jobs to process in parallel
    jobs = [
        (case_dir_by_id[case_id], split_name)
        for split_name, ids in splits.items()
        for case_id in ids
    ]

    # Process the cases in parallel, collecting the results
    per_split_images = defaultdict(list)
    per_split_annotations = defaultdict(list)
    case_id_to_split = {case_id: s for s, ids in splits.items() for case_id in ids}
    failed = []
    empty_cases = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all jobs to the executor and map futures to case names for error reporting
        futures = {
            executor.submit(
                process_case,
                case_dir,
                split_name,
                images_dir,
                args.window_level,
                args.window_width,
                args.empty_ratio,
                args.seed,
            ): case_dir.name
            for case_dir, split_name in jobs
        }
        # Wait for the futures to complete and collect results
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Converting cases"
        ):
            try:
                case_id, image_records, annotation_records = future.result()
            except Exception as exc:  # noqa: BLE001
                failed.append((futures[future], repr(exc)))
                continue
            if not image_records:
                empty_cases.append(case_id)
            split_name = case_id_to_split[case_id]
            per_split_images[split_name].extend(image_records)
            per_split_annotations[split_name].extend(annotation_records)

    for split_name in splits:
        # Sort the images and annotations by file_name to ensure consistent ordering
        images = sorted(per_split_images[split_name], key=lambda r: r["file_name"])
        annotations = sorted(
            per_split_annotations[split_name], key=lambda r: r["file_name"]
        )
        coco = assemble_coco(images, annotations)
        out_path = annotations_dir / f"instances_{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(coco, f)
        n_cases = len({rec["case_id"] for rec in coco["images"]})
        print(
            f"{split_name}: {n_cases} cases, "
            f"{len(coco['images'])} images, {len(coco['annotations'])} annotations -> {out_path}"
        )

    # Save the split.json file for reference
    with open(annotations_dir / "split.json", "w") as f:
        json.dump(splits, f, indent=2)

    # Warn about any cases that produced no images or failed to process
    if empty_cases:
        print(
            f"WARNING: {len(empty_cases)} case(s) produced no images (no annotated slices): "
            f"{', '.join(sorted(empty_cases))}"
        )

    # Warn about any cases that failed to process
    if failed:
        print(f"WARNING: {len(failed)} case(s) failed and were skipped:")
        for case_id, err in sorted(failed):
            print(f"  {case_id}: {err}")


if __name__ == "__main__":
    main()
