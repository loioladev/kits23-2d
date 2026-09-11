"""Report dataset metrics for the KiTS23 2D COCO dataset produced by convert.py.

For each split (train/val/test) this prints:
    - dataset size (cases, images, annotations)
    - per-class instance counts and area statistics
    - instances-per-image distribution
    - how many images have no annotated instance at all ("empty" images)

It also renders example images to disk: a handful of annotated slices per
class (mask overlay + bbox + label) and a handful of empty slices, so the
dataset can be sanity-checked visually.

Usage:
    uv run kits23_2d/stats --dataset-dir kits32-2d --output-dir stats_output
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

SPLITS = ("train", "val", "test")

# BGR colors per category id.
CATEGORY_COLORS = {
    1: (60, 180, 75),  # kidney - green
    2: (0, 0, 220),  # tumor - red
    3: (0, 165, 255),  # cyst - orange
}
DEFAULT_COLOR = (255, 255, 255)


def load_coco(annotations_dir: Path, split: str) -> dict | None:
    """Load one split's COCO annotation file, if it exists.

    Parameters
    ----------
    annotations_dir : Path
        The directory containing instances_{split}.json files.
    split : str
        The split name (train/val/test).

    Returns
    -------
    dict | None
        The parsed COCO dictionary, or None if the split's file is missing.
    """
    path = annotations_dir / f"instances_{split}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_split_stats(coco: dict) -> dict:
    """Compute size, class-distribution, and empty-image metrics for one split.

    Parameters
    ----------
    coco : dict
        A COCO-format dictionary with images, annotations, and categories.

    Returns
    -------
    dict
        Summary metrics for the split, plus indices (prefixed with "_") used
        to render example images without recomputing them.
    """
    images = coco["images"]
    annotations = coco["annotations"]
    categories = {c["id"]: c["name"] for c in coco["categories"]}

    image_id_to_anns = defaultdict(list)
    for ann in annotations:
        image_id_to_anns[ann["image_id"]].append(ann)

    empty_images = [img for img in images if img["id"] not in image_id_to_anns]

    category_counts = Counter(ann["category_id"] for ann in annotations)
    category_areas = defaultdict(list)
    for ann in annotations:
        category_areas[ann["category_id"]].append(ann["area"])

    instances_per_image = np.array(
        [len(image_id_to_anns.get(img["id"], [])) for img in images]
    )

    return {
        "num_cases": len({img["case_id"] for img in images}),
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_empty_images": len(empty_images),
        "empty_images_pct": 100.0 * len(empty_images) / len(images) if images else 0.0,
        "category_counts": {
            categories[cid]: n for cid, n in sorted(category_counts.items())
        },
        "category_area_stats": {
            categories[cid]: {
                "min": float(np.min(areas)),
                "max": float(np.max(areas)),
                "mean": float(np.mean(areas)),
                "median": float(np.median(areas)),
            }
            for cid, areas in sorted(category_areas.items())
        },
        "instances_per_image": {
            "mean": float(instances_per_image.mean()) if len(images) else 0.0,
            "max": int(instances_per_image.max()) if len(images) else 0,
        },
        "empty_image_examples": [img["file_name"] for img in empty_images[:10]],
        "_images": images,
        "_categories": categories,
        "_image_id_to_anns": image_id_to_anns,
        "_empty_images": empty_images,
    }


def draw_annotations(image: np.ndarray, anns: list, categories: dict) -> np.ndarray:
    """Overlay masks, bboxes, and class labels for a list of annotations.

    Parameters
    ----------
    image : np.ndarray
        The BGR image to draw on (not modified in place).
    anns : list
        COCO annotation dicts (with RLE segmentation and bbox) for this image.
    categories : dict
        Mapping of category_id -> category name.

    Returns
    -------
    np.ndarray
        A copy of the image with annotations drawn on top.
    """
    overlay = image.copy()
    for ann in anns:
        color = np.array(CATEGORY_COLORS.get(ann["category_id"], DEFAULT_COLOR))
        mask = mask_utils.decode(ann["segmentation"]).astype(bool)
        overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)

        x, y, w, h = (round(v) for v in ann["bbox"])
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color.tolist(), 1)

        label = categories[ann["category_id"]]
        cv2.putText(
            overlay,
            label,
            (x, max(y - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color.tolist(),
            1,
            cv2.LINE_AA,
        )
    return overlay


def save_examples(
    stats: dict, images_dir: Path, split: str, out_dir: Path, num_examples: int
) -> None:
    """Render annotated and empty example images for one split to disk.

    Parameters
    ----------
    stats : dict
        The output of compute_split_stats for this split.
    images_dir : Path
        The dataset's images/ directory (containing a subdir per split).
    split : str
        The split name (train/val/test).
    out_dir : Path
        The root output directory for the stats report.
    num_examples : int
        The number of example images to save per category, and for empty images.
    """
    categories = stats["_categories"]
    image_id_to_anns = stats["_image_id_to_anns"]
    images_by_id = {img["id"]: img for img in stats["_images"]}
    split_images_dir = images_dir / split
    out_split_dir = out_dir / "examples" / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    for cat_id, cat_name in categories.items():
        saved = 0
        for image_id, anns in image_id_to_anns.items():
            if saved >= num_examples:
                break
            cat_anns = [a for a in anns if a["category_id"] == cat_id]
            if not cat_anns:
                continue
            img_rec = images_by_id[image_id]
            image = cv2.imread(str(split_images_dir / img_rec["file_name"]))
            if image is None:
                continue
            vis = draw_annotations(image, anns, categories)
            cv2.imwrite(str(out_split_dir / f"{cat_name}_{img_rec['file_name']}"), vis)
            saved += 1

    for img_rec in stats["_empty_images"][:num_examples]:
        image = cv2.imread(str(split_images_dir / img_rec["file_name"]))
        if image is None:
            continue
        cv2.imwrite(str(out_split_dir / f"empty_{img_rec['file_name']}"), image)


def print_report(split_stats: dict) -> None:
    """Print a human-readable summary of every split's stats to stdout.

    Parameters
    ----------
    split_stats : dict
        Mapping of split name -> compute_split_stats output.
    """
    for split, stats in split_stats.items():
        print(f"\n=== {split} ===")
        print(f"cases:        {stats['num_cases']}")
        print(f"images:       {stats['num_images']}")
        print(f"annotations:  {stats['num_annotations']}")
        print(
            f"empty images: {stats['num_empty_images']} "
            f"({stats['empty_images_pct']:.1f}%)"
        )
        print(
            f"instances/image: mean={stats['instances_per_image']['mean']:.2f}, "
            f"max={stats['instances_per_image']['max']}"
        )
        print("class counts:")
        for name, count in stats["category_counts"].items():
            area = stats["category_area_stats"][name]
            print(
                f"  {name:>8}: {count:>6}  "
                f"(area min={area['min']:.0f} median={area['median']:.0f} "
                f"mean={area['mean']:.0f} max={area['max']:.0f})"
            )

    total_cases = sum(s["num_cases"] for s in split_stats.values())
    total_images = sum(s["num_images"] for s in split_stats.values())
    total_annotations = sum(s["num_annotations"] for s in split_stats.values())
    total_empty = sum(s["num_empty_images"] for s in split_stats.values())
    print("\n=== total ===")
    print(f"cases:        {total_cases}")
    print(f"images:       {total_images}")
    print(f"annotations:  {total_annotations}")
    if total_images:
        print(f"empty images: {total_empty} ({100.0 * total_empty / total_images:.1f}%)")


def parse_args() -> argparse.Namespace:
    """Create the CLI.

    Returns
    -------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("kits32-2d"))
    parser.add_argument("--output-dir", type=Path, default=Path("stats_output"))
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Example images to save per class (and for empty images), per split.",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Skip rendering example images; only print/save the metrics.",
    )
    return parser.parse_args()


def main():
    """Compute and report dataset statistics."""
    args = parse_args()
    annotations_dir = args.dataset_dir / "annotations"
    images_dir = args.dataset_dir / "images"

    split_stats = {}
    for split in SPLITS:
        coco = load_coco(annotations_dir, split)
        if coco is None:
            continue
        split_stats[split] = compute_split_stats(coco)

    if not split_stats:
        raise SystemExit(f"No instances_*.json found under {annotations_dir}")

    print_report(split_stats)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        split: {k: v for k, v in stats.items() if not k.startswith("_")}
        for split, stats in split_stats.items()
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics summary to {args.output_dir / 'summary.json'}")

    if not args.no_examples:
        for split, stats in split_stats.items():
            save_examples(stats, images_dir, split, args.output_dir, args.num_examples)
        print(f"Saved example images to {args.output_dir / 'examples'}")


if __name__ == "__main__":
    main()
