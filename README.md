# KiTS23 2D models

Detection and segmentation models for the [KiTS23](https://github.com/neheller/kits23)
challenge, trained on 2D axial slices instead of full 3D volumes.

## Setup

```bash
uv sync
```

> **GPU note.** This project pins `torch`/`torchvision` to the **CUDA 12.6**
> index in `pyproject.toml`. The default PyPI wheels are built against CUDA
> 13.x, which dropped support for Pascal cards such as the GTX 1080 (sm_61);


## 1. Download KiTS23

The raw data comes from the official
[neheller/kits23](https://github.com/neheller/kits23) repository. The
annotations (`segmentation.nii.gz` and the per-annotator `instances/` masks)
are versioned inside that repository, while the CT volumes (`imaging.nii.gz`)
are fetched by its download script.

```bash
# From the root of this repository
git clone https://github.com/neheller/kits23
cd kits23

# The kits23 package has its own dependencies, so give it its own venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Download imaging.nii.gz for all 489 cases into kits23/dataset/
kits23_download_data

deactivate
cd ..

# kits23-convert reads ./dataset by default
ln -s kits23/dataset dataset
```


## 2. Build the 2D dataset

Converts each case's `imaging.nii.gz` into windowed axial PNGs and fuses the
per-annotator instance masks into COCO annotations (bbox + RLE).

```bash
uv run kits23-convert --dataset-dir dataset --output-dir kits23-2d
```

The defaults are the ones used for the reported results: abdominal window
(level 40, width 400 HU), a 70/15/15 case split with `--seed 42`, and every
empty slice kept (`--empty-ratio 1.0`).

```
kits23-2d/
├── images/{train,val,test}/case_XXXXX_zNNNN.png
└── annotations/instances_{train,val,test}.json, split.json
```

Splits are drawn at the **case** level, so no patient appears in two splits.
With the defaults this produces:

| Split | Cases | Slices | Annotations |
| --- | ---: | ---: | ---: |
| train | 342 | 43,815 | 50,800 |
| val | 73 | 8,359 | 9,805 |
| test | 74 | 11,815 | 13,550 |

## 3. Inspect the dataset

```bash
uv run kits23-stats --dataset-dir kits23-2d --output-dir stats_output
```

Things worth knowing about the data, because they shape the training code:

- **Classes overlap.** A kidney annotation covers the whole kidney *including*
  the tumor or cyst growing inside it. The segmentation target paints masks in
  ascending category order, so lesions end up on top of the kidney.
- **About half the slices are empty** (48% of the training split have no
  annotation). Metrics are aggregated over the whole split rather than averaged
  per image, and `--empty-ratio` subsamples them.
- **Annotations can be tiny** (cysts down to 1 px). Detection drops boxes
  smaller than `--min-box-size`, since degenerate boxes make the torchvision
  loss produce NaNs.
- **Sizes are not uniform** (mostly 512x512, but also 512x796/651/632). Every
  pipeline resizes by the longest side and pads to a square rather than
  stretching.

## 4. Search hyperparameters with Optuna

```bash
uv run kits23-tune-seg --config configs/tune_seg.yaml
uv run kits23-tune-det --config configs/tune_det.yaml
```

## 5. Train

Segmentation — a UNet from `segmentation_models_pytorch` predicting a 4-class
label map (background, kidney, tumor, cyst):

```bash
uv run kits23-train-seg --config configs/seg_unet.yaml
```

Detection - Faster R-CNN from torchvision over the same slices:

```bash
uv run kits23-train-det --config configs/det_frcnn.yaml
```

## 6. Track runs with MLflow

Every run logs its parameters, per-epoch metrics, and a ground-truth vs
prediction image grid.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```


## 7. Evaluate on the test split

```bash
uv run kits23-evaluate --task seg --checkpoint runs/seg-unet-resnet34/seg/best.pt --split test
uv run kits23-evaluate --task det --checkpoint runs/det-frcnn-r50/det/best.pt --split test
```

Segmentation reports Dice and IoU per class plus the KiTS hierarchical
evaluation classes (kidney and masses / masses / tumor). Detection reports COCO
mAP via `pycocotools`. Both write a `metrics.json` and an example overlay image
to `--output-dir`.

## Reproducing the reported results

The full sequence of commands behind the checkpoints and metrics below, run on
a single GTX 1080 (8 GB). The training configs in `configs/` already contain
the hyperparameters the Optuna searches selected, so steps 4–5 can be skipped
if you only want to retrain the final models.

```bash
# 1. Environment
uv sync

# 2. Raw data (see section 0)
git clone https://github.com/neheller/kits23
(cd kits23 && python3 -m venv .venv && .venv/bin/pip install -e . && .venv/bin/kits23_download_data)
ln -s kits23/dataset dataset

# 3. 2D dataset + report
uv run kits23-convert --dataset-dir dataset --output-dir kits23-2d
uv run kits23-stats --dataset-dir kits23-2d --output-dir stats_output

# 4. Hyperparameter search (optional; resumes from optuna-*.db)
uv run kits23-tune-seg --config configs/tune_seg.yaml --n-trials 20
uv run kits23-tune-det --config configs/tune_det.yaml --n-trials 15

# 5. Full-scale training
uv run kits23-train-seg --config configs/seg_unet.yaml
uv run kits23-train-det --config configs/det_frcnn.yaml

# 6. Test-split evaluation
uv run kits23-evaluate --task seg --checkpoint runs/seg-unet-resnet34/seg/best.pt --split test
uv run kits23-evaluate --task det --checkpoint runs/det-frcnn-r50/det/best.pt --split test
```

## Layout

| File | Role |
| --- | --- |
| `convert.py` | KiTS23 3D volumes to a 2D COCO dataset |
| `stats.py` | Dataset report and example overlays |
| `config.py` | argparse + YAML merging, Optuna search-space parsing |
| `datasets.py` | COCO index and the segmentation/detection datasets |
| `transforms.py` | Albumentations pipelines for CT slices |
| `models.py` | Model, optimizer, and scheduler builders |
| `losses.py` | Dice / cross-entropy / focal / Tversky combinations |
| `metrics.py` | Confusion-matrix Dice and HEC, COCOeval wrapper |
| `engine.py` | Train and validation loops, warmup, AMP |
| `tracking.py` | MLflow logging and prediction visualizations |
| `train_seg.py`, `train_det.py` | Training entry points |
| `tune_seg.py`, `tune_det.py`, `tuning.py` | Optuna studies |
| `evaluate.py` | Scoring a checkpoint on a held-out split |
