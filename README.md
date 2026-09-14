# KiTS23 2D models

Detection and segmentation models for the [KiTS23](https://github.com/neheller/kits23)
challenge, trained on 2D axial slices instead of full 3D volumes.

The pipeline has five stages:

```
convert  ->  stats  ->  train  ->  tune  ->  evaluate
 (3D->2D)   (report)   (UNet /    (Optuna)   (test split)
                       Faster R-CNN)
```

## Setup

```bash
uv sync
```

> **GPU note.** This project pins `torch`/`torchvision` to the **CUDA 12.6**
> index in `pyproject.toml`. The default PyPI wheels are built against CUDA
> 13.x, which dropped support for Pascal cards such as the GTX 1080 (sm_61);
> installing those instead fails at runtime with `CUDA error: no kernel image
> is available for execution on the device`. On a newer GPU you can remove the
> `[[tool.uv.index]]` and `[tool.uv.sources]` blocks.

Verify the install before training:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# 2.14.0+cu126 True (6, 1)
```

## 1. Build the 2D dataset

Converts each case's `imaging.nii.gz` into windowed axial PNGs and fuses the
per-annotator instance masks into COCO annotations (bbox + RLE).

```bash
uv run kits23-convert --dataset-dir dataset --output-dir kits32-2d
```

```
kits32-2d/
├── images/{train,val,test}/case_XXXXX_zNNNN.png
└── annotations/instances_{train,val,test}.json, split.json
```

Splits are drawn at the **case** level, so no patient appears in two splits.

## 2. Inspect the dataset

```bash
uv run kits23-stats --dataset-dir kits32-2d --output-dir stats_output
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

## 3. Train

Segmentation — a UNet from `segmentation_models_pytorch` predicting a 4-class
label map (background, kidney, tumor, cyst):

```bash
uv run kits23-train-seg --config configs/seg_unet.yaml
uv run kits23-train-seg --encoder resnet50 --batch-size 4 --loss focaldice
```

Detection — a Faster R-CNN from torchvision over the same slices:

```bash
uv run kits23-train-det --config configs/det_frcnn.yaml
uv run kits23-train-det --arch mobilenet --batch-size 4
```

Command-line flags override the YAML, which overrides the built-in defaults.
Both write `best.pt`, `last.pt`, and the resolved `config.json` under
`--output-dir`.

## 4. Track runs with MLflow

Every run logs its parameters, per-epoch metrics, and a ground-truth vs
prediction image grid.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

MLflow 3.16 retired the `./mlruns` file store, so tracking defaults to a local
SQLite database; point `--mlflow-uri` elsewhere to use a server.

## 5. Search hyperparameters with Optuna

```bash
uv run kits23-tune-seg --config configs/tune_seg.yaml --n-trials 20
uv run kits23-tune-det --config configs/tune_det.yaml --n-trials 15
```

The search space is the `search_space:` block of the YAML, so ranges can be
changed without touching code. Studies are stored in SQLite and resume if you
rerun the same `--study-name`; trials report after every epoch and are pruned
by a median pruner. Each trial is an MLflow run nested under the study.

**Trials run on a subset on purpose.** A full UNet epoch over 43,815 slices at
512x512 is roughly half an hour on a GTX 1080, and Faster R-CNN is several
times slower, so a 20-trial search over the full data would not finish. The
tuning configs cap `max_train_images` and `epochs`; treat the result as a
*ranking*, then retrain the winner at full scale with the printed flags:

```bash
uv run kits23-tune-seg --config configs/tune_seg.yaml
# ... retrain at full scale with:
#   --lr 0.000133 --encoder resnet34 --arch unetplusplus ...
uv run kits23-train-seg --config configs/seg_unet.yaml --lr 0.000133 --arch unetplusplus
```

## 6. Evaluate on the test split

```bash
uv run kits23-evaluate --task seg --checkpoint runs/seg-unet-resnet34/seg/best.pt --split test
uv run kits23-evaluate --task det --checkpoint runs/det-frcnn-r50/det/best.pt --split test
```

Segmentation reports Dice and IoU per class plus the KiTS hierarchical
evaluation classes (kidney and masses / masses / tumor). Detection reports COCO
mAP via `pycocotools`. Both write a `metrics.json` and an example overlay image
to `--output-dir`.

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

## Notes on the 8 GB budget

- `--amp` is on by default. Pascal has no tensor cores, so on a GTX 1080 mixed
  precision buys **memory headroom rather than speed** — it is what lets the
  UNet run at batch 8 and Faster R-CNN at batch 4 at 512x512.
- Use `--accum-steps` to simulate a larger batch when a configuration does not
  fit, and `--arch mobilenet` for a much lighter detector.
