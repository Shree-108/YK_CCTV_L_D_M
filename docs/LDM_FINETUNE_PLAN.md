# Fine-tuning plan: v3_store_cctv → candidate v4 (NOT APPROVED, NOT RUN)

Status 2026-10-12: **blocked**. There are no verified annotations, no built dataset and no held-out test set (see
section 1). Nothing below has been run. Training starts only after the blockers are cleared **and** the
configuration is approved.

## 1. Readiness checks

| check | result |
|---|---|
| Production checkpoint | `models/litter_yolo/v3_store_cctv/best.pt`, 19,158,106 bytes, sha256 `d1de9cb6…862bc6`, active in the app |
| Backups | git commit `574c7e1` (byte-identical, pushed to origin/main) **and** local `models/backups/v3_store_cctv_20261012/` (sha256 verified, git-ignored, not scanned by the model registry) |
| Architecture | YOLO11s detect (`yolo11s.yaml`, scale s, 9.43 M params), trained at 640 px, from v1 with `freeze=10` |
| Classes | 0 plastic_bag · 1 plastic_packaging · 2 food_wrapper · 3 paper · 4 cardboard · 5 plastic_bottle · 6 cup · 7 newspaper · 8 plastic_piece · 9 other_litter (checkpoint, class_names.json, data.yaml and model_meta.json agree) |
| Dataset YAML | ❌ `dataset/cctv_ldm_v4/data.yaml` does not exist: the build refuses because 0 images are verified |
| Annotation validation | workspace: 0 errors, but **0 of 215 images verified** (0 label files) |
| Splits / leakage | ❌ none yet. The build tool enforces session-level splits, near-duplicate ties and train-only synthetic data (tested) |
| Held-out test set | ❌ none. The two real recordings are the same camera; one was probably in v3's training data |
| Old training data | ❌ `cctv_store_v2` images and labels are still missing (risk of the model forgetting what it learned) |
| Hardware | Intel 2-core / 4-thread CPU, 7.8 GB RAM (**0.7 GB free** with the app + browser running), no CUDA GPU, 331 GB disk free |
| Software | Python 3.14.8, Ultralytics 8.4.173 (v3 trained with 8.4.152), torch 2.14.1+cpu, OpenCV 5.0.0, `yolo` CLI available |
| Production inference speed (v3, this laptop) | 640x368 store frames: median 132 ms, p90 171 ms. 1280x720: median 140 ms, p90 200 ms |

## 2. Baseline (to run once a test split exists)

The project's evaluator (`backend/services/evaluation.py`, Model Lab → Evaluation) runs the **production engine and
settings**: imgsz 640, weak/candidate/confirmed 0.30/0.45/0.60, tiling auto. It reports P/R/F1, mAP50 and
mAP50-95, per-class and per-size recall, a threshold sweep, false positives on litter-free images, hard examples and
latency. The baseline is v3 on `dataset/cctv_ldm_v4` split `test`. The split manifest records each image's
distractor tags (product_display, car, motorcycle, tyre…), so false positives can be broken down by distractor.

## 3. Proposed training configuration

Start from the production checkpoint (it is only read; Ultralytics never writes to the starting weights) and save
everything to a new experiment folder.

GPU (recommended: Google Colab T4 or any NVIDIA GPU with 8 GB or more; copy the dataset folder there):

```bash
yolo detect train model=models/litter_yolo/v3_store_cctv/best.pt data=dataset/cctv_ldm_v4/data.yaml imgsz=640 epochs=60 patience=15 batch=16 device=0 workers=4 optimizer=SGD lr0=0.002 lrf=0.1 momentum=0.937 weight_decay=0.0005 warmup_epochs=2 cos_lr=True freeze=10 seed=0 deterministic=True hsv_h=0.015 hsv_s=0.5 hsv_v=0.45 degrees=5.0 translate=0.1 scale=0.5 shear=1.0 perspective=0.0003 flipud=0.0 fliplr=0.5 mosaic=1.0 mixup=0.05 copy_paste=0.0 close_mosaic=10 plots=True exist_ok=False project=storage/runs name=ldm_v4_ft_from_v3
```

This laptop (CPU only; close the browser and the app first): the same command with `batch=4 device=cpu workers=2`.

| setting | value | in plain words |
|---|---|---|
| model | v3 `best.pt` | keep everything v3 already knows and adjust it, rather than starting from scratch |
| epochs | 60 | how many times the model sees the whole training set. Small datasets need a few dozen passes |
| patience | 15 | early stopping: if validation results don't improve for 15 epochs, stop and keep the best epoch |
| imgsz | 640 | same size v3 was trained at and the app runs at, so production behaviour stays comparable. Larger sizes help tiny litter but would also change inference |
| batch | 16 (GPU) / 4 (CPU) | images per learning step. Smaller uses less memory |
| optimizer / lr0 | SGD / 0.002 | learning rate = how big each correction is. 0.002 is 5× smaller than the default, for gentle fine-tuning that doesn't undo v3 |
| lrf, cos_lr | 0.1, on | the learning rate slowly falls to 10 % by the end |
| warmup_epochs | 2 | the first 2 epochs ease in, to avoid a shock to the existing weights |
| freeze | 10 | the first 10 layers (the general "seeing" part) stay fixed. Only the parts that decide "litter or not" learn. Faster, and safer with little data. v3 used the same setting |
| augmentation | the project's `CCTV_AUG` | brightness/colour shifts, small zoom, horizontal flip, mosaic. Never upside-down: the camera is fixed |
| seed / deterministic | 0 / on | repeatable runs |
| output | `storage/runs/ldm_v4_ft_from_v3/` | candidate `weights/best.pt`, `last.pt`, `results.csv`, plots, `args.yaml`. Git-ignored; `exist_ok=False` refuses to overwrite |

The model is **not registered** in the app automatically. It is registered (`training/register_model.py`, which
never overwrites a version and does not activate it) only after the comparison is approved.

### Risks

- **CPU only, 0.7 GB free RAM:** training here is likely to run out of memory or take many hours (rough estimate:
  10+ h for ~400 images × 60 epochs at batch 4; not measured). Use a GPU machine or Colab.
- **Tiny dataset:** with fewer than ~300 real litter images and ~200 hard negatives, the model can over-fit to one
  scene. Early stopping and the frozen backbone limit this, but they don't fix it.
- **Forgetting:** without `cctv_store_v2` in the mix, detection on the original scenes may get worse.
  Rebuild with `--include-dataset dataset/cctv_store_v2` once it is restored.

## 4. Acceptance criteria (proposal: approve or change before training)

Same test split, same production settings for both models. The candidate is acceptable only if **all** hold:

1. False positives on litter-free test images at the operating threshold (0.60) drop by **≥ 30 %**, reported
   per distractor type (product display / vehicle / tyre).
2. Litter recall at 0.60 drops by **≤ 2 points** (absolute), and no class with ≥ 20 test boxes loses > 10 points.
3. mAP50 is not lower by more than 0.02.
4. Median latency is within +10 % (same architecture, so this is expected).
5. A person spot-checks the candidate's prediction images on the test set.

## 5. Rollback

The candidate never replaces `v3_store_cctv`. Deploying it is a separate, approved step in the Model Lab, which only
flips the active version. Rolling back means re-activating v3. The v3 files are also in git and in
`models/backups/v3_store_cctv_20261012/`.
