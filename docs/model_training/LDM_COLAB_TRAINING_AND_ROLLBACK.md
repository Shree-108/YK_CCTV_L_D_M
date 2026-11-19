# One-class litter model: dataset, Colab training, install, rollback

Status (2026-11-17): the dataset is prepared and the notebook passed a local CPU **smoke test** (1 epoch, 320 px; it
proves the code path only). **No model has been trained yet.** `v3_store_cctv` is still the only production model and
stays active.

## 1. Why a separate one-class model
* `CCTV_Litter_100_Images` has no annotation files. The only box labels for these images are the class-0 `litter` labels
  in `dataset/exp_aug100` (homography transfer from 2 reference frames, AI-reviewed, not human-annotated).
* `v3_store_cctv` has 10 specific classes. Mapping one generic class onto them would invent labels, so fine-tuning v3 is
  not possible. The new model is separate and versioned, and it does not replace v3.

## 2. Dataset: `C:\SHREE\DATASET\LDM_TRAINING_READY` (+ `.zip`)
`python -m training.prepare_training_ready` builds it. It refuses to write into an existing folder, hash-checks the source
before and after, and refuses unverified labels.

| split | images | boxes | scene | purpose |
|---|---|---|---|---|
| train | 40 | 760 | A_shopfront (includes the reference frame) | training |
| val | 10 | 190 | A_shopfront | checkpoint selection only (optimistic) |
| test | 50 | 914 | B_roadside_bin | independent scene |

There are only 2 scenes, so a random 70/20/10 split would leak transforms of the test scene into training. That is
why the split is by scene. 1,112 of the 1,864 boxes are under 16 px, and every image contains litter (no negative
images). Details are in `dataset_summary.csv`, `annotation_audit.csv` and `split_summary.csv`.

## 3. Train on Colab GPU: `training/colab/LDM_YOLO11s_Training.ipynb`
1. Upload `C:\SHREE\DATASET\LDM_TRAINING_READY.zip` to Google Drive at `MyDrive/LDM/`.
2. Optional, for `model_comparison.csv`: upload `models/litter_yolo/v3_store_cctv/best.pt` as `MyDrive/LDM/v3_store_cctv_best.pt`.
3. Open the notebook in Colab, set Runtime → GPU, then Run all. The configuration is cell 1: epochs 100, patience 25,
   batch auto, imgsz 640, SGD with lr0 0.01, seed 0, and mild CCTV augmentation with no vertical flip, mixup or
   copy-paste. Given the tiny boxes, a second run with `IMG_SIZE = 960` or `1280` is worth comparing.
4. Outputs go to `MyDrive/LDM/runs/<RUN_NAME>/`:
   * checkpoints, `best.pt` and `last.pt`
   * curves and confusion matrices
   * `reports/` with `training_summary`, `evaluation_summary`, `per_class_metrics`, `prediction_results`,
     `false_positives`, `false_negatives` and `model_comparison` (all `.csv`)
   * `ldm_model_<MODEL_VERSION>.zip`

   Predictions inside `ignore_regions.json` areas are neither TP nor FP.
5. If Colab disconnects, set `RESUME = True` and Run all again.

## 4. Install in LDM without replacing production
```
python -m training.install_trained_model --package "<path>\ldm_model_v4_aug100_litter1c.zip"
```
* The package is copied to `models/litter_yolo/v4_aug100_litter1c/`. The installer refuses an existing folder or the
  name `v3_store_cctv`, and it verifies SHA256SUMS and that `best.pt` contains exactly `['litter']`.
* It does not activate the model or write to the database. After the next backend restart, the registry lists the
  model as **not active**. Test it in **Image Analysis → Model version**: the CSV metadata (`analysis_metadata.json`)
  records which model produced each result.
* A one-class model reports the class `litter` instead of v3's specific class names.

## 5. Model selection and rollback
* Selection is per analysis, through the Model version dropdown. Activation is a separate manual admin action on the
  Models page, and it should happen only after reviewing `model_comparison.csv` and testing on real store footage.
* Rollback depends on the state:
  * Never activated: delete `models/litter_yolo/v4_aug100_litter1c/`.
  * Activated: re-activate `v3_store_cctv` on the Models page (`POST /api/models/activate {"version": "v3_store_cctv"}`).
  * v3 weights are hash-protected. `tests/test_image_analysis_outputs.py::test_v3_weights_unchanged_against_backup_checksum`
    checks them against `models/backups/v3_store_cctv_20261012/SHA256SUMS`.
