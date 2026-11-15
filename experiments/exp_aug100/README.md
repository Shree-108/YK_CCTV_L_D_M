# AUG100 exploratory YOLO11s experiment

Exploratory only: **2 synthetic scenes**, cross-scene transfer within one generator. Results are **not** a measure of
real CCTV accuracy or of generalisation across stores. Plan: `docs/model_training/AUG100_EXPERIMENT_PLAN.md`.
Status: `docs/model_training/AUG100_PREP_PILOT_REPORT.md`.

## Rules enforced in code
- The original dataset (`CCTV_Litter_100_Images`, outside the repository; location via `AUG100_SOURCE_DIR`) is only read; copies are SHA-256 checked.
- `models/litter_yolo/v3_store_cctv/best.pt` is hash-checked before and after every step (`common.check_v3`).
- `dataset/ldm_v4_workspace/` is only read. Production scripts (`training/*`) are not used or changed.
- Training refuses unverified labels, a failed validation, changed start weights, or an existing run dir.
- Nothing is registered as a model version.

## Pipeline
| step | command | output |
|---|---|---|
| copy + transforms + manifest | `AUG100_SOURCE_DIR=<original folder> python -m experiments.exp_aug100.prepare` | `dataset/exp_aug100/{images,manifest.csv,source_sha256.txt,data.yaml}` |
| reference proposals (aid only) | `python -m experiments.exp_aug100.propose_refs` | `annotations/proposals_*.json`, `review/ref_*` |
| reference decisions | `python -m experiments.exp_aug100.reference_review` | `annotations/reference_*.json` (accept / ambiguous / reject + zone) |
| transfer to variants | `python -m experiments.exp_aug100.transfer_boxes --roi pav_kerb_gutter` | `annotations/per_image/*.json`, `labels/train/*.txt` |
| review sheets | `python -m experiments.exp_aug100.review_sheets --what boxes` / `--what uncovered` | `review/boxes_*.jpg`, `review/uncovered_*.jpg` |
| record verification | `python -m experiments.exp_aug100.verify` | manifest `label_status=verified` |
| validate + folds | `python -m experiments.exp_aug100.validate` | `dataset/exp_aug100/{AtoB,BtoA}_{train,val}.txt` (entries `./images/train/<file>`), `data_AtoB.yaml`, `data_BtoA.yaml` (relative paths) |
| clean-frame candidates | `python -m experiments.exp_aug100.clean_candidates --top 12` | `annotations/clean_candidates.json` (user confirms) |
| train | `python -m experiments.exp_aug100.run --config S1 --fold AtoB` | `storage/runs/exp_aug100_S1_AtoB/` + `run_meta.json` |

## Label semantics
- **label**: identifiable litter on the ground inside the ROI (class 0 `litter`).
- **ignore region** (in `per_image/*.json`, never a label): ambiguous objects, litter outside the chosen ROI,
  objects clipped below 50 % visibility, edge slivers. At evaluation a detection there is neither TP nor FP.
- Changing the ROI only re-runs `transfer_boxes.py --roi ...` (+ `verify`, `validate`); no re-annotation.

Images and labels are git-ignored (`dataset/**/images/`, `dataset/**/labels/`): back up `dataset/exp_aug100/` as a whole.
