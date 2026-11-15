# AUG100: preparation and pilot report

Date: 2026-11-13. Status: **preparation complete and pilot passed. S1–S5 not started; waiting on 3 decisions (§6).**

Exploratory experiment on 2 synthetic scenes. Nothing here measures real CCTV accuracy.

## 1. Labels

| | Scene A (shop-front) | Scene B (roadside + bin) | total |
|---|---|---|---|
| Images (all verified) | 50 | 50 | **100** |
| Boxes, class `litter` (single class) | 950 | 914 | **1,864** |
| Reference boxes, labelled by visual review | 19 (frame 083) | 19 (frame 006) | 38 |
| Transferred boxes | 931 | 895 | 1,826 |
| Transferred boxes corrected | 0 | 0 | **0** |
| Clipped (< 50 % visible) → ignore region | 0 | 8 | 8 |
| Ambiguous ignore regions (instances) | 463 | 1,197 | 1,660 |
| Edge slivers outside the reference view → ignore | 4 | 0 | 4 |

- Box sizes by shorter side: **1,112 boxes < 16 px**, 526 boxes 16–32 px, 226 boxes ≥ 32 px.
- **Reference labelling.** v3 proposals were used as an aid only (45 in scene A, 81 in scene B). Every proposal was accepted, marked ambiguous or rejected from zoomed crops. No missed litter was found.
  - Rejections: overlay text, foliage, vehicles, signage, **bin contents**, duplicates and texture.
  - Ambiguous: specks, leaf-like items, debris smears and paper inside the planter.
- **Transfer.** Each variant gets its own homography from the reference. All 98 variants passed the guard rules (inliers ≥ 500, RMS ≤ 2 px, rotation ≤ 0.5°, no perspective or flip).
  - Measured transforms: scale 0.994–1.13, RMS 1.1–1.3 px.
  - Per-box alignment check (NCC): every one of the 1,826 transferred boxes scored ≥ 0.853, median 0.976.
- **Visual verification.**
  - All 1,826 transferred boxes were inspected as zoomed tiles (`review/boxes_00..16.jpg`). Every box is on its object and none needed correcting.
  - All 112 border strips the reference does not show (≤ 8 px wide) were inspected (`review/uncovered_*.jpg`). They contain no labelable litter; 4 object slivers at the bottom edge became ignore regions.
- **Who verified.** The verification was done by Claude, not a person. A human spot-check in the Ground Truth page (dataset `exp_aug100`) is recommended before results are shared.
- **Validation** (`annotations/validation.json`): all checks pass.
  - Every image has a label file; class 0 only; coordinates valid.
  - No duplicate boxes; each copy's hash matches its original.
  - No scene appears on both sides of a split.

## 2. ROI: provisional, needs approval

No approved pavement ROI existed for these scenes. Zones (pavement / kerb / gutter / road) were drawn on each reference frame and stored per box, so switching ROI needs no re-annotation.

| ROI option | Scene A labels / frame | Scene B labels / frame | kerb + gutter litter |
|---|---|---|---|
| `pavement` | 12 | 14 | ignore regions |
| `pav_kerb_gutter` (**used provisionally**) | 19 | 19 | labelled |

Road (carriageway) litter is out of ROI in both options. Visual: `dataset/exp_aug100/review/roi_options_*.jpg`.

## 3. Train / validation split

| fold | train | val (= reported evaluation) |
|---|---|---|
| AtoB | Scene A, 50 images, 950 boxes | Scene B, 50 images, 914 boxes |
| BtoA | Scene B, 50 images, 914 boxes | Scene A, 50 images, 950 boxes |

No test split exists, because there are only two scenes. False positives on clean frames will be measured once you confirm the frames (§5).

## 4. Pilot (640, S1 settings, 2 epochs, fold AtoB)

| check | result |
|---|---|
| Completion | completed, no errors |
| Time | 133 s total; ~53 s per epoch including validation on 50 images |
| Checkpoints | `best.pt`, `last.pt` (19.1 MB each) in `storage/runs/exp_aug100_pilot_AtoB/weights/` |
| Logs | `results.csv`, `args.yaml`, plots, `run_meta.json` |
| Training process RAM, peak | **1.65 GB** |
| System RAM available, minimum | **0.18 GB** of 7.8 GB (about 2.1 GB free when idle) |
| Swap | 2.25 GB peak, but ~2.2 GB was already in use before training, so this is baseline |
| CPU | 204 % mean (all 4 threads) |
| v3 hash | unchanged (`d1de9cb6…62bc6`) |
| Registered | no |

The 2-epoch metrics (scene B val: P 0.59, R 0.59, mAP50 0.54) only confirm that the pipeline works. **They are not a result.**

**Memory assessment:** 640 / batch 8 fits, but with only ~0.2 GB to spare. The 960 (S5) and 1280 (S6) runs need more activation memory; on this machine they risk heavy paging or an out-of-memory error. Closing other applications (VS Code, browser) before S5 is required, or S5 should run at batch 2.

## 5. Clean frames: candidates only, not confirmed

All 46 real, non-overlay frames in `ldm_v4_workspace` come from **one camera** (`store_entrance`, 640×368). None gets zero v3 detections at confidence 0.10. Top 12 candidates (`dataset/exp_aug100/review/clean_candidates_0.jpg`, details in `annotations/clean_candidates.json`, kept locally and not committed):

| id | frame | session | t (s) | v3 detections at 0.10 (max conf) | note |
|---|---|---|---|---|---|
| C00 | (real store frame, local only) | footfall | 116.2 | 1 (0.11) | crowded |
| C01 | (real store frame, local only) | footfall | 94.2 | 1 (0.22) | crowded |
| C02 | (real store frame, local only) | footfall | 40.1 | 1 (0.27) | crowded |
| C03 | (real store frame, local only) | footfall | 74.1 | 2 (0.19) | crowded |
| C04 | (real store frame, local only) | footfall | 170.3 | 2 (0.20) | crowded |
| C05 | (real store frame, local only) | footfall | 158.3 | 2 (0.20) | crowded |
| C06 | (real store frame, local only) | footfall | 58.1 | 2 (0.52) | crowded |
| C07 | (real store frame, local only) | poc3min | 0.0 | 2 (0.83) | **litter visible: probably not clean** |
| C08 | (real store frame, local only) | poc3min | 22.0 | 2 (0.93) | **litter visible: probably not clean** |
| C09 | (real store frame, local only) | footfall | 44.1 | 3 (0.23) | **burned-in green box: not flagged as overlay** |
| C10 | (real store frame, local only) | footfall | 96.2 | 3 (0.31) | crowded |
| C11 | (real store frame, local only) | footfall | 72.1 | 3 (0.34) | crowded |

Caveats:
- One camera only, so a false-positive count here measures one store entrance.
- `poc3min` was probably in v3's training data. That doesn't matter for the COCO-started models, but **v3's false-positive count on these frames is contaminated**.

## 6. Decisions needed before S1–S5

1. **ROI:** `pav_kerb_gutter` (current) or `pavement`. This changes the training labels, so it must be fixed before S1.
2. **Clean frames:** which candidates are genuinely litter-free (expected C07, C08 and C09 to be excluded)?
3. **Memory for S5 (960):** close other applications, or approve batch 2 for S5. That changes the planned config, so it needs your approval.

## 7. Files

New (nothing committed; git branch unchanged):

| path | tracked by git? |
|---|---|
| `experiments/__init__.py`, `experiments/exp_aug100/*.py`, `README.md`, `configs/*.yaml`, `data_AtoB.yaml`, `data_BtoA.yaml` | yes (untracked, new) |
| `docs/model_training/AUG100_PREP_PILOT_REPORT.md` | yes (untracked, new) |
| `dataset/exp_aug100/manifest.csv`, `data.yaml`, `source_sha256.txt`, `annotations/**`, `splits/**` | yes (untracked, new) |
| `dataset/exp_aug100/images/**`, `labels/**`, `review/**` | git-ignored |
| `models/pretrained/yolo11s.pt` (19,313,732 bytes, SHA-256 `85a76fe8…d502d5`, from `github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt`) | git-ignored |
| `models/pretrained/yolo11s.SOURCE.txt` | yes (untracked, new) |
| `storage/runs/exp_aug100_pilot_AtoB/**` | git-ignored |

Modified: **none.** No existing file was changed; v3, the production scripts, `ldm_v4_workspace` and the original dataset are untouched.
