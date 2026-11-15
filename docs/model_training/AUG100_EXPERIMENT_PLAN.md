# AUG100: dataset preparation plan and proposed YOLO11s experiment configuration

Date: 2026-11-13. Status: **PLAN ONLY, awaiting approval.** No data has been copied, no labels written, no weights downloaded, no training run.
Builds on `docs/model_training/AUG100_DATASET_AUDIT.md`. Evidence scripts and results: `docs/model_training/aug100_audit/`.

---

## 0. New measurements behind this plan

### 0.1 Box transfer: is it geometrically valid?

Each variant was registered to the widest-view frame of its scene (ORB features, MAGSAC homography at full 1280×720). The transforms were decomposed, then each variant was warped back and the local alignment checked patch by patch (gradient NCC).

| | Scene A (shop-front) | Scene B (roadside + bin) |
|---|---|---|
| Reference frame (widest view) | `cctv_litter_083.jpg` | `cctv_litter_006.jpg` |
| Variants registered | 49 / 49 | 49 / 49 |
| Scale (zoom) | 1.001 – 1.132 | 0.994 – 1.128 |
| Rotation | −0.02° – +0.01° | −0.01° – +0.01° |
| Anisotropy / shear | 1.000–1.001 / ≤ 0.001 | 0.999–1.002 / ≤ 0.001 |
| Perspective term | ≈ 0 | ≈ 0 |
| Flips | 0 | 0 |
| Inlier reprojection RMS | 1.13 – 1.32 px | 1.11 – 1.26 px |
| Median local NCC (worst variant) | 0.85 | 0.93 |

**Conclusion:** every variant is a **crop + uniform zoom (≤ 13 %)** of the reference, with no rotation, perspective, flip or content change. A 2.5× zoomed overlay of the warped variant on the reference shows no misalignment on any litter item. Transfer error is ≈ 1 px, which is about IoU 0.85 on a 12 px box and ≥ 0.95 on boxes ≥ 30 px.

- The ~10 % of patches with low NCC lie in flat pavement and areas whose blur or noise differs. They are not on objects.
- The camera-name and timestamp overlay is re-burned at fixed screen positions in each variant. It is not part of the geometry and never gets labelled.
- Scene B variants with zoom < 1 (down to 0.994) show a ≤ 0.6 % border strip that is **not** in the reference. Any litter there would be missed by transfer, so this is caught in verification.

### 0.2 Compute budget (this machine)

The machine has **2 physical cores / 4 threads, 7.8 GB RAM (≈ 0.7 GB free during the test), and no GPU.**
These timings are forward + backward + SGD step of the YOLO11s architecture on random tensors. They are an approximation that excludes the dataloader, mosaic and the real loss.

| imgsz | batch | freeze | train s / img | infer s / img | est. min / epoch (50 train + 50 val) |
|---|---|---|---|---|---|
| 640 | 8 | 10 | 0.85 | 0.33 | **1.0** |
| 640 | 8 | 0 | 2.15 | 0.33 | 2.1 |
| 960 | 4 | 10 | 1.74 | 0.81 | **2.1** |
| 1280 | 2 | 10 | 3.15 | 1.58 | 3.9 |
| 1280 | 4 | 10 | 5.11 | 1.46 | 5.5 (memory pressure) |
| 1280 | 4 | 0 | 18.47 | 1.52 | 16.7 (not feasible) |

Close other applications before training: free RAM is the binding limit.

### 0.3 Ultralytics pitfalls with only 50 training images (checked in installed v8.4.173)

- `nbs=64` (the default) accumulates gradients to 64 images. With batch 8 that is 7 batches per epoch against an accumulation of 8, i.e. **fewer than one weight update per epoch.** We set **`nbs = batch`**, which also keeps `weight_decay` unscaled.
- `optimizer=auto` silently picks AdamW and ignores `lr0`, so the optimizer is always set explicitly.
- Warmup is `warmup_epochs × batches` in this version, with no 100-iteration floor. 2 warmup epochs means ≈ 14 steps.

---

## 1. Preserve baseline (Step 1)

| asset | protection |
|---|---|
| `models/litter_yolo/v3_store_cctv/best.pt` | never opened for writing. SHA256 checked before and after every experiment against `models/backups/v3_store_cctv_20261012/SHA256SUMS` |
| v1, v2, model_meta, results.csv | untouched |
| Original dataset (`CCTV_Litter_100_Images`, outside the repository) | read-only. Files are **copied**, then each copy's SHA256 is checked against the original. Folder SHA256 list recorded before and after |
| `dataset/ldm_v4_workspace/manifest.csv` and all its labels | **not modified.** It still lists the 100 images as one session `street_aug100`. That is harmless there, because one group always lands in one split. The scene correction lives in the new manifest |
| Production workflow (`training/train.py`, registry, backend) | not used and not changed. The experiment has its own runner and never calls `register_run` |
| Git | branch `security/step1-local-only` kept as is. Only the experiment files listed in §9 are added. Nothing is committed without your go-ahead |

## 2. Dataset organisation (Step 2)

New experimental dataset at `dataset/exp_aug100/`. It appears in the Ground Truth page because it has a `data.yaml`.

```
dataset/exp_aug100/
  data.yaml              names: {0: litter}; train: images/train (annotation view only, never used to train directly)
  README.md
  manifest.csv           one row per image (schema below), committed to git
  source_sha256.txt      SHA256 of all 100 originals at copy time
  images/train/          100 COPIES, original filenames kept      (git-ignored)
  labels/train/          YOLO labels                               (git-ignored; backed up as a zip with the manifest)
  negatives/             verified clean / hard-negative images + EMPTY .txt labels (see §3.4)
  splits/                AtoB_train.txt, AtoB_val.txt, BtoA_train.txt, BtoA_val.txt, neg_probe.txt (image path lists)
```

Each image is copied once. The two experiments use path lists, so nothing is duplicated.

**manifest.csv columns:**

| column | example | meaning |
|---|---|---|
| `image_id` | `aug100_A_083` | stable ID: scene + original number |
| `file` | `cctv_litter_083.jpg` | original filename (unchanged) |
| `source_path` | `cctv_litter_083.jpg` | original file name (folder via `AUG100_SOURCE_DIR`) |
| `image_path` | `dataset/exp_aug100/images/train/cctv_litter_083.jpg` | copy |
| `label_path` | `dataset/exp_aug100/labels/train/cctv_litter_083.txt` | |
| `sha256` | … | identical for copy and original (verified) |
| `scene` | `A_shopfront` / `B_roadside_bin` | source scene |
| `role` | `reference` / `variant` | reference = widest view. **No unaugmented original exists**; the reference is itself an augmented frame |
| `reference_id` | `aug100_A_083` | augmentation parent |
| `aug_scale`, `aug_tx`, `aug_ty`, `aug_rot_deg`, `aug_rms_px`, `aug_inliers` | `1.08, -42.1, -18.6, 0.00, 1.21, 2797` | measured reference→image transform |
| `label_status` | `none` → `transferred_candidate` → `verified` / `manual_verified` / `rejected` | **only `verified` / `manual_verified` may enter training or evaluation** |
| `verified_by`, `verified_at` | | the person who looked |
| `n_boxes` | | after verification |
| `dataset_version` | `exp_aug100_v1` | bumped on any label change |
| `source_type` | `synthetic` | |
| `notes` | | |

## 3. Annotation (Step 3)

### 3.1 Recommended: approach A, label 2 references and transfer, then verify every image

1. You annotate **2 images** in the Ground Truth page (dataset `exp_aug100`, class `litter`): `cctv_litter_083.jpg` (scene A) and `cctv_litter_006.jpg` (scene B).
2. `transfer_boxes.py` maps each reference box to every variant with its measured transform. It clips boxes to the frame and **drops** a box when < 50 % of its area remains visible or a side is < 4 px; those drops go to a list. The script refuses a variant (falls back to manual) if RMS > 2 px, inliers < 500, rotation > 0.5°, scale outside 0.9–1.2, or any perspective or flip. Output: `label_status = transferred_candidate`.
3. **Verification:** for every variant, a review sheet (full frame + 3× zoom strips of small boxes) and the image in the Ground Truth page. Fix or confirm, then `verified`. This includes Scene B's border strip, where transfer cannot see litter.
4. Cost: about 2 × (20–40) boxes drawn, plus 98 images checked (minutes each, not boxes each). Box tightness is consistent across variants, which is better label quality than redrawing 3,000 boxes by hand.

### 3.2 Approach B, manual on all 100

Valid but ~3,000 boxes, about 50× the drawing effort, and box tightness drifts between variants, which adds label noise the model then learns. It is the fallback **per image** wherever 3.1 refuses a variant (currently none would be refused).

### 3.3 Annotation policy (needs your decision, see §10 Q2)

Proposed definition of `litter`: any discarded man-made object (packet, wrapper, paper, cup, bottle, carton, plastic piece) lying on the ground, **including the kerb and gutter**, whose extent is ≥ 6 px on the reference.

- **Not** litter: leaves, stains, items **inside** the bin, objects in shops or on display stands, parked vehicles.
- Overflow from the bin on the ground **is** litter.
- Ambiguous debris (Scene B has a lot) is listed in `ambiguous.txt`, not silently skipped or guessed.
- Small objects are reported separately at evaluation (§6.3).

### 3.4 Clean-background negatives

**The AUG100 dataset contains no clean-background image.** No clean images will be fabricated (no inpainting, no generation). Honest sources:

| source | use | status |
|---|---|---|
| Real store frames already in `dataset/ldm_v4_workspace` (116 real CCTV) that a person marks litter-free | **FP probe set**: evaluation only, empty `.txt` labels | needs your review. Copied, not moved; the ldm workspace is untouched |
| Crops of AUG100 frames containing no labelled litter (road, storefront, cars, bin body) | hard negatives, **train-only within the same scene/fold** | generated only after labels are verified; each crop human-checked; probably few in Scene B |
| New clean pavement footage | proper negatives | not available yet (§8) |

An empty YOLO label is a zero-byte `.txt` with the same stem. The validator checks that each negative has one, and that the manifest marks it `category=clean|hard_negative` and `label_status=verified`.

## 4. Training data checks (Step 4)

`validate_exp_aug100.py` must pass before any run:

- every image has exactly one label and vice versa; each copy's SHA256 equals the original's
- class ID is 0 only; 5 values per line; coordinates within (0, 1]; w, h > 0; no box under 2 px; no duplicate boxes (IoU > 0.9)
- every training or evaluation image is `verified`
- **no scene appears in both train and val of the same fold** (checked from the manifest, not filenames)
- `reference_id` relationships recorded, references identifiable via `role`

Folds:

| experiment | train | val (and the reported evaluation) |
|---|---|---|
| **AtoB** | Scene A, 50 images (+ verified A-crops as negatives) | Scene B, 50 images |
| **BtoA** | Scene B, 50 images (+ verified B-crops as negatives) | Scene A, 50 images |
| both | | + `neg_probe` (real clean store frames), FP count only |

**All 50 variants are kept in training** as built-in photometric/zoom augmentation. They are reported as **1 scene, not 50 samples.**

## 5. Model configuration (Step 5)

- Start: **official COCO `yolo11s.pt`.** It is not on disk; download once from the Ultralytics GitHub release into `models/pretrained/yolo11s.pt` (git-ignored) and record its SHA256.
- v3 is used only as a reference baseline, labelled **contaminated**.

**Fixed for all runs:**

| | value |
|---|---|
| classes | `single_cls=True` (litter) |
| nbs | = batch |
| optimizer | explicit |
| weight_decay | 0.0005 |
| warmup | 2 epochs |
| cos_lr | True, lrf 0.1 |
| early stopping | patience 8 |
| epochs | max 30 |
| seed / deterministic | 0 / True |
| workers / cache | 0 / False (RAM) |
| amp | False (CPU) |
| augmentation (mild) | hsv_h 0.01, hsv_s 0.3, hsv_v 0.3, degrees 2, translate 0.1, scale 0.3, shear 0, perspective 0.0002, fliplr 0.5, flipud 0, mosaic 0.5, close_mosaic 5, mixup 0, cutmix 0, copy_paste 0, erasing 0 |

Augmentation is mild because the images already carry blur, noise, JPEG, zoom and exposure changes.

### 5.1 Screening grid: fold AtoB only, one variable changed at a time

| run | imgsz | batch | freeze | optimizer, lr0 | ≈ max time |
|---|---|---|---|---|---|
| S1 (anchor) | 640 | 8 | 10 (backbone) | SGD 0.002, mom 0.937 | 30 min |
| S2 | 640 | 8 | 10 | SGD 0.01 (Ultralytics default) | 30 min |
| S3 | 640 | 8 | 10 | AdamW 0.0005 | 30 min |
| S4 | 640 | 8 | 0 (full) | best of S1–S3 | 65 min |
| S5 | 960 | 4 | best of S1/S4 | best of S1–S3 | 65 min |
| S6 (only if S5 beats S1 on small-object recall by ≥ 5 pts) | 1280 | 2 | 10 | best | ~2 h |

Screening total ≈ 3.7 h worst case (early stopping usually shortens it), plus S6 if triggered.

Selection rule, fixed in advance: **highest mAP50-95 on AtoB val**, tie-break small-object recall then speed.

### 5.2 Final runs

The chosen config is run on **both folds with fixed epochs** (the screening's best epoch for that config), **no early stopping**, and `last.pt` reported. Reasons:

- Screening selected the config on Scene B as validation. **AtoB numbers are therefore optimistically biased.**
- **BtoA is the cleaner number**: Scene A was never used to choose anything.
- `best.pt` (selected on the val scene) is reported too, but labelled "selected on test".

## 6. Experiments, evaluation and logging (Steps 6–7)

### 6.1 Layout

```
experiments/exp_aug100/                 (committed)
  README.md                             exact commands + rules
  configs/base.yaml, S1..S6.yaml        hyper-parameters (one file per run)
  data_AtoB.yaml, data_BtoA.yaml        point at dataset/exp_aug100/splits/*.txt, names {0: litter}
  prepare.py                            copy + hash + manifest + transform measurement
  transfer_boxes.py                     §3.1 step 2
  make_review_sheets.py                 §3.1 step 3
  validate.py                           §4
  run.py                                train one config/fold (no registry; refuses if v3 hash changes or labels unverified)
  evaluate.py                           metrics + small objects + FP/FN + speed + overlays
storage/runs/exp_aug100_<run>_<fold>/   (git-ignored) Ultralytics run, weights, plots, eval/
storage/runs/exp_aug100_results.csv     one row per run x fold x weights (git-ignored; summary copied to docs/)
```

### 6.2 Commands (after approval)

```bash
python -m experiments.exp_aug100.prepare                       # copy, hash, manifest
#   -> annotate 083 + 006 in Ground Truth page (dataset exp_aug100)
python -m experiments.exp_aug100.transfer_boxes
python -m experiments.exp_aug100.make_review_sheets            # -> verify all 98 variants
python -m experiments.exp_aug100.validate
python -m experiments.exp_aug100.run --config S1 --fold AtoB   # ... S2..S5
python -m experiments.exp_aug100.run --config <best> --fold AtoB --final
python -m experiments.exp_aug100.run --config <best> --fold BtoA --final
python -m experiments.exp_aug100.evaluate --runs exp_aug100_*  # + baselines
```

### 6.3 Reported per run, fold and weights

| metric | how |
|---|---|
| Precision, Recall, **F1**, mAP50, mAP50-95 | Ultralytics `val` on the fold's val scene (`single_cls`, imgsz as trained); F1 at the max-F1 confidence, plus P/R/F1 at a fixed conf 0.25 |
| Per-scene results | AtoB → scene B, BtoA → scene A, shown side by side, never averaged into one headline |
| False positives / false negatives | at the chosen conf, IoU 0.5: per-image lists + overlay images (TP green, FP red, FN yellow) for **all** val images; FP count on `neg_probe` |
| Small objects | GT buckets by shorter side at 1280×720: < 16 px, 16–32 px, ≥ 32 px. Recall and AP per bucket |
| Inference speed | ms/img (CPU, median of 50) at 640 / 960 / 1280 and tiled (with cross-tile NMS) |
| Train / val curves | box / cls / dfl loss per epoch; overfitting flagged when val loss rises 3 epochs while train loss falls |
| Baselines | (a) **COCO yolo11s untrained**, class-agnostic (any COCO detection = "litter"), which shows the effect of fine-tuning; (b) **v3, class-agnostic, marked CONTAMINATED**, reference only |

**Manual inspection:** I look at all overlay images per final run and write the observed failure modes (e.g. overlay text, vehicles, bin, kerb debris) into the results report.

### 6.4 Reporting rules

- Every result table carries the line: "Synthetic, 2 scenes, cross-scene transfer within one generator. Not a measure of real CCTV accuracy."
- No single "accuracy" number.

## 7. Limitations (stated up front)

- n = 2 scenes from one synthetic generator. A cross-scene result says how far one synthetic street transfers to another synthetic street, nothing about real stores, night, IR or other cameras.
- Each fold trains on ~20–40 distinct objects. High variance between folds is expected and is itself a finding.
- The negatives probe uses real store frames possibly seen by v3. That is fine for the COCO-started model, but a v3 FP count there is contaminated.

## 8. Future real-world data strategy (Step 8)

The target is a dataset where **whole cameras and whole stores are held out for testing.**

| need | target (first fine-tune) | how |
|---|---|---|
| Real store CCTV with litter | ≥ 300 frames, ≥ 8 recording sessions, ≥ 4 stores | `training/extract_frames.py --every 2` → ldm workspace → Ground Truth page |
| Clean pavement / store surroundings | ≥ 100 frames, same cameras | record the same cameras when clean (after cleaning rounds); `category=clean`, empty labels |
| Camera angles | ≥ 3 heights/angles (overhead, entrance, wide street) | one session per camera per day |
| Store locations | ≥ 4; **≥ 1 entire store locked to test** (`split_lock=test` on all its sessions) | |
| Day / dusk / night / IR / rain | ≥ 30 frames each | tag `conditions`; one night or IR camera kept entirely in test |
| Small / distant litter | ≥ 150 boxes < 32 px | prioritise far-field frames; evaluate with tiled inference |
| Hard negatives | ≥ 200: product displays, cars, tyres, motorbikes, shadows, reflections, pavement markings, people carrying bags | from the same footage; false alarms from the live system go to review |

**Evaluation protocol:**
- test = held-out store(s) and camera(s), never seen in training, locked before any training
- val = different sessions of training cameras
- report per store and per condition, plus false positives per hour on clean footage

**Continual loop:** the existing ldm_v4 workflow: uncertain frames → dedupe → human verification → versioned dataset → candidate fine-tune → locked-test gate → manual promotion. **Unverified predictions never become labels.**

## 9. Files touched by this stage

| file | status |
|---|---|
| `docs/model_training/AUG100_DATASET_AUDIT.md` | created in the previous step |
| `docs/model_training/AUG100_EXPERIMENT_PLAN.md` | **new** (this file) |
| `docs/model_training/aug100_audit/*` | new: audit, benchmark, geometry and compute scripts plus JSON results |

Nothing else has been created or modified. After approval, the execution stage adds `experiments/exp_aug100/**`, `dataset/exp_aug100/**` (images and labels git-ignored), `models/pretrained/yolo11s.pt` (git-ignored) and `storage/runs/exp_aug100_*` (git-ignored).

## 10. Decisions needed

1. **Approve the plan:** approach A (2 references + transfer + verify all 98), the dataset layout and the manifest schema.
2. **Annotation policy (§3.3):** count kerb/gutter litter? Minimum size 6 px? Bin overflow on the ground = litter?
3. **Negatives:** may I list candidate clean frames from `ldm_v4_workspace` for you to confirm as the FP probe?
4. **Download** official `yolo11s.pt` (~19 MB) from the Ultralytics GitHub release.
5. **Screening grid S1–S5** (≈ 3.7 h CPU worst case) and the S6 trigger rule.
6. **Commit:** add only the docs and `experiments/exp_aug100/` code to the current branch when done?
