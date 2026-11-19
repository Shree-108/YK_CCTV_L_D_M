# LDM seven-class detector (`v4_ldm7_yolo11s`): end-to-end workflow

Status (2026-11-19): the **tooling is ready and tested** on synthetic data (`tests/test_ldm7_tools.py`). **No
seven-class model has been trained**, and none can be trained honestly yet; see the blockers in section 0.
`v3_store_cctv` remains the production model and stays active.

Code: `training/ldm7/` (schema, verify_dataset, split_by_group, train_yolo11s, small_object_experiment,
compare_models). Installation goes through the existing `training/install_trained_model.py`.

---

## 0. BLOCKERS (today)

| # | Blocker | Consequence |
|---|---|---|
| B1 | **No labelled seven-class CCTV data exists.** The 100 "CCTV" images are unlabeled synthetic augmentations of about 2 scenes. `dataset/exp_aug100` has 1-class `litter` boxes that were AI-verified, not human-annotated. SpotGarbage has 337 coarse pile-level boxes. The multi-class garbage set is image-level only. | Nothing to train on. Converting any of these sets would invent labels. |
| B2 | **No independent labelled test set** (no held-out camera, day or session with 7-class boxes). | `compare_models.py` will report *INSUFFICIENT EVIDENCE*, and no v3-vs-v4 decision is possible. |
| B3 | **No `glass` and no `food_waste` examples** in any source. v3 has no class for either. | Those classes cannot be learnt or evaluated until footage is collected and annotated. |
| B4 | Small-object gap is unmeasured on real labelled data. The 68x32 px wafer packet (2560x1440 frame) scored 0.01 to 0.10 with v3; 960, 1280 and tiling did not help. | The 1280 training size is a hypothesis to be tested (section 7), not a decision. |
| B5 | App integration: `backend/core/class_metadata.json` has no entry for `plastic_packet`, `food_waste` or `glass`. | The lead needs a separate, approved backend change before the model is used for alerts. Installing it as a non-active version is unaffected. |
| B6 | `training/ldm_common.py` workspace (`dataset/ldm_v4_workspace`) is locked to the v3 10-class list and refuses other names. | ldm7 annotation needs its **own pool folder**. Never reuse the v3 workspace. |

---

## 1. Class schema (fixed: never re-order or rename)

| id | name | label | do NOT label as this class |
|---|---|---|---|
| 0 | `plastic_packet` | wafer, chips and snack packets; plastic or foil wrappers; sachets (shampoo, gutka, ketchup) | carry bags (→ `other_litter`); packets still on a shelf or in a hand |
| 1 | `plastic_bottle` | discarded PET water or soft-drink bottles, with or without cap | glass bottles (→ `glass`); bottles held, in a crate, or on display |
| 2 | `paper` | loose paper, tissues, newspaper, paper wrappers, receipts, paper cups flattened beyond recognition | cardboard (→ `cardboard`); posters or signboards fixed to walls |
| 3 | `food_waste` | food scraps, fruit and vegetable peels, discarded food, spilt rice or curry heaps | food on plates or stalls; fallen leaves and flowers (not food, not litter → background) |
| 4 | `other_litter` | clearly discarded items not covered above: carry bags, cups, cigarette packets, cans, foam, rags, straws | anything you are unsure is litter (leave unlabelled and mark the image `ambiguous`) |
| 5 | `glass` | discarded glass bottles, visible broken-glass fragments | windows, glass doors, display glass, reflections |
| 6 | `cardboard` | discarded cardboard, cartons, packaging pieces lying on the ground or pavement | cartons stacked as **merchandise** or deliveries waiting at a door |

Box rules:
* One tight box per object, around the visible part only. Partially occluded items are still labelled.
* Piles: box separable items individually. Box an inseparable heap once with its dominant class.
* Minimum: label anything you can identify at 100 % zoom, even 10x10 px. Small objects are the point of this model.
* If it is not clearly discarded litter, do not label it. A missed box hurts recall a little; a wrong box teaches the
  model false positives.

### What NOT to label: these go in as hard-negative BACKGROUND images
Shelf products and merchandise (including packets and bottles on display), merchandise cartons, clothing, shoes,
bags people carry, logos, signboards, posters, vehicles, tyres, pavement cracks, stains, shadows, wet patches,
reflections, road markings, leaves, flowers, and people.

Frames that show such things **and no litter** are saved with an **existing, empty** label file (`<stem>.txt`, 0
bytes). A missing label file is an ERROR in `verify_dataset.py` and is never read as "clean". Aim for at least
25 to 35 % background images, most of them hard negatives from the same cameras.

---

## 2. Footage collection plan

| What | Target |
|---|---|
| Cameras | **every** production camera (store-front CAM_A and all others in the CCTV registry). Reserve **at least 1 camera, or at least 2 full days on every camera**, as the independent TEST group, chosen before annotation starts. |
| Conditions | day, dusk, night/IR, rain or wet ground, harsh shadow, crowded, empty. Note these in `manifest.csv` `notes`. |
| Volume | about 4 h per camera per condition to start (≈ 2 weekdays + 1 weekend day, morning/noon/evening/night windows). Extract with `training/extract_frames.py` at 1 frame / 5 to 10 s, plus extra frames around litter events. Drop near-duplicates (the verifier flags them). |
| Staged litter | if real `glass` and `food_waste` never appear, **staged** drops (real objects placed in front of real cameras, then removed) are acceptable. Mark the session id `staged_*` so it can be reported separately. Synthetic images are **not** used in val/test. |
| Target instances | **train ≥ 300 per class** (warning below 50), from at least 3 cameras and 10 sessions. **test ≥ 50 per class**, ≥ 30 small (< 32x32 px) boxes, ≥ 100 images and ≥ 20 clean images (`compare_models.py` refuses to conclude below these). |
| Small objects | over-sample distant pavement areas. Keep the native resolution (do not resize frames before annotation). |

`manifest.csv` (pool root), one row per image: `image, camera_id, session_id, date` (+ optional `notes`).
`session_id` = one continuous recording (e.g. `CAM_A_2026-12-03_eve`); `date` = `YYYY-MM-DD`.

---

## 3. Annotation tool

Use a **local** tool (frames must not leave the machine; this branch is local-only):
* **CVAT** (self-hosted Docker): create a task with the 7 labels **in schema order**, then export as **"YOLO 1.1"**.
  Empty frames are exported as empty `.txt` files.
* **Label Studio** (local `pip` install) with a RectangleLabels config in schema order; export as "YOLO". Check that
  empty label files are written for reviewed background frames.
* Offline alternative: **X-AnyLabeling / labelImg** in YOLO mode, with `classes.txt` = the 7 names in order.

After export, put the files in the pool layout: `<pool>/images/<file>`, `<pool>/labels/<stem>.txt`,
`<pool>/manifest.csv`. Review rule: a second person reviews ≥ 10 % of images per class, and **all** test images.

---

## 4. Verification

```
py -m training.ldm7.verify_dataset --root D:/LDM7/ldm7_v1 [--min-instances 50] [--group-by session_id,camera_day]
```
This writes `report.json` and `report.md` (default `<root>/verify_report/`) and exits 1 on any ERROR.

* **ERRORS**: wrong data.yaml names; bad class id, coordinates or size; unreadable image; missing label file;
  exact or near duplicates across splits; a session or camera-day spanning splits; an image without a manifest row.
* **WARNINGS**: fewer than N instances per class, no test split, a class absent from val/test, no background images
  in val/test.
* **Statistics**: instances and images per class per split, and size buckets (< 32², 32²–96², > 96² px).

Use `--group-by camera_id` for a strict unseen-camera test.

## 5. Group split

```
py -m training.ldm7.split_by_group --pool D:/LDM7/pool --out D:/LDM7/ldm7_v1 \
    --test-group camera_id=CAM_C --test-group session_id=CAM_A_2026-12-07_day --val-frac 0.2 --seed 0
```
* Groups are formed from the manifest (session and camera-day) plus near-duplicate images, using union-find.
* A group is never split across sets.
* The designated test groups go to test, and nothing else does.
* The split is deterministic for a given seed.
* Files are copied, the output folder must be new, and `verify_dataset` runs automatically.
* Without `--test-group`, the report marks the test set as **not independent**.

The written `data.yaml` has no `path:` key, so Ultralytics resolves the dataset relative to the yaml (relocatable).

## 6. Training (Colab GPU)

1. Zip the dataset folder and upload it to Drive (`MyDrive/LDM7/ldm7_v1.zip`). Upload `training/` (at least
   `training/__init__.py`, `training/install_trained_model.py` and `training/ldm7/`) and
   `models/pretrained/yolo11s.pt`.
2. In Colab (Runtime → GPU), run:
   ```
   !pip install ultralytics==8.4.*
   !unzip -q /content/drive/MyDrive/LDM7/ldm7_v1.zip -d /content
   !python -m training.ldm7.train_yolo11s --data /content/ldm7_v1 --base /content/yolo11s.pt \
        --staging /content/drive/MyDrive/LDM7/staging --device 0 --imgsz 1280 --epochs 150 --zip
   ```
   For the size experiment, also run `--imgsz 640 --version v4_ldm7_yolo11s_640`, using **the same dataset**.
3. What the script does:
   * It refuses to start if verification fails.
   * It starts from COCO `yolo11s.pt` only, and refuses any `models/litter_yolo/*` weights (v3).
   * It uses mild CCTV augmentation: no flipud, mosaic 0.5, mixup 0.05, scale 0.4.
   * It evaluates val and test (if present).
   * It writes `<staging>/v4_ldm7_yolo11s/` with `best.pt`, `class_names.json`, `data.yaml`, `train_args.json`,
     `args.yaml`, `results.csv`, `dataset_info.json` (dataset version + hash), `verify_report.json`,
     `evaluation.json`, `model_meta.json` and `SHA256SUMS`.
4. A local CPU `--smoke` run (1 epoch, 320 px) only proves the code path. Its package is marked `smoke_test` and the
   installer refuses it without `--allow-smoke`. Do not run it on this PC while RAM is short.

## 7. Evaluation

* **Small-object experiment** (any model, labelled ldm7 images; v3 via mapping):
  ```
  py -m training.ldm7.small_object_experiment --weights <best.pt> --images <root>/images/test \
      --labels <root>/labels/test --out D:/LDM7/exp/<name> --imgsz 640 960 1280 --tiling off on \
      --tile-size 640 --overlap 0.2 --thresholds 0.05 0.1 0.2 0.3 --match-iou 0.3
  ```
  For each GT box it records the best IoU and confidence. It reports recall at each threshold by size bucket and
  class, and FP per image. Outputs are `per_gt.csv`, `summary.csv` and `summary.json`. Keep the 1280 training size
  only if it beats the 640 run on **small-bucket recall at 0.30**, without more FP per clean image.
* **Model comparison** (same independent test set):
  ```
  py -m training.ldm7.compare_models --root D:/LDM7/ldm7_v1 --split test \
      --candidate <staging>/v4_ldm7_yolo11s/best.pt --candidate-imgsz 1280 --out D:/LDM7/compare_v1
  ```
  It reports per-class P/R at 0.30 and AP50, class-agnostic P/R/AP50, recall by size bucket, FP per clean image,
  unmapped-v3 FP and a calibration table. If the test set is missing, too small, lacks clean images or classes, or
  its independence is not verified, the report says **INSUFFICIENT EVIDENCE** and draws no conclusion.

### v3 → ldm7 comparison mapping (`training/ldm7/schema.py`; comparison only, never applied to live output)
| v3 | ldm7 | quality |
|---|---|---|
| plastic_packaging | plastic_packet | approximate (rigid plastic packaging would be other_litter) |
| food_wrapper | plastic_packet | approximate (paper wrappers are mis-attributed) |
| paper, newspaper | paper | exact |
| cardboard | cardboard | exact |
| plastic_bottle | plastic_bottle | exact |
| other_litter | other_litter | exact |
| plastic_bag | **unmapped** | carry bag vs packet ambiguity; v3 confuses them |
| cup | **unmapped** | material unknown |
| plastic_piece | **unmapped** | fragment of anything |

Predictions from unmapped v3 classes are counted in a separate `unmapped_v3` FP bucket for per-class scores. They
still count in the class-agnostic score. v3 can never predict `food_waste` or `glass`; its per-class recall there is
0 by construction, and the report flags it as such.

## 8. Packaging and installation (non-active)

```
py -m training.install_trained_model --package D:/LDM7/staging/v4_ldm7_yolo11s      # or ldm_model_v4_ldm7_yolo11s.zip
```
* The installer checks `SHA256SUMS`, that `model_meta.version` equals the folder name, and that `model_meta.classes`
  equals `class_names.json`, which equals the classes inside `best.pt`.
* It has **no class-count assumption**, so a 7-class package needs **no extra flag**.
* It refuses `v3_store_cctv` and any existing folder.
* It copies the package to `models/litter_yolo/v4_ldm7_yolo11s/` and **never activates it**. After a backend restart
  the registry lists the model as non-active, and it can be chosen per analysis in Image Analysis → Model version.
* Do **not** create `models/litter_yolo/v4_*` by hand, and never point `--staging` into `models/`; the trainer
  refuses that.

## 9. Rollback

* While not active, delete `models/litter_yolo/v4_ldm7_yolo11s/` and restart the backend.
* If someone activated it, re-activate `v3_store_cctv` on the Models page (`POST /api/models/activate`).
* The v3 weights are never modified: the installer hash-checks them before and after copying, and a backup exists
  in `models/backups/v3_store_cctv_20261012/`.

## 10. Order of work

1. Owner approves the camera and test-group plan (section 2).
2. Record footage.
3. Extract frames.
4. Annotate in CVAT.
5. Review.
6. Pool + `manifest.csv`.
7. `split_by_group`.
8. `verify_dataset` (0 errors).
9. Colab training at 640 and at 1280.
10. `small_object_experiment` + `compare_models` on the independent test set.
11. Decision documented.
12. Install non-active.
13. Backend class-metadata change (separate approval).
14. Pilot side by side with v3.
