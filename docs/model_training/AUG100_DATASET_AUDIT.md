# CCTV_Litter_100_Images: dataset audit and YOLO11s experiment proposal

Date: 2026-11-13. Status: **audit complete. Training NOT started; it waits for annotation and approval.**

Dataset: the original AUG100 folder `CCTV_Litter_100_Images` (outside the repository; set `AUG100_SOURCE_DIR`) (opened read-only; nothing modified).
Audit scripts and raw output: `docs/model_training/aug100_audit/`.

---

## Summary

- The 100 images are **2 independent synthetic scenes × 50 augmented variants each**. That is not 100 images and not 1 image.
- There are **no annotations at all** (0 label files). A supervised training run is therefore impossible until the images are labelled and a person has checked the labels.
- The production model `v3_store_cctv` was **very likely trained on these same two scenes**. Evidence: its training README lists "street scenes A+C", the LDM workspace manifest marks this folder `used_in_previous_training: likely`, and v3 finds almost every visible item in scene 1 at 0.85–0.95 confidence. **Any score of v3 on these images is contaminated and means nothing.**
- An honest estimate of how well a model generalises **is not possible from this dataset**. The best it can give is a 2-fold leave-one-scene-out test. Both scenes come from the same synthetic generator, so even that says nothing about real stores.

---

## 1. Dataset inventory

| item | value |
|---|---|
| Files | 100 × `.jpg` (`cctv_litter_001…100`) + `README.txt` |
| Size / mode | all 1280×720, RGB, JPEG, 72–216 KB |
| Labels (`.txt` YOLO) | **none** |
| `data.yaml` / classes | none |
| Source README | "generated from **two** CCTV-style source scenes with crop, zoom, lighting, blur, sensor-noise and JPEG variation… not 100 independent real-world CCTV captures" |

## 2. Original vs augmented

| | count | basis |
|---|---|---|
| Independent source scenes | **2** | ORB + RANSAC matching, confirmed by eye |
| Scene G1: shop-front pavement, "CAM 01 - Main Entrance", 08:14:27 | 50 (all **odd** file numbers) | |
| Scene G2: roadside pavement with green bin, 17:24:38 | 50 (all **even** file numbers) | |
| Augmented variants | 100 (all of them) | No unaugmented master exists; even `001` is cropped, because its overlay text is cut off. |
| AI-generated | 2 scenes (all 100 images derive from them) | per user + source README |
| Real CCTV captures | **0** | |

Effective information: 2 backgrounds, 2 litter layouts, and roughly 15–25 litter objects per scene (exact count after annotation). Each object appears ~50 times under photometric and geometric changes.

## 3. Duplicate analysis

| check | result |
|---|---|
| Exact duplicates (MD5) | 0 |
| pHash ≤ 6 bits (near-duplicate) | 499 pairs |
| dHash ≤ 6 bits | 771 pairs |
| ORB inliers ≥ 25 (same scene geometry) | 2,450 / 4,950 pairs = exactly every within-scene pair; **0 cross-scene pairs** |
| Grouping | 2 connected groups of 50 |

Hashing alone under-groups the set, because crops and zooms move pHash by up to ~30 bits. Feature matching is the reliable signal here. The LDM workspace manifest currently files all 100 under one session (`street_aug100`). It should be split into `street_aug100_G1` and `street_aug100_G2` so the two scenes can be used as separate folds.

## 4. Image and annotation quality

- Sharpness (Laplacian variance) 80–916, brightness 80–118, noise σ 2.1–6.9. These are the intended blur, noise and exposure augmentations. The images look like CCTV frames, with timestamp overlays.
- **Domain:** both scenes are **outdoor street pavements**, not store interiors, which differs from v3's "store" target domain.
- Litter is visible and reasonably detailed in the near field (25–90 px wide at 1280). Far-field items near the kerb are **< 20 px**, and at 640 input they shrink below 10 px.
- **No clean-background images and no hard negatives.** Every frame contains the same litter.
- **Annotations: 0.** Bounding boxes, class IDs and image-label correspondence cannot be checked because none exist.

**Verdict:** the dataset **fails** the requirements for a training run. It can support one after annotation.

## 5. Existing model analysis

| | v1_litter_s1_synthetic | v3_store_cctv (production) |
|---|---|---|
| Architecture | YOLO11s | YOLO11s (19.2 MB) |
| Started from | `yolo11s.pt` (COCO) | v1 `best.pt` |
| Data | 120 synthetic train / 20 val | `cctv_store_v2`: street scenes A+C, store clips, synthetic, negatives |
| Classes | 13 | 10: plastic_bag, plastic_packaging, food_wrapper, paper, cardboard, plastic_bottle, cup, newspaper, plastic_piece, other_litter |
| imgsz / epochs / freeze | 640 / 8 / none | 640 / 30 (best 22) / 10, CPU, cos LR, mosaic 1.0 |
| Reported val mAP50 | 0.984 | 0.995 (mAP50-95 0.839) |

The v3 validation split is the 'realistic' store clip, **the same scene as the '3min' training clip** (per its README). 0.995 mAP50 therefore measures memorisation of a known scene, not generalisation. The dataset `cctv_store_v2` itself is not present on this machine (only its `data.yaml`; images were on `D:\…`).

Production weights are protected. Copies exist in `models/litter_yolo/v3_store_cctv/best.pt` and `models/backups/v3_store_cctv_20261012/` (with SHA256SUMS). The experiment will not write to either.

### Benchmark of v3 on these images (CPU, 10 frames, 5 per scene)

| mode | conf | detections / img | median ms / img |
|---|---|---|---|
| full @640 | 0.25 | 19.8 | 415 |
| full @1280 | 0.25 | 26.9 | 1,343 |
| tiled 640 (6 tiles, 25% overlap) | 0.25 | 92.1* | 3,803 |

\* This test did no cross-tile NMS, so the count includes duplicate boxes in tile overlaps.

The output images show that on G1 at 640, v3 finds almost every litter item at high confidence (the contamination sign noted above). On G2, tiling adds false positives on a distant vehicle, the storefront signage and the bin top. Precision and recall cannot be measured without labels.

## 6. Recommended starting checkpoint

**For this experiment: B, COCO `yolo11s.pt`.** Reasons:

1. v3 has very likely already seen these two scenes. Fine-tuning it on them adds almost no information, and nothing it does on them can be measured honestly.
2. v3's own lineage is synthetic-heavy (v1 synthetic, then v3 with synthetic + 2 street scenes + 1 store camera). Its features are not proven better than COCO's, and it carries its own biases (e.g. firing on pavement texture when tiled, as noted in `training/propose.py`).
3. A COCO start yields a clean, controlled answer to the real question: *how much can 2 synthetic scenes teach a detector?*

**For the production path: A (fine-tune v3), but only on genuinely new real data** (the 116 real frames already in `dataset/ldm_v4_workspace`, plus new footage). It should then be checked on a locked camera or session it has never seen. The 100 synthetic images may join that set as **train-only** supplement, never val/test.

`yolo11s.pt` is not on disk (only `yolo11n.pt`), so option B needs one download of the official Ultralytics weights.

## 7. Recommended experiment configuration (proposal, pending approval)

| setting | value | why |
|---|---|---|
| model | `yolo11s.pt` (COCO) | see §6 |
| classes | **`single_cls=True` ("litter")** | ~35 objects total across 10 classes is ~3 per class; per-class learning is not feasible |
| split | **leave-one-scene-out, 2 folds:** train G1 → val G2, then train G2 → val G1 | never mixes relatives across splits |
| imgsz | 1280 | far-field items are < 20 px; 640 halves them |
| epochs / patience | 40 / 10 | early stopping; more epochs only repeat the same 2 scenes |
| freeze | 10 (backbone) | ~25 objects cannot retrain a backbone; trains the neck and head |
| optimizer | SGD, lr0 0.002, lrf 0.1, cos_lr, warmup 2 | conservative; `auto` picks AdamW with its own LR on tiny sets |
| batch | 4 (CPU) | |
| augmentation | hsv_h 0.01, hsv_s 0.3, hsv_v 0.3, degrees 2, translate 0.1, scale 0.3, shear 0, perspective 0.0002, fliplr 0.5, flipud 0, mosaic 0.5, close_mosaic 10, mixup 0, copy_paste 0 | the images already contain blur, noise, JPEG, crop and exposure augmentation, so online augmentation stays mild to avoid stacking distortions |
| seed / deterministic | 0 / true | |
| output | `storage/runs/exp_aug100_coco_G1toG2` and `…_G2toG1`; **`--no-register`** | never touches `models/litter_yolo/` |
| baseline comparison | untrained COCO yolo11s (single-class, any-object) and v3 (contaminated, reference only) on each fold's val scene | |

Estimated cost on this CPU: about 2–4 min per epoch per fold at 1280, so roughly 1–3 h total with early stopping.

## 8. Augmentation strategy

- The dataset's built-in augmentation (crop, zoom, light, blur, noise, JPEG) already covers most realistic CCTV changes. Counting it again as "more data" would be double counting.
- **Recommended addition, with approval, after labels exist:** tile crops that contain no box, used as **hard negatives** (pavement, kerb, cars, bin, signage). They cost nothing, need no generated content, and attack exactly the false positives seen in §5.
- **Not recommended now:** generative synthesis or inpainting of new litter, or copy-paste of litter onto new backgrounds. It is possible, but it adds more synthetic signal from the same ~35 objects and needs segmentation masks. It should be proposed separately, if at all.
- Avoid: flipud, large rotations, heavy mixup or cutmix (ghost litter), strong perspective (fixed cameras).

## 9. Overfitting risks

- **2 independent scenes.** Every metric from this dataset is an n=2 estimate on synthetic images from one generator. **An honest independent generalisation evaluation is not possible with this dataset alone.**
- Random splitting would put relatives of the same scene in both train and val and produce a meaningless ~0.99 mAP50. Splits must be by scene.
- Leave-one-scene-out uses the val scene both for early stopping and for reporting. With only 2 groups, no third held-out group exists, so the reported number is optimistically biased.
- Overlay text (camera name and timestamp) is constant per scene and could become a shortcut feature. Watch for detections on the overlay.
- Monitor: the train/val loss gap per fold, and val box/cls loss rising while train loss falls.

## 10–11. Experimental results and model comparison

**Not run.** This section stays empty until (a) both scenes are annotated and a person has checked the labels, and (b) the §7 configuration is approved.

## 12. Missing data requirements

The existing gap report (`dataset/ldm_v4_workspace/reports/gap_report.md`) already sets the targets, and none are met yet:

- ≥ 300 real litter images, ≥ 200 real hard negatives, ≥ 100 clean scenes
- ≥ 50 boxes per class; ≥ 150 small (< 32 px) boxes
- day / dusk / night-IR / rain / glare conditions (30 each)
- **≥ 8 independent recording sessions** from different stores and camera angles, with at least one whole camera locked to test (`split_lock`)

Fastest progress: annotate the 116 real frames already imported in `ldm_v4_workspace`, then add footage from new stores.

## 13. Continual-learning architecture

Most of this pipeline already exists in the repo. Proposed flow:

| step | component | status |
|---|---|---|
| 1. New footage arrives | `training/extract_frames.py` → `ldm_v4_workspace/manifest.csv` (provenance, session, camera) | exists |
| 2. Inference | backend pipeline (`backend/services/pipeline.py`) | exists |
| 3. Collect uncertain frames | frames with detections at conf 0.10–0.40, v3 vs YOLOE disagreement, or new false-alarm reports from the Alerts page | **to build** |
| 4. Dedupe | dHash grouping (in `build_ldm_dataset.py` / `import_images.py`); add ORB grouping for crop/zoom relatives | partly exists |
| 5. Operator verifies | Ground Truth page; `training/propose.py` boxes are aids, never labels | exists |
| 6. Store labels | `labels/train/*.txt` + `review_status` in the manifest | exists |
| 7. Versioned dataset | `build_ldm_dataset.py` (session-grouped split, `split_lock`) + `split_manifest.csv`; snapshot by hash | exists; add dataset version tag |
| 8. Periodic candidate fine-tune | `training/train.py --no-register` from the current production checkpoint | exists |
| 9. Evaluate on independent test | locked-camera test split + hard-negative false-positive rate; Evaluation page | exists; **add promotion gate** |
| 10. Approve and replace | `register_model.py` + manual deploy in Model Lab; previous weights kept in `models/backups/` | exists |

Hard rule: **unverified predictions never become training labels.** Only frames with `review_status=reviewed` reach `build_ldm_dataset.py`.

## 14. Small-object detection

- At 640, a 1280×720 frame is downscaled 2×, so far-field litter (< 20 px) drops below the reliable size for a stride-8 head.
- Options, cheapest first:
  - **1280 inference:** about 3.2× CPU cost.
  - **Tiling restricted to the far-field band:** only the top half of the frame, where the perspective makes objects smallest, with cross-tile NMS.
  - **ROI masking:** the repo already has `models/roi_segmenter/upernet-convnext-tiny`. Restricting detection to walkable ground would remove the road, vehicle and signage false positives seen on G2.
  - **Per-camera confidence calibration**, once labelled data exists.
- Full-frame tiling costs about 9× CPU (3.8 s/frame) and adds false positives. It is not justified without ROI masking.
- Speed and quality can only be measured properly once a labelled, independent test set exists (mAP on small boxes and false positives per hour on clean footage).

## 15. Production-readiness assessment

**Not production-ready, and no defensible production metric exists today, for v3 or for any model trained on this dataset.**

- This dataset: 2 synthetic outdoor scenes with no labels. It is useful only as a train-only supplement or as a smoke test of the training pipeline.
- v3: the reported 0.995 mAP50 comes from a validation clip of the same scene as its training clip. Its behaviour on unseen stores is unknown.
- What would change this: labelled real footage from ≥ 8 independent sessions, with whole cameras held out for test.
