# Model evaluation summary

## Models

| Version | Status | Files |
|---|---|---|
| `v3_store_cctv` | **production, active**. YOLO11s, 10 classes, 640 px, trained 2026-09-30 from v1 on `cctv_store_v2` | `best.pt` present, sha256 `d1de9cb6…862bc6`, backed up |
| `v1_litter_s1_synthetic` | historical (synthetic data) | weights **missing** |
| `v2_yoloe_openvocab` | historical zero-shot labelling helper | weights **missing** |
| fine-tuned candidate (v4) | **does not exist** | not trained: no annotated data (B-01..B-03) |

Classes: 0 plastic_bag · 1 plastic_packaging · 2 food_wrapper · 3 paper · 4 cardboard · 5 plastic_bottle · 6 cup ·
7 newspaper · 8 plastic_piece · 9 other_litter.

## Metrics: what exists and what it means

- **v3 training-time validation:** P 0.96 / R 0.99 / mAP50 0.995 / mAP50-95 0.84 (`model_meta.json`). **Not a
  real-world measure:** the validation clip came from the same scene as the training clips, and the separate test
  clip was never evaluated.
- **Measured in this UAT:** none of precision, recall or mAP. There is no annotated, unseen test set (M15 BLOCKED).
- **Measured speed (CPU, this laptop):** 218–272 ms per frame (stills), 132–140 ms median in the production engine
  benchmark; a 10 s clip processed in 4 s; a 60 s clip in 31 s at 2 analysed frames/s.

## Qualitative UAT (11 cases; visual review by the developer assistant, not ground truth)

`handover/evidence/uat_model/` (annotated images, git-ignored) and `model_uat_raw.json`:

| Behaviour | Observation |
|---|---|
| Pavement litter (store camera) | found in M01/M02 at 0.82–0.91 |
| Fridge bottles and products | **not** detected as litter in any of the 5 store frames |
| Cars, tyre, motorbikes, scooter | **not** detected as litter (M03–M07) |
| Degraded images (dark, blur, low resolution, heavy JPEG) | the wrapper was still found at 0.79–0.83 |
| False positive | **shoe sole → `plastic_piece` 0.63 confirmed** (M04, D-07) |
| Not covered | clean pavement without any litter, night/IR, rain, other cameras, small distant litter on real footage |

These frames come from the same store camera as v3's training data, so good results here are expected and **do not
show generalisation**. The original concerns (shop products, bottles and vehicles mistaken for litter; small litter
missed) can only be confirmed or ruled out on unseen, annotated footage.

## Recommended confidence threshold

**Keep the production values (weak 0.30 / candidate 0.45 / confirmed 0.60).** No validation set exists to support
any other value. The project's evaluator includes a threshold sweep, to be run on the future test split.
