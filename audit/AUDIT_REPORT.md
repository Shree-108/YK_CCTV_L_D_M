# LDM Phase 1 audit: why small litter is missed

**Date:** 2026-11-20. **Scope:** read-only. **Status:** waiting for owner approval before Phase 2.

Nothing in the project was edited, moved, retrained or activated. The production weights, database, datasets and
settings were not changed. The only new project file is this report. The experiments read existing frames and ran
the existing model. Their scripts and outputs are in the session scratchpad, outside the project:
`C:\Users\chinm\AppData\Local\Temp\claude\360c24f91df0d034\da3682a2-1b61-4dbf-8735-7407ae3ad5b7\scratchpad\audit\`.

Each claim below is marked as one of:

- **[verified]**: checked in this audit.
- **[from files]**: read from a project file, not independently re-checked.
- **[unverified]**: could not be checked. The reason is given.

---

## 0. Summary

1. **The main cause of the missed green wrapper is a learning gap in the model.** Resolution, thresholds, NMS and
   filters are not the cause. The production model (`v3_store_cctv`) finds the wrapper's position: its candidate
   box overlaps the true box at IoU 0.75–0.94. But the model scores that box no higher than **0.059**. This holds at
   every input size from 640 to 2560, with tiling, and with CLAHE. The production floor is 0.30. Higher-resolution or
   tiled inference with the current model therefore cannot recover the wrapper. A retrained model is needed.
2. **The model's high-confidence boxes in this scene are mostly false positives.** On the same frame, the boxes at or
   above 0.30 are mainly CRAX logos and printed text on the stacked cartons, labelled `food_wrapper` at 0.40–0.71,
   plus pavement specks and slivers at the frame edge. The model appears to have learned "printed graphics" rather
   than "discarded wrapper". This is also why lowering the confidence threshold is not a fix: at conf 0.05 the
   wrapper ranks 9th, behind 8 other boxes.
3. **The reported accuracy is not real-world accuracy.** The model card shows mAP50 0.995, but its validation clip
   is the same store scene as its training clip. No held-out real CCTV test set exists. The gap was therefore
   invisible.
4. **The v3 training data is not on this PC.** The data YAML points to `D:\YATRIKART\...`, and no D: drive exists.
   The audit could not check dataset size, class balance, box sizes or annotation quality for v3. The 1440p
   camera's footage is dated 2026-11-19, and v3 was trained on 2026-09-30, so this camera's footage was certainly not
   in v3's training data.
5. **Four defects found along the way** (section 3.3). They need separate approval:
   - The summary CSV's per-class columns drop 6 of v3's 10 class names.
   - Static-frame reuse lets one model output count as several temporal "confirmations".
   - Still images and video frames use different tiling paths.
   - The two annotation guides use two different class taxonomies.
6. **Hardware limits the plan.** This PC is a 2-core i3-1115G4 with 7.8 GB RAM and no CUDA GPU, so training here
   is not practical. The existing plan trains on Google Colab, which means uploading CCTV frames. That conflicts with
   your rule against external services, so you need to decide how training will be done (section 5.7).

---

## 1. Current technical condition

### 1.1 Active model [verified]

| Item | Value |
|---|---|
| Active version (DB `models.active=1`) | `v3_store_cctv` |
| Path | `models/litter_yolo/v3_store_cctv/best.pt` |
| Size, SHA-256 | 19,158,106 bytes, `d1de9cb6…62bc6`. Identical to the backup `models/backups/v3_store_cctv_20261012/` |
| Trained | 2026-09-30 09:06 IST, Ultralytics 8.4.152 (installed now: 8.4.173) |
| Architecture | YOLO11**s** detect, 9.43 M parameters, `nc=10` |
| Detection heads | 3 heads, P3/P4/P5, strides **8 / 16 / 32**. No P2 head |
| Training input | **640**, `rect=False`, mosaic 1.0, `multi_scale=0` |
| Training regime | Fine-tuned from `v1_litter_s1_synthetic` (13 classes, synthetic data) with `cls_remap`. **`freeze: 10`** (backbone frozen). 30 epochs, best at epoch 22. **device cpu**, batch 8, 8,812 s |
| Reported metrics | P 0.962, R 0.994, mAP50 0.995, mAP50-95 0.839, on its own val split. **Not trustworthy** (section 2, cause 3) |
| Other versions on disk | `v1_litter_s1_synthetic` (no weights file), `v2_yoloe_openvocab` (metadata only), `models/pretrained/yolo11s.pt`. Also `models/roi_segmenter/` (left over from the removed ROI feature; unused) |

**Runtime environment:** Python 3.14.8, torch 2.14.1+cpu, `torch.cuda.is_available()` = False, OpenCV 5.0.0,
onnxruntime 1.31.0. SAHI is not installed.

### 1.2 Class names and mapping

| Source | Classes |
|---|---|
| Weights (`model.names`), `class_names.json`, `model_meta.json`, `data.yaml`, DB `models.classes_json` | **10, identical [verified]:** plastic_bag, plastic_packaging, food_wrapper, paper, cardboard, plastic_bottle, cup, newspaper, plastic_piece, other_litter |
| `docs/LDM_ANNOTATION_GUIDE.md` | The same 10 v3 classes |
| `training/ldm7/schema.py`, `backend/services/litter_reports.py` (`LDM7_CLASSES`, "the finalized seven-class schema") | **7 different classes:** plastic_packet, plastic_bottle, paper, food_waste, other_litter, glass, cardboard |
| `backend/core/class_metadata.json` | A superset of 31 names (v1, v3, ldm7 and extras). Priorities only |
| `dataset/class_mapping.yaml` | A 25-class mapping from an older project. Not used by the active model |
| `dataset/cctv_litter_dataset_v1`, `cctv_real_v1_autolabel` | 13 v1 classes |

**Mismatches:**

- **M1 [verified]. The summary CSV loses most v3 classes.** `litter_reports.summary_rows()` fills its per-class
  columns using only the 7 ldm7 names. The active model emits v3 names. As a result, `food_wrapper`,
  `plastic_packaging`, `plastic_bag`, `cup`, `newspaper` and `plastic_piece` events are counted in
  `total_unique_litter_events` but in **no** class column, so the class columns do not add up to the total.
  `food_wrapper` is v3's most frequent class: 18 of the 24 events across the 1440p, 1080p and 368p jobs listed in
  section 6. The metadata JSON records the problem (`classes_outside_ldm7_schema`), but the CSV does not. The
  six-column litter report CSV is not affected, because it writes the raw class name.
- **M2 [from files]. Two "approved" taxonomies exist.** The annotation guide says to label with v3's 10 classes.
  The report code and the training tooling call the 7-class ldm7 schema "finalized". I cannot tell which one you
  approved. **You need to decide this before any annotation starts.**
- No mismatch was found between the weights, the YAML and the registry.

### 1.3 Training data

- **v3's training set (`dataset/cctv_store_v2`) is not on this machine [verified].** The folder contains only
  `README.txt` and `data.yaml`, and the path inside the YAML is `D:\YATRIKART\CCTV-LD\...`. There is no D: drive.
  - The README describes the split **[from files]**:
    - train: street scenes A+C, store clips `3min` and `real_life`, synthetic images, negatives
    - val: store clip `realistic`. The README itself says this is the "same scene as 3min".
    - test: store clip `clear_recognizable`
  - The split is **by clip, but val shares its scene with train**.
  - **[unverified, data absent]:** image count, annotation count, class distribution, box-size distribution,
    negative-image count and annotation quality for v3.
- **The labelled data that is on this PC [verified]:**

| Dataset | Images | Labels | Notes |
|---|---|---|---|
| `dataset/exp_aug100` (= `C:\SHREE\DATASET\LDM_TRAINING_READY`) | 100 (1280×720) | 1,864 boxes, **1 class (`litter`)**, 0 negative images | Synthetic augmentations of **2 street scenes**, so not independent. Labels were made **ROI-scoped**: litter outside the `pav_kerb_gutter` ROI was set to "ignore". They are not valid full-frame labels without a re-review. |
| `dataset/ldm_v4_workspace` | 261 (215 in the manifest) | **0 labels**. All 215 manifest rows are `unreviewed` | 28 frames are from the `poc3min` clip, which v3 probably trained on. 88 frames are footfall frames, 70 of them flagged for burned-in overlay boxes. 99 frames are from street_aug100. |
| `dataset/cctv_litter_dataset_v1` | Folder holds only a README and YAML | – | Synthetic starter set |
| `dataset/cctv_real_v1_autolabel` | Folder holds only a README and YAML | – | Unreviewed auto-labels. Must not be used |

- **Box sizes in exp_aug100** (largest side of each box, in pixels, at native 1280×720) **[verified]**:

| Size band | Boxes | Cumulative |
|---|---|---|
| < 16 px | 283 | 283 (15 %) |
| 16–32 px | 737 | 1,020 (55 %) |
| 32–64 px | 694 | 1,714 (92 %) |
| 64–128 px | 150 | 1,864 |
| ≥ 128 px | 0 | – |

  After resizing the frame to 640, **1,020 of the 1,864 boxes (55 %) are under 16 px**. This set really is a
  tiny-object set, but it covers only 2 scenes and 1 class.

### 1.4 Annotation quality

- **v3 labels: [unverified]**. The data is absent.
- **exp_aug100: sampled from `dataset/exp_aug100/review/boxes_00.jpg`** (96 crops).
  - **Good:** Boxes are generally tight. The objects are plausible litter: wrappers, cups, packets, paper. No
    shadows or shop stock were seen labelled.
  - **Concern 1:** Many targets are blurry blobs of 10–20 px whose litter identity cannot be confirmed by eye, for
    example `002#75`, `004#75` and `004#76`. They need an "ignore/difficult" rule, not a positive label.
  - **Concern 2:** The same physical objects repeat across augmented images. The effective number of unique
    instances is far below 1,864.
  - **Concern 3:** The labels are ROI-scoped, as noted in section 1.3.

### 1.5 Preprocessing [verified, code]

- **Decoding.** `cv2.VideoCapture` reads frames sequentially.
- **Sampling.** `video.sample_fps = 2.0`, so the pipeline analyses every `round(fps/2)`-th frame: every 10th frame
  at 20 fps.
- **Static-frame skip.** The frame is shrunk to a 96×54 grey thumbnail and compared with the previous one. If the
  mean absolute difference is below 2.0, the previous detections are reused, for at most 5 s.
- **Resizing.** If a frame's resolution changes mid-stream, it is resized to the first frame's size.
- **No enhancement.** There is no CLAHE, denoising or sharpening.
- **Resize to model input.** Ultralytics letterboxes each frame to `imgsz` 640 (long side), keeping the aspect
  ratio. A 2560×1440 frame becomes about 640×360, a scale factor of **0.25**.

### 1.6 Inference settings

The `settings` DB table is **empty**, so the code defaults in `backend/core/settings.py` apply.

| Setting | Value |
|---|---|
| `imgsz` | 640 |
| conf floor (`weak_threshold`) | **0.30** |
| `candidate_threshold` / `confirmed_threshold` | 0.45 / 0.60 |
| NMS | IoU 0.50, class-aware (`agnostic_nms=False`) |
| `max_det` | 300 |
| device | auto → **cpu** |
| `half` | True, but ignored on CPU |
| Tiling, **still images** and the four-screenshot sampler | `tiling.enabled=True`, `mode=auto`. Tiling starts when the long side is over 1,280 px. Then 1,024 px tiles with 20 % overlap **plus** a full frame at 640 |
| Tiling, **video files** | Forced **off** (`video.tiling=False`) |

### 1.7 Post-processing filters [verified, code: `backend/detection/filters.py`, `backend/tracking/tracker.py`]

| Filter | Rule | Can it remove a valid small detection? |
|---|---|---|
| Confidence floor | below 0.30 → dropped inside `model.predict` | **Yes. This is where the wrapper is lost** (score 0.059) |
| Class allow/block lists | None set | No |
| `min_box_area_px` | 36 px² (6×6) | Only below about 6×6 px native. The wrapper is about 2,850 px² |
| `max_box_area_frac` | 0.25 of the frame | No (large-object rule only) |
| `max_aspect_ratio` | 12 | Rarely (long thin straws) |
| Tiled merge | class-aware NMS 0.5, then cross-tile containment 0.75 | Can drop a partial box at a tile seam; not an issue at the wrapper's position |
| Tracker confirmation | ≥ 3 hits at ≥ 0.45 within 5 sampled frames, mean ≥ 0.60, stationary within 5 % of the frame diagonal | **Yes:** a real item scoring 0.30–0.45 never confirms. It stays UNDER_REVIEW |
| Static reuse | see defect D2 below | Inflates confirmations rather than removing detections |

There is no ROI, polygon or region filter. The detection docstrings confirm this. `settings.strip_legacy()` drops
old ROI keys.

### 1.8 Video pipeline [verified]

**Source videos on disk** (unique files, probed with OpenCV):

| Video | Resolution | fps | Codec | Duration | Bitrate | Bits per pixel |
|---|---|---|---|---|---|---|
| Store front (green wrapper / wafer packet) | 2560×1440 | 20 | HEVC | 285 s | **2.1 Mbps** | **≈ 0.028** |
| Same camera, second file | 2560×1440 | 20 | HEVC | 286 s | 3.2 Mbps | ≈ 0.043 |
| Second camera | 1920×1080 | 15 | HEVC | 288–292 s | ≈ 1.0 Mbps | ≈ 0.032 |
| POC store entrance (`a228…`, `cctv_litter_poc_3min_h264`) | 640×368 | 20 | H.264 | 180 s | 0.5 Mbps | ≈ 0.11 |
| Footfall | 640×368 | 20 | H.264 | 187 s | 1.0 Mbps | ≈ 0.22 (burned-in overlay boxes) |
| Short clip | 1150×644 | 30 | H.264 | 14 s | 3.4 Mbps | – |

The high-resolution cameras are **heavily compressed**. At about 0.03 bits per pixel, fine texture on small objects
is smeared.

**Path through the system:**

1. The upload is kept in `storage/media/`.
2. `pipeline.detect_video` runs. It decodes every frame, samples at 2 fps (with static reuse), runs YOLO full-frame at
   640, then filters, then the tracker.
3. It writes:
   - `annotated.mp4` (every frame annotated and re-encoded)
   - `detections.json`, `events.json`, `stats.json`
   - an evidence JPEG per confirmed track
4. `video_evidence.py` then writes:
   - 4 random screenshots (one per quarter of the video), re-detected with tiling off
   - the six-column `litter_report.csv`
5. `litter_reports` writes the detection CSV and the summary CSV, and the daily consolidated CSV is appended.
6. Events go to the `litter_events` DB table.

The sampled "four-screenshot" jobs (`backend/sampling/runner.py`) instead call `detect_image`, which **tiles
automatically** on frames over 1,280 px.

### 1.9 ROI / polygon logic

There is none in the detection path **[verified]**. The leftovers are:

- `roi_versions` table in the DB (history)
- `models/roi_segmenter/` (unused weights)
- ROI-scoped labels in `exp_aug100`

None of them affect inference. **The plan below does not reintroduce ROI.**

---

## 2. Why small wrappers are missed: root causes ranked by evidence

### 2.1 Test case

- **Job:** `sv_20261119_184427_d1b60d`, frame s4 (video time 256 s), 2560×1440.
- **Green wrapper box:** `[751,1012,824,1051]` = **73×39 px native**. It lies in building shadow on wet pavers,
  next to small white paper scraps.
- **Visually it is clearly identifiable.** It is green with white print and distinct from the grey pavers.

### 2.2 Experiment A: input size, tiling, CLAHE

Production weights, conf floor 0.01, CPU. "Wrapper best conf" is the highest score of any box overlapping the
wrapper at IoU ≥ 0.3.

| Setting | Wrapper size at model input | Wrapper best conf (IoU) | Boxes ≥ 0.30 | Boxes ≥ 0.10 | ms per frame |
|---|---|---|---|---|---|
| full @ 640 (production) | 18×10 | **0.059** (0.75, `plastic_bag`) | 4 | 5 | 156 |
| full @ 960 | 27×15 | none ≥ 0.01 | 0 | 13 | 376 |
| full @ 1280 | 37×20 | none | 4 | 11 | 589 |
| full @ 1920 | 55×29 | none | 4 | 9 | 1,288 |
| full @ 2560 (native) | 73×39 | none | 3 | 18 | 2,214 |
| tiles 1024 + full 640 (the app's own still-image tiling) | 73×39 | 0.059 (0.75) | **14** | 46 | 3,456 |
| tiles 640, native pixels | 73×39 | 0.020 (0.94, `food_wrapper`) | 11 | 43 | 2,963 |
| CLAHE + full @ 640 | 18×10 | none | 2 | 7 | 127 |
| CLAHE + full @ 1280 | 37×20 | none | 3 | 14 | 470 |

### 2.3 Experiment B: what the boxes at or above 0.30 are (tiled run, inspected visually)

- **CRAX logo / carton print:** `food_wrapper` 0.71, 0.62, 0.53, 0.40, and `other_litter` 0.52. These are **false
  positives (stock).**
- **Paver spot** at (1076,893): 0.55. Probably a false positive.
- **Far-pavement speck**, 14×7 px: 0.73.
- **Slivers at the left frame edge**, 13×9 and 6×7 px: 0.52 and 0.36. **False positives**, and they come from tiles
  only.
- **Small object near the bin base:** `plastic_piece` 0.38. Unclear.
- **Green wrapper and white scraps:** not detected.

### 2.4 Experiment C: what lowering the confidence would cost (full @ 640)

| Conf floor | Boxes |
|---|---|
| 0.30 | 4 |
| 0.20 | 5 |
| 0.10 | 5 |
| 0.05 | 11 |
| 0.03 | 19 |

The wrapper (0.059) ranks **9th**. At least 4 of the boxes above it are carton logos.

An earlier session (2026-11-19) recorded the same result for the raw model output before NMS: the wrapper's best
class score was ≤ 0.06 at every scale and ≤ 0.003 on zoomed crops. The wafer packet in the same scene showed the
same pattern (0.01–0.10).

### 2.5 Causes, ranked

| # | Candidate cause | Evidence | Verdict |
|---|---|---|---|
| 1 | **Model limitation: appearance not learned** | The model localises the wrapper (IoU 0.75–0.94) but its class score is ≤ 0.06 at every scale, with tiling and with CLAHE. Native-pixel tiles do not help either. This camera's footage post-dates training. The training-data descriptions list no forecourt scene and nothing in shadow on wet pavers. | **Primary cause. Strong evidence [verified]** |
| 2 | **Training regime / domain gap** | Backbone frozen (`freeze=10`), only 30 CPU epochs, started from a synthetic-data model, trained at 640. Training data: a few clips from one 640×368 store camera, plus street scenes and synthetic images. | **Strong contributing cause.** The settings are verified; their exact effect is unverified |
| 3 | **Validation leakage hid the problem** | Val clip = same scene as a train clip (README). mAP50 0.995 is not plausible for real CCTV. No independent test set exists. | **Strong [from files]**. This does not cause misses directly, but it is why nobody saw them |
| 4 | **Learned false-positive bias (printed graphics → `food_wrapper`)** | Carton logos score 0.62–0.71 while real litter scores ≤ 0.06. | **Strong [verified]**. It blocks the "lower the threshold" workaround and points to a lack of hard negatives (stock, cartons) in training |
| 5 | **Resolution / downscaling** | At 640 the wrapper is 18×10 px, about 2×1 cells on the stride-8 head. That is marginal for any model. For **this** model, more pixels did not help (cause 1 dominates). For a **retrained** model, resolution matters: a 1440p frame at 640 shrinks small litter 4×. | **Contributing, secondary.** It becomes primary once cause 1 is fixed |
| 6 | **Compression / low light / shadow** | About 0.028 bits per pixel HEVC. The wrapper is in shadow on a wet surface; edges are smeared but the colour is preserved. Motion blur does not apply (static object). Not tested: night/IR footage (none on disk). | **Contributing, moderate.** Night: unverified |
| 7 | **Tiny-object scarcity in training** | v3 data absent. The local 1-class set has many tiny boxes, but only 2 scenes. | **[unverified]**. Likely, given cause 2 |
| 8 | **Annotation quality** | v3 labels absent. exp_aug100 is mostly good, with some ambiguous blobs and ROI-scoping. | **[unverified] for v3** |
| 9 | **Class imbalance** | v3 class counts unavailable. 18 of the 24 events in the existing job stats are `food_wrapper`. | **[unverified]** |
| 10 | **Class-mapping error** | Weights, YAML and registry are consistent. The miss is a low score for **every** class, not the wrong class. | **Ruled out as a cause of the miss [verified].** A separate reporting bug exists (M1) |
| 11 | **conf / NMS settings** | The conf floor of 0.30 is where the wrapper is dropped, but the model's score is the root problem (Experiment C). NMS: no stronger overlapping box suppresses it. | **Not the root cause [verified]** |
| 12 | **Post-processing filters** | Area, aspect and class filters do not touch a 73×39 box. The tracker needs ≥ 0.45 hits, which the wrapper never reaches. | **Ruled out for this case [verified]** |

### 2.6 Wrapper size by input resolution

Wrappers similar to this one (about 70×40 px native at 1440p) end up this size at the model input:

| Model input | Size | Stride-8 cells |
|---|---|---|
| 640 | 18×10 | 2×1 |
| 960 | 27×15 | – |
| 1280 | 37×20 | 4.6×2.5 |
| Native / tiles | 73×39 | – |

A cigarette-pack-sized item about 30 px wide at native resolution would be **about 7 px at 640**, below what P3
can reliably detect. That is the case where native-resolution tiles or a 1280 input become necessary.

---

## 3. Other findings that affect accuracy

### 3.1 Inconsistent inference paths

The same 1440p frame gets **different detectors** depending on how it arrives:

- **Still image or sampled screenshot:** tiling runs automatically (1024-px tiles plus a full frame at 640).
- **Video file:** full frame at 640 only.

In Experiment A, the tiled path tripled the number of boxes at or above 0.30 on this frame, mostly false positives
and edge slivers. Existing sampled reports show these slivers too, for example 13×9 px `food_wrapper` boxes at
x = 0. A user comparing a screenshot with the video result will see disagreement.

### 3.2 Training would need an external service

`docs/model_training/LDM_COLAB_TRAINING_AND_ROLLBACK.md` and `training/colab/` assume Google Colab. That uploads
CCTV frames to Google, which your rules forbid. **Decision needed** (section 5.7).

### 3.3 Defects found (none fixed; each needs approval)

- **D1. Summary CSV class columns (M1).** Fix with an explicit, approved v3 → report-class mapping, or by reporting
  v3's own names until a 7-class model exists. File: `backend/services/litter_reports.py`.
- **D2. Static reuse counts as temporal evidence.**
  - When a frame is "static", `pipeline.process(reuse_previous=True)` feeds a copy of the **previous model output**
    to the tracker as a new hit.
  - One inference can therefore supply all 3 required "confirmations" (reuse lasts up to 5 s = 10 sampled frames).
  - In the 1440p job, 305 of 563 analysed frames were reused.
  - This weakens the main false-positive defence: a carton logo seen once at 0.71 can become a CONFIRMED event.
  - Fix: count only fresh model runs as hits. Files: `backend/services/pipeline.py`, `backend/tracking/tracker.py`.
- **D3. Inconsistent tiling** (section 3.1). Files: `backend/services/pipeline.py`, `backend/detection/engine.py`,
  `backend/sampling/runner.py`, `backend/services/video_evidence.py`.
- **D4. Two class taxonomies (M2).** Documentation and owner decision only.

---

## 4. Proposed improvement plan

Each step is gated by your approval and by measurement on the fixed evaluation set (section 7). **Nothing ships on
the strength of a training-split metric.**

### 4.1 Higher-resolution inference (Phase 3, no retraining)

- Benchmark `imgsz` 640 / 960 / 1280 for the current model on the evaluation set: recall per size bucket, false
  positives per image, and ms per frame.
- **Expectation, from Experiment A:** little gain for v3, which was trained at 640. Do not expect it to fix the
  wrapper.
- The real gain comes from a model **trained** at the size it is run at (section 4.3).

### 4.2 Tiled / sliced inference (Phase 3)

- The project already has a SAHI-equivalent tiler (`backend/detection/tiling.py`). It has overlapping tiles,
  coordinate remapping, class-aware NMS and cross-tile containment suppression.
- **Recommendation:** benchmark the existing tiler first rather than adding the `sahi` dependency.
- Measure tile size 640 vs 1024, overlap 0.2, and with vs without the full frame.
- Add **edge-sliver suppression**: drop boxes that touch a tile border and are under N px. Experiment B showed the
  slivers come from tiles.
- Make video and still images use **one** consistent path (D3).
- For v3, tiling **raised false positives** without recovering the wrapper. So tiling is only worth enabling with a
  model trained at the tile scale.

### 4.3 Small-object training (Phase 4, needs your annotations)

- **Separate model version:**
  - name: `v4_<schema>_yolo11s_<imgsz>`
  - trained into a staging folder (`training/ldm7/train_yolo11s.py` already stages and never activates)
  - full fine-tune, no `freeze=10`
  - base: COCO `yolo11s.pt`, or v3 if the 10-class taxonomy is kept
- **Training on tile crops:**
  - Cut 1440p/1080p frames into 640–1024 px crops at native resolution, so small items keep their pixels.
  - Keep empty crops as negatives (target 30–50 % of crops).
  - Make sure every annotated object appears whole in at least one crop.
  - Train at imgsz = crop size, and run inference with the same tiling.
- **Full-frame training at 1280:** run in parallel as the alternative, then compare.
  - **Cost on this PC: about 3.8× the per-frame time of 640** (Experiment A: 589 ms vs 156 ms).
- **P2 head (stride 4):** only if, after the above, recall on objects under 16 px at model input is still poor.
  - The wrapper at 1280 or in native tiles is 37–73 px, well inside P3's range, so P2 is **not justified by the
    current evidence**.
  - It also adds substantial compute on a CPU-only machine. Exact cost unmeasured; to be benchmarked.
- **Augmentation:**
  - mild mosaic, `copy_paste=0`
  - **no** heavy blur or sharpen
  - HSV jitter for shadow and sun
  - optional JPEG/HEVC-like compression augmentation, matched to about 0.03 bits per pixel

### 4.4 Validated preprocessing

- **CLAHE:** in Experiment A it made no difference to the wrapper and lowered counts slightly. Test it **only** as
  an evaluation-set A/B on low-light and shadow strata, never as a blanket default.
- **Denoise:** same rule. Mild `fastNlMeans` or bilateral filtering, A/B only, on night strata.
- **No aggressive sharpening:** it amplifies HEVC block edges, which look like litter texture.
- **Rule:** any preprocessing used in inference must be identical in training.

### 4.5 False-positive reduction

1. **Hard-negative mining, with a human in the loop.**
   - Run the current and new models over unlabelled footage of your cameras and collect every box ≥ 0.30.
   - A human marks each one as litter or not.
   - Rejected crops go into training as **empty-label images**: cartons, stock, tyres, shadows, paver cracks,
     timestamp/OSD text, logos.
   - Nothing is auto-labelled into training.
2. **Telling litter apart from stock.**
   - Annotation rules, not new classes: stocked, stacked, held, sealed or displayed items are never labelled.
   - Include many frames of the shop front with stock and **no** litter.
3. **Optional second-stage verifier, off by default.**
   - A small crop classifier (litter / not-litter, or litter class) runs only on candidate boxes, at about 64–128 px.
   - Trained on the human-reviewed mined crops.
   - Adds cost only per candidate, not per frame. Estimated tens of ms per candidate on this CPU (unmeasured).
   - Justified only if the evaluation shows the detector's false positives cluster on a few patterns (as logos do
     here).
4. **Temporal tracking.** Fix D2 so that only fresh inferences count as confirmations. Keep the 3-hit rule and the
   stationarity rule. Optionally require hits spread over a minimum time span (for example ≥ 3 s).

### 4.6 Event de-duplication across frames

The current tracker already gives one event per track, and alerts are de-duplicated (900 s window, IoU/centre
match). Proposed additions:

- **Re-identification after occlusion.** A lost track that reappears at the same place (IoU ≥ 0.3 or centre within
  4 % of the diagonal, same class group) within N minutes becomes **the same event**, not a new one. Today, a
  person standing on the wrapper for longer than `max_missed_frames` (15 sampled frames = 7.5 s) creates a second
  event.
- **Daily CSV.** De-duplicate at the event level by (camera, location cluster, class group, day), keeping first and
  last seen.

---

## 5. Files that would be modified or created (after approval only)

### Phase 2: evaluation set and baseline

| Action | File |
|---|---|
| create | `training/eval/build_eval_set.py` (manifest of strata; copies only; checks camera/session separation) |
| create | `training/eval/run_eval.py` (wraps `training/ldm7/evalcore.py`; per-stratum precision/recall, FP/FN, mAP50/50-95, small-object recall, threshold sweep, ms/frame, FPS, example-image sheets) |
| create | `docs/model_training/EVAL_SET_SPEC.md` |
| modify (small) | `training/ldm7/evalcore.py` (per-stratum and negative-image FP reporting) |
| reuse unchanged | `training/ldm7/split_by_group.py`, `verify_dataset.py`, `compare_models.py`, `small_object_experiment.py` |

### Phase 3: imgsz and tiling benchmark

| Action | File |
|---|---|
| create | `training/eval/benchmark_inference.py` (imgsz × tiling × conf grid; ONNX/OpenVINO CPU speed option) |
| modify only if adopted | `backend/core/settings.py` (defaults), `backend/detection/engine.py` (tile batching, edge-sliver filter), `backend/detection/tiling.py`, `backend/services/pipeline.py` + `backend/services/video_evidence.py` + `backend/sampling/runner.py` (one consistent path, D3) |

### Phase 4: data and training

| Action | File |
|---|---|
| create | `training/ldm7/make_tile_crops.py` |
| create | `training/ldm7/mine_hard_negatives.py` (writes a review queue, never labels) |
| create | `training/ldm7/yolo11s-p2.yaml` (only if justified) |
| create, optional | `training/ldm7/train_crop_verifier.py`, `backend/detection/verifier.py` (off by default) |
| modify | `training/ldm7/train_yolo11s.py` (tile-crop mode; `freeze` option; local-GPU path) |
| output | `models/litter_yolo/v4_…/` staging, **not registered as active** |

### Defect fixes (separate approval each)

| Defect | Files |
|---|---|
| D1 | `backend/services/litter_reports.py` + tests |
| D2 | `backend/services/pipeline.py`, `backend/tracking/tracker.py` + tests |
| D3 | see Phase 3 |
| D4 | `docs/LDM_ANNOTATION_GUIDE.md` |
| Event re-identification | `backend/tracking/tracker.py`, `backend/services/litter_store.py` |

### Not touched

- dashboard features unrelated to detection
- AWS / storage
- the production `v3_store_cctv` weights and their backups
- the database schema

### 5.7 Decisions needed from you

1. **Taxonomy.** Is the approved taxonomy v3's 10 classes or ldm7's 7 classes (M2)? Annotation cannot start until
   this is settled.
2. **Training compute.** This PC cannot realistically train at 1280. The options are:
   - (a) a local or owned GPU machine (recommended, at least 8 GB VRAM)
   - (b) explicit permission to use Colab or another cloud service, which means uploading CCTV frames and breaks your
     current rule
   - (c) CPU-only training at 640 on tile crops: slow (days) and limited
3. **v3 training data.** Can you supply the `D:\YATRIKART\CCTV-LD\...\cctv_store_v2` dataset? Without it, v3's data
   quality stays unverified and cannot be reused.

---

## 6. Footage and annotations needed from you

**Why so much:** the current labelled data on this PC is 0 real-CCTV frames (beyond 2 synthetic scenes). The
counts below are minimums for a credible first model plus an independent test set. They are not guarantees of
accuracy.

### 6.1 Raw footage

- **Cameras:** at least 4 (ideally from 2–3 stores). Include the 1440p store-front camera and the 1080p camera.
- **Sessions:** per camera, at least **4 sessions on different days**, covering:
  - morning, midday (hard shadows), evening/dusk, and night (IR or artificial light)
  - at least 1 wet or rainy session
  - at least 1 busy session with people, vehicles and tyres
- **Length:** 10–30 min each, as **original NVR exports** (not screen recordings, and without burned-in overlay
  boxes). Keep the original resolution and codec.
- **Placed litter for controlled tests, if possible:** green and other small wrappers, wafer packets, cigarette
  packs, paper scraps, bottles, cups. Place them at near, mid and far distances, in sun and in shadow, and note
  where and when.

### 6.2 Annotation targets (human-verified labels, every image)

| Item | Minimum |
|---|---|
| Labelled frames, training pool (not the test cameras/sessions) | 1,500 |
| …of which negative frames (no litter: clean pavement, stock, cartons, tyres, shadows) | ≥ 40 % |
| Instances per litter class used | ≥ 300 (≥ 150 for rare classes) |
| Small instances (largest side < 32 px **native**) | ≥ 1,000 |
| Green / plastic wrappers specifically | ≥ 150, across ≥ 3 cameras or sessions |
| Hard-negative crops from mining (human-rejected) | ≥ 500 |

### 6.3 Annotation guide (summary; full version would extend `docs/LDM_ANNOTATION_GUIDE.md`)

1. **Litter = discarded.** Lying on the ground, kerb, steps or ledge, and nobody holding, selling, storing or
   stacking it.
2. **Never label:**
   - shop stock or displays
   - sealed or stacked cartons
   - items in bins or hands
   - vehicle parts or tyres
   - shadows, stains, wet patches, paver cracks, leaves
   - logos, posters, camera timestamp/OSD text
3. **Box every visible litter item, including tiny ones, down to about 6×6 px native.** The box should be tight to
   the visible extent.
4. **Ignore regions, not labels,** for:
   - items you cannot identify as litter
   - items more than 50 % occluded
   - edge slivers
   - motion-blurred items
5. **Use one class per item,** according to the approved taxonomy (decision 1). When torn between two classes, use
   `other_litter`. Never guess.
6. **Record metadata per frame:** `camera_id`, `session_id`, date, lighting (day/shadow/dusk/night/IR), wet yes/no.
7. **Sample frames sparsely.** At most one frame per 10 s or per scene change, and remove near-duplicates by dHash.
   Never label consecutive frames.
8. **Quality control:**
   - A second person re-checks 10 % of images.
   - Disagreements are settled and the guide updated.
   - Model-generated proposals may be shown as an aid, but **every** saved label must be human-confirmed, and
     proposals are never saved unreviewed.

---

## 7. Proposed fixed evaluation set

### 7.1 Ground rules

- **Independent.** The test set uses **held-out cameras and/or sessions** only. No session that contributes
  training frames may appear in it. The split is made with `training/ldm7/split_by_group.py` using
  `--test-group`, never at random.
- **Frozen.** Once built, its manifest and SHA-256 hashes are fixed. Old and new models are scored on **identical**
  images.
- **Matching.** IoU 0.5 for mAP50 and IoU 0.3 for small-object recall. Ignore regions are neither true nor false
  positives.

### 7.2 Strata

| Stratum | Content | Minimum |
|---|---|---|
| Tiny litter | largest side < 32 px native | 150 instances |
| Green wrappers | including in shadow and on wet surfaces | 50 instances, ≥ 2 cameras |
| Large litter | ≥ 96 px | 80 instances |
| Filled cartons / stock | must produce no detection | 100 frames / 200 objects |
| Tyres / vehicle parts | negative | 60 frames |
| Shadows / road markings / paver cracks / stains | negative | 80 frames |
| Clean pavement | negative | 100 frames |
| Difficult lighting (dusk, night/IR, backlight, rain/wet) | positives and negatives | 100 frames |

**Total: about 500–650 frames.**

### 7.3 Metrics reported per model, per stratum

- precision, recall, mAP50, mAP50-95
- false-positive and false-negative counts
- per-class results
- small-object recall
- false positives per negative frame
- threshold sweep at 0.05 / 0.10 / 0.20 / 0.30 / 0.45 / 0.60
- average ms per frame and FPS on this PC
- example sheets: correct detections, missed small litter, false positives, correctly ignored non-litter

---

## 8. Processing-time impact on the current hardware

**Hardware:** Intel i3-1115G4 (2 cores / 4 threads), 7.8 GB RAM, Intel UHD graphics. CPU inference only.

**Baseline, measured from `stats.json` of a real job:** a 281 s 2560×1440 video took **279 s wall time**, about
real time.

- 563 frames were analysed; 258 of them needed a model run (305 were reused).
- Mean inference was 208 ms, about **54 s** in total.
- The remaining **~225 s** was decoding, annotating every frame and re-encoding the 1440p output video.
- A 287 s 1080p video took 190 s.

**Estimates for the same 1440p video:** 258 model runs, with only the model time changing. The per-frame
milliseconds come from Experiment A.

| Inference mode | ms per model run | Model time | Est. total wall | vs. real time |
|---|---|---|---|---|
| full @ 640 (today) | 156–208 | ~54 s | ~280 s | 1.0× |
| full @ 960 | ~376 | ~97 s | ~320 s | 1.1× |
| full @ 1280 | ~589 | ~152 s | ~375 s | 1.3× |
| full @ 1920 | ~1,288 | ~332 s | ~560 s | 2.0× |
| tiles 640 (native) | ~2,963 | ~765 s | ~990 s | 3.5× |
| tiles 1024 + full 640 | ~3,456 | ~890 s | ~1,115 s | 4.0× |

**Caveats:**

- A busy scene needs more model runs: up to 563 here, since static reuse helps less. That would roughly double the
  tiled times.
- The live-camera budget at 2 fps is 500 ms per frame, so on this PC tiling cannot keep up live. Only `imgsz` ≤ 960
  fits one live camera.
- **Possible speed-ups to benchmark in Phase 3, all unverified:**
  - export to OpenVINO or ONNX (Intel CPU, often 1.5–3× faster)
  - batching tiles in one call
  - running tiles only on sampled frames with motion or change
  - an option to skip annotating every output frame. That is a dashboard behaviour, so it would be your call, not
    part of this work.
- A second-stage verifier adds cost only per candidate box.
- A P2-head model would add compute on every frame, by an amount not yet measured.

---

## 9. Completed, proposed, pending

**Completed in Phase 1, read-only:**

- code and config audit
- model and checkpoint inspection
- class-mapping cross-check
- statistics of the local datasets
- label-quality sample
- video probe
- Experiments A–C on the existing frame with the production model
- timing from existing job stats

**Proposed, not started:**

- Phases 2–5 (sections 4–7)
- defect fixes D1–D3
- annotation guide update (D4)

**Pending on you:**

- the decisions in section 5.7 (taxonomy, training compute, v3 data)
- the footage and annotations in section 6
- approval to start Phase 2

**Limits of this audit:**

- Experiments used **one** frame and one scene. They explain the wrapper miss but are **not** an accuracy
  measurement. That requires the Phase 2 evaluation set.
- Night/IR, rain and other cameras were not tested; no such labelled footage exists on disk.
- v3's training data could not be inspected.
- No claim of improvement or production readiness is made.
