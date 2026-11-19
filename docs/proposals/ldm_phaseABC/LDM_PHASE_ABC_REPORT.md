# LDM Phase A/B/C: camera ROI proposal, independent dataset plan, E0 plan

Status: **proposal for owner review. Nothing is activated, trained, committed or deleted.**
Prepared 2026-11-16 on branch `security/step1-local-only` (after commit `5c33a72`).

## What was and wasn't changed

| Item | State |
|---|---|
| `cameras` table | unchanged (0 rows before, 0 rows now) |
| saved `roi:<camera>` settings | unchanged (0 rows) |
| detection settings, weights, thresholds, NMS, severity, dedup | unchanged |
| production pipeline code | unchanged |
| New files | `docs/proposals/ldm_phaseABC/*` (untracked) and the preview images under `storage/proposals/roi_cam/` (git-ignored, real CCTV frames) |

---

## 1. Camera registration details

**Camera identified (internal reference `CAM_A_IPCAM2560_COUNTER_VIEW`)**

| Property | Value (evidence) |
|---|---|
| View | Indoor camera behind the shop counter, looking out through the shopfront at interlocking pavers and the road opposite |
| Overlay | Clock top-left, "IP Camera" bottom-right |
| Resolution / fps / codec | 2560×1440, 20 fps, HEVC |
| Recording on disk | `20261110_20261110102031_20261110102517_102031.mp4`, 2026-11-10 10:20:31–10:25:17, 285.85 s, 5,696 frames, sha256 `60b59b93…a62a`, stored copy `storage/media/80de1ba8e8d849e1.mp4` |
| Analysis job | `sv_20261115_232628_e603d5` (store_id and camera_id were left blank) |

**Not the same camera** as these other cameras that appear in the repo:
- the EZVIZ outdoor camera (1920×1080, looks down at the awning from outside; the 2026-11-10 23:19 night clip comes from it);
- `indoor_V0_cam2560` (2026-10-29; same "IP Camera" overlay and resolution, so probably another channel on the same NVR, but it shows indoor shelves);
- the store-entrance camera (640×368: `poc3min`, `footfall`).

**Registration was prepared, not written.** See `camera_registration_draft_CAM_A.json`. There are two reasons:
1. The **store ID and the NVR channel/camera name** do not appear in the footage, the filenames or the repo.
2. `cameras.id` is the primary key. The API can't rename it, and deleting a wrong record is not approved.

Proposed record, applied with `POST /api/cameras` once you confirm the values:

```
id:       <STORE_ID>-cam-counter-pavement-01
name:     Counter view - pavement opposite shop
source:   env:LM_CAM_COUNTER_PAVEMENT_01   (stream URL kept out of the DB)
location: store:<STORE_ID>; indoor camera behind counter, looking out at pavers/road opposite the shop
enabled:  false      -> no live worker is started
roi:      []         -> analysis stays NOT_CONFIGURED (full frame, review required)
```

Notes:
- The `cameras` table has **no store_id column**. The store goes into `location`, and each upload must pass `store_id`.
- **Activation risk.** Writing the polygons into this record's `roi` field *is* activation: `auto_roi.decide()` returns `APPROVED / REGISTERED_CAMERA` (confirmed in `roi_validation_log.txt`). It also switches on ROI-crop inference, so the model sees a 1942×906 crop instead of the full frame.

## 2. Proposed ROI polygon (NOT approved, NOT active)

Previews (git-ignored):
- `storage/proposals/roi_cam/01_original_frame_axes_2560x1440.jpg`: original frame, axes and resolution
- `storage/proposals/roi_cam/02_proposed_roi_preview_clean_f3702.jpg`: proposal on a clean frame, vertices numbered P1–P36
- `storage/proposals/roi_cam/03_proposed_roi_preview_sweeping_f0570.jpg`: same proposal with a person on the pavement

Coordinates are pixels on the 2560×1440 frame, origin top-left. `roi_proposal_CAM_A.json` also holds the normalised `polygon` values in the LDM `ROIZone` format.

**Include: `outdoor_pavement` (ground).** Inset about 10–14 px from the visible ground edge.

```
P1 (648,38)   P2 (830,25)   P3 (878,80)   P4 (922,25)   P5 (926,25)   P6 (926,146)
P7 (1100,147) P8 (1280,172) P9 (1450,202) P10 (1640,229) P11 (1820,254) P12 (2000,284)
P13 (2085,305) P14 (2150,342) P15 (2180,360) P16 (2160,500) P17 (2153,560) P18 (2143,625)
P19 (2124,688) P20 (2100,790) P21 (1900,752) P22 (1900,684) P23 (1868,655) P24 (1775,655)
P25 (1745,684) P26 (1745,733) P27 (1716,726) P28 (1695,772) P29 (1285,676) P30 (1102,632)
P31 (1122,566) P32 (785,505)  P33 (700,541)  P34 (668,531)  P35 (686,432)  P36 (648,414)
```

**Exclude zones.** These are needed because production keeps any foot point within **0.02 of an include zone** (about 51 px horizontally, 29 px vertically). Without them, items standing in the trays would pass. Exclude zones always win and have no tolerance.

| Zone | Covers |
|---|---|
| `shop_left_rack_and_storage_box` | snack racks, storage-box lid and its reflections, door pillar |
| `counter_trays_devices_glass` | counter and food-warmer cabinet, both green trays, POS terminal, scanner, jar, the counter's glass top (pavement is visible *through* it), lower shelves, the bottom-left outdoor sliver seen through the side glass |
| `shop_right_racks` | right-hand racks and hanging merchandise |
| `awning_and_hanging_sign` | awning flaps (white flaps could look like paper) and the hanging sign at the top |

**Validation.** Run with the production code (`ROIZone`, `ROISet.zone_for`, foot point, margin 0.02), in memory only: **26/26 checks pass.**
- All **8 real detections** from the 2026-11-10 job are dropped: the counter shelf object (×3), the storage-box lid edge (×4) and the side-glass sliver.
- These are kept: pavement centre, small far litter on the road, litter right at the counter edge, at the tray rim, at the right edge and just under the awning.
- These are dropped: tray contents, the tray rim, the jar, the scanner head, pavement seen through the counter glass, awning flaps, the sign, racks, the box lid, the cabinet and the shelves.

**Decisions for you:**
1. The far **road strip** (top, y ≈ 25–200) is currently *included*. It's ground, but litter there is tiny and may be outside the store's responsibility. Exclude it?
2. A strip of about 10–14 px along each hard edge (counter, trays, awning) is not ground-truth pavement. Litter whose foot point is hidden behind the counter edge is still kept through the 0.02 tolerance, unless it falls inside an exclude zone.

## 3. Dataset collection and annotation manifest

**What exists today for CAM_A:** one 285 s recording.
- At about 28 s a person sweeps the pavement (broom, dustpan, blue debris). After that the pavement is clean.
- **There is no confirmed real litter from this camera on disk.**
- All 8 v3 detections in that clip are on non-ground surfaces.

**Proposed manifest:** `dataset_manifest_proposed_CAMA_20261110.csv`. It has 143 rows, one per 2 s, built from the source video by sequential decode. No frames were saved, and the workspace manifest is untouched.

| Result | Rows |
|---|---|
| Selected (distinct) | **64**: 20 clean-pavement candidates, 11 ambiguous sweeping frames, 33 with pavement activity (people/vehicles) |
| Near-duplicate of the previous kept frame (whole-frame dHash < 10 of 256 bits, the project rule) | 70, in no split |
| Burned-in NVR person box (green rectangle from the recorder's smart detection; 13 sampled frames, 9 of which would otherwise be selected) | 9, excluded |
| `annotation_status` | `not_annotated` for all rows |
| `human_review_status` | `pending` for all rows |
| `litter_class` | `pending_annotation` for all rows |
| Proposed split | `test` for all 64 (whole session held out: the only CAM_A recording, made after v3 was trained on 2026-09-30) |

Columns include:
- source file, sha256, media ID, session, recording date and start, frame index, timestamp, camera-clock time, dimensions;
- whole-frame dHash, pavement-only dHash, near-duplicate group;
- burned-in-overlay flag, pavement-change %, sharpness;
- proposed category (marked *heuristic, not a label*), litter class, annotation status and source, AI pre-label, human-review status, reviewer and date;
- selected flag, proposed split and the reason for it.

Issues found while building it:
- **Seeking in this HEVC file decodes corrupted frames.** Sequential decoding is required. `training/extract_frames.py` already decodes sequentially, but its `--start` option seeks once, so don't use `--start` on these exports.
- **`training.ldm_common.overlay_suspect()` flags 100 % of CAM_A frames.** It reacts to rack wire, awning stripes and paver lines. A colour-specific NVR-box check (pure-green lines, calibrated on f560 against f570/f2000/f4560) was used instead.
- **Whole-frame dHash will tie every CAM_A session together.** The static counter fills about half the frame. As a result `build_ldm_dataset` would put all days of this camera into one split. Proposed fix: compare frames on the **pavement-only dHash** (already in the manifest) and assign whole days explicitly with `split_lock`.

**Collection plan (needs new footage):**

| Need | Target | Source |
|---|---|---|
| Real litter, CAM_A, different days | ≥ 5 days, morning / afternoon / evening, 10–30 min exports containing litter, about 150 instances | NVR export of CAM_A |
| Small / distant litter | ≥ 60 instances < 32×32 px (far pavers / road edge) | same exports |
| Clean pavement | ≥ 3 days × 3 time slots, 5 min each (e.g. right after sweeping) | CAM_A |
| Hard negatives | ≥ 100 frames: counter-glass reflections, cabinet tops, product racks, afternoon shadows, wind-blown awning flaps, people carrying bags, vehicles | CAM_A |
| Night / low light | ≥ 1 session | CAM_A |
| Additional angle | ≥ 2 sessions (day and night) of the **EZVIZ outdoor camera** as an original export (only an annotated output exists today, and the tools rightly refuse it) | EZVIZ |

**Annotation rules:**
- AI proposals (`training/propose.py`, v3 output) are stored as `annotation_source=ai_proposal` and never count as labels.
- A label becomes ground truth only when a named person sets `human_review_status=approved` with a date.
- Every test image gets a second human review.
- **AUG100 is excluded from all test evidence.** It is synthetic, its labels are `verified_by: claude…` (AI-verified, not human-approved), and it is locked to train only.

## 4. Independent test-set plan

- **Test split:** two whole held-out CAM_A days (testing a known camera on unseen days) plus the whole EZVIZ camera (testing generalisation to a new camera). Today's 2026-11-10 session goes into test as clean and hard-negative evidence.
- **Validation split:** one further CAM_A day. Train gets everything else, and only later if training is approved; no training is planned now.
- **Leakage rules:**
  - Each session stays in one split.
  - Near-duplicates are tied on pavement-only dHash ≤ 10 bits, across sessions too.
  - Clean frames are capped at 20 per session, at least 30 s apart.
  - A near-duplicate that crosses splits is removed from train, never moved out of test.
  - Recordings must be later than 2026-09-30.
- **Minimum size before E0 numbers count as evidence:**
  - ≥ 100 litter instances, of which ≥ 50 are small;
  - ≥ 100 clean frames across ≥ 3 days;
  - ≥ 100 hard-negative frames.
  - With 100 instances a recall estimate is still about ±10 points at 95 % confidence.

## 5. E0 comparison configuration and resource estimate

Full configuration: `E0_comparison_config.yaml`. Everything runs offline with in-memory settings, so production is unchanged.

| Config | What it is | Passes per 2560×1440 frame |
|---|---|---|
| C1 screenshot / tiled (current) | `detect_image`, auto tiling | 7 (6 tiles + full); 4 with the ROI crop |
| C2 full video (current) | `pipeline.py:353` forces tiling off | 1 at 640 px (an 11×4 px object becomes about 3×1 px) |
| C3 full video, tiled (experimental) | `video.tiling=true` in memory. Variant a: production tile parameters. Variant b (optional, needs approval): tile 1280 / overlap 0.15 | 7 / 4 (a), 3 (b) |

- **Level A (per frame, same labelled frames for all configs):**
  - precision and recall at 0.45 (and 0.30);
  - mAP50 and mAP50–95;
  - small-object recall (< 32² px and < 16² px);
  - false positives per clean frame;
  - median and p95 time per frame;
  - duplicate boxes after tile merging.
  - C3a gives the same output as C1 per frame, so it is run once as a sanity check.
- **Level B (per clip):** object recall, false alerts per clean clip-hour, and wall time per clip, for the 4-screenshot job, C2 and C3.
- **ROI conditions:** R0 = no ROI. R1 = the proposed ROI applied offline, only after you approve it.

**Estimate on this PC** (4 CPU cores, no GPU). Measured anchors: 8.3 s per tiled 2560×1440 frame, 0.18–0.21 s per untiled video pass.

| Run | Time |
|---|---|
| 350 frames (Level A): C1 | about 48 min (about 27 min with the ROI crop) |
| 350 frames (Level A): C2 | about 2.5 min |
| 350 frames (Level A): C3b | about 21 min |
| 10 five-minute clips (Level B): C2 | about 20 min |
| 10 five-minute clips (Level B): C3 | **about 9 h** (about 5 h with the ROI crop); run overnight or on a GPU |
| Annotation | about 9 h first pass plus about 4 h second review |

## 6. Needs more footage or your approval

1. **Store ID** for this camera, and the **NVR channel/camera name**, needed before registration.
2. **ROI boundary approval**, including the road-strip decision (§2). Activating it means writing the ROI to the camera record.
3. **CAM_A footage from other days** containing real litter (there is none on disk today), plus clean, night and hard-negative sessions.
4. **An original EZVIZ export** (day and night) as the second camera angle.
5. **The proposed split policy:** pavement-only dHash, whole days held out, EZVIZ test-only.
6. **The E0 C3b speed variant** (tile 1280 / overlap 0.15). It is optional.
7. **Fixing the tooling gaps** found here (the `--start` seek, the overlay detector, region-based dHash). These are code changes to dataset tools only, as a separate approved task.
8. **Annotators:** who performs the first annotation pass and the second review on the test set.
