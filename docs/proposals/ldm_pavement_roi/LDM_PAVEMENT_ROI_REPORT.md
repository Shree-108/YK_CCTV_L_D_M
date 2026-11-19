# LDM pavement-only detection: revised patch (review corrections applied)

Status: **revised patch for owner approval. Not applied to the active application. No ROI saved, approved or active.
No camera registered. Global pavement mode NOT activated. Nothing committed.**
Revision 2, 2026-11-16, against commit `5c33a72`. It replaces revision 1 of this report and the patch.

| File | What it is |
|---|---|
| `LDM_PAVEMENT_ROI.patch` | The complete diff (20 files, +1574 / −46). `git apply --check` passes on the current tree. |
| `roi_draft_EZVIZ_for_review.json` | A **draft** EZVIZ polygon, used offline for the evidence only. It is not approved. |
| `evidence/pavement_regression.py` | The real-camera regression (in memory; nothing is written to the database). |
| `evidence/pavement_regression_results.json` | Every detection in every configuration, with its outcome, reason, measurements and timing. |
| `evidence/pytest_*.txt`, `evidence/frontend_tests.txt` | Test output. |
| `storage/proposals/pavement_roi/` | Annotated images and hard-negative crops. These are real CCTV frames, so the folder is git-ignored. |

## What changed in revision 2 (one line per correction)

1. **Full-frame inference is the default.** `filters.roi_crop_inference` now defaults to **false** and is marked
   EXPERIMENTAL. Once a camera has an ROI, the ROI acts only as a post-detection filter. A test proves inference
   coordinates are identical with and without the ROI. In production the crop default was `true`; it is now
   `false`. No camera currently has an ROI, so nothing that runs today changes.
2. **Explicit analysis modes**, decided per frame by `pavement.decide_mode()`:

   | Mode | When | Behaviour |
   |---|---|---|
   | `CAMERA_ROI_MODE` | A **registered** camera with an approved include polygon whose frame shape matches | Pavement gate: membership, exclusions, size rule. The only mode that is pavement-validated. |
   | `GENERAL_IMAGE_MODE` | No camera association (`camera_id` empty or `upload`) | The **existing behaviour, unchanged**: filters, confirmation and alerts. Outcomes are `GENERAL_CONFIRMED` / `GENERAL_CANDIDATE`, labelled "not pavement-validated". |
   | `ROI_NOT_CONFIGURED` | A camera is named but is not registered, or has no approved ROI | Nothing confirmed, no alerts. Detections kept as `UNVERIFIED_ROI_NOT_CONFIGURED`. |
   | `ROI_FRAME_MISMATCH` | The frame's aspect ratio differs from `roi_reference_size` | As `ROI_NOT_CONFIGURED`. |

   - **My interpretation, please confirm:** a camera ID that is *named but not registered* is treated as
     `ROI_NOT_CONFIGURED`, not as general mode. That covers an `roi:<id>` saved without a camera record, which the
     existing automatic-ROI policy calls `SAVED_CAMERA_ROI`. It is the safe reading of "CAMERA_ROI_MODE requires a
     registered camera", and it stops an unknown camera from picking anything up.
   - **Scheduled captures** of non-litter models (the indoor uniform task) have pavement mode switched off, so they
     are never blocked.
3. **Size rule.** It stays configurable per camera (shorter side ≥ 0.012, area ≥ 0.00015) and is **experimental**.
   Every detection records why it was accepted, demoted or rejected; for example:
   - `accepted:INSIDE+significant_size+confirmed_by_existing_rule`
   - `below_min_size`, `below_confirmation_rule`, `low_confidence`
   - `outside_pavement_roi`, `exclusion_zone:<name>`

   Raw detections are kept. YOLO confidence thresholds are unchanged.
4. **EZVIZ false positives: investigated and recorded, no new rules** (§3).
5. **Camera isolation.** Each camera reads only its own record. Tests cover:
   - two registered cameras each keep their own polygons;
   - the same box is confirmed on one camera and rejected on the other;
   - updating one camera leaves the other unchanged;
   - an unknown camera gets nothing.

   No ROI saved, no camera registered, no identifier guessed.
6. **Nine new required tests added** (§4).
7. **Full suites and the real-camera regression rerun** (§2, §4); protected state verified (§6).

---

## 1. Design (summary; unchanged parts as in revision 1)

- **Membership:** the bottom-centre of the box (ground contact), in original-frame pixels. Inside the polygon, or
  within 0.01 × the frame's shorter side of it, or at least 50 % of the box area inside. A box crossing the boundary
  is not rejected for that alone.
- **Exclusions** are applied after membership: foot point inside an exclusion zone, or at least 50 % of the box inside.
- **Demotion happens before confirmation.** Only `candidate` detections can be confirmed (still images: confidence
  ≥ 0.60; video and live: the tracker), so detections outside the ROI, too small, or from unverified cameras can
  never reach an alert.
- **Unchanged:** weights, confidence thresholds, NMS, tiling, severity, alert de-duplication, the four-screenshot
  positions and stages, and the full-video stages. With `pavement.enabled = false` (the default), every path runs
  the original code.
- **Reports:**
  - CSV columns appended: `analysis_mode`, `pavement_validated`, `pavement_outcome`, `pavement_reason`, foot point,
    size fractions, `size_check`.
  - Summary counts kept separate: `confirmed_litter_detections` (ROI-validated) and `general_confirmed_detections`.
  - The JSON `pavement` block carries `pavement_validated: true/false`.
- **UI:** a mode badge ("Camera ROI mode · pavement-validated" or "General image mode · not pavement-validated"), the
  camera's own ROI drawn over the untouched original, and an optional diagnostic view showing rejected boxes.

## 2. Real-camera regression (repeated with revision 2)

Totals over the 4 screenshots per camera. Real v3 model, CPU, **full-frame tiled inference** in every row except X.
The polygons are proposals and are not approved; each camera uses only its own.

| Configuration | CAM_A raw | CAM_A confirmed | CAM_A candidates | CAM_A rejected | EZVIZ raw | EZVIZ confirmed | EZVIZ candidates | EZVIZ rejected |
|---|---|---|---|---|---|---|---|---|
| R0 production today | 15 | **3** | 5 (+7 weak) | 0 | 61 | **21** | 14 (+26 weak) | 0 |
| G general image mode (no camera) | 15 | 3 general | 12 general | 0 | 61 | 21 general | 40 general | 0 |
| C1 camera ROI mode, no size rule | 15 | 0 | 1 | 14 | 61 | 7 | 13 | 41 |
| **C2 camera ROI mode, proposed** | **15** | **0** | **1** | **14** | **61** | **2** | **18** | **41** |
| N registered camera, no ROI | 15 | 0 | 0 (15 unverified) | 0 | 61 | 0 | 0 (61 unverified) | 0 |
| X experimental ROI crop | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |

- **Raw detections retained:** C2 keeps all of them (15 / 61). Boxes are **byte-identical to R0 on all 8
  screenshots**, which confirms the ROI is a post-filter only.
- **General mode** reproduces production exactly: same boxes, same statuses, same confirmed count.
- **What happens to today's production confirmed and candidate detections under C2:**

  | Camera | Change | Count |
  |---|---|---|
  | CAM_A | confirmed → rejected outside the ROI (counter object) | 3 |
  | CAM_A | candidate → rejected (storage-box lid, side glass) | 5 |
  | EZVIZ | confirmed → rejected outside the ROI (clock, logo, vehicles, fence, awning) | 14 |
  | EZVIZ | candidate → rejected outside the ROI | 10 |
  | EZVIZ | confirmed → small candidate (7–16 px specks, conf 0.64–0.77) | 5 |
  | EZVIZ | candidate → still candidate | 4 |
  | EZVIZ | confirmed → still confirmed (§3) | 2 |

- **Genuine pavement detections lost: none found.**
  - No detection inside the pavement polygon was rejected.
  - The five rejections whose foot point lies within 80 px of the boundary are: the storage-box lid (×3), a weak box
    at the edge of the counter tray, and a 21×6 px weak box on the EZVIZ carriageway about 43 px above the kerb.
  - The five size-demoted specks stay visible as candidates for review.
  - The one plausible real small item (EZVIZ s3, a 32×22 px crumpled object, conf 0.55) is a candidate before and
    after. The size rule keeps it; the existing confidence rule does not confirm it.
  - **Limitation:** recall can't be measured without labels, and CAM_A has no real pavement litter on disk.
- **Experimental crop (X)** is now off by default. On real frames it removes almost every raw detection, so it is not
  deployable until recall has been evaluated independently.

**Processing time** (median per frame, this 4-core CPU, measured in one run):

| | Full-frame tiled inference | Gate only | Experimental crop |
|---|---|---|---|
| CAM_A | 10.9–13.4 s | 2.5 ms | 5.6 s |
| EZVIZ | 10.8–11.1 s | 10.1 ms | 0.66 s |

- The pavement gate adds **< 0.1 %**.
- The spread across full-frame configurations is machine noise: every row runs the same inference. Absolute times
  are about 30 % higher than in revision 1's run for every configuration, including unchanged production, which
  points to background load on the PC.

## 3. Remaining EZVIZ false positives (recorded; no rule added)

| | Detection A | Detection B |
|---|---|---|
| Screenshot / frame | s1 / f431 | s3 / f2800 |
| Class, confidence | `food_wrapper` 0.602 | `food_wrapper` 0.769 |
| Box | 1681,1040–1714,1054 (33×14 px) | 1278,1056–1350,1080 (71×24 px), **clipped by the frame's bottom edge** |
| Size (side / area) | 0.0128 / 0.00022 | 0.0217 / 0.00081 |
| ROI membership | INSIDE `outdoor_pavement` | INSIDE `outdoor_pavement` |
| Why accepted | foot point inside the ROI, significant size, conf ≥ 0.60 | same |
| Recurring? | **Yes, same spot**: s4 at 0.456 (candidate). Confirmed in 1 of 4 screenshots. | **Yes, same spot**: s1 at 0.336 (weak). Confirmed in 1 of 4. |
| What the crop shows | Night paver joint with small light specks | Paver joint and specks at the bottom edge |

- The crops are in `storage/proposals/pavement_roi/hard_negative_candidates/`; the full records are under
  `remaining_confirmed_hard_negative_candidates` in the results JSON.
- **Status: HARD_NEGATIVE_CANDIDATE, pending human review.** They are not added to any dataset.
- Both recur at the same position at varying confidence. That fits a static pavement texture rather than transient
  litter, but static real litter would recur too, so a person must decide.
- Removing them with a size limit would also drop the real small item, so no rule is proposed. The right fix is a
  human-reviewed hard-negative set for future fine-tuning.

## 4. Tests

| Suite | Result |
|---|---|
| Complete backend suite with the patch | **394 passed, 7 skipped** (the same historical v1 / clean-fixture skips as before), `PYTHONHASHSEED=0` |
| `tests/test_pavement_roi.py` + `tests/test_roi_upload.py` | **60 passed** |
| Frontend | **27/27 passed**; `tsc --noEmit` clean |

The nine required additional tests (all in `tests/test_pavement_roi.py`):

| # | Requirement | Test |
|---|---|---|
| 1, 2 | Full-frame is the default; crop is off by default | `test_full_frame_inference_is_the_default_and_roi_crop_is_off` |
| 3 | Post-filter keeps original coordinates | `test_roi_post_filtering_does_not_alter_inference_coordinates`; also on real frames (§2) |
| 4 | Manual upload works without registration (real model, API, same result as pavement off) | `test_manual_upload_without_camera_registration_still_works`, `test_general_image_mode_keeps_existing_results_and_alerts` |
| 5, 6 | Unknown camera inherits nothing; a registered camera without an ROI gives no confirmed result or alert (API) | `test_unknown_camera_cannot_inherit_another_cameras_roi_and_registered_camera_without_roi_confirms_nothing` |
| 7 | Two cameras keep separate ROIs | `test_two_cameras_keep_separate_roi_configurations` |
| 8 | Full-video and four-screenshot workflows still work | `test_full_video_workflow_works_with_and_without_pavement_mode`, the two zero-click job tests, and the unchanged existing workflow tests |
| 9 | CSV/JSON distinguish general from ROI-validated | `test_csv_and_json_distinguish_general_from_roi_validated_detections` |

`tests/test_roi_upload.py` was adjusted: its two crop tests now switch the experimental crop on explicitly, and one
assertion checks the new default.

## 5. Files changed

| File | Change |
|---|---|
| `backend/detection/pavement.py` (new) | `decide_mode`, the gate, general tagging, outcomes and reasons. |
| `backend/core/settings.py` | `PavementSettings` (off by default); **`roi_crop_inference` default → false (experimental)**. |
| `backend/services/pipeline.py` | `filter_detections(mode)` for image, video and live; general view labelled "not pavement-validated". |
| `backend/api/routes.py`, `backend/services/state.py`, `backend/cameras/scheduler.py`, `backend/sampling/runner.py` | Pass `camera_registered`. The scheduler switches pavement mode off for non-litter tasks. |
| `backend/sampling/detection_report.py`, `backend/sampling/reports.py`, `backend/api/routes_sampled.py` | `analysis_mode` and `pavement_validated` columns and summaries, `diagnostic.jpg`. |
| `backend/services/visualization.py`, `backend/cameras/manager.py`, `backend/detection/types.py` | Overlays; outcome fields. |
| `frontend/…` (5 files) | Mode badges, ROI overlay on the original, diagnostic toggle, tests. |
| `tests/test_pavement_roi.py` (new), `tests/test_roi_upload.py` | Tests. |

## 6. Protected state: verified unchanged

| Item | Result |
|---|---|
| Commit | `5c33a72`. No tracked file modified; the work was built and tested in an isolated scratch copy. |
| `storage/litter_monitor.db` | sha256 `10336d65…`, identical to the pre-work baseline: 0 cameras, 0 settings, 0 `roi:*`, 23 alerts, 3,259 detections. |
| `best.pt` | sha256 `d1de9cb6…`, identical. |
| `.env` | sha256 `cd9a990c…`, last modified 2026-10-05, before this work. |
| `backend/core/settings.py` in the active tree, `class_metadata.json` | Identical. |
| Cameras, ROIs | None registered, saved or approved. |
| Other | No training, global activation, commit, push or deletion. |

## Decisions for you

1. Approve applying the revised patch. It stays inert until `pavement.enabled` is set; per-camera activation is
   recommended.
2. Confirm the interpretation in correction 2: a named but unregistered camera gets `ROI_NOT_CONFIGURED`.
3. Approve or redraw both polygons, and supply the camera registration details (store ID, camera ID, NVR channel).
4. Human review of the two hard-negative candidates.
5. Phase 5: an independent recall evaluation (full-frame vs experimental crop vs tiled) on labelled CAM_A and EZVIZ
   footage.
