# LDM litter-focus implementation report (2026-11-19)

Scope: owner brief "Remove ROI dependency and make the application fully focused on litter, garbage and trash
detection". Contract: [API_CONTRACT.md](API_CONTRACT.md). Nothing was committed, pushed, deployed, trained or activated;
the live servers were not restarted.

## 1. Phase 1 - audit and backup

**Backup** (taken before any change): `C:\Users\chinm\LDM_backups\ldm_backup_20261119_175138\`
- `db\litter_monitor.db`: SQLite online backup. Also `models\litter_yolo\` and `storage\alerts\`, with SHA256SUMS, made by `deploy\backup-ldm.ps1`.
- `working_tree\`:
  - `BASE_COMMIT.txt`: c0b9e94.
  - `uncommitted_tracked.patch`: the 36 modified tracked files as they were.
  - `untracked_files.tgz`: 34 untracked files, including the ROI modules that were deleted later.
  - `full_source_snapshot.tgz`
  - `storage_reports.tgz`: existing `storage/sampled` and `storage/analysis_outputs`.

**ROI footprint found:**
- About 1,500 references in about 80 files.
- Backend modules:
  - `detection/pavement.py`
  - `detection/roi_status.py`
  - `services/roi_registry.py`
  - `services/image_analysis.py`, which rendered the ROI preview
  - `sampling/auto_roi.py`
  - `sampling/roi_calibration.py`
  - `sampling/roi_proposal.py`
  - `sampling/segmenter.py` (UperNet)
- `filters.ROIZone`/`ROISet` and the `outside_roi` rejection.
- Pipeline gate `filter_detections` → `pavement.gate`.
- Settings `pavement.*` and `filters.roi_*`.
- Routes:
  - `/api/roi/*`
  - `/api/sampled/jobs/{id}/roi`
  - `/api/sampled/jobs/{id}/roi/approve`
- Sampled-job stages `identifying_roi`/`roi_confirmation` (state `awaiting_roi`).
- ROI and pavement CSV columns.
- Frontend `roi.tsx`, `roiPolygon.ts`, plus ROI badges and editors across about 25 files.

**Historical data depending on ROI** (live DB, read-only check):

| item | count |
|---|---|
| `cameras` | 0 |
| `roi_versions` | 0 |
| `settings` rows | 0 (no stored `pavement`/`roi_*` keys) |
| `sampled_jobs` | 9, all `roi_status=not_configured`, `full_frame_fallback` |
| detections | 3,688 |
| alerts | 23 |
| confirmed objects | 24 |

**Store identity:** no stores table existed. Store ID was free text on sampled jobs only.

**Baseline tests** (before changes, with the earlier uncommitted work in place): 432 passed, 28 failed, 7 skipped. Most
of the failures were ROI-gate tests that contradicted that day's earlier uncommitted "detection first" change.

## 2. What changed

### Removed (active application)
- Deleted modules:
  - `backend/detection/pavement.py`
  - `backend/detection/roi_status.py`
  - `backend/services/roi_registry.py`
  - `backend/services/image_analysis.py`
  - `backend/sampling/auto_roi.py`
  - `backend/sampling/roi_calibration.py`
  - `backend/sampling/roi_proposal.py`
  - `backend/sampling/segmenter.py`
- ROI tests deleted.
- Removed from the code:
  - `ROIZone`/`ROISet`, the `outside_roi` filter and the ROI crop inference.
  - Settings `pavement` and `filters.roi_*`.
  - Config `LM_ROI_SEGMENTER*`.
- Removed routes: `/api/roi/*`, `PUT /api/sampled/jobs/{id}/roi` and `POST /api/sampled/jobs/{id}/roi/approve`.
- Removed request fields: `use_roi`, `roi_mode` and `auto_accept`.
- Removed from camera schemas: `roi` and `roi_reference_size`. Sending them now returns 422.
- Removed response fields: every `roi_*`, `pavement*`, `outside_roi*` and `diagnostic*` field.
- Removed CSV columns: the ROI and pavement columns.
- Sampled stages: the ROI stages are gone. Manual and automatic quick checks now go straight from sampling to analysis.
- Frontend: the ROI UI was removed (frontend agent, see section 6).
- No replacement region feature was added.
- Legacy settings keys (`pavement`, `filters.roi_*`) are dropped when settings are read (`settings.strip_legacy`), so old
  stored settings can never fail validation or affect detection.

### Kept as history (never read or written by active code)
- `cameras.roi_json`
- the `roi_versions` table
- `sampled_jobs.roi_*`/`proposal_json`
- `sampled_frames.roi_*`
- old event-log message patterns

Old outputs in `storage/analysis_outputs/` and `storage/sampled/` were not touched. `models/roi_segmenter/` weights
remain on disk, unused.

### Database (additive migration, schema version 1 → 2, runs automatically at the next backend start)
- New tables:
  - `stores` (id, name)
  - `analyses`: one row per image, video or quick check. Holds identity, `processing_status`, counts, files and error.
  - `litter_events`: one row per unique DETECTED object. Holds identity, class, confidences, bbox, first and last seen,
    duration, `observation_count`, `review_status` and `timestamp_source`.
- New column `cameras.store_id`.
- No column or table was dropped or rewritten.
- `upsert_camera` no longer overwrites `roi_json`.
- **The migration has run only on test databases.** It will run on `storage/litter_monitor.db` at the next restart, which
  needs owner approval. Take a backup first.

### Detection and video engine
- Full-frame inference only. Every detection at or above `confidence.weak_threshold` (0.30, unchanged) is kept, tagged
  and drawn: DETECTED is solid, UNDER_REVIEW is dashed amber, REJECTED is dotted grey with its reason.
- Statuses (`litter_reports.detection_status`):
  - DETECTED: image confidence at or above `confirmed_threshold`, or the video track passed temporal confirmation.
  - TRACKED: any other observation of an event's object.
  - UNDER_REVIEW
  - REJECTED
  - VERIFIED, RESOLVED and REJECTED: operator review via `PATCH /api/litter/events/{id}`.
- Full video (`pipeline.detect_video`):
  - Validation: codec, fps sanity, frames reported vs decoded, truncation warning, resolution changes and the
    `max_video_seconds` limit.
  - Configurable sampling interval (`sample_interval_sec`).
  - ByteTrack-style tracking (existing `LitterTracker`).
  - One evidence JPEG per event.
  - `video_litter()` builds observations, tracks and events. One object seen in N frames is one event with
    `observation_count` N.
  - The original upload is kept in `LDM_OUTPUT/.../input/` (`video.keep_original_video`, default on).
- Live cameras: `live_events.LiveEventRecorder` records one event per confirmed track and updates it when the track is
  lost.

### Store, camera, reports, analytics
- Identity (`litter_store.resolve_identity`):
  - A registered camera's store comes from its record.
  - A conflicting store returns 422.
  - An unregistered store ID is kept, with name UNKNOWN and a warning.
  - Missing values become UNKNOWN with a warning.
- Standard outputs per analysis:
  - Location: `storage/LDM_OUTPUT/<STORE>/<YYYY>/<MM>/<DD>/<CAMERA>/{input,annotated,detections,reports,metadata}`.
  - File names carry the analysis id and are never overwritten.
  - Detection CSV (22 columns), summary CSV (15 columns) and metadata JSON.
- A failed analysis writes metadata (FAILED) only. No CSV is written.
- Zero DETECTED objects gives `COMPLETED_NO_LITTER_DETECTED`, never "clean".
- New routes:
  - `GET/POST/PUT /api/stores`
  - `GET/PATCH /api/litter/events`
  - `GET /api/litter/analyses`
  - `GET /api/litter/analytics`: hour, day or week buckets; store-wise, camera-wise and class-wise; unique events vs
    frame observations; average visible time; processing status counts. Operator-rejected events are excluded.
  - `GET /api/analyses/{id}[/files/{kind}]`
  - `GET /api/jobs/{id}/evidence/{event}.jpg`
- `class_metadata.json`: added `plastic_packet`, `food_waste` and `glass`, with provisional priorities, owner to confirm.

## 3. Phase 4 - small-object findings (not re-run today)

The 2026-11-19 investigation measured that v3_store_cctv scores the 68×32 px wafer packet at only 0.01–0.10. Image
size 960 or 1280, tiling and zoomed crops did not raise it. The cause is a model gap, not resizing, NMS or the class
map.

Lowering the global floor would mostly add false positives: v3's detections at or above 0.30 in that scene are mostly
carton logos, a head and paver cracks. So the floor stays at 0.30 until it is calibrated on a labelled validation set.

`training/ldm7/small_object_experiment.py` is the controlled-experiment tool for any new model. It sweeps image size,
tiling and threshold, and reports per-GT-box recall by size bucket and false positives per image. It has not been run
on real data, because no labelled CCTV set exists.

## 4. Phase 7/8 - dataset and training (tooling only; nothing trained)

`training/ldm7/` and [`docs/model_training/ldm7/LDM7_TRAINING_WORKFLOW.md`](../model_training/ldm7/LDM7_TRAINING_WORKFLOW.md):
- Schema: exact IDs 0–6.
- `verify_dataset`: images, labels, class IDs, missing label files, duplicates and near-duplicates across splits, and
  group leakage.
- `split_by_group`: splits by camera, session or day.
- `train_yolo11s`: starts from COCO YOLO11s, not v3. It writes a versioned package to a staging folder and never
  installs or activates it.
- `small_object_experiment`
- `compare_models`: v3 against the candidate on the same test set. It reports "INSUFFICIENT EVIDENCE" when the test set
  is too small or not independent.

The v3→7-class mapping is used for comparison only and is never applied to live output. Tooling tests: 23 passed.

**Model comparison (Phase 8) is blocked:** there is no seven-class model and no labelled independent test set.

## 5. Blockers needing the owner

1. **Labelled data:** no labelled seven-class CCTV data exists, so the model cannot be trained.
2. **Missing classes:** there are no glass or food_waste examples in any source.
3. **Test set:** no independent labelled CCTV test set exists (held-out camera or day), so accuracy claims are
   impossible.
4. **Store master data:** no stores or cameras are registered. Results are UNKNOWN until an admin adds stores and
   assigns cameras.
5. **Restart:** applying this to the live app needs approval. The steps are backup, stop, `start-ldm -Rebuild`, and the
   additive DB migration runs at the restart.
6. **Staff and captain mapping (Part 9):** not implemented. No roster or shift data exists, and CCTV alone does not
   establish responsibility.
7. **Frequency alerts:** "N events per hour per camera or store" alerts are not implemented. The existing per-event
   alerts with track and IoU de-duplication remain.
8. **AWS S3:** not started (out of scope). The folder layout is S3-key compatible.

## 6. Tests and acceptance

### Test runs

| Suite | Result |
|---|---|
| Backend, full suite (2026-11-19, after the changes) | **375 passed, 7 skipped, 0 failed** (6 min). The 7 skips are existing ones: v1 artefacts and clean frames are not on this machine. |
| Frontend: `npm run test:frontend` | 35/35 |
| Frontend: `tsc --noEmit` | clean |
| Training tooling: `tests/test_ldm7_tools.py` | 23 passed (part of the full suite) |
| Before the changes | 432 passed, 28 failed (old ROI-gate tests) |

The backend count is lower than before because the ROI-only test files were removed. Their behaviour no longer exists.

### Live run on 2026-11-19 18:26

1. Fresh backup taken: `ldm_backup_20261119_182547`.
2. Started with `deploy/start-ldm.ps1 -Rebuild` (backend :8000, UI :3010).
3. The live DB migrated to `user_version` 2. History was unchanged: 3,688 detections and 23 alerts.

**Smoke image analysis:**
- One analysis on the fixture image (`img_20261119T182743_28ac26`, store/camera UNKNOWN, alerts off).
- It gave 21 detections: 19 DETECTED and 2 UNDER_REVIEW.
- Processing status: `COMPLETED_LITTER_DETECTED`.
- All 5 files downloaded, and both CSV headers are exact.
- This is one smoke record in the live DB, with its files under `storage/LDM_OUTPUT/UNKNOWN_STORE/...`.

**UI check (headless Chrome):** Overview, Image Analysis, Video Analysis and Cameras all render live data, with no console errors, no HTTP errors and 0 ROI mentions.

### Acceptance tests (brief Part 13)

| # | Scenario | Result | Evidence and limitation |
|---|---|---|---|
| 1–7 | Wafer packet, chips packet, bottle, paper, food, glass, cardboard on real CCTV | **NOT RUN, blocked** | No labelled real CCTV footage per class and no 7-class model. v3 is known to miss the wafer packet (scores 0.01–0.10, 19 Nov). v3 cannot output food_waste or glass. |
| 8 | Clean pavement false positives | **NOT RUN, blocked** | No human-verified clean frames (`tests/fixtures/clean` is empty). |
| 9 | Shelf merchandise not classified as litter | **NOT RUN, blocked** | No labelled shelf footage. On 19 Nov, v3 boxed CRAX carton logos, so this is a known weakness. |
| 10 | Same litter in consecutive frames | **PASS** | `test_litter_api` video test: events = confirmed tracks, observations ≥ 3× events, each event has 1 DETECTED row and the rest TRACKED. |
| 11 | Two stores and cameras | **PASS** | `test_litter_api`: correct IDs and names in identity, CSV, events and `by_store`. A mismatching store gives 422 and nothing is stored. |
| 12 | Missing metadata | **PASS** | UNKNOWN with warnings (`test_litter_api`, `test_litter_reports`, live smoke). |
| 13 | CSV matches DB and inference | **PASS** | CSV row counts and totals equal the job result, `/litter/events` and `/litter/analytics`. |
| 14 | Corrupt or failed video | **PASS** | Upload gives 400 with no job. A failure during processing gives job `failed`, analysis FAILED and no CSV on disk. |
| 15 | ROI-free operation | **PASS** | No ROI key at any depth of the API responses. Removed routes give 404/405, and ROI fields give 422. Live UI shows 0 ROI mentions. |

### Production readiness: not production-ready

**Passed:**
- ROI-free full-frame pipeline.
- Statuses.
- Event and observation separation.
- Store and camera identity.
- CSV and metadata outputs.
- Failure handling.
- Analytics API and UI.
- Authorization matrix tests.
- Additive migration on the live DB.

**Failed or blocked:**
- Seven-class recognition.
- Small-object detection.
- False-positive control on shelves.
- Calibrated thresholds.
- Independent CCTV evaluation.

**Untested:**
- Night-time footage, shadows and reflections, camera movement, long recordings over 1 h, multiple resolutions on real NVR exports, GPU throughput, concurrent jobs (one video worker by design), and live RTSP event recording against a real camera.
- Memory under load. The machine runs at about 87% RAM idle.

### Contract notes (code is authoritative)
- Video job CSVs are listed in `job.files` and are served from `/api/analyses/{analysis_id}/files/{kind}`, not
  `/api/jobs/{id}/files/{kind}`.
- Quick-check standard CSVs are under `standard_outputs.urls`. `reports` still lists that job's legacy per-job reports
  (without ROI columns).
