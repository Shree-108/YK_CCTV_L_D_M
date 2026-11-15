# Four-screenshot video analysis — audit and implementation plan

Status: **plan, not yet implemented** (2026-11-10). Scope: LDM only. No AWS, no uniform/clothing/staff logic.

## 1. Audit summary

### UDM (reference: `YK_CCTV_U_D_M_main/YK_CCTV_U_D_M`)

UDM has **two separate video pipelines**:

| | A. Quick sampled analysis (default) | B. Full-video analysis |
|---|---|---|
| Entry | `POST /api/media` → `POST /api/extractions` (`run_detection=true`) | `POST /api/analyses` |
| Worker | `ExtractionRunner` (`backend/extraction/runner.py`), one thread + in-memory queue, state in SQLite | `JobRunner` (`backend/analysis/jobs.py`) |
| Frames | 4 random frames, one per equal segment (`plan.py:111`), seed stored | every sampled frame, tracker |
| ROI | **none** — whole frame | operator-drawn polygon, bottom-centre point test |
| Output | annotated JPGs + per-job CSV + master CSV + Results Dashboard | annotated MP4 + alerts |

The step flow "Upload → Preview → Select camera → Store ROI → Analyse → Results" belongs to **B**; the
4-frame/CSV/dashboard workflow belongs to **A**. UDM never combined them. **UDM has no automatic ROI** — the
operator draws it. Automatic outdoor ROI is therefore new work, not a port.

Reusable UDM patterns (adapt, not copy):

- `probe()` metadata: packet-count frame length (NVR containers lie), readability check, preview frames.
- `plan_segments()`: stratified sampling, `random.Random(seed)`, refuses "too short" instead of returning fewer.
- `SequentialReader`: decode in order without seeking (seeking gives grey frames on HEVC NVR exports) +
  `frame_quality.problem()` grey/blank check; an unreadable frame is recorded as a failure, never replaced.
- Duplicate protection: SHA-256 on upload, 409 for a queued/running job on the same media, reuse only with the
  same explicit seed.
- Restart recovery: queued/running → `interrupted`.
- Reports: `/api/reports/overview`, `/months/{YYYY-MM}`, `/records`, hourly buckets that separate
  video-relative time from recording time (`results.py:_bucket`).
- Recording start parsed from NVR file names (`^\d{8}_(\d{14})_\d{14}_`), shown as an editable, pre-filled field.

Not copied: everything under `backend/uniform/`, uniform outcomes/remarks/CSV columns, compliance and vote logic,
all S3 modules, routes, DB tables and runner hooks (`s3_*`, `card_id`, `on_run_done`).

### LDM (this repository)

- Video today: `POST /api/detect/video` → `AppState.submit_video` → `detect_video` (tracker, annotated MP4).
  One shared `ThreadPoolExecutor(max_workers=1)`; table `jobs`; no duplicate protection; upload deleted after the job.
- Single-frame inference to reuse: `detect_image()` (`backend/services/pipeline.py:254`) — ROI crop, tiling,
  `apply_filters`, `confirm_single_image`, `annotate_litter`. **Defect:** writes detection rows with
  `job_id=None` even when a job id is passed (`pipeline.py:273`).
- ROI: `ROIZone` (normalised polygon, include/exclude) and `apply_filters` test the **bottom-centre foot point**
  with `roi_margin_frac=0.02`; exclude zones win. Stored per camera (`cameras.roi_json`) or `settings["roi:<id>"]`.
  **No ROI proposal exists.**
- No store field anywhere (camera has `location` only). Recording time is not parsed in the backend.
- Existing CSV schema (`/api/export/detections.csv`): `id, ts, camera_id, source_type, job_id, frame_idx,
  video_time_sec, class, confidence, x1, y1, x2, y2, status, reason, track_id, model_version, session_id`.
  No `job_id` filter.
- Model: only `v3_store_cctv/best.pt` exists (10 classes). Validation-split metrics only; no independent test set
  (audit blocker B1); 0/215 v4 frames annotated (B2); shoe sole confirmed as `plastic_piece` (D-07); shelf
  products never tested with zones (T09). CPU only.
- No segmentation libraries or weights are installed (`transformers`, SAM, `*-seg.pt` all absent).
- Every new route must be added to `tests/test_authz.py` `MATRIX`; new event texts to `_SAFE_EVENTS`.

## 2. Target workflow

```
Received → Validating → Preview → Sampling frames → Identifying ROI → ROI confirmation
        → Analysing → Generating images → Generating CSV → Completed      (or Failed / Interrupted)
```

The job **pauses at ROI confirmation** unless the automatic proposal is confident enough *and* the operator
allowed auto-accept. A stored camera ROI that still fits the sampled frames can be reused.

UI steps: **Upload → Preview → Select camera → Store ROI → Analyse → Results**. The existing full-video
pipeline stays available as "Full video (tracking)" — nothing is removed.

## 3. Design

### 3.1 Backend modules (new: `backend/sampling/`)

| Module | Responsibility |
|---|---|
| `media.py` | Streamed upload with SHA-256 and size limit; `probe()` (fps, frames via packet count, duration, resolution, readability). |
| `plan.py` | `plan_segments(frames, fps, count=4, seed)` — one random frame per quarter, plus a minimum gap between picks (default ≥ 5 % of duration) so neighbouring segments cannot produce clustered frames; raises `TooShort` instead of returning fewer than 4. |
| `extract.py` | Sequential decode up to the last sample, grey/blank check, save `frame_<i>_t<HHMMSSmmm>.jpg`; unreadable → recorded failure. |
| `roi_proposal.py` | Outdoor-ground proposal per frame + cross-frame consistency (see 3.3). |
| `runner.py` | Single worker thread + queue, state machine above, 409 on duplicates, restart → `interrupted`, cancel. |
| `csv_report.py` | Per-job CSV + append-only master CSV (ledger, dedup by `job_id, screenshot`). |
| `reports.py` | Summary, latest analysis, months, records, hourly (only with recording time). |

Routes: `backend/api/routes_sampling.py` (`/api/sampled/*`, `/api/reports/*`), each with exactly one permission:
uploads/runs/ROI confirm = OPERATOR, reads = VIEWER, CSV downloads = ADMIN_IF_LOGIN (same as today's export).

### 3.2 Database (additive migrations, existing `_migrate()` pattern)

- `media(id, created_at, filename, path, sha256, size_bytes, width, height, fps, frames, duration_sec)`
- `sampled_jobs(id, media_id, created_at, started_at, finished_at, stage, state, error, store_id, camera_id,
  recording_start, seed, sample_count, roi_json, roi_source[auto|camera|manual], roi_confidence,
  roi_status[pending|proposed|confirmed|uncertain], model_version, settings_json, output_dir, counts_json)`
- `sampled_frames(job_id, idx, frame_idx, video_time_sec, status[saved|unreadable|failed], original_path,
  annotated_path, roi_status, detection_count, error)`
- Detections reuse the existing `detections` table with `source_type='sample'`, `job_id`, `frame_idx`,
  `video_time_sec` (fixes the `job_id=None` defect for this path only, without changing image uploads).

Storage: `storage/sampled/<job_id>/{frames,annotated,roi}/`, `report_<job_id>.csv`; master CSV in
`storage/exports/litter_sampled_master.csv`. The original video is **kept** (UDM behaviour) under
`storage/uploads/` — unlike today's video jobs, which delete it.

### 3.3 Automatic outdoor ROI

Evidence: the 4 frames. Per frame, produce an "outdoor ground" mask, then:

1. Static-camera check: compare frames (edge/structure similarity). Static → fuse masks (majority vote) into one
   camera ROI; moved → per-frame ROI and flag.
2. Polygon: largest ground component → simplified polygon (≤ 12 points), normalised; indoor regions become an
   exclude zone.
3. Confidence = agreement across frames × mask quality (size, compactness, contact with the frame bottom,
   separation from indoor classes). Below threshold → `uncertain` → manual confirmation required.

How the mask is produced is the key open decision (see §5).

### 3.4 Detection on the 4 frames

`detect_image()` with the confirmed ROI (existing foot-point rule, `roi_margin_frac`), existing weights,
classes and thresholds. Boundary rule: a box whose foot point is outside but ≥ 50 % of its area is inside the
include zone is kept as `candidate` (not confirmed) and marked `roi_boundary` for review — default off until
measured. Per frame outcome: `LITTER_DETECTED | NO_LITTER | ROI_UNCERTAIN | UNREADABLE | FAILED`.

### 3.5 CSV

One row per detection (existing detection columns, same order), plus job fields, plus one row per screenshot
with no detection so all 4 screenshots are always traceable:

`video_name, store_id, camera_id, recording_date, video_timestamp, recording_timestamp, screenshot,
frame_idx, class, confidence, x1, y1, x2, y2, status, detection_count, roi_status, remark,
evidence_image, job_id, model_version`

Unknown store/camera/date stay blank — never guessed.

## 4. Phases

1. **Backend pipeline** — upload/probe, sampling, extraction, runner/states, manual ROI confirm, detection,
   annotated images, CSV, tests (sampling, short/corrupt video, unreadable frames, duplicates, restart, CSV).
2. **Frontend workflow** — Video Analysis steps, progress, 4 result cards, CSV preview/download.
3. **Results Dashboard** — summary, latest, latest CSV, monthly, history, hourly.
4. **Automatic ROI** — per the decision in §5, with measured agreement against hand-drawn ROIs.
5. **Model evaluation** — point the skipped tests at v3; build an indoor hard-negative test set; measure
   false positives before any retraining. **Retraining needs labelled data that does not exist yet** (B1, B2).

## 5. Decisions needed

1. Automatic ROI method (new dependency/weights vs classical heuristic vs manual-only).
2. Whether sampled analyses create alerts.
3. Store identity (free-text job field vs camera registry).
4. Which phases to do now.

## 6. Known limitations

- Validated on one camera at 640×368, daytime only; no pavement view yet (blocker B4).
- ROI accuracy and indoor false-positive rate cannot be claimed until measured on labelled frames.
- CPU only: 4-frame analysis is cheap, but a segmentation model adds seconds per frame.
