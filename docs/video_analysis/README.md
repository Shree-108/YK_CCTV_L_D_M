# Four-screenshot video analysis

Default workflow on **Video Analysis** (`/detect/video`). The earlier full-video tracking analysis is still there
under **Full video (tracking)** (`/detect/video?mode=full`). Results are on the **Results Dashboard** (`/results`,
also reachable from Video Analysis and with Ctrl+K).

Litter only, local storage only. No uniform/clothing/staff logic and no cloud storage.

## Zero-click analysis (default since 2026-11-15)

Dropping a video on **Video Analysis** is the only action. The page checks the file type and size, then sends one
request: `POST /api/sampled/media` with `auto_analyze=true` (and the optional remembered Camera ID / Store ID
defaults). The server stores and checks the video and creates the analysis (`mode=auto`) in the same request. The
page then only reads the job until it finishes. The manual workflow below is still available under
**Review settings and ROI before analysis** (`/detect/video?flow=manual`). The full-video tracking analysis is
unchanged.

| | Automatic (`mode=auto`) | Operator workflow (`mode=manual`) |
|---|---|---|
| Screenshots | fixed: 10 / 35 / 65 / 90 % of the frames (`plan_fixed`) | random, one per quarter, seeded |
| Unreadable planned frame | nearest readable frame within about 1 s (never overlapping a neighbour's window) is used; the planned frame, the frame used and the reason are recorded (`sampled_frames.planned_frame_idx`, `sampling_note`). No readable frame nearby: the screenshot is `UNREADABLE` | recorded as `UNREADABLE`/`FAILED`, never replaced |
| ROI | `backend/sampling/auto_roi.py`: 1. ROI of the registered camera, 2. ROI saved for that camera ID, 3. otherwise the whole frame with `roi_status=NOT_CONFIGURED`. The ROI segmenter is not run, and nothing is approved automatically | proposal + operator confirmation |
| Alerts | only when an approved ROI was applied; full-frame detections are marked `review_required` instead | as chosen by the operator |
| Pauses | never | at ROI confirmation |

Job lifecycle (`lifecycle` field, derived from the stored state/stage): `QUEUED → PROCESSING →
EXTRACTING_SCREENSHOTS → RUNNING_DETECTION → GENERATING_REPORT → COMPLETED`, or `FAILED` / `CANCELLED`
(`UPLOADING` exists only in the browser). A server restart during processing gives `FAILED` (stored as
`interrupted`). A video that cannot be opened, or is too short for four separate screenshots, is refused at upload
(HTTP 415 or 422) and no analysis is created.

Duplicate protection: the page ignores the same drop delivered twice while it is uploading. The server returns the
running job, not a new one, when a file with the same SHA-256 is already being analysed automatically
(`duplicate_of_active: true`). It checks this before probing and again under the runner lock.

### Reports (every completed job, written before it is marked completed)

`storage/sampled/<job_id>/reports/`:

- `detection_report.csv` has one `SCREENSHOT` row per screenshot (always 4) plus one `DETECTION` row per detected
  object per screenshot: confirmed litter, possible litter (candidate) and detections removed as outside the ROI.
  The columns are the requested sections A–E plus `screenshot_analysis_status` and `overall_analysis_status`.
  Traceability columns follow: `planned_frame_number`, `sampling_adjustment`, `object_sightings`,
  `object_max_confidence`, `rejection_reason`, `model_version`.
- `analysis_summary.csv` has one row with the job-wide totals: `total_objects_detected`,
  `unique_objects_estimated` and the `total_*_severity` counts. These are not repeated on detection rows, so a
  spreadsheet sum cannot multiply them.
- `analysis_summary.json` holds the summary, the ROI decision, the per-screenshot status and the estimated objects.

Definitions:
- **Severity** uses the existing rules (`compute_severity`, Settings → severity). The severity totals count each
  estimated unique object once, at its highest severity.
- **Estimated unique objects** links detections across screenshots with the existing alert de-duplication rule:
  similar class group, IoU ≥ `alerts.dedup_iou` or centre distance ≤ `alerts.dedup_center_frac`. This is not
  the tracker's stationary confirmation; four screenshots have no consecutive frames.
  `stationary_confirmation_status` is therefore `REPEATED_AT_SAME_POSITION` or `SINGLE_OBSERVATION`.
- `confidence_score` is the model's confidence for that detection. Cross-screenshot evidence is in separate columns.
- All text cells are protected against spreadsheet formula injection (`backend/core/csv_safety.py`).

The legacy per-job `litter_report_<job_id>.csv` is still written. The monthly and all-time CSVs and the Results
Dashboard use it.

Downloads: `GET /api/sampled/jobs/{id}/reports/{detection_report.csv|analysis_summary.csv|analysis_summary.json}`
(admin when login is on, same as the existing CSV). For screenshots, add `?download=1` to the frame image URLs.

## Operator workflow (`?flow=manual`)

| UI step | Backend stage(s) | What happens |
|---|---|---|
| Upload | `received`, `validating` | `POST /api/sampled/media` streams the file, checks it decodes, counts the real frames (NVR containers often report wrong lengths) and stores the SHA-256. A previously received identical file is flagged. |
| Preview | (upload time) | Preview frame + file name, size, resolution, duration, fps, total frames. |
| Select camera | — | Optional Store ID, Camera ID, recording start (pre-filled from NVR names such as `20261029_20261029183009_…`), ROI method, alerts. |
| Store ROI | `sampling`, `identifying_roi`, `roi_confirmation` | 4 random frames, one per quarter of the video, decoded in order (never by seeking). The outdoor area is proposed; the operator confirms, edits, redraws, clears, skips a screenshot or explicitly chooses the whole frame. |
| Analyse | `analysing`, `generating_images`, `generating_csv` | Existing YOLO model on the confirmed ROI of each screenshot; annotated images; CSV. |
| Results | `completed` | 4 result cards, CSV preview/download, Results Dashboard entry. |

States: `queued`, `running`, `awaiting_roi`, `done`, `failed`, `cancelled`, `interrupted` (server restarted).
One active analysis per uploaded video (HTTP 409 otherwise). The same video + explicit seed + same settings reuses the
finished result instead of running again.

## Sampling rules (`backend/sampling/plan.py`)

- Exactly 4 frames. Frame 0 is never used.
- One random frame per equal quarter of the usable frames (`random.Random(seed)`), consecutive picks at least
  5 % of the video (minimum 2 frames) apart.
- The seed is stored on the job; the same seed gives the same frames.
- A video too short for 4 well-separated frames is refused (HTTP 422). Never fewer or more than 4.
- A planned frame that cannot be decoded, or decodes as grey fill / blank picture, is recorded as
  `UNREADABLE`/`FAILED` and never replaced by another frame.

## Outdoor ROI (`backend/sampling/roi_proposal.py`, `segmenter.py`)

Scene segmentation: UperNet + ConvNeXt-tiny trained on ADE20K (`openmmlab/upernet-convnext-tiny`, MIT licence),
stored in `models/roi_segmenter/upernet-convnext-tiny/` (the 241 MB weights file is git-ignored). Runs on CPU,
no network access at run time.

1. Walkable ground (road, sidewalk, path, earth, grass, … and floor) minus shop-interior structures (shelving,
   display cases, boxes, counters, ceiling) → largest region → one include polygon.
2. View check across the 4 frames: same view → one camera ROI from the averaged maps; changed view → one ROI per
   screenshot and per-screenshot confirmation.
3. Confidence = agreement × plausible size × separation from interior × segmenter certainty × outdoor specificity.
   ADE20K labels outdoor paver pavement mostly as "floor", which cannot be told apart from a shop floor, so a
   region that is only "floor" scores low and **always goes to the operator**.
4. Auto-accept only when the operator enabled it **and** the proposal is consistent with confidence ≥ 0.70.

Detection filtering uses the existing LDM rule: the bottom-centre of a box (ground-contact point) must lie inside
an include zone (2 % margin); exclude zones always win.

Configuration (`.env`): `LM_ROI_SEGMENTER=off` disables the proposal (manual ROI only);
`LM_ROI_SEGMENTER_DIR` points to other weights. Re-download the weights with:

```
python -c "from huggingface_hub import snapshot_download; snapshot_download('openmmlab/upernet-convnext-tiny', local_dir='models/roi_segmenter/upernet-convnext-tiny', allow_patterns=['*.json','*.safetensors','*.md'])"
```

Python dependency added: `transformers` (see `requirements.txt`).

## Outputs and storage

| What | Where |
|---|---|
| Uploaded video (kept) + preview | `storage/media/<media_id>.<ext>`, `<media_id>_preview.jpg` |
| 4 screenshots | `storage/sampled/<job_id>/frames/<video>_s<n>_f<frame>_t<HHMMSSmmm>.jpg` |
| ROI proposal images | `storage/sampled/<job_id>/roi/proposal_s<n>.jpg` |
| Annotated screenshots | `storage/sampled/<job_id>/annotated/…_annotated.jpg` |
| CSV report | `storage/sampled/<job_id>/litter_report_<job_id>.csv` |
| Job, frames, media | SQLite tables `sampled_media`, `sampled_jobs`, `sampled_frames` |
| Detection rows | existing `detections` table, `source_type='sample'`, `job_id`, `frame_idx`, `video_time_sec` |
| Alerts | existing `alerts` table (one per object per job; the same object in several screenshots is merged) |

Existing paths and naming (`storage/uploads`, `storage/videos`, `storage/alerts`) are unchanged.

## CSV columns

`video_name, store_id, camera_id, recording_date, video_timestamp, recording_timestamp, screenshot, frame_idx,
class, confidence, x1, y1, x2, y2, status, detection_count, outcome, roi_status, remark, evidence_image, job_id,
model_version`

One row per confirmed litter detection; a screenshot without one still gets exactly one row (outcome
`NO_LITTER`, `ROI_UNCERTAIN`, `UNREADABLE` or `FAILED`). Class names, confidence (0–1) and pixel boxes follow the
existing `/api/export/detections.csv`. Unknown store/camera/recording date stay blank. `evidence_image` is relative to
`storage/`. Monthly and all-time CSVs use the same columns.

## API (all under `/api/sampled`, permissions in `tests/test_authz.py`)

| Method | Path | Permission |
|---|---|---|
| POST | `/media` (optional form fields `auto_analyze`, `camera_id`, `store_id`) | operator |
| GET | `/media/{id}`, `/media/{id}/preview.jpg` | viewer |
| POST | `/jobs` | operator |
| GET | `/jobs`, `/jobs/{id}`, `/jobs/{id}/frames/{n}/{original\|annotated\|proposal}.jpg` | viewer |
| PUT | `/jobs/{id}/roi` (saving a *registered* camera's ROI additionally needs admin) | operator |
| POST | `/jobs/{id}/cancel` | operator |
| GET | `/jobs/{id}/report.csv`, `/jobs/{id}/reports/{name}`, `/reports/months/{YYYY-MM}.csv`, `/reports/all.csv` | admin when login is on |
| GET | `/reports/overview`, `/reports/months/{YYYY-MM}`, `/reports/history`, `/reports/detections`, `/reports/hourly` | viewer |

## Time rules

- *Video timestamp* = position in the file. Always known.
- *Recording timestamp* = recording start + video timestamp, only when a recording start was given.
- Monthly reports: by analysis month (server local time) or by recording month (analyses without a recording start
  are excluded).
- Hourly summary: recording time only; analyses without a recording start are counted as excluded, never shown as
  clock times.

## Tests

`tests/test_sampled.py` (sampling, decoding, ROI proposal with a stub segmenter, full API workflow with the real v3
model), `tests/test_sampled_cancel.py` (a single cancel during ROI identification stops the job),
`tests/test_csv_export_safety.py` (spreadsheet-formula protection of every CSV export), `tests/test_authz.py`
(permission matrix), `frontend/tests/csv.test.mjs`.

**Evaluation on real footage: not documented yet.** `EVALUATION.md` (the measured ROI-proposal results that
`backend/sampling/segmenter.py` refers to) was never written, and those measurements are not in the repository. Until
an independent evaluation on real store footage is recorded, treat the class mapping and the auto-accept threshold as
unvalidated defaults: automatic acceptance is off by default and the operator confirms every ROI.
