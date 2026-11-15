# Four-screenshot video analysis

Default workflow on **Video Analysis** (`/detect/video`). The earlier full-video tracking analysis is still there
under **Full video (tracking)** (`/detect/video?mode=full`). Results are on the **Results Dashboard** (`/results`,
also reachable from Video Analysis and with Ctrl+K).

Litter only, local storage only. No uniform/clothing/staff logic and no cloud storage.

## Workflow

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
| POST | `/media` | operator |
| GET | `/media/{id}`, `/media/{id}/preview.jpg` | viewer |
| POST | `/jobs` | operator |
| GET | `/jobs`, `/jobs/{id}`, `/jobs/{id}/frames/{n}/{original\|annotated\|proposal}.jpg` | viewer |
| PUT | `/jobs/{id}/roi` (saving a *registered* camera's ROI additionally needs admin) | operator |
| POST | `/jobs/{id}/cancel` | operator |
| GET | `/jobs/{id}/report.csv`, `/reports/months/{YYYY-MM}.csv`, `/reports/all.csv` | admin when login is on |
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
model), `tests/test_authz.py` (permission matrix), `frontend/tests/csv.test.mjs`. Measured results on real footage:
[EVALUATION.md](EVALUATION.md).
