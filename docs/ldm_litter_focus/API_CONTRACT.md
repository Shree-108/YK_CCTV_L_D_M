# LDM litter-focus API contract (2026-11-19)

ROI has been removed from the active application. No request takes an ROI, polygon or approval; no response carries
`roi_*`, `pavement*`, `outside_roi*` or ROI statuses. Historical DB columns (`cameras.roi_json`, `roi_versions`,
`sampled_jobs.roi_*`) are kept untouched as history but are never read or written by the active code.

## Shared vocabulary

`detection_status` (per detection row / event):

| value | meaning |
|---|---|
| `DETECTED` | High confidence. Image: confidence >= `confidence.confirmed_threshold`. Video: the track passed temporal confirmation. Only DETECTED objects become litter events. |
| `TRACKED` | Video only: a later observation of an object that is already a litter event (the same track) - NOT a new event. |
| `UNDER_REVIEW` | Model output between `weak_threshold` and the DETECTED rule (candidate/weak tier, or a video track not (yet) confirmed). Shown and reported, never counted as an event. |
| `REJECTED` | Removed by a filter (`reason`: too_small, too_large, bad_aspect_ratio, class_blocked, class_not_allowed, moving). |
| `VERIFIED` / `RESOLVED` | Set by an operator on a litter event (PATCH below), never by the model. `REJECTED` can also be set by an operator (false positive). |

`processing_status` (per analysis): `RUNNING`, `COMPLETED_LITTER_DETECTED`, `COMPLETED_NO_LITTER_DETECTED`
(the model found no DETECTED object - this is NOT "clean"), `FAILED`. A failed analysis never gets detection or summary
CSVs.

Identity block (every analysis result):

```json
"identity": {"store_id": "STORE_001" | "UNKNOWN", "store_name": "Main St" | "UNKNOWN",
             "camera_id": "CAMERA_001" | "UNKNOWN", "camera_name": "Front door" | "UNKNOWN",
             "recording_id": "<job/analysis id or video id>",
             "warnings": ["Store ID not provided: results are recorded under UNKNOWN.", ...]}
```

Rules: a registered camera's store comes from the camera record (`cameras.store_id`). If the caller sends a different
store for a registered camera -> HTTP 422 (never attach a detection to the wrong store). A store ID that is not in the
`stores` table is accepted only as free text if it matches `^[A-Za-z0-9._ -]{1,64}$`; its name is then `UNKNOWN` and a
warning is returned. Missing values -> `UNKNOWN` + warning.

## Stores and cameras

- `GET /api/stores` (viewer) -> `{"stores": [{"id", "name", "created_at", "cameras": [{"id", "name"}]}]}`
- `POST /api/stores` (admin) body `{"id": "STORE_001", "name": "Main Street"}` -> store (409 if exists)
- `PUT /api/stores/{id}` (admin) body `{"name": "..."}` -> store
- Cameras: `POST/PUT /api/cameras` accept `store_id` (must exist in `stores`, or null). The `roi` and
  `roi_reference_size` fields are GONE (sending them -> 422, extra fields forbidden). `GET /api/cameras[/{id}]` returns
  `store_id`, `store_name`; no `roi`, no `roi_status`.
- Removed routes: `GET/PUT /api/roi/{camera_id}`, `GET /api/roi/{camera_id}/versions`, `POST /api/roi/{camera_id}/propose`,
  `PUT /api/sampled/jobs/{id}/roi`, `POST /api/sampled/jobs/{id}/roi/approve`.

## Image analysis

`POST /api/detect/image` (operator), multipart form:
`file`, `store_id` (optional), `camera_id` (optional, default unknown), `model_version`, `confidence`, `image_size`,
`tiling` (auto|on|off), `create_alerts` (bool), `save_outputs` (bool, default true), `return_image` (bool).
(`use_roi` is gone.)

Response:

```json
{
  "detections": [{"class": "plastic_bag", "confidence": 0.8731, "bbox": [x1, y1, x2, y2],
                  "status": "confirmed|candidate|weak|rejected", "detection_status": "DETECTED|UNDER_REVIEW|REJECTED",
                  "reason": "too_small"?, "track_id": 1, "priority": "high"?, "event_id": "<id>"?}],
  "count": 1, "counts": {"confirmed": 1, "candidate": 0, "weak": 0, "rejected": 0, "raw": 1},
  "by_class": {"plastic_bag": 1}, "highest_confidence": 0.8731, "average_confidence": 0.8731,
  "alerts": [...], "inference_ms": 312.5, "tiled": false, "tiles": 0, "image_size": [w, h], "model_version": "v3_store_cctv",
  "identity": {...},
  "analysis": {"analysis_id": "img_20261119T181500_ab12cd", "processing_status": "COMPLETED_LITTER_DETECTED",
               "unique_events": 1, "frame_observations": 1,
               "urls": {"original": "...", "annotated": "...", "detections_csv": "...", "summary_csv": "...", "metadata": "..."}},
  "annotated_image": "<base64 jpeg>"
}
```

Every detection the model returned above the weak floor is in `detections` and drawn on the annotated image (including
UNDER_REVIEW and REJECTED, drawn dashed/grey).

## Video analysis (full video, tracking) - the primary video workflow

`POST /api/detect/video` (operator), multipart form: `file`, `store_id`, `camera_id`, `recording_start` (optional ISO
local date-time of the first frame, e.g. `2026-11-19T14:05:00`), `sample_interval_sec` (optional, 0.1-60; default from
settings = 1/`video.sample_fps`), `tiling`, `confidence`, `image_size`, `model_version`, `temporal_window`,
`min_confirmations`. -> `{"job_id", "status": "queued", "status_url"}`. 422 on a bad store/camera combination.

`GET /api/jobs/{job_id}`: `status` queued|running|done|failed, `progress` 0..1, `result`:
- while running: `{"progress": {frames_read, frames_total, frames_analysed, processing_fps, eta_sec, detections, confirmed_objects, new_alerts}}`
- done: 

```json
{
  "stats": {...existing stats...},
  "events": [...tracker events...], "alert_ids": [..],
  "identity": {...},
  "video_validation": {"duration_sec", "width", "height", "fps", "codec", "frames_reported", "frames_read",
                       "read_failures", "truncated": false, "warnings": []},
  "litter": {"unique_events": 2, "frame_observations": 31, "tracked_objects": 5,
             "by_class": [{"class": "paper", "events": 1, "observations": 12}],
             "events": [{"event_id", "track_id", "class", "detection_status": "DETECTED",
                         "first_seen", "last_seen", "first_seen_sec", "last_seen_sec", "duration_sec",
                         "observation_count", "max_confidence", "mean_confidence", "bbox": [..],
                         "evidence_url": "/api/jobs/{id}/evidence/{event_id}.jpg"}]},
  "analysis": {"analysis_id", "processing_status", "unique_events", "frame_observations", "urls": {...}}
}
```

`files` (when done): `video`, `detections` (json), `events` (json), `stats` (json), `detections_csv`, `summary_csv`,
`metadata` -> `/api/jobs/{id}/files/{kind}`. Evidence: `GET /api/jobs/{id}/evidence/{event_id}.jpg`.

Failed jobs: `status: failed`, `error`, and `result.analysis.processing_status = "FAILED"`; no CSV files.

### Four evidence screenshots + six-column litter report (added 2026-11-20)

Implementation: `backend/services/video_evidence.py`. The video is still analysed in full at the sampling interval;
the screenshots are extra evidence, not the analysis.

- Optional form field `screenshot_seed` (int >= 0; setting `video.screenshot_seed`). Default seed = CRC32 of the
  analysis id, so the selection is reproducible and recorded.
- Selection: the decoded duration is split into 4 equal segments and one random **analysed** frame is taken from each
  (distinct, spread, valid decoded frames). Only a video with fewer than 4 analysed frames uses other decoded frames;
  those are flagged `analysed_frame: false` and their boxes drawn as unconfirmed. A video with fewer than 4 decodable
  frames FAILS (exactly four screenshots are guaranteed for every completed analysis).
- Boxes are the pipeline's own result for that frame: litter-event tracks solid (`Class 87%`), review candidates dashed
  amber (`REVIEW ...`). Weak-only tracks and filter rejections are not drawn. A frame without a litter-event box shows
  a "No litter detected" banner.
- Files: `storage/analysis_outputs/<analysis_id>/screenshots/screenshot_01.png .. _04.png`, `screenshots.json`,
  `review_candidates.json`. No per-video CSV file: the rows are appended to the day's consolidated CSV (below).
- Daily CSV: `storage/analysis_outputs/daily_reports/litter_report_<YYYY-MM-DD>.csv`. The first video analysed on a
  date creates it (header + rows); every later video that day reads it back, appends its rows and saves it again
  (temp file + rename). Date = processing date (local date when the analysis finished). Rows are a snapshot at
  analysis time (operator reviews are reflected in the per-video download, not here). A file whose header is not the
  six columns is never overwritten.
- `result.screenshots[]`: `index, kind, file, url, frame_idx, video_time_sec, timestamp ("YYYY-MM-DD HH:MM:SS"),
  timestamp_source, analysed_frame, litter_detected, under_review, detections[]`.
  `result.screenshot_selection`: `seed, method, duration_sec, frames_read`.
- `result.report`: `columns`, `rows`, `counts {rows, DETECTED, UNDER_REVIEW, VERIFIED, classes}`, `url`,
  `daily {date, path, rows_appended, total_rows, url}`.
- Downloads: `GET /api/analyses/{id}/files/screenshot_01..04` (PNG), `screenshots_json`, `report_csv` (this video
  only, **rebuilt from the database on every download**) and `daily_report_csv` (the whole day's file).
- `GET /api/reports/daily` -> `{columns, days: [{date, rows, size_bytes, url}]}` (newest first);
  `GET /api/reports/daily/<YYYY-MM-DD>.csv` -> that day's file (viewer).

Report columns (exactly): `store_id,camera_id,timestamp,litter_class,confidence_score,detection_status`. One row per
unique object:
- `DETECTED`: a litter event (temporally confirmed track) in `litter_events`.
- `VERIFIED`: an event an operator set to VERIFIED (or RESOLVED).
- `UNDER_REVIEW`: a track that never confirmed but reached `confidence.candidate_threshold` at least once. Weak-only
  tracks are not reported.

Events an operator REJECTED are left out. `timestamp` is the first sighting: recording time when `recording_start` was
given, else processing time. `confidence_score` is the max confidence with 2 decimals. `litter_class` is the active
model's own class name, never relabelled. A missing store or camera is `UNKNOWN`, with an identity warning. No litter
gives a CSV with the header line only.

## Four-screenshot quick check (secondary)

Same routes as before (`/api/sampled/...`) minus the ROI routes. `POST /api/sampled/jobs` no longer accepts `roi_mode`
or `auto_accept`. Stages: received -> validating -> sampling -> analysing -> generating_images -> generating_csv ->
completed. State `awaiting_roi` no longer occurs for new jobs. Frame outcome: LITTER_DETECTED | NO_LITTER (the model
found nothing - not a verified clean result) | UNREADABLE | FAILED. Each completed job also has `reports/<...>_detections.csv` and `<...>_summary.csv` in the standard
format (report names listed in `report_urls`).

## Litter events and analytics (new)

- `GET /api/litter/events?store_id&camera_id&class&status&start&end&limit&offset` (viewer)
  -> `{"events": [{event_id, analysis_id, source_type, store_id, store_name, camera_id, camera_name, recording_id,
  track_id, class, max_confidence, mean_confidence, bbox, first_seen, last_seen, observation_count, duration_sec,
  detection_status, review_status, model_version, image_url}], "total"}`
- `PATCH /api/litter/events/{event_id}` (operator) body `{"review_status": "VERIFIED|REJECTED|RESOLVED|DETECTED", "note"?}`
- `GET /api/litter/analyses?limit` (viewer) -> recent analyses with status, identity, counts and urls.
- `GET /api/litter/analytics?start&end&store_id&camera_id&granularity=hour|day|week` (viewer) ->

```json
{"totals": {"unique_events": 18, "frame_observations": 240, "events_today": 3, "avg_visible_sec": 41.2,
            "analyses": 7, "analyses_failed": 1},
 "most_frequent_class": "paper",
 "by_class": [{"class", "events", "observations"}],
 "by_store": [{"store_id", "store_name", "events", "observations"}],
 "by_camera": [{"camera_id", "camera_name", "store_id", "events", "observations"}],
 "series": [{"bucket": "2026-11-19T14", "events", "observations", "by_class": {"paper": 2}}],
 "hourly_today": [{"hour": 0..23, "events"}],
 "processing": {"COMPLETED_LITTER_DETECTED": 4, "COMPLETED_NO_LITTER_DETECTED": 2, "FAILED": 1}}
```

Events whose `review_status` is `REJECTED` (operator: false positive) are excluded from analytics counts.
Event time = recording time when `recording_start` was given, else processing time (`timestamp_source` says which).

## CSV files (exact columns)

Detection CSV `<STORE>_<YYYY_MM_DD>_<CAMERA>_<analysis_id>_detections.csv`:
`store_id, store_name, camera_id, camera_name, recording_id, timestamp, frame_timestamp, frame_number, event_id,
track_id, litter_class, confidence_score, detection_status, bbox_x, bbox_y, bbox_width, bbox_height,
observation_count, first_seen, last_seen, model_version, image_path`

Summary CSV `<...>_summary.csv` (one row per store/camera/date/hour that has observations; one row with zeros and the
processing status when nothing was detected):
`store_id, store_name, camera_id, date, hour, plastic_packet_count, plastic_bottle_count, paper_count, food_waste_count,
other_litter_count, glass_count, cardboard_count, total_unique_litter_events, total_frame_observations,
processing_status`

The seven `*_count` columns count unique DETECTED events of exactly those class names. A model with other class names
(v3_store_cctv) is NOT relabelled: its events are counted in the totals but only matching names fill the class columns
(v3's `paper`, `cardboard`, `plastic_bottle`, `other_litter`). The metadata JSON lists every class actually counted.

Output folder: `<storage>/LDM_OUTPUT/<STORE>/<YYYY>/<MM>/<DD>/<CAMERA>/{input,annotated,detections,reports,metadata}/`
(UNKNOWN_STORE / UNKNOWN_CAMERA when unknown). File names carry the analysis id, so nothing is overwritten.
