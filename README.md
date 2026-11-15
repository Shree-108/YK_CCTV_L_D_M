# CCTV Litter Monitor

YOLO-based litter detection for CCTV: images, videos, RTSP / HTTP / webcam streams,
with tracking, temporal confirmation, deduplicated alerts, snapshots, and a REST API.

> **Honest status (verified 2026-11-13, see `LDM_POC_READINESS_REPORT.md`):** the software pipeline works end to end
> (upload -> detection -> tracking/confirmation -> alerts -> annotated H.264 video -> JSON/CSV -> dashboard), tested
> through the API and in a browser. **The detector's accuracy on real CCTV is not measured.**
> The production model `v3_store_cctv` (YOLO11s, 10 classes, 640 px) reports val mAP50 0.995, but its validation clip is
> the same scene as its training clip, so that number is not independent. All real footage on this machine comes from one
> store-entrance camera; there is no labelled clean-scene, small/distant-litter or unseen-camera footage yet.
> False alerts were observed on non-litter footage (boxes cut by the frame edge). Do not quote an accuracy figure.

## Status by milestone
| Milestone | Scope | State |
|---|---|---|
| M1 | Backend: YOLO engine, tiling, filters/ROI, tracking, temporal confirmation, alerts/dedup/snapshots, DB, cameras + reconnect, API, tests | **done** |
| M2 | Frontend pages wired to the real API (dashboard, image/video analysis, alerts, history, analytics, reports/CSV, cameras, models, settings) | **done** (browser walkthrough 2026-11-13: 17/17 steps) |
| M3 | Dataset tools, evaluation, threshold sweep, hard examples, model deploy/rollback | tools **done**; **no labelled real evaluation data yet** (0/215 workspace frames labelled) |
| M4 | Login (optional, `LM_AUTH_MODE=on`), local-only guard, retention/cleanup, notifications hardening | login + local-only **done**; retention cleanup and video-job cancellation **not implemented** |

## Quick start
```bash
pip install -r requirements.txt
cp .env.example .env              # set ADMIN_API_TOKEN
```
Normal instance (API :8000 + UI :3010, cameras off): `deploy\start-ldm.cmd`, stop with `deploy\stop-ldm.cmd`
(details: `deploy/README.md`, `deploy/SETUP.md`, `deploy/TROUBLESHOOTING.md`).
Backend only: `uvicorn backend.main:app --host 127.0.0.1 --port 8000`.

**Demo instance (separate data):** `deploy\start-ldm-demo.ps1` / `deploy\stop-ldm-demo.ps1` (API :8100 + UI :3110). It uses its
own database, storage, dataset folder and a verified copy of the model outside the project, does not read `.env` (no
notification or camera credentials, separate demo admin token), and checks on stop that production `storage/`, `dataset/`,
`models/` and `.env` are unchanged. Guide: `deploy/DEMO_ENVIRONMENT.md`. Use it for demonstrations and test uploads:
uploads to the normal instance become real alerts in `storage/litter_monitor.db`.

**Tests:** `python -m pytest` -> 317 passed, 0 failed, **7 skipped** (latest verification, 2026-11-15; earlier
result on 2026-11-13: 284 passed, 0 failed, 7 skipped, before the CSV-export, ROI-cancel and dashboard regression
tests were added). The integration tests use the
current `v3_store_cctv` model and a synthetic fixture (`tests/fixtures/v3/`); upload-size (413) limits are covered by
`tests/test_upload_limits.py`. The 7 documented skips: 6 v1-only tests in `tests/test_v1_legacy.py` (the old
`v1_litter_s1_synthetic` weights and `cctv_litter_dataset_v1` images were never committed and are not on this machine)
and 1 clean-frame test that waits for human-verified litter-free CCTV frames in `tests/fixtures/clean/` (kept out of Git).
Frontend: `npm run test:frontend` (17 tests; 13 on 2026-11-13) and `node node_modules/typescript/bin/tsc --noEmit` (0 errors).

**Local only.** Bind to `127.0.0.1` (as above or via the deploy scripts); do not use `--host 0.0.0.0`. The API rejects
every request that does not come from this computer (`backend/api/local_only.py`). Optional login/sessions exist
(`LM_AUTH_MODE=on`, off by default); with login off, viewer and operator actions are open to local users and admin
actions need `ADMIN_API_TOKEN`.

**Report downloads are admin-only when login is on (by design).** With `LM_AUTH_MODE=on`, an operator can upload
images and videos, start analyses and view all results in the browser, but downloading reports (the alerts/detections
CSV export, the four-screenshot report CSVs, the monthly and all-time report CSVs, and any JSON report download) needs
an admin account (`ADMIN_IF_LOGIN` in `backend/api/authz.py`). With login off, these downloads stay open to local
users, as before.

Interactive API docs (`/docs`, `/redoc`, `/openapi.json`) are **off by default**. For local development set
`LM_ENABLE_API_DOCS=true` in `.env` and open http://localhost:8000/docs (still local-only).

## Architecture
```
 source (image | video file | rtsp:// | http:// | webcam)
   │  CameraWorker: capture thread ──queue(1, drop stale)──► processing thread   (one pair per camera)
   ▼
 YoloEngine.detect()  ── full frame (+ overlapping tiles when frame > tile_size*1.25)
   │                     tiles → map coords → class-aware global NMS → cut-off-box suppression
   ▼
 apply_filters()      ── confidence tier (weak / candidate), class allow/block, size, aspect, ROI zones
   ▼                     (rejected detections are TAGGED with a reason and logged, never silently dropped)
 LitterTracker        ── ByteTrack-style two-stage IoU association, persistent IDs,
   │                     temporal window/min confirmations, stationarity (carried items never confirm)
   ▼
 AlertManager         ── only CONFIRMED objects; dedup by track + time window + IoU/centre distance + class group;
                         severity rules; full-frame + crop snapshots + JSON sidecar; notifications (env-configured)
```
Code layout:
```
backend/
  main.py                FastAPI app factory
  api/                   routes.py (REST), deps.py (admin auth, safe uploads)
  core/                  config.py (env), settings.py (runtime settings, pydantic), class_metadata.json (25 classes)
  detection/             engine.py (YOLO + device + tiling), tiling.py, filters.py (tiers/ROI/size), types.py
  tracking/              tracker.py (tracking + temporal confirmation)
  alerts/                manager.py (dedup, severity, snapshots), notifier.py (email/webhook/telegram)
  cameras/               manager.py (workers, reconnect, MJPEG frames)
  database/              db.py (SQLite/WAL, Postgres-portable schema)
  services/              pipeline.py (detect_image / detect_video / StreamPipeline), state.py, model_registry.py
  sampling/              "four screenshots" analysis: ROI proposal (UperNet segmenter), job runner, CSV reports
  legacy/                previous classical-ML engine (not imported by the app)
models/
  litter_yolo/<version>/ best.pt, data.yaml, class_names.json, model_meta.json, results.csv
                         (on this machine only v3_store_cctv has best.pt; v1 and v2 folders hold metadata only)
  backups/               v3_store_cctv_20261012 (best.pt + SHA256SUMS)
  roi_segmenter/         upernet-convnext-tiny weights for the ROI proposal
training/                dataset tools, train.py, register_model.py
storage/                 alerts/<date>/<camera>/alert_00001.jpg + _crop.jpg + .json, videos/<job>/, DB
```

## The legacy `litter_ml_model.pkl`
An earlier scikit-learn crop **classifier** (not an object detector) that `backend/legacy/` was written for.
The file is **not present** in this project (checked 2026-11-13); `backend/legacy/` is kept as code only and is not
used by the application.

## Classes
The **active model's own class list is authoritative.** Production `v3_store_cctv` has 10 classes:
plastic_bag, plastic_packaging, food_wrapper, paper, cardboard, plastic_bottle, cup, newspaper, plastic_piece,
other_litter. `backend/core/class_metadata.json` holds 25 classes (priority, bin, similarity groups) for display and
severity; classes the active model was not trained on cannot be detected. `dataset/class_mapping.yaml` uses a different
ID order and is not used by the pipeline.

## Confidence states
| state | rule |
|---|---|
| rejected | conf < weak, or failed class / size / aspect / ROI filter (reason stored) |
| weak | weak ≤ conf < candidate: logged, keeps an existing track alive, never confirms or alerts |
| candidate | conf ≥ candidate: counts as a temporal hit |
| confirmed | **video/live:** ≥ `min_confirmations` hits in the last `temporal_window` frames, mean hit conf ≥ confirmed threshold, not moving. **still image:** passed all filters and conf ≥ confirmed threshold |

Defaults (0.30 / 0.45 / 0.60, window 5, min 3) are placeholders until the threshold sweep (M3) is run on real validation data.

## Severity (unchanged in this release)
Default rules (`backend/core/settings.py`, editable in Settings): **HIGH** if the class priority is `critical` and
conf ≥ 0.75, **or** if at least 5 confirmed objects are visible (`min_active_objects=5`); **MEDIUM** for `critical`
or `high` priority classes; otherwise **LOW**. Review finding (2026-11-15): the crowd rule counts confirmed objects
per camera view (video/live) or per image/screenshot (image uploads and four-screenshot analyses), so when a screenshot
contains five or more confirmed objects, **every** object in it is HIGH, including low-priority items. Severity here
describes each object, while the five-object rule really measures how much litter is in the scene. This is kept as is
for this release; separating object severity from a scene-level litter-volume indicator is a future enhancement.

## ROI zones
Per camera, a list of normalized polygons: `{"name","type": sidewalk|road_edge|ground|parking|entrance|custom,
"mode": include|exclude, "polygon": [[x,y],...]}`. The foot-point (bottom-centre) of each box is tested;
exclude zones win; with no include zones the whole frame is allowed; `roi_margin_frac` (2%) keeps litter
on the zone border from being dropped.

## API (JSON)
| method | path | notes |
|---|---|---|
| GET | /api/health | model, device, cameras, CPU/RAM/GPU, hardware recommendation |
| GET | /api/summary | live counters: confirmed litter, raw, rejected, active tracks, new alerts |
| GET/POST | /api/cameras | POST = admin |
| GET/PUT/DELETE | /api/cameras/{id} | PUT/DELETE = admin |
| GET | /api/cameras/{id}/frame.jpg, /stream | latest annotated frame / MJPEG |
| GET | /api/detections | filters: camera_id, status, job_id, start, end |
| POST | /api/detect/image | multipart `file`, `camera_id`; returns detections + annotated JPEG (base64) |
| POST | /api/detect/video | returns `job_id` (202); poll GET /api/jobs/{id}; files under /api/jobs/{id}/files/{video,detections,events,stats} |
| GET | /api/alerts, /api/alerts/{id} | filters: date range, camera, class, min_confidence, severity, status |
| PATCH | /api/alerts/{id} | status: new / acknowledged / resolved / false_positive |
| GET | /api/alerts/{id}/snapshot?kind=full\|crop | |
| GET | /api/analytics | by camera/class/hour/day/severity, detections by status, rejection reasons, confidence histogram |
| GET | /api/models · POST /api/models/activate | activate = admin; failed load keeps the previous model |
| GET | /api/settings · PUT /api/settings | PUT = admin, partial update, validated |
| GET | /api/events, /api/jobs, /api/classes | |

Admin endpoints need `Authorization: Bearer $ADMIN_API_TOKEN` (or an admin session when login is on). If no token is
configured they are disabled (503). Also available: `/api/export/{alerts,detections}.csv`, `/api/stream` (SSE),
`/api/sampled/*` (four-screenshot analysis + CSV), `/api/auth/*` (login), and the dataset / evaluation / test-lab routes
in `backend/api/routes_lab.py`. There is no per-video-job CSV (job outputs are JSON) and no video-job cancellation.

## Known limitations (2026-11-13)
- Accuracy on real CCTV is unmeasured; the only litter clip is probably from v3's training scene.
- False alerts observed on non-litter footage, both on boxes touching the frame edge (cause not yet investigated).
- No ROI configured by default: uploads are analysed on the full frame ("ROI NOT SET - FULL FRAME").
- Small/distant litter is limited at the 640 px default; tiling helps images but is off for video by default.
- CPU only on this machine (7.8 GB RAM): a 3-minute 640x368 clip takes ~45 s, a 91 s 2304x1296 clip ~81 s.
- One worker thread processes all video jobs in order; no cancellation; no storage retention/cleanup.
- 7 tests skipped by design: 6 need the unavailable v1 model/images, 1 needs verified clean CCTV frames (see Quick start).

## Cameras
`source` may be `rtsp://…`, `http(s)://…`, a video file (looped, for demos), a webcam index (`0`), or
`env:VARNAME` so RTSP credentials stay in `.env` (credentials are also masked in API responses).
Each camera runs in its own threads; a failing camera reconnects with exponential backoff (1→30 s,
5 s open/read timeouts) without affecting the others.

## Live CCTV (CP Plus NVR) and scheduled screenshots
Full guide: [docs/LIVE_CCTV_SETUP.md](docs/LIVE_CCTV_SETUP.md). Summary:

* **Connection** - Live CCTV page (admin): NVR IP/host, RTSP port, channel, main/sub stream, username,
  password, RTSP path. The path defaults to the common CP Plus/Dahua pattern
  `/cam/realmonitor?channel={channel}&subtype={subtype}`, which **must be confirmed** for the CP-UNR-104F1.
  The password goes to the gitignored `.env` (`CCTV_<CAMERA_ID>_PASSWORD`); it is never stored in the
  database, returned by the API, shown in the browser or written to logs (URLs are masked `user:********@`).
* **Test Connection** - TCP reachability -> RTSP DESCRIBE with Digest/Basic auth -> one valid decoded frame.
  Safe errors: unable to reach NVR, network timeout, authentication failed, channel unavailable,
  RTSP stream unavailable, invalid configuration.
* **Scheduled screenshots** - backend scheduler (runs without the dashboard open): random time between
  min and max minutes (min = max for a fixed interval, e.g. 45 or 120), Start/Pause/Resume/Stop/Capture Now,
  one capture per camera at a time, state persisted (a missed capture runs once after a restart).
* **Fault tolerance** - every camera read runs in a separate process (`backend/cameras/grabber.py`) with a
  hard timeout and is killed if it hangs; bounded retries with exponential backoff; blank/corrupted frames
  are rejected (`invalid_frame`) and never reach the model; failures are recorded and retried after
  `retry_minutes`; image/video uploads are unaffected by camera problems.
* **Results** - `storage/captures/<camera>/<id>/original.jpg`, `annotated.jpg`, `result.json` + `captures`
  table (timestamp, camera, model version, detections, confidence, status, alert ids). Repeated detections
  of the same litter update one alert (`occurrences`). A notification is logged as delivered only after the
  provider call succeeded.
* Keep the NVR on the LAN/VPN. Never port-forward RTSP or the NVR web UI.

## Registering a trained model
```bash
python -m training.register_model --run runs/detect/<run> --version v2_real_cctv --data <data.yaml> [--activate]
```
Versions are immutable; nothing is deleted automatically.
