# CCTV Litter Monitor

YOLO-based litter detection for CCTV: images, videos, RTSP / HTTP / webcam streams,
with tracking, temporal confirmation, deduplicated alerts, snapshots, and a REST API.

> **Honest status:** the pipeline is real and tested end-to-end with a real trained YOLO model.
> The only model available so far (`v1_litter_s1_synthetic`) was trained on the **synthetic starter
> dataset** (rendered scenes, 120 training images). Its validation numbers (P 0.89 / R 0.88 /
> mAP50 0.98) are measured on synthetic images and **say nothing about real CCTV performance**.
> Real accuracy requires real, annotated footage from your cameras (see Milestone 3: training).

## Status by milestone
| Milestone | Scope | State |
|---|---|---|
| M1 | Backend: YOLO engine, tiling, filters/ROI, tracking, temporal confirmation, alerts/dedup/snapshots, DB, cameras + reconnect, API, tests | **done** |
| M2 | Frontend pages wired to the real API (dashboard, live, detection, alerts, history, analytics, cameras, models, settings) | next |
| M3 | Dataset validation/stats, training/retraining, evaluation (per-class, small/medium/large recall), threshold sweep, hard-example mining, model deploy/rollback | planned |
| M4 | Notifications hardening, auth for the UI, docker-compose, retention/cleanup, performance tuning | planned |

## Quick start (backend)
```bash
pip install -r requirements.txt
cp .env.example .env              # set ADMIN_API_TOKEN (a random one was generated locally)
uvicorn backend.main:app --host 0.0.0.0 --port 8000
python -m pytest                  # 38 tests, uses the real model
```
Interactive API docs: http://localhost:8000/docs

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
  legacy/                previous classical-ML engine (baseline only)
models/
  litter_ml_model.pkl    legacy classifier (kept, never overwritten)
  litter_yolo/<version>/ best.pt, last.pt, data.yaml, class_names.json, model_meta.json, results.csv
training/register_model.py
storage/                 alerts/<date>/<camera>/alert_00001.jpg + _crop.jpg + .json, videos/<job>/, DB
```

## The legacy `litter_ml_model.pkl`
Inspected without unpickling: a joblib-compressed **scikit-learn Pipeline (StandardScaler + classifier)**
over hand-crafted crop features, fed by OpenCV contour/MSER region proposals. It is a crop
**classifier**, not an object detector, and cannot be converted into a YOLO model. It is kept in
`models/` (moved, not modified) and the old engine lives in `backend/legacy/` as a baseline only.

## Classes
The **active model's own class list is authoritative** (13 classes, from the dataset `data.yaml`):
chips_packet, food_wrapper, plastic_bottle, plastic_cup, metal_can, paper, cardboard, plastic_bag,
food_container, tissue, cigarette_packet, plastic_packaging, other_litter.
All 25 legacy classes are preserved in `backend/core/class_metadata.json` (priority, bin, similarity
groups). The 12 extra ones (bottle_cap, cigarette_butt, …) have **no training labels yet**, so no current
model can detect them. Note: `dataset/class_mapping.yaml` uses a *different* ID order than `data.yaml`;
it is not used by the pipeline.

## Confidence states
| state | rule |
|---|---|
| rejected | conf < weak, or failed class / size / aspect / ROI filter (reason stored) |
| weak | weak ≤ conf < candidate: logged, keeps an existing track alive, never confirms or alerts |
| candidate | conf ≥ candidate: counts as a temporal hit |
| confirmed | **video/live:** ≥ `min_confirmations` hits in the last `temporal_window` frames, mean hit conf ≥ confirmed threshold, not moving. **still image:** passed all filters and conf ≥ confirmed threshold |

Defaults (0.30 / 0.45 / 0.60, window 5, min 3) are placeholders until the threshold sweep (M3) is run on real validation data.

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

Admin endpoints need `Authorization: Bearer $ADMIN_API_TOKEN`. If no token is configured they are disabled (503).

## Cameras
`source` may be `rtsp://…`, `http(s)://…`, a video file (looped, for demos), a webcam index (`0`), or
`env:VARNAME` so RTSP credentials stay in `.env` (credentials are also masked in API responses).
Each camera runs in its own threads; a failing camera reconnects with exponential backoff (1→30 s,
5 s open/read timeouts) without affecting the others.

## Registering a trained model
```bash
python -m training.register_model --run runs/detect/<run> --version v2_real_cctv --data <data.yaml> [--activate]
```
Versions are immutable; nothing is deleted automatically.
