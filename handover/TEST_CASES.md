# UAT test cases and results: LDM, 2026-10-12

Status values: **PASS** (executed, met the expectation) · **FAIL** (executed, did not) · **BLOCKED** (cannot be
executed until a dependency is resolved) · **NOT TESTED** (possible, but not executed). Evidence paths are relative
to the project folder. Model under test: production `v3_store_cctv` (the only model; no candidate exists).

## 1. Readiness

| ID | Requirement | Status | Evidence |
|---|---|---|---|
| R1 | Original production checkpoint preserved | **PASS** | `best.pt` sha256 `d1de9cb6…862bc6` = local backup `models/backups/v3_store_cctv_20261012/` = git blob of commit 574c7e1 (pushed to origin/main) |
| R2 | Fine-tuned candidate has documented evaluation results | **BLOCKED** | no candidate exists: no `storage/runs`, only `v3_store_cctv` registered; 0 of 215 dataset images annotated (`dataset/ldm_v4_workspace`) |
| R3 | Real CCTV validation completed | **BLOCKED** | no labelled test set; only 2 store recordings, both the same camera, one probably in v3's training data (`docs/LDM_FINETUNE_PLAN.md`) |
| R4 | Frontend and backend communicate | **PASS** | UI `:3010` → API `:8000` CORS allowed; upload from the UI origin HTTP 200 (A07) |
| R5 | Startup and shutdown procedures work | **PASS** | `deploy/TEST_RESULTS.md` (clean start, re-run, stop, restart, error cases); re-use path re-checked in this UAT |
| R6 | CCTV retrieval only through a verified, authorised interface | **PASS** | only the export-folder ingester was built; NVR retrieval not implemented because unverified (`docs/CCTV_RECORDING_RETRIEVAL.md`) |
| R7 | Configuration and credentials handled securely | **PASS** (local demo) | S01–S04 |

## 2. Model (v3, production settings: imgsz 640, thresholds 0.30/0.45/0.60; outputs `handover/evidence/uat_model/`)

These are **visual checks by the developer assistant, not ground truth**. No accuracy metric can be computed until
annotated test data exists. "Confirmed" = passed all filters and confidence ≥ 0.60 (still image).

| ID | Case | Status | Result |
|---|---|---|---|
| M01 | Genuine litter on pavement, fridge with bottles in frame | **PASS** | 1 food_wrapper 0.83 on the pavement item; no box on fridge products |
| M02 | Several litter items + fridge products | **PASS** | 3 food_wrapper 0.82–0.91 on pavement items; fridge products not detected |
| M03 | People, car and tyre, fridge products | **PASS** | 0 confirmed (1 candidate 0.56 at the frame edge); car, tyre and products not detected |
| M04 | People, motorbikes, fridge products | **FAIL** | **false positive:** `plastic_piece` 0.63 *confirmed* on the white sole of a pedestrian's sneaker (enlarged crop reviewed). Motorbikes and products not detected. → D-07 |
| M05 | People, car, motorbike, fridge products | **PASS** | 0 confirmed (2 weak ≤ 0.37) |
| M06 | Street: much litter + parked cars (synthetic image) | **PASS** (qualitative) | 17 confirmed on pavement litter; cars not detected. Synthetic image, probably seen in training |
| M07 | Street, other scene, cars + scooter (synthetic) | **PASS** (qualitative) | 8 confirmed on litter; cars, scooter and dustbin not detected |
| M08 | Poorly lit (M01 darkened) | **PASS** | wrapper still found (0.79) |
| M09 | Blurred | **PASS** | wrapper found (0.83, looser box); 1 extra weak 0.36 |
| M10 | Low resolution (1/3, upscaled) | **PASS** | wrapper found (0.80) |
| M11 | Heavy JPEG compression | **PASS** | wrapper found (0.79) |
| M12 | Boxes, class labels and confidence scores drawn and returned | **PASS** | annotated images M01–M11; API returns class/confidence/bbox (A04) |
| M13 | Clean pavement with no litter at all | **NOT TESTED** | no litter-free pavement recording available |
| M14 | Night / IR / rain footage | **NOT TESTED** | none available (only simulated darkening, M08) |
| M15 | Litter recall and precision on unseen footage | **BLOCKED** | needs an annotated held-out test set |
| M16 | Speed | **PASS** | 218–272 ms per still on this CPU (first call 2.7 s including warm-up) |

## 3. Application

| ID | Case | Status | Evidence |
|---|---|---|---|
| A01 | Frontend loads (`http://localhost:3010`), pages render | **PASS** | Overview, Image/Video Analysis, Alerts, Analytics and Settings rendered; Settings shows "Backend status ok, Active model v3_store_cctv" |
| A02 | Backend health endpoint | **PASS** | `/api/health` ok, model v3, cameras 0 |
| A03 | Model loads | **PASS** | health + `evidence/uat_app/app_smoke_18_checks.txt` |
| A04 | Image upload and inference (API) | **PASS** | HTTP 200, class/confidence/bbox + annotated image |
| A05 | Video upload and processing | **PASS** | 10 s real clip: 202 → done in 4 s, timestamped detections, annotated H.264 video |
| A06 | Annotated results and output files accessible | **PASS** | `/api/jobs/{id}/files/video` and `detections` 200 |
| A07 | Upload from the browser UI origin (`:3010`) | **PASS** | in-page `fetch` from `http://localhost:3010`: HTTP 200, 717 ms (grey test image, 0 detections, camera `uat_test`) |
| A08 | Corrupt image | **PASS** | 400 "Could not decode image (corrupt or not an image)" |
| A09 | Unsupported file type | **PASS** | 415 "Unsupported file type '.txt'. Allowed: …" (API and from the UI origin) |
| A10 | Model files missing | **PASS** | API starts and reports it; detection 503 with message; `start-ldm` stops with "No model weights found …" |
| A11 | Unknown model activation / activation without admin | **PASS** | 404 (v3 stays active) / 401 |
| A12 | Backend down: UI status | **PASS** | sidebar + top bar "Backend offline" within ~20 s; recovers without reload (`deploy/TEST_RESULTS.md`) |
| A16 | Video above the upload limit | **PASS** | 413 "File too large (limit 1 MB)" (isolated instance with the limit set to 1 MB) |
| A17 | Video longer than `max_video_seconds` | **PASS** | job finishes safely (10 s clip, 4 s limit) |
| A17b | Truncation visible to the operator | **FAIL** | stats and job result do not say the video was cut; `video.duration_sec` reports the analysed 3.96 s instead of 10 s → D-06 |
| A18 | Restart keeps configuration and model | **PASS** | setting changed, app restarted on the same storage: value kept, model v3 still active |
| A19 | Processing timeout of a very long job | **NOT TESTED** | no multi-hour recording available; jobs run in one background worker (no time limit in the API) |
| A20 | Regression suite | **PASS** | `pytest`: 48 passed, 38 skipped (skipped tests need the missing v1 model/images, unchanged since before this work); `tsc` 0 errors; `next build` OK |

## 4. CCTV integration

| ID | Case | Status | Evidence |
|---|---|---|---|
| C01 | Automatic retrieval of recordings from the NVR | **BLOCKED** | interface not verified by CP PLUS; no NVR address/credentials/network path (`docs/CCTV_RECORDING_RETRIEVAL.md` §1, §4) |
| C02 | Camera + time-range selection on the NVR | **BLOCKED** | as C01 |
| C03 | Export-folder ingestion of a recording | **PASS** | `evidence/phase9_export_ingest/run1.txt`: 2 clips processed through the app |
| C04 | Camera id and recording time | **PASS** (synthetic names) | camera from the sub-folder, start from the file name; `events.csv` wall time = start + offset. **Real KVMS naming not yet seen** |
| C05 | Frame timestamps | **PASS** | `detections.csv` `video_time_sec` + `wall_time` |
| C06 | Inference and output generation | **PASS** | `summary.json`, CSVs, snapshots per recording |
| C07 | Duplicate recordings not reprocessed | **PASS** | renamed copy and second run: 0 uploads (`run2.txt`) |
| C08 | Network failure during processing | **PASS** | API killed mid-job → "stays pending", exit 3; processed on the next run (`run3.txt`) |
| C09 | Corrupt / unsupported (`.dav`) exports | **PASS** | reported in log + ledger, not uploaded |
| C10 | A real KVMS Pro Lite export | **NOT TESTED** | no export file available |
| C11 | `--watch` unattended mode | **NOT TESTED** | deliberately not run unattended before pilot approval |
| C12 | Live RTSP from the real NVR | **BLOCKED** | no NVR address/credentials; never tested against the real device |

## 5. Security and operations

| ID | Check | Status | Evidence |
|---|---|---|---|
| S01 | No secrets in source / committed files | **PASS** | the real admin-token value was searched in 253 git-tracked, untracked and log files: 0 hits; `.env` not tracked; credential-looking matches are only empty template keys and the test suite's dummy token |
| S02 | Uploads validated | **PASS** | A08, A09, A16; camera ids validated (regex) |
| S03 | Logs do not reveal credentials | **PASS** | ingest and demo logs scanned: 0 hits for password/token/rtsp://; RTSP URLs are masked in code (`backend/cameras/cctv.py`) |
| S04 | Access restricted | **PASS for the local demo / FAIL for a multi-user pilot** | API and UI bind to 127.0.0.1 (this PC only); changes need the admin token; **read endpoints have no login** → D-08 |
| S05 | Model and data restorable | **PASS** | model backup checksum verified; `deploy/backup-ldm.ps1` tested: database online copy passes `integrity_check` (15 alerts), model checksum matches |
| S06 | Processing failures visible | **PASS with gap** | job status/error in the UI, ingest log + ledger, start-script messages; truncation is not visible (D-06) |
| S07 | Alerts / notifications disabled | **PASS** | email, webhook and Telegram all "configured: false"; no messages can be sent |
| S08 | CCTV recordings stored in approved locations | **PASS** | exports only read in place; app copies in `storage/` (git-ignored); evidence images git-ignored |
