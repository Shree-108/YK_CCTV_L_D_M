# LDM: local POC readiness report

| | |
|---|---|
| Date | 2026-11-13 |
| Scope | the local application only (no AWS, no cloud, no multi-store) |
| Branch | `security/step1-local-only`, with uncommitted work present; this is the code as audited |
| Method | read-only inspection by 3 parallel code reviews plus my own checks; existing test suites run; isolated end-to-end API tests |
| Changes made | **none** to source, configuration, datasets or model weights |
| New files | this report, plus evidence in `docs/poc_audit/evidence/` |

> **Status update, 2026-11-13 (after this audit).** The test figures below (203 passed, 38 skipped) are the
> results *at audit time* and are kept unchanged as historical evidence. Since then the skipped tests were modernised
> for the current `v3_store_cctv` model, upload-size (413) tests were added, explicit class metadata was added for
> `cup`, `newspaper` and `plastic_piece` together with 34 severity and dedup tests, and two scheduler tests were made
> robust to frame-decoding differences. Current result: `python -m pytest` -> **284 passed, 0 failed, 7 skipped**
> (6 v1-only tests whose artifacts are unavailable, 1 test waiting for verified clean CCTV frames). An isolated demo
> environment now exists (`deploy/DEMO_ENVIRONMENT.md`). Real CCTV footage, ROI approval and independent model
> evaluation remain open, as listed in section 7. Privacy: the real-CCTV evidence images and the raw E2E evidence
> files (`evidence/e2e.py`, `e2e.txt`, `e2e_results.json`, which contain machine paths and real-frame names) are kept
> locally and are not committed to Git.

**Evidence labels used throughout:**

- **TESTED**: executed by me in this audit.
- **CODE**: code present with file:line evidence, but not executed by me.
- **PARTIAL**: implemented with gaps.
- **MISSING**: not implemented.
- **PROPOSED**: a recommendation, not implemented.

---

## 1. Summary

The **software pipeline is demonstrable today.** Each of these was executed against an isolated copy of the app and works (33/33 checks):

- upload a video or image
- detection, tracking and confirmation
- alerts with snapshots
- annotated H.264 video
- JSON and CSV outputs
- dashboard and analytics data

On top of that, 203/203 runnable backend tests and 13/13 frontend unit tests pass, and the TypeScript check is clean.

The **AI evidence is not demonstrable today:**

- There is no accuracy measurement on any unseen footage.
- The only litter-positive video is probably from v3's training scene.
- No genuine clean-scene video or small/distant-litter video exists on this machine.
- Both non-litter videos tested produced a false alarm.

The POC is therefore blocked by **demo data and evidence**, not by code. The required code changes are small (§8).

---

## 2. Current architecture

```
Browser (Next.js 16.3.3 / React 19, port 3010 via deploy\start-ldm.ps1, or 3000 for dev)
   |  direct cross-origin fetch to NEXT_PUBLIC_API_URL (default http://localhost:8000); no Next rewrites
   |  proxy.ts = CSP nonce + security headers only
   v
FastAPI (uvicorn, 127.0.0.1:8000) - LocalOnlyMiddleware: loopback clients only; CORS allow-list; API docs off
   |-- routes.py          detect/image, detect/video -> jobs, alerts, detections, analytics, ROI, cameras, models, settings
   |-- routes_lab.py      CSV export, SSE stream, datasets / ground truth, evaluations, test lab, hard examples
   |-- routes_sampled.py  "four-screenshot" analysis: segmenter-proposed ROI -> operator confirms -> 4 annotated frames + CSV
   |-- routes_auth.py     login / sessions (LM_AUTH_MODE=off by default)
   v
Services: YoloEngine (Ultralytics, CPU) -> filters (thresholds, ROI footpoint, size/aspect) -> tracker (IoU, Hungarian)
          -> stationary confirmation -> AlertManager (dedup, snapshot, crop) -> optional notifier (email/webhook/Telegram, none configured)
   v
SQLite storage/litter_monitor.db (WAL): detections, confirmed_objects, alerts, jobs, models, settings, cameras, sampled_*
Files: storage/videos/<job>/ {annotated.mp4, detections.json, events.json, stats.json}, storage/alerts/<date>/<cam>/*.jpg
Model: models/litter_yolo/v3_store_cctv/best.pt (the only registered / loadable model; YOLO11s, 10 classes, imgsz 640)
```

**Detection defaults** (`backend/core/settings.py`, no overrides in the DB):

| setting | value |
|---|---|
| confidence thresholds | weak 0.30 / candidate 0.45 / confirmed 0.60 |
| image size | 640 |
| tiling for images | auto (1024 px tiles), only when the long side > 1280 px |
| tiling for video | off |
| video sampling | 2 frames/s |

**Confirmation rule** (`backend/tracking/tracker.py:158-162`): within the last 5 sampled frames (about 2.5 s), at least 3 candidate hits, mean confidence ≥ 0.60, and a centre shift ≤ 5 % of the frame diagonal.

**Alert dedup:** within 900 s, by IoU ≥ 0.30 or centre distance within 4 %.

---

## 3. Feature implementation status

### 3.1 AI model

| Item | Status | Evidence |
|---|---|---|
| Production weights `v3_store_cctv` load at startup | **TESTED** | `/api/health` → v3, CPU, 10 classes; SHA-256 `d1de9cb6…62bc6` unchanged; backup in `models/backups/` |
| Other versions | MISSING | v1 `best.pt` and v2 (YOLOE) `best.pt` are not on disk; the registry only loads folders with `best.pt` |
| Training/val history | CODE | v3 val mAP50 0.995. Its val clip is the **same scene** as its training clip (`dataset/cctv_store_v2/README.txt`), so it is not independent. v3's training dataset itself is not on this machine |
| Ground-truth evaluation on real footage | **MISSING** | `storage/eval/baseline_v3_20261112/predictions.csv`: 674 predictions, **0 verdicts**. `handover/MODEL_EVALUATION_SUMMARY.md`: no precision/recall measured on unseen data |
| Dataset / annotation status | PARTIAL | `ldm_v4_workspace` 0/215 frames labelled; AUG100 experiment 100 synthetic images labelled (2 scenes, exploratory) |
| Image detection | **TESTED** | store frame: 1 detection in 3.7 s (first call, warm-up); synthetic street: 19 in 1.3 s; footfall frame: 0 in 0.3 s |
| Video detection | **TESTED** | 3-min 640×368 clip: done in **45 s**, 6 events, 3 wrappers visible at 70–92 % (`evidence/video_frame_1800.jpg`); 1 white item in the same frame not detected (possible miss, not verified) |
| False positives on clean scenes | **TESTED, failing / unknown** | Footfall clip (crowded, no visible litter): **1 alert**, `paper` 60 % on an unidentifiable object at the bottom frame edge. Indoor "clean" clip: **1 alert**, `food_wrapper` 68 % on decode artefacts at the bottom edge; the source file decodes as corrupted frames, so it is **not a valid clean test**. The 2026-11-12 baseline shows up to 27 confirmed detections on indoor negatives (unreviewed) |
| Small / distant litter | **TESTED (synthetic only)** | In the AUG100 work, 60 % of labelled litter was < 16 px; at 640 inference far-field items shrink below ~10 px. No real small-litter footage exists |
| Generalisation to unseen CCTV | **MISSING** | No unseen-camera footage with labels; every real clip is from one store-entrance camera |

### 3.2 Backend

| Feature | Status | Evidence |
|---|---|---|
| `POST /api/detect/image`: validation, inference, alerts, annotated image | **TESTED** | `e2e.txt`; corrupt → 400, `.txt` → 415, bad `camera_id` → 422, traversal filename sanitised |
| `POST /api/detect/video` → 202 → background job → progress → done | **TESTED** | progress 0 → 0.09 → … → 1.0; 3 videos all `done`, none `failed` |
| Job outputs (annotated MP4 in **H.264/avc1**, detections, events, stats) | **TESTED** | All 4 files returned 200. Codec confirmed `avc1` for the new jobs and for both existing outputs |
| Tracking + stationary confirmation + dedup | CODE + **TESTED (indirect)** | 6 events from the litter clip; unit tests in `test_units.py` pass |
| ROI: draw/save per `camera_id`, footpoint filter | CODE (+ pytest `test_roi_upload` partly) | `filters.py:74-94`, `routes.py:477-500`. **No ROI is configured**, so the demo shows "ROI NOT SET – FULL FRAME" |
| ROI auto-proposal (UperNet segmenter) | PARTIAL | only wired into the four-screenshot flow, not full video. The 2026-11-12 ROI baseline returned "uncertain" for all 7 scenes |
| SQLite storage and retrieval | **TESTED** | jobs, alerts, detections, events and analytics all retrieved after processing |
| Alert snapshot, status update | **TESTED** | snapshot JPEG 200; PATCH to `acknowledged` 200 |
| CSV export (alerts, detections) | **TESTED** | `/api/export/alerts.csv` 47 rows, `/api/export/detections.csv` 4,457 rows, `text/csv` |
| Per-video-job CSV | MISSING | job files are JSON only; CSV exists for the four-screenshot jobs and the global exports |
| Video job cancellation | MISSING | no endpoint; one worker thread is shared by all jobs (`state.py:43`) |
| Error handling / status | CODE | failures recorded on the job with the error; upload always deleted (`state.py:242-246`) |
| Local-only access, CORS, docs off, admin token | **TESTED** | `/docs` 404; admin route without token 401; CORS allows :3000, blocks others |
| Login / sessions | CODE + pytest | off by default; `test_auth.py` / `test_authz.py` pass |
| Live RTSP cameras / scheduler | CODE | 0 cameras configured; a local file can act as a looping camera (untested) |
| Notifications (email/webhook/Telegram) | CODE | none configured in `.env` (`/api/health` reports `configured:false`) |
| Storage retention / cleanup | MISSING | `SettingsPage.tsx:186` says "not implemented" |

### 3.3 Frontend

All frontend features below were confirmed from code only. I did **not** click through the UI in a browser, because starting Next.js rewrites tracked files such as `next-env.d.ts`.

| Feature | Status | Evidence |
|---|---|---|
| Image upload + boxes + confidence + threshold slider + compare view | CODE | `ImageDetectionPage.tsx`, `Overlay.tsx:64,93` |
| Video upload, progress, annotated playback, timeline, events | CODE | `VideoDetectionPage.tsx` (reached via `/detect/video?mode=full`; the default view is the four-screenshot flow) |
| Processing history / alerts / dashboard on real API data | CODE | no mock data found; polled every 15–30 s |
| CSV download (backend-generated) | CODE (backend **TESTED**) | `ExportCsvButton.tsx:29` → `/api/export/*.csv` |
| Frontend ↔ backend contract | **TESTED (partially)** | CORS preflight from :3000 OK; all `endpoints.ts` paths exist in the backend; `tsc --noEmit` 0 errors; 13/13 node tests pass |
| Frontend component / E2E tests | MISSING | only 3 unit-test files; no lint script |
| Known UI defects (from code review) | CODE | 1. `<a download>` to the cross-origin API (`VideoDetectionPage.tsx:182,203,303`, `ReportsPage.tsx:59`): the browser may open the file instead of downloading it. 2. Hooks inside `.map` (`AlertsPage.tsx:43`). 3. Two pages titled "Live CCTV". 4. Object URL not revoked |

---

## 4. End-to-end workflow (as exercised)

1. Upload the 3-minute store CCTV clip (kept locally, not in Git) to `POST /api/detect/video`. Response: **202** with `job_id`.
2. The single worker samples 2 frames/s and runs YOLO (640, CPU). It filters by threshold, size and ROI (none set, so the full frame), then tracks with IoU and confirms stationary objects. Detections go to SQLite.
3. Each confirmed object becomes an alert, deduplicated, with a snapshot and crop under `storage/alerts/`.
4. The job is written as `done`, with `annotated.mp4` (H.264), `detections.json`, `events.json` and `stats.json`.
5. Results are read through `GET /api/jobs/{id}`, `/files/{kind}`, `/api/alerts`, `/api/analytics`, `/api/detections` and `/api/export/*.csv`. All **TESTED**.
6. The UI renders steps 1 and 5 through `frontend/api/endpoints.ts`. Code only; not clicked through.

## 5. Test results and evidence

| Test | Result | Evidence |
|---|---|---|
| Backend `pytest` (isolated temp dirs) | **203 passed, 38 skipped, 0 failed** (212 s) | `docs/poc_audit/evidence/pytest_summary.txt`. All skips: v1 model / `cctv_litter_dataset_v1` images missing (`tests/conftest.py:13-17`) |
| Frontend node tests | **13/13 passed** | `node --test frontend/tests/*.test.mjs` |
| TypeScript | **0 errors** | `tsc --noEmit --incremental false` |
| Isolated E2E API (port 8010, scratch storage, DB snapshot, notifications blanked) | **33/33 passed** | `evidence/e2e.txt`, `e2e_results.json` |
| Video: litter clip (180 s, 640×368) | done in 45 s; 6 events; avc1 | `job_stats.json`, `job_events.json`, `video_frame_1800.jpg` |
| Video: footfall clip (180 s) | done in 61 s; **1 confirmed (`paper` 0.66, frame edge)**; 3 confirmed / 79 candidate / 116 weak detections | `job_footfall_stats.json`, `snap_test_footfall.jpg` |
| Video: indoor clip (91 s, 2304×1296) | done in 81 s; **1 confirmed (`food_wrapper` 0.65, decode artefacts, frame edge)**; source file corrupt | `job_indoor_clean_stats.json`, `snap_test_indoor_clean.jpg` |
| Production DB / weights untouched | confirmed | `storage/litter_monitor.db` unchanged (11-11-2026 17:55); v3 hash unchanged |

Prior evidence (older code, 2026-10-12): `deploy/TEST_RESULTS.md` covers the start/stop scripts and an 18/18 API check.

---

## 6. POC requirements

**Implemented and tested:**

- Image and video detection
- Background jobs with progress
- Tracking and confirmation
- Alerts with snapshots and status updates
- Annotated H.264 video
- JSON outputs and CSV exports (alerts, detections)
- Analytics, summary and history APIs
- Upload validation
- Local-only access, CORS and admin-token protection
- Start/stop scripts (tested earlier, 2026-10-12)

**Implemented, untested here:**

- The full browser UI: dashboard, alerts, history, reports, image and video pages
- Four-screenshot analysis with ROI proposal (pytest `test_sampled` passes)
- ROI drawing
- Login mode
- Live/RTSP cameras
- Notifications

**Partially implemented:**

- ROI: works, but not configured for any demo camera; auto-proposal is not in the full-video flow and is uncertain on every tested scene.
- Model evaluation tooling: exists, but has no labelled data.
- Integration tests: 38 skipped.

**Missing:**

- Measured accuracy on unseen footage
- Clean-scene and small-litter demo footage
- Per-video-job CSV
- Video job cancellation
- Storage retention
- Frontend E2E tests

**Bugs and blockers:** see §7.

**Manual steps currently required:**

1. Start with `deploy\start-ldm.cmd`. The first UI build takes 1–3 min.
2. Use the admin token from `.env` for admin actions.
3. Draw and save an ROI per `camera_id` before the demo.
4. Remember that demo uploads become real alerts in the **live** DB.
5. Stop with `deploy\stop-ldm.cmd`.

**Can safely wait until after the POC:**

- Login mode on, LAN access, RTSP/NVR cameras, notifications
- Retention cleanup, job cancellation
- Model retraining
- Lockfile cleanup, dependency pinning, removing `@vercel/analytics`
- Live-page title, hooks refactor
- AWS (out of scope)

---

## 7. Critical blockers, ranked by severity

| # | Severity | Blocker | Why it matters for the demo | Fix type |
|---|---|---|---|---|
| B1 | **Critical** | No valid **clean** CCTV video and no **small/distant-litter** video on this machine (the indoor "clean" clips decode corrupted) | 2 of the 3 required demo clips do not exist | Data: record or obtain footage |
| B2 | **Critical** | **No ground-truth measurement** of v3 on real footage; the litter clip is probably from v3's training scene | Any accuracy claim in the demo would be unsupported | Data + labelling: label events in the 3 demo clips; use the existing evaluation tools |
| B3 | **High** | **False alarms** on non-litter footage: 2/2 alerts at the bottom frame edge (partial or garbled objects); unreviewed baseline shows confirmed detections on indoor scenes | The clean-video demo would show false alerts | Config (ROI excluding frame edges and shop interior) and/or a small code filter for edge-truncated boxes, plus a measured re-test |
| B4 | High | **No ROI configured**: the UI shows "ROI NOT SET – FULL FRAME"; detections anywhere count | Weakens the demo; increases false positives (fridges, people, vehicles) | Operator step (feature exists) |
| B5 | High | Demo uploads go into the **live** DB (alerts 23 → 49 in the test copy after a few uploads) | Test and demo data mix with real history | Config: dedicated `LM_STORAGE_DIR` + `LM_DB_PATH` for the demo (variables exist) |
| B6 | Medium | Memory: ~90 % RAM in use (7.8 GB machine); a 2304×1296 video took 81 s for 91 s of footage | Risk of slowdowns if the browser + VS Code + UI build run together | Operational: close apps; use ≤ 1080p clips |
| B7 | Medium | UI never clicked through in this audit; cross-origin `download` links may open files instead of saving them | Possible demo surprise | Browser walkthrough (PROPOSED); small code fix if confirmed |
| B8 | Medium | 38 skipped tests (fixture model missing) | Detection-path regressions are not caught by CI | Code: point fixtures at v3 + small test images |
| B9 | Low | Stale `README.md` (v1-only, "38 tests"); `docs/video_analysis/EVALUATION.md` linked but missing | Confusing for reviewers | Docs |
| B10 | Low | No video-job cancellation; single worker queue | A long upload blocks later jobs | Post-POC |

## 8. Recommended implementation order (PROPOSED; nothing done yet)

1. **Demo footage** (B1). Record at the store-entrance camera, or a second camera if possible:
   - a) litter-positive on a **different day** than v3's training clip
   - b) the same view **after cleaning**, 2–3 min, no litter
   - c) a wide view with small or distant items

   Keep them ≤ 1080p, H.264.
2. **Isolated demo environment** (B5): demo storage and DB through the existing env variables, set in a copy of the start script or a demo `.env` section. Config only.
3. **ROI per demo camera** (B4): draw and save in the UI for each demo `camera_id`, excluding shop interior, road and frame edges.
4. **Ground truth for the 3 demo clips** (B2): list the true litter events per clip, then run the clips and score:
   - alerts: TP / FP / FN
   - false alerts per hour on the clean clip
   - detection time to confirmation

   Use the existing evaluation tooling (`backend/services/evaluation.py`) or a simple event sheet.
5. **Edge false-alarm fix** (B3), if step 4 confirms the pattern: reject boxes touching the frame border below a minimum visible size. Small, unit-tested change, then re-run step 4.
6. **Browser walkthrough** (B7) with `deploy\start-ldm.cmd` against the demo environment. Fix the download links if they misbehave.
7. **Restore integration tests** (B8) and refresh `README.md` (B9).

## 9. Proposed POC demonstration (PROPOSED)

| Part | Input | Show | Expected (to be confirmed in step 8.4) |
|---|---|---|---|
| 1. Litter-positive | clip (a), unseen day, ROI set | Video page: progress → annotated playback → timeline → alerts with snapshots | Each true litter event alerts once; no alert outside the ROI |
| 2. Clean | clip (b), same view after cleaning | Same flow | **0 alerts** (or a stated false-alerts-per-hour figure) |
| 3. Difficult | clip (c), small/distant litter | Same flow; then an image-page comparison of one frame at 640 vs tiling "on" | Honest result: which items are found and missed; tiling trade-off (time vs recall) |
| 4. Outputs | n/a | Download annotated MP4; alerts CSV and detections CSV from Reports/Alerts | CSVs open in Excel, with timestamps and bounding boxes |
| 5. Admin dashboard | n/a | Dashboard KPIs, 14-day chart, top classes, recent alerts; acknowledge an alert | Counts match the three runs |
| 6. Limitations slide | n/a | §11 of this report | n/a |

## 10. POC acceptance criteria (PROPOSED; to be agreed)

1. **Pipeline:** all 3 demo clips process to `done` with no error; each processes faster than real time on this laptop.
2. **Outputs:** annotated MP4 plays in the browser; alerts CSV and detections CSV download and open; dashboard counts match the API.
3. **Litter clip:** ≥ 80 % of hand-listed true litter events raise an alert, each at most once (dedup).
4. **Clean clip:** ≤ 1 false alert in the clip, with the false-alerts-per-hour figure stated.
5. **Difficult clip:** results reported honestly (found / missed per item); no accuracy claim.
6. **Reproducible:** start → demo → stop from the documented script, in an isolated demo DB; production weights hash unchanged.
7. **UI walkthrough:** every demo step done in the browser without console errors.

## 11. Remaining limitations (true even after the POC)

- One model (v3) trained on very limited data. Its 0.995 validation mAP is not independent.
- No measured accuracy on unseen cameras, night/IR, rain or glare.
- All real footage is from **one** store-entrance camera; results do not generalise to other stores.
- Small or distant litter is limited at 640 inference. Tiling helps but costs CPU time (§3.1).
- CPU only, 7.8 GB RAM. Speed figures are for this laptop.
- Local-only, login off by default, no retention policy, no cancellation.
- The AUG100 experiment (2 synthetic scenes) is exploratory and contributes no production evidence.
