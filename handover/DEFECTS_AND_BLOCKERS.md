# Defect and blocker register: LDM (2026-10-12)

Severity: **High** blocks a pilot · **Medium** must be fixed or accepted before a pilot · **Low** cosmetic / housekeeping.

## Defects

| ID | Title | Severity | Status |
|---|---|---|---|
| D-01 | "Backend offline" when the UI runs on :3010 | High | **Fixed** (Phase 8), re-tested |
| D-02 | `start-ldm.ps1` never returns when its output is captured | Medium | **Fixed**, re-tested |
| D-03 | Demo build rewrites `tsconfig.json` / `next-env.d.ts` | Low | **Fixed**, re-tested |
| D-04 | Ingester `--dry-run` wrote "given up" records to the ledger | Medium | **Fixed** (Phase 9), re-tested |
| D-05 | Sidebar showed "Connecting…" forever when the API was unreachable; health check had no timeout | Medium | **Fixed** (Phase 8), re-tested |
| D-06 | Video truncation (> `max_video_seconds`) is silent | Medium | **Open: fix proposed, awaiting approval** |
| D-07 | False positive: shoe sole detected as `plastic_piece` (confirmed) | Medium | **Open (model)** |
| D-08 | Read endpoints have no login | High for a multi-user pilot, n/a for the local demo | **Open** |
| D-09 | Activating a model silently applies its `recommended_settings` to the global thresholds | Medium (risk) | **Open: procedural control** |
| D-10 | Overlay detector misses frames with only partial overlay lines | Low | **Open** |
| D-11 | 38 tests skipped | Low | **Open** |
| D-12 | While offline, the dashboard keeps showing the last loaded figures | Low | **Open** |
| D-13 | "Failed to load OpenH264 library" in the API log | Low | **Open (cosmetic)** |

### D-01: "Backend offline" when the UI runs on :3010 (High, fixed)

- **Expected:** the UI shows "Connected" when the API is running.
- **Actual:** "Backend offline". `.env` `CORS_ORIGINS` allowed only `http://localhost:3000`, so the browser blocked
  every response, and the upload pre-check (preflight) returned 400.
- **Fix:** `start-ldm.ps1` adds its UI origin and verifies it before starting the UI. The status badge is now based
  on `/api/health`.
- **Re-test:** origin :3010 allowed; preflight 200; an unknown origin is still refused. Evidence:
  `deploy/TEST_RESULTS.md`.

### D-02: `start-ldm.ps1` never returns when its output is captured (Medium, fixed)

- **Expected:** the script exits when the demo is up.
- **Actual:** with the output piped or captured, the call waited until the servers stopped, because the children
  inherited the console handles.
- **Fix:** children start via `cmd /c … > log` through ShellExecute.
- **Re-test:** returned in 46 s with output captured.

### D-03: Demo build rewrites `tsconfig.json` / `next-env.d.ts` (Low, fixed)

- **Expected:** the developer files are unchanged.
- **Actual:** `next build` with its own build folder edited both files.
- **Fix:** the script restores both after the build.
- **Re-test:** `git diff` is empty.

### D-04: Ingester `--dry-run` wrote "given up" records to the ledger (Medium, fixed)

- **Expected:** a dry run writes nothing.
- **Actual:** corrupt and `.dav` files were recorded, so a later real run skipped them silently.
- **Fix:** no ledger writes in a dry run.
- **Re-test:** unit test `test_dry_run_writes_nothing…` passes.

### D-05: Sidebar showed "Connecting…" forever when the API was unreachable (Medium, fixed)

- **Expected:** "offline" when the health check fails.
- **Actual:** the badge was driven only by the event stream, and the health check had no timeout.
- **Fix:** a status hook based on health, with an 8 s timeout.
- **Re-test:** offline within ~20 s, then recovers.

### D-06: Video truncation is silent (Medium, open)

- **Expected:** if a video is longer than `video.max_video_seconds` (default 3600 s), the job result says that only
  part of it was analysed.
- **Actual:** the job ends "done", and `stats.video.duration_sec` reports only the analysed part (3.96 s for a 10 s
  clip with a 4 s limit). There is no truncation flag (UAT A17b).
- **Reproduce:** set `video.max_video_seconds` = 4 (Settings → Detection, or `PUT /api/settings`) and upload a 10 s
  video. The stats show `duration_sec` ≈ 4.
- **Evidence:** `handover/evidence/uat_app/app_extra_checks.txt`; code `backend/services/pipeline.py:329,367,399`
  (`frames = idx`, the file's `total` is never reported).
- **Proposed fix (needs approval; it touches the production video pipeline, not inference):** add
  `source_frames`, `source_duration_sec` and `truncated` to `stats.video`, and show a warning on the Video Analysis
  page. It only adds output fields.
- **Mitigation now:** the export ingester checks the real file length itself and logs or records a warning. Keep UI
  uploads ≤ 1 h.

### D-07: Shoe sole detected as `plastic_piece` (Medium, open, model)

- **Expected:** no litter on a pedestrian's shoe.
- **Actual:** `plastic_piece` 0.63, confirmed, on a white sneaker sole (UAT M04).
- **Reproduce:** Image Analysis on `dataset/ldm_v4_workspace/images/train/store_entrance_footfall__f0001640.jpg`.
- **Evidence:** `handover/evidence/uat_model/M04.jpg`, `model_uat_raw.json`.
- **Proposed fix:** add footwear and pedestrian frames as hard negatives in the fine-tuning dataset (Phase 2–4
  workflow). For video, the stillness/tracking filter usually rejects moving objects, but that is not verified for
  this case. **Do not raise thresholds blindly:** 0.63 is above the 0.60 operating point, and real litter in M02 sits
  at 0.82+, but a threshold must be chosen on a validation set.

### D-08: Read endpoints have no login (High for a pilot, open)

- **Expected:** only authorised users see alerts, snapshots and detections.
- **Actual:** anyone who can reach the API can read them; only changes need the admin token. Today the API binds to
  127.0.0.1, so only this PC can reach it.
- **Proposed fix:** before any multi-PC access, put the app behind the company's authenticated reverse proxy / SSO,
  or add user login and roles (development work). Do not use `-Lan` with real data until then.

### D-09: Activation applies `recommended_settings` silently (Medium risk, open)

- **Expected:** thresholds change only with approval.
- **Actual:** `backend/services/state.py` (`activate_model`) saves a model's `recommended_settings` as global settings.
- **Proposed fix:** procedure: before activating a new model, check its `model_meta.json` has no
  `recommended_settings`, or that the change is approved. Optional code change: require explicit confirmation.

### D-10: Overlay detector misses partial overlay lines (Low, open)

- **Expected:** frames with burned-in overlay are flagged.
- **Actual:** a footfall frame with only horizontal green line fragments (no box corner) was not flagged (seen in
  M04).
- **Proposed fix:** a human reviews `overlay_suspect=no` frames from that recording; better, obtain a raw NVR export
  of that footage.

### D-11: 38 tests skipped (Low, open)

- **Expected:** the full test suite runs.
- **Actual:** the skipped tests need the old v1 model and `cctv_litter_dataset_v1` images, which are not on this PC.
- **Proposed fix:** restore those files, or re-point the fixtures to v3 plus a small committed test set.

### D-12: Dashboard keeps showing the last figures while offline (Low, open)

- **Expected:** stale data is marked as stale.
- **Actual:** the badges turn red, but the figures stay unchanged.
- **Proposed fix:** grey out or label cards when health fails.

### D-13: "Failed to load OpenH264 library" in the API log (Low, cosmetic)

- **Expected:** a clean log.
- **Actual:** the message appears when writing annotated video. The fallback still produces playable H.264 (A05).
- **Proposed fix:** none needed; it is documented in TROUBLESHOOTING.

## Blockers (outside the app's code; need people, data or vendor input)

| ID | Blocker | Owner | Needed to unblock |
|---|---|---|---|
| B-01 | No annotated dataset; 0 of 215 images reviewed | Developer / annotator | confirm class definitions; annotate the 46 priority real frames (`dataset/ldm_v4_workspace/reports/annotation_queue.csv`) |
| B-02 | No unseen, representative real footage (night, rain, clean pavement, other cameras) | Manager / site | authorise raw NVR exports from several days/cameras; keep one full session as the test set |
| B-03 | No fine-tuned candidate, so no measured improvement | Developer (after B-01/B-02) | build + validate the dataset, v3 baseline, approved GPU training (`docs/LDM_FINETUNE_PLAN.md`) |
| B-04 | Automatic NVR retrieval unverified | CP PLUS support / installer | answers to `docs/CCTV_RECORDING_RETRIEVAL.md` §4 (ONVIF Profile G / SDK / playback RTSP, view-only account, LAN/VPN access) |
| B-05 | `cctv_store_v2` (v3's training data) missing | Whoever trained v3 (paths mention `D:\YATRIKART\…`, user "PATIL") | restore images and labels into `dataset/cctv_store_v2/` |
| B-06 | No GPU: training this laptop is impractical (0.7 GB RAM free during the demo) | Manager / IT | Colab or a GPU machine approved for this data |
| B-07 | Multi-PC access for QA and manager | IT | an approved internal server/VM or authenticated reverse proxy, plus D-08 |
