# Live CCTV + scheduled screenshots — setup

Status: implemented and tested with a **sample video / mock stream only**. It has **not** been tested
against the real CP Plus NVR yet — the authorised RTSP details are still required (see section 1).

## 1. Information required from the CCTV administrator

| Item | Notes |
|---|---|
| NVR LAN (or VPN) IP address | The device serial/PSN shown in the CP Plus app (e.g. `01J05HAR58G8S9T7`) is **not** an IP. |
| RTSP port | Often 554 — confirm. |
| Channel number(s) | One per camera to monitor; say which are **outdoor** (litter) and **indoor** (uniform). |
| Main or sub stream | Sub stream is lighter; main stream gives more detail for small litter. |
| RTSP path for model CP-UNR-104F1 / firmware | The app defaults to `/cam/realmonitor?channel={channel}&subtype={subtype}` (common CP Plus/Dahua pattern) — **unconfirmed** for this device. |
| A view-only NVR account | Username + password. Do not use the NVR admin account. |
| Network access | The AI computer must be on the NVR's LAN or the company VPN. Never port-forward the NVR. |

## 2. Configure a camera (Live CCTV page)

1. Sign in as admin (Settings → Security) — all CCTV/scheduler controls need the admin token.
2. Live CCTV → Camera: **+ New CCTV camera** → enter the details above → **Test Connection**.
   It reports `CONNECTED` only after real frames were decoded; otherwise one of: unable to reach NVR,
   network timeout, authentication failed, channel unavailable, RTSP stream unavailable, invalid configuration.
3. **Save**. The password is written to the gitignored `.env` as `CCTV_<CAMERA_ID>_PASSWORD` —
   never to the database, API responses, logs or the browser.
4. Optional: **Connect Live CCTV** for continuous live detection; **Disconnect** stops it.
5. For outdoor cameras draw the pavement ROI (Detection ROI panel) — litter outside it is ignored.

## 3. Scheduled screenshots (Scheduled screenshots panel)

* **Min / max wait** (minutes) — each next capture is a random time in this range (default 120–180 = 2–3 h).
* **Model** — `Outdoor · litter` (active litter model + camera ROI) or `Indoor · uniform`.
* **Start / Pause / Resume / Stop / Capture Now**; the panel shows camera status, next capture (countdown),
  last result and recent screenshots (click for original vs annotated + detections).
* Each capture stores `storage/captures/<camera>/<id>/original.jpg`, `annotated.jpg`, `result.json` and a
  database row (timestamp, model version, detections with confidence, counts, alert ids, duration).
* Alerts reuse the existing alert manager. Litter still present at the next capture updates the existing
  alert (`occurrences`) instead of creating a new one.
* Reliability: one capture at a time per camera (Capture Now during a capture → "already running");
  3 connection attempts per capture; failures retry after `retry_minutes` (10) instead of waiting 2–3 h;
  schedule state is persisted — after a restart a missed capture runs **once** 1–3 min later and
  captures interrupted by a crash are marked `interrupted`.
* If the live stream is connected, the screenshot is taken from it (no second RTSP session).

### Uniform model
No uniform-detection model exists in this project yet. A camera set to `Indoor · uniform` saves the
screenshot and records `skipped — Uniform model not installed`. To enable it: register the trained model
(`python -m training.register_model --run <run dir> --version uniform_v1 ...`) and set the camera's
schedule `model_version` to it (`PUT /api/schedule/<camera> {"task":"uniform","model_version":"uniform_v1"}`).

## 4. Testing without the NVR
A camera whose source is a video file (e.g. `storage/test_sources/store_sample.mp4`, camera
`test_store`) appears under **Test sources** and is labelled TEST SOURCE. Scheduled screenshots pick a
random moment of the video. `tests/test_scheduler.py` covers capture, controls, random intervals,
overlap protection, failure/retry, restart recovery, uniform-without-model and alert de-duplication.

## 5. Security rules for online deployment
* Keep the NVR and this AI service inside the office network/VPN; publish only the dashboard/hub.
  No port forwarding of RTSP (554) or of the NVR web UI.
* Secrets live only in `.env` / the host's secret manager: `ADMIN_API_TOKEN`, `CCTV_*_PASSWORD`.
  **Set a long random `ADMIN_API_TOKEN` before exposing anything** (an empty value followed by a comment
  makes the comment text the token).
* `.env`, `.env.*` and `storage/` (screenshots, alerts, database) are gitignored.
* Before the dashboard is reachable from the internet, add user logins/roles: today read endpoints
  (alerts, captures, snapshots) are open to anyone who can reach the API; only changes need the admin token.
* Set `CORS_ORIGINS` to the dashboard's exact URL and serve the API over HTTPS only.
