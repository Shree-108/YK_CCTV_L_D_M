# API security remediation: Step 0 and Step 1 (2026-10-15)

Approved scope: Step 0 (snapshot and baseline) and Step 1 (local-only access, docs off, no LAN startup) only.
No user accounts or route-level authentication. No change to the model, settings, inference or invoice feature.

## Step 0: snapshot and baseline

| Item | Value |
|---|---|
| Snapshot | commit `d0ea84d` on branch `snapshot/pre-api-security-20261015`: the full working state as found (129 files), committed unchanged; `main` (`95c151b`) not modified; nothing pushed |
| Pre-commit checks | no file > 1 MB; no `.env`, weights, database, video or CCTV images (git-ignored); the only images are 4 synthetic invoice samples and 2 brand logos; real secret values from `.env` searched in all 129 files: 0 hits |
| Work branch | `security/step1-local-only` (from the snapshot) |
| Tests (baseline) | `pytest`: **88 passed, 38 skipped** (2026-10-15 22:02 IST; isolated temporary storage) |
| Model | `models/litter_yolo/v3_store_cctv/best.pt`, active, sha256 `d1de9cb6ca7a8fe4348bd9a8c81bc3fdc5d0699965c263639b494486e2862bc6`, 10 classes |
| Settings | the live DB has **no settings overrides**, so the effective settings are the code defaults (weak/candidate/confirmed 0.30/0.45/0.60, image size 640, video sample_fps 2.0, …). Full dump: `baseline_settings_model_20261015.json` |

## Step 1: changes

| File | Change |
|---|---|
| `backend/api/local_only.py` (new) | pure-ASGI middleware: any request whose client is not a loopback address (127.0.0.0/8, ::1, IPv4-mapped loopback) gets **403** "Remote access … disabled until login protection is enabled" before reaching any route, whatever `--host` uvicorn uses. Logs at most one line per client per minute. Streaming responses are unaffected for local clients |
| `backend/main.py` | registers the middleware (outermost); `/docs`, `/redoc`, `/openapi.json` **off unless `LM_ENABLE_API_DOCS=true`**; docstring now uses `--host 127.0.0.1` |
| `backend/core/config.py` | `enable_api_docs` (env `LM_ENABLE_API_DOCS`, default false); `local_extra_hosts` (test-only, not settable from the environment) |
| `tests/conftest.py` | the shared test config treats Starlette's TestClient name `"testclient"` as local (one line) |
| `tests/test_local_only.py` (new) | 35 tests |
| `deploy/start-ldm.ps1` | `-Lan` now **refuses to start** (exit 1) with an explanation; summary no longer advertises `/docs` |
| `README.md` | quick start uses `--host 127.0.0.1`, with a local-only warning; docs off by default |
| `.env.example` | documents `LM_ENABLE_API_DOCS` |
| `deploy/README.md`, `deploy/SETUP.md`, `deploy/TROUBLESHOOTING.md`, `handover/DEPLOYMENT_AND_ROLLBACK.md` | LAN access described as disabled until login protection exists; `/docs` 404 explained |

Not changed (verified empty `git diff`): `invoice_extractor/`, `backend/api/routes_invoice.py`, `frontend/`, `app/invoices/`,
`backend/services/` (pipeline, state), `backend/detection/`, `backend/core/settings.py`, `models/`, `requirements.txt`.

## Step 1: test results (actual)

| Test | Result |
|---|---|
| `tests/test_local_only.py` | **35 passed**: loopback detection (13 cases, including IPv4-mapped, link-local, "localhost" string); local `/api/health` and `/api/classes` 200 via 127.0.0.1 and ::1; **remote client 403** on health, alerts, snapshot, camera stream and frame, capture image, dataset image, CSV export, SSE stream, invoices, `PATCH` alert, image upload, evaluations, docs; remote CORS preflight 403; rejected remote writes leave 0 detections / 0 alert events in the DB; docs 404 by default; docs 200 locally when enabled and 403 remotely; env flag read, extra hosts not settable from env |
| Real server (uvicorn `--host 0.0.0.0`, isolated temp storage, port 8766) | loopback `/api/health` **200**; this PC's LAN address `/api/health` and `/api/alerts` **403**; LAN request with spoofed `X-Forwarded-For: 127.0.0.1` **403**; `/docs` and `/openapi.json` **404**; 1 throttled log line; server stopped, temp storage deleted |
| `start-ldm.ps1 -Lan` | `[FAIL] -Lan is disabled …`, exit 1; no process started (0 listeners on 8000/3010 before and after) |
| Full suite | **123 passed, 38 skipped** (2026-10-15 22:08 IST): the 88 baseline passes + 35 new; same 38 skips (missing v1 model) |
| Model and settings | sha256 unchanged; effective detection settings identical to the baseline |

## Known effects and limits

- **Local only:** every client that is not this PC gets 403, including the invoice endpoints, which are covered without changing the invoice code. This is intended until login exists.
- **The UI must be opened on this PC** (`http://localhost:3010`). This is the same as before, since `start-ldm` was already local.
- **Reverse proxies:** do not put one in front of the API before Step 4. A proxy on this PC that does not pass the real client address would look local.
- **Old evidence scripts:** the scripts in `handover/evidence/uat_scripts/` use TestClient without the test flag, so they would now get 403. They are records of past runs and were not changed.
- **Rollback:** `git checkout snapshot/pre-api-security-20261015 -- <file>`, or discard the work branch.
