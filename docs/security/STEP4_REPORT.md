# API security — Step 4 report: route permissions (backend authorization)

Status: **implemented, reviewed and hardened; not committed** (awaiting human review). Branch `security/step1-local-only`, based on
85e365c. Login stays **off by default** (`LM_AUTH_MODE=off`). LAN access stays off; the local-only safeguard is
unchanged.

## Pre-change checks

- **Checkpoint:** branch `security/step1-local-only`, HEAD `85e365c`, clean working tree.
- **Baseline:** `133 passed, 38 skipped`.
- **Reviewed before writing code:**
  - auth service, sessions, routes and cookie;
  - the Step 3 report and frontend gate;
  - API client, `require_admin` and `ADMIN_API_TOKEN` usage;
  - all 69 API operations.
- **No WebSocket endpoints exist.** There are two streaming responses: the SSE `/api/stream` and the MJPEG `/api/cameras/{id}/stream`.

## Design

- **[backend/api/authz.py](../../backend/api/authz.py)** is the single place where permissions are decided.
  - Every route declares exactly one permission: `PUBLIC`, `VIEWER`, `OPERATOR`, `ADMIN` or `ADMIN_IF_LOGIN`.
  - Role order: viewer < operator < admin.
  - `tests/test_authz.py` fails if a route has no permission, has more than one, or differs from the reviewed matrix below.
- **Login on:**
  - The caller is identified by the session cookie, or by `ADMIN_API_TOKEN`, which identifies an admin.
  - No identity → **401** `Authentication required: sign in`.
  - A role that is too low → **403**, naming the role the action needs.
  - Session validity (idle 30 min, absolute 12 h, logout, disabled user, password or role change) is checked on the server on every request.
- **Login off (the default): exactly the pre-Step-4 behaviour.**
  - `ADMIN` routes need `ADMIN_API_TOKEN` as before, and answer 503 when no token is configured.
  - All other routes are open, as before, and only reachable from this PC (local-only safeguard).
- **Cross-site writes:**
  - For cookie-authenticated `POST/PUT/PATCH/DELETE`, an `Origin` header that is not in `CORS_ORIGINS` and is not the API's own origin gets **403**.
  - `SameSite=Strict` alone does not block other pages on `localhost`, because any port counts as "same-site".
  - Requests without `Origin` (non-browser clients) and token-authenticated requests are not affected, since a header token cannot be forged cross-site.
- **Streams:**
  - Both streams need at least viewer.
  - While open they re-check the session every 30 s and end when it has expired or been revoked.
  - The re-check does not extend the idle timer (`session_principal(..., touch=False)`).
- **Health:**
  - `/api/health` is public.
  - With login on, anonymous callers get only `{status, time}`.
  - Signed-in users and the admin token get the existing details, which the dashboard uses.
  - `/api/system/health` (CPU, memory, disk, database size, per-camera errors) is admin-only with login on.
- **Logging:** a denied request is logged with username, role, method, path and the required role. Session IDs, passwords and tokens are never logged or returned.
- **Session lifetimes and lock-out** are unchanged, and match the approved values: idle 30 min, absolute 12 h, 5 failures → 15 min lock-out.
- **Rate limiting** beyond login is deferred to Step 5.

## Role–permission matrix

`admin*` = needs `ADMIN_API_TOKEN` even with login off (unchanged). `admin (login)` = admin with login on, open with
login off (as before Step 4).

| Permission | Routes |
|---|---|
| **public** | `GET /api/health` (minimal for anonymous callers when login is on); `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me`, `POST /api/auth/activity` (added in the fourth pass; needs a valid session and an allowed Origin) |
| **viewer** (read) | `GET` on: `/api/classes`, `/summary`, `/events`, `/cameras`, `/cameras/{id}`, `/cameras/{id}/frame.jpg`, `/cameras/{id}/stream`, `/cameras/{id}/live`, `/schedule/{id}`, `/captures`, `/captures/{id}/image`, `/roi/{id}`, `/detections`, `/jobs`, `/jobs/{id}`, `/jobs/{id}/files/{kind}`, `/alerts`, `/alerts/{id}`, `/alerts/{id}/snapshot`, `/alerts/{id}/timeline`, `/analytics`, `/models`, `/settings`, `/stream` (SSE), `/search`, `/datasets`, `/datasets/{name}`, `/datasets/{name}/images`, `/datasets/{name}/image`, `/datasets/{name}/annotation`, `/experiments`, `/experiments/{id}`, `/experiments/{id}/artifacts/{name}`, `/evaluations`, `/evaluations/{id}`, `/hard-examples` |
| **operator** (daily work) | `POST /api/detect/image`, `POST /api/detect/video`, `PATCH /api/alerts/{id}` (acknowledge/resolve), `PUT /api/roi/{id}` for an upload context, `POST /api/testlab/run`, `POST /api/testlab/upload`, `POST /api/testlab/reviews`, `PATCH /api/hard-examples/{id}` |
| **admin\*** (configuration; unchanged) | `POST/PUT/DELETE /api/cameras…`, `POST /api/cameras/{id}/restart`, `POST /api/cctv/test`, `PUT /api/cctv/{id}`, `POST /api/cctv/{id}/connect`, `POST /api/cctv/{id}/stop`, `PUT /api/schedule/{id}`, `POST /api/schedule/{id}/{action}`, `POST /api/models/activate`, `POST /api/models/register`, `PUT /api/settings`, `PUT /api/datasets/{name}/annotation`, `POST /api/datasets/{name}/versions`, `GET /api/datasets/{name}/export`, `GET /api/auth/check`; `PUT /api/roi/{id}` for a **registered camera** |
| **admin (login)** (new) | `GET /api/export/{alerts,detections}.csv` (approved: exports admin only), `GET /api/system/health` (detailed diagnostics), `GET /api/cctv/defaults` (NVR host/username from `.env`), `POST /api/evaluations` (compute-heavy; decided in the third pass) |

Totals (final): 5 public, 36 viewer, 8 operator, 17 admin\*, 4 admin (login), 70 in all. Step 4 kept the original 69 operations and added one: the public `POST /api/auth/activity` (fourth pass). No invoice route exists.

## Compatibility decisions

- **`ADMIN_API_TOKEN`:**
  - With login off it behaves exactly as before.
  - With login on it identifies an admin, so it can call every route. Every non-admin route was already open to it before Step 4, so its reach is not expanded.
  - It still does not bypass the local-only safeguard.
  - With login on and no token configured, admin routes answer 401 (sign in) instead of 503.
- **Frontend** (small, needed for working admin sessions):
  - `isAdmin` is now true for a verified admin token **or** a signed-in admin session, so admin buttons unlock after an admin signs in.
  - Any 401 makes the login gate re-check at once, so an expired session shows the sign-in screen within about a second instead of 60 s.
  - These are display aids only; the server decides.

## Tests

- **New: `tests/test_authz.py` (28 tests).**
  - The matrix: every route declared exactly once and equal to the reviewed table; no unclassified or invoice routes.
  - Login off by default, with pre-Step-4 behaviour on **every** route, including the admin-token requirement and the 503 without a token.
  - With login on:
    - 401 on every protected route;
    - viewer, operator and admin boundaries on **every** route;
    - the admin token as admin, and a wrong token → 401;
    - admin-only CSV exports;
    - ROI operator vs admin;
    - health disclosure;
    - invalid cookies;
    - idle and absolute expiry (fake clock);
    - revoked sessions (logout, disable, password change, role change), including a replayed cookie;
    - lock-out → 429 with protected routes staying closed;
    - cross-origin cookie writes refused;
    - stream authentication and stream termination when the session ends;
    - stream checks don't extend sessions;
    - viewer/operator workflows, including a real `detect/image`.
- **Updated:** the Step 2 test that asserted routes were still unprotected now asserts 401 without a login and 200 with the token.
- **Results:**
  - Full backend suite (isolated temporary storage): **161 passed, 38 skipped, 0 failed** (before the review below; after it: 165 passed).
  - `tsc --noEmit`: pass.
  - Production build: pass (23/23 pages).
- **Skipped (38), unchanged from the baseline:**
  - integration 19, scheduler 8, lab_api 5, capture_faults 4, roi_upload 2;
  - 37 with reason "trained model/test images not present", 1 "no test images";
  - they run only with login off, so the admin-token paths inside them were not exercised. The new matrix test covers those routes with login off.
- **Pre-existing intermittent test:** `tests/test_ldm_dataset_tools.py::test_export_batch_puts_real_first_and_skips_overlay`. The root cause is in the review section below.

## Browser checks (isolated stack, synthetic accounts, all removed afterwards)

- **Anonymous:** `/api/health` returns `{status, time}` only, and `/api/alerts` returns 401.
- **Viewer:** the dashboard, alerts, analytics and cameras load. Detection, alert update, export, diagnostics and settings changes return 403.
- **Operator:**
  - `POST /api/detect/image` returned 200 (`v3_store_cctv`), alert update reaches the route, upload ROI 200, and the SSE stream opens;
  - export, diagnostics, CCTV defaults and settings return 403;
  - the Live CCTV page handles its 403 without errors.
- **Admin:** the "AD" chip shows and admin buttons are enabled. CSV export returns 200 (`text/csv`), and diagnostics, CCTV defaults and `auth/check` return 200.
- **Session revoked on the server mid-use:** the next API call's 401 showed "Your session has expired" after 0.8 s.
- **Login off (default):**
  - no login screen and the "OP" chip, as before;
  - alerts, export, diagnostics and CCTV defaults are open as before, while `auth/check` and `PUT /api/settings` need the token;
  - no `auth.db` is created.
- **Leak check:** no password, session cookie or bearer token appears in the backend log, frontend log or browser console.

## Integrity

- Model sha256 `d1de9cb6…` is unchanged.
- The live database fingerprint `f7c83c2dafd7ea24` is unchanged, with 0 settings rows.
- No real `storage/auth.db` exists.
- No change to detection, CCTV, alerts or analytics logic, settings, `local_only.py` or invoice-removal work. Route handler bodies are unchanged except the health detail switch, the stream re-checks and the ROI admin call.
- No dependencies were added.

## Review and hardening (second pass)

### Frontend

- **Role label:**
  - The top-bar chip shows the real role: Viewer (VW), Operator (OP) or Admin (AD), with a tooltip saying what the role allows.
  - Login on: the account's role, or admin when a verified admin token is present. Login off: Operator (or Admin with the token), as before.
  - Signed-out and expired sessions never reach the shell; the sign-in screen is shown instead.
- **CSV exports:**
  - The plain links are replaced by `ExportCsvButton` (Alerts, History, Reports), which downloads through fetch with the cookie or token.
  - Failures show fixed messages ("Permission denied: You do not have permission to download this. Ask an administrator.", "Your session has ended...") and never the server's JSON.
  - With login on, the button is disabled for non-admins ("Only administrators can export CSV files").
  - `downloadAuthed` (also used for dataset ZIP export) now maps every failure to a fixed message and signals the login gate on 401.
- **Role-specific controls:** operator-only actions are disabled for viewers, with the tooltip "Your role (viewer) is read-only":
  - Analyze Image and Analyze Video;
  - ROI Define, Reset and Save;
  - alert Acknowledge, Resolve, Reject and Save notes (bulk and drawer);
  - Test Lab Run, and the verdict buttons;
  - Ground Truth "Show predictions";
  - Evaluate;
  - hard-example Approve, Reject and Ignore.
- **Admin-only controls** already used `isAdmin`, which now reflects the role. With login on, camera pages and Live CCTV say "Only administrators can do this" / "needs an administrator account" instead of pointing at the admin token.
- **System health:** the page shows "Administrators only" for non-admins when login is on, instead of erroring.
- **Settings → Security:** shows the signed-in role. An admin account is told no token is needed, and the old "no ADMIN_API_TOKEN configured" warning no longer appears by mistake for non-admins.
- These are display aids only. The server remains the security boundary.

### Backend (least privilege, login on only; admins and login-off mode unchanged)

- **Cameras** (`GET /api/cameras`, `GET /api/cameras/{id}`):
  - Non-admins no longer receive `rtsp` (NVR host, port, username, path), `rtsp_url` or `has_password`.
  - `source` is reduced to its kind: `cctv:<id>`, `rtsp://(hidden)`, `env:(hidden)`, `file:(hidden)` or `webcam`.
  - A new `rtsp_configured` boolean (sent to everyone) keeps the Live CCTV page's status correct.
  - Passwords were never returned, and still aren't.
- **Health:** signed-in non-admins get only what the dashboard uses: status, time, model version, device, classes and image size, camera counts, notifications. They do not get the model file path, uptime, host CPU/memory/GPU, hardware or `admin_enabled`, and the model-load error is replaced by a generic message. Anonymous callers still get `{status, time}`.
- **Models:** for non-admins, `path` is relative to the models folder (e.g. `litter_yolo/v3_store_cctv/best.pt`) instead of an absolute Windows path.
- **New tests (4):** non-admins (viewer and operator) get no NVR host, username, camera host, credentials or stream path; admins and login-off mode still get full details (never passwords); model paths are relative for non-admins and absolute for admins; health trimming.

### Route review (all 69 routes at the time; 70 after the fourth pass)

The table above is unchanged and verified by `test_every_route_declares_exactly_one_permission_and_matches_the_reviewed_matrix`.

| Area | Status |
|---|---|
| CSV exports | admin with login on |
| Detailed diagnostics | admin with login on |
| CCTV defaults | admin with login on |
| Camera, CCTV, schedule, settings, model and label configuration | admin, and the admin token with login off, as before |
| Registered-camera ROI | admin; upload ROI is operator |
| Streams | viewer, re-checked every 30 s |
| `ADMIN_API_TOKEN` | admin, with no new reach |

### Flaky test: root cause (not changed in this patch)

- **The defect is not file order.** `_populate()` in `tests/test_ldm_dataset_tools.py` draws its synthetic images with `scene(hash((s, i)) % 10**6)`. Python salts `str` hashing per process (`PYTHONHASHSEED`), so the image content differs on every run.
- **Reproduction:** with a fixed seed the result is deterministic. Seeds 0–55 and 57–70 pass. **`PYTHONHASHSEED=56` fails every time**: the import step's overlay detector flags one of the random synthetic images as `overlay_suspect`, so together with the row the test marks itself, two images are excluded and the batch has 2 rows instead of 3.
- **Scope:** a test-fixture defect, not a product defect. The overlay heuristic is working as designed on an image that happens to contain box-like shapes. No LDM code path depends on `hash()` here.
- **Smallest deterministic fix:** in the test helper, replace `hash((s, i))` with a stable value such as `zlib.crc32(f"{s}/{i}".encode())`, and/or have the test clear `overlay_suspect` on every other row before marking one.
- It lies outside Step 4, and a separate task is fixing it in its own worktree, so it is **not** part of this patch.

### Results after the review

- Full backend suite (isolated storage): **165 passed, 38 skipped, 0 failed**. The flaky test passed in this run.
- `tsc --noEmit`: pass. Production build: pass (23/23 pages, no invoice route).
- **Skipped (38), unchanged:** integration 19, scheduler 8, lab_api 5, capture_faults 4, roi_upload 2 ("trained model/test images not present" ×37, "no test images" ×1). These scenarios are **not** verified.
- **Browser** (isolated stack, synthetic accounts, all removed afterwards):
  - **Viewer:** chip "Viewer". Export, Analyze, Evaluate and camera actions are disabled with reasons. No NVR host or username anywhere; Live CCTV shows "Configured (connection details are visible to administrators)". System health shows "Administrators only".
  - **Operator:** chip "Operator". Analyze has no role lock; export, camera setup and diagnostics are locked.
  - **Admin:** chip "Admin". Export produced `alerts_<time>.csv` via blob, with no raw JSON. Full diagnostics, NVR details visible and editable. After a server-side revocation, Export showed the sign-in screen ("Your session has expired"), with no download and no JSON.
  - **Login off:** chip "Operator", export works without a token, full diagnostics and camera details, as before.
  - No JavaScript errors in the console, and no password, session cookie or token in any log.
- **Integrity:**
  - model sha256 `d1de9cb6…` and live DB `f7c83c2dafd7ea24` unchanged (0 settings rows);
  - no real `storage/auth.db`;
  - `local_only.py` unchanged;
  - commits `85e365c` and `265e98a` unchanged;
  - `main` and the snapshot branch unchanged;
  - nothing pushed.

## Final decisions and risk review (third pass)

### Decisions applied

- **Evaluations: admin-only with login on.**
  - `POST /api/evaluations` is now `ADMIN_IF_LOGIN`: admin with login on, open with login off as before.
  - Ordinary operator work (image/video detection, alert review, upload ROI, test lab) is unchanged.
  - Frontend: the Evaluate button is disabled for viewers and operators with login on ("Only administrators can do this").
  - Tests: viewer and operator get 403, the operator's `detect/image` is still authorized, admin is allowed, and login off is unchanged.
- **Login off (development mode): unchanged.**
  - Login-off behaviour is exactly what existed before Step 4: CSV exports, diagnostics, camera details, frame grabs and evaluations stay open to callers on this PC, and `ADMIN` routes still need `ADMIN_API_TOKEN`.
  - Nothing is opened beyond that, and the local-only safeguard is untouched.
  - **Development-mode limitation:** with login off, anyone using this PC (or any process on it) has the access described above. Login must be on before the application is used by more than one person or exposed beyond this PC.
- **Sessions: unchanged policy (idle 30 min, absolute 12 h), now with more tests.**
  - *(Superseded by the fourth pass.)* Ordinary requests and background polling never extend the idle deadline. Only real user activity does, reported through `POST /api/auth/activity`. The server records it at most every 30 s, and it can never move the deadline past the 12-hour absolute limit.
  - Stream re-checks never refresh it.
  - Idle and absolute expiry, logout, disable, password change and role change all end access to every protected route.
  - Open streams close when the session is revoked or times out (new test: idle timeout with no logout ends the SSE stream).

### Risk review

| # | Risk | Realistic impact | Exploitable? | Mitigation | Status |
|---|---|---|---|---|---|
| 1 | Event and error text visible to viewers | Information disclosure to a signed-in low-privilege user: NVR/camera IP addresses and host names, NVR user names, RTSP URLs (passwords already masked), and server file paths. These appear in camera errors, capture/schedule errors, job/evaluation errors and system events, and the events are also pushed over the live SSE stream. Useful for reconnaissance of the CCTV network; no direct access gained. | Yes, by any viewer, via `/api/events`, `/api/stream`, `/api/cameras` (`last_error`), `/api/cameras/{id}/live`, `/api/schedule/{id}`, `/api/captures`, `/api/jobs`, `/api/jobs/{id}`, `/api/evaluations`. | **Implemented:** `text_redactor` (backend/api/authz.py) replaces URLs, IPv4 addresses (with port), Windows/UNC/POSIX paths, and the configured camera host and user names with `[hidden]` in all of those responses, for non-admins with login on. Stored data is unchanged; admins and login-off mode see the original text. | Fixed; 4 tests (REST endpoints for viewer and operator, SSE stream, admin and login-off unchanged) |
| 2 | Viewing a camera frame makes the server connect to the camera | A viewer can repeatedly make the server open a new RTSP/HTTP connection to the NVR. Each attempt holds a server worker thread until the open/read timeouts expire and uses an NVR connection slot (NVRs allow few). This can be used to exhaust server threads or NVR sessions, and to probe connectivity. | Yes, by any viewer, via `GET /api/cameras/{id}/frame.jpg` for a camera that is not running (frames of running cameras come from memory and need no connection). | **Implemented:** opening a new connection is admin-only with login on. Frames from a running camera remain available to viewers. The only UI callers are the admin ROI editors. The Cameras editor's fetch now sends the session cookie (it did not, so it would have failed even for admins with login on); the Live CCTV ROI panel shows "No frame available…" instead of loading forever. CCTV configuration and connectivity are unchanged. | Fixed; test (viewer and operator 403 before any connection, admin allowed, login off unchanged); browser check (403 logged, no connection attempt) |
| 3 | Dashboard polling keeps a session alive (superseded: fixed in the fourth pass below) | Background polling (every 15–30 s) and the live stream count as activity. An unattended, signed-in browser therefore stays signed in until the 12-hour absolute limit, so anyone at that PC can use the session. The idle timeout only takes effect once the tab is closed or hidden. | Only with physical or remote-desktop access to an unlocked PC with the app open; not remotely. | **Proposed, not implemented, because this is a product decision.** Count only user-initiated requests (clicks, navigation, changes) as activity: the frontend would mark polling requests and the server would not refresh the idle timer for them (`touch=False`, already used by streams). Optionally also add a client-side inactivity sign-out with a warning. **Trade-off:** a wall-mounted control-room dashboard that is watched but not touched would be signed out after 30 minutes. Needs your decision (for example: enforce for all roles; or exempt a dedicated "display" account; or keep as is). | Open: awaiting approval |

### Results

- **Backend suite (isolated temporary storage):** 173 passed, 0 failed, 38 skipped.
  - New in this pass: 8 tests, covering evaluations, redaction (×4), frame grabs, stream idle-expiry and idle-refresh granularity.
  - `tests/test_authz.py`: 40 tests.
- **Skipped (38), unchanged:**
  - integration 19, scheduler 8, lab_api 5, capture_faults 4, roi_upload 2;
  - reasons: "trained model/test images not present" ×37, "no test images" ×1;
  - these scenarios are **not** verified.
- **`tsc --noEmit`:** pass.
- **Production build:** pass (23/23 pages, no invoice route).
- **Browser (isolated stack, synthetic accounts, all removed afterwards):**
  - **Admin:** Evaluate enabled.
  - **Operator:** Evaluate disabled ("Only administrators can do this"), Analyze not role-locked, no NVR host visible on Live CCTV. `GET /api/cameras/nvr1/frame.jpg?raw=1` → 403, logged as an authz denial, with no connection attempt.
  - No password, session cookie or NVR host in any log.
- **Dataset-tools test:** `test_export_batch_puts_real_first_and_skips_overlay` passes with `PYTHONHASHSEED=0` and fails with `PYTHONHASHSEED=56`, as documented above.
  - It is a test-fixture defect (hash-salted synthetic images), not a product defect.
  - It is being handled in a separate workstream (branch `claude/elastic-shtern-f65dfa`, which had no changes at the time of this report) and was **not** modified here.
  - It passed in this pass's full run.
- **Integrity:**
  - model sha256 `d1de9cb6…` and live DB `f7c83c2dafd7ea24` unchanged;
  - no real `storage/auth.db`;
  - `local_only.py` unchanged;
  - invoice removal intact;
  - commits `85e365c` and `265e98a` unchanged;
  - `main` and the snapshot branch unchanged;
  - nothing staged, committed or pushed.

## Final role–permission matrix

Totals: 5 public, 36 viewer, 8 operator, 17 admin\*, 4 admin (login), 70 in all. The fourth pass added the public `POST /api/auth/activity`.

| Permission | Routes |
|---|---|
| public | `GET /api/health` (status only for anonymous callers; trimmed for non-admins), `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me`, `POST /api/auth/activity` (needs a valid session and an allowed Origin; the only request that extends the idle timeout) |
| viewer | all read-only `GET` routes listed in the matrix above, including both streams. Free text is redacted and infrastructure fields are hidden for non-admins. `frame.jpg` needs admin when it would open a new connection. |
| operator | `POST /api/detect/image`, `POST /api/detect/video`, `PATCH /api/alerts/{id}`, `PUT /api/roi/{id}` (upload context), `POST /api/testlab/run`, `POST /api/testlab/upload`, `POST /api/testlab/reviews`, `PATCH /api/hard-examples/{id}` |
| admin\* (token also needed with login off) | camera, CCTV, schedule, settings, model and dataset configuration (17 routes, unchanged) and the ROI of a registered camera |
| admin (login) | CSV exports, `GET /api/system/health`, `GET /api/cctv/defaults`, `POST /api/evaluations` |

## Open items for review

1. **Risk 3 (polling keeps sessions alive):** decided (Option A) and implemented in the fourth pass below.
2. **Carried forward:**
   - no rate limiting beyond login (Step 5);
   - the admin token is still kept in browser sessionStorage;
   - *(resolved in the fifth pass: a dedicated test now covers the MJPEG camera stream closing on logout and on idle timeout)*;
   - 38 skipped tests;
   - the dataset-tools fixture defect (separate workstream).
3. **Residual (accepted, low):** requests without an `Origin` header (non-browser clients) are not subject to the cross-origin check. Browsers always send `Origin` on cross-origin writes.

## Approved idle-timeout policy (fourth pass)

**Decision (Option A):** the 30-minute idle timeout applies to everyone: every role and every dashboard session, with no display-account exemption. Only real user activity extends a session.

### Implementation

- **Server: checking a session never extends it.**
  - `AuthService.session_principal()` is now read-only. It is used for every API request (reads and writes), background polling, `/api/health`, `/api/auth/me` and the stream re-checks.
  - Before this change, any authenticated request moved the idle deadline.
- **Server: one way to extend.**
  - `AuthService.record_activity()` sets the idle deadline to now + 30 min, never past the 12-hour absolute limit, and writes at most once every 30 s per session.
  - It is reachable only through **`POST /api/auth/activity`**, which needs:
    - the session cookie (the admin token alone has no session, so 401);
    - an `Origin` header that is in `CORS_ORIGINS` or is the API's own origin (missing, `null` or foreign → 403);
    - login on (404 with login off).
  - It returns only the new `idle_expires_at`, with `Cache-Control: no-store`.
  - An expired session cannot be revived through it (401).
- **Browser (`AuthGate`):** while signed in, trusted `pointerdown`, `keydown`, `wheel` and `touchstart` events in a visible tab report activity at most once a minute.
  - Events dispatched by scripts (`isTrusted === false`) are ignored.
  - Polling, timers, streams and page loads never report activity.
  - If the report finds the session gone, the gate re-checks and shows the sign-in screen.
- **Expiry handling:**
  - The backend rejects an expired session on every route (401).
  - The frontend shows "Your session has expired. Please sign in again." as soon as a polling request gets 401, or at the next 60 s re-check.
  - Open SSE and MJPEG streams re-check every 30 s (read-only) and close once the session has expired or been revoked.
- **Unchanged:**
  - idle 30 min, absolute 12 h, lock-out 5 failures / 15 min;
  - roles and the permission matrix;
  - read-only monitoring and camera viewing (they keep working until the session expires);
  - login-off mode (no sessions, no change).

### What "cannot be bypassed by arbitrary requests" means here

- No GET, polling, health check, stream or ordinary write (detection, alert update, ROI save, settings change) extends a session. Tests check polling GETs and writes explicitly.
- The single activity endpoint refuses requests without an allowed browser Origin and refuses sessions that have already expired.
- **Limitation (inherent):** whoever holds the session cookie *and* deliberately sends an allowed `Origin` header can report activity, just as the real user's clicks do. The server cannot tell a person from a program that holds the user's own cookie.
  - The cookie is HttpOnly, so page scripts cannot read it.
  - Script-generated input events in the page are ignored.
  - The policy therefore protects against the realistic case, an unattended open browser, not against someone who already controls the session.
  - The 12-hour absolute limit still applies in every case.

### Tests (new or rewritten)

- **`tests/test_authz.py`:**
  - polling GETs (`/api/alerts`, `/api/health`, `/api/auth/me`, `/api/summary`, `/api/cameras`, `/api/events`) never extend the timeout;
  - writes do not extend it either;
  - `POST /api/auth/activity` extends it and returns only the deadline;
  - the same policy applies to viewer, operator and admin;
  - activity is refused without a session, without an Origin, with a foreign or `null` Origin, via GET, and for a token without a session, and refused reports extend nothing;
  - activity is recorded at most every 30 s;
  - activity never moves the deadline past 12 h after login, and the session ends at 12 h even with continuous activity;
  - login off has no activity endpoint;
  - the event stream closes on logout and on idle timeout;
  - an open, watched-but-untouched stream does not keep the session alive.
- **`tests/test_auth.py`:** checking a session never extends it; `record_activity` does; the 12 h limit holds under continuous activity.

### Results

- **Auth suites** (`test_authz`, `test_auth`, `test_local_only`, `test_app_routes`): 127 passed.
- **Full backend suite** (isolated temporary storage): **181 passed, 0 failed, 38 skipped.**
- **Skipped (38), unchanged:**
  - integration 19, scheduler 8, lab_api 5, capture_faults 4, roi_upload 2;
  - "trained model/test images not present" ×37, "no test images" ×1;
  - **not verified**.
- **`tsc --noEmit`:** pass.
- **Production build:** pass (23/23 pages, no invoice route).
- **Browser** (isolated stack with `LM_AUTH_IDLE_MINUTES=2`, synthetic viewer, all removed afterwards):
  - Signed in through the form.
  - For 84 s the dashboard made 31 polling requests and received script-dispatched input events, with **0** activity reports and an unchanged deadline.
  - One real mouse click on blank page text produced exactly one `POST /api/auth/activity` (200), and the deadline moved from 05:26:50 to 05:28:44.
  - With no further input, the backend began answering 401 at about 05:28:52, just after that deadline, and the page showed "Your session has expired. Please sign in again."
  - No password or session cookie in any log.
- **Not verified in the browser:** the *server-side* closing of an open stream. In the browser run the page itself closed its SSE connection when the sign-in screen replaced the app. Server-side closing on expiry is covered by the automated tests.
- **Integrity:**
  - model sha256 and live DB `f7c83c2dafd7ea24` unchanged;
  - no real `storage/auth.db`;
  - `local_only.py` unchanged;
  - the flaky-test worktree (`claude/elastic-shtern-f65dfa`, `95c151b`) untouched and without changes;
  - nothing staged, committed or pushed.
- **Change set:** 32 files, the 31 reviewed earlier plus `frontend/api/auth.ts` (new `recordActivity()`).

### Remaining limitations

1. A signed-in user who only watches the dashboard (no input) is signed out after 30 minutes. This is the intended effect of Option A. Long-running jobs such as a video analysis keep running on the server, but the user has to sign in again to see the result.
2. The cookie-holder limitation described above.
3. A page reload is not counted as activity; a click or key press after the reload is.
4. Activity is recorded at most every 30 s on the server and reported at most once a minute by the browser, so the effective deadline can be up to about a minute earlier than "last click + 30 min".
5. Carried forward:
   - no rate limiting beyond login (Step 5);
   - the admin token is still kept in sessionStorage;
   - *(resolved in the fifth pass: dedicated MJPEG stream-closure test added)*;
   - 38 skipped tests;
   - the dataset-tools fixture defect (separate workstream; seed 56 still fails).

## Final security review (fifth pass)

### Scope

- The complete working-tree diff (all 32 files) was reviewed: the backend file by file, and the frontend through the full diff.
- An exploratory scan called every viewer-readable `GET` route as a signed-in viewer against the real (read-only) model, dataset and experiment folders, with temporary storage, and searched the responses for absolute server paths.

### Findings and fixes

| # | Finding | Severity | Fix |
|---|---|---|---|
| F1 | `GET /api/datasets` and `GET /api/datasets/{name}` returned absolute Windows paths to viewers: in the dataset `path` field, and inside README/description text (e.g. source recordings under `C:\SHREE\…`). Found by the scan. | Medium (information disclosure to signed-in non-admins) | New `redact_json()` (authz.py) applies the existing text redactor to every string in these documents for non-admins with login on. Applied to datasets and dataset stats, and defensively to experiments, experiment detail, the evaluations list (`summary`) and evaluation detail, since those are read from training/evaluation files that normally hold absolute paths (no such data existed locally to scan). The UI only tests `readme` for the word "synthetic", which still works. Scan after the fix: 0 path leaks on all viewer routes. |
| F2 | The redactor hid IPv4 addresses but not IPv6 ones (e.g. `fe80::1%eth0`, `[2001:db8::7]:554`). | Low | IPv6 pattern added (full form, or compressed with `::`, with optional zone and port). Times (`12:30:45`), ratios (`16:9`) and ISO timestamps are not affected. Side effect: a code-style token such as `a::b` is also hidden. |
| F3 | No automated test covered the server closing the MJPEG camera stream. | Test gap | New test with an in-memory fake camera worker (no camera, no network): the stream closes on logout and on idle timeout, and a reconnect gets 401. |
| F4 | Frontend activity tracker: after a reload, the first click waited up to a minute before being reported, so a session with seconds left could expire despite the click. | Low (usability) | The first trusted input after the tracker starts is reported at once. The server still records at most every 30 s. |
| F5 | The `routes_auth.py` docstring said "These three are public"; there are four. | Cosmetic | Corrected. |

**Reviewed with no change needed:**
- the origin check;
- role ranking and the 401/403 split;
- admin-token handling;
- the stream guard (read-only re-checks every 30 s, closing on revocation or expiry);
- the activity endpoint (needs a session and an allowed Origin, returns only the deadline);
- `record_activity` (never past the absolute limit);
- trimmed camera, health and model responses for non-admins;
- frame-grab and evaluation permissions;
- the frontend role gating and export flow.

Camera `notify` settings hold only flags and provider names (credentials are in the environment), so they are not sensitive.

### Verified in this pass (all actually run)

- **Server-side stream closure:**
  - SSE: tests for logout, idle timeout, and a watched-but-untouched stream.
  - MJPEG: new tests for logout and idle timeout.
- **Session expiry:**
  - 30 min idle; polling and writes don't extend it; activity extends it.
  - 12 h absolute, even with continuous activity.
  - Activity is refused without a session, without an Origin, with a foreign or `null` Origin, via GET, and for a token without a session.
  - Activity is recorded at most every 30 s, and login off has no endpoint.
- **Authorization and redaction:**
  - every route for each role;
  - free text (events, jobs, evaluations, SSE);
  - dataset and evaluation documents;
  - camera and NVR details;
  - health detail;
  - model paths.
- **Frames and evaluations:**
  - frames that would open a new connection: admin-only with login on (viewer/operator 403 before any connection);
  - evaluations: admin-only with login on;
  - the Cameras ROI editor's fetch sends the session cookie (code reviewed; with login on a non-admin gets a fixed message).

### Results

- **Security suites** (`test_authz`, `test_auth`, `test_local_only`, `test_app_routes`): 131 passed. `test_authz.py` now has 51 tests (4 new in this pass).
- **Full backend suite** (isolated temporary storage): **185 passed, 0 failed, 38 skipped.**
- **Skipped (38), unchanged:**
  - integration 19, scheduler 8, lab_api 5, capture_faults 4, roi_upload 2;
  - "trained model/test images not present" ×37, "no test images" ×1;
  - **not verified**.
- **`tsc --noEmit`:** pass.
- **Production build:** pass (23/23 pages, no invoice route).
- **Viewer path scan:** 0 leaks.
- **Dataset-tools fixture defect** (separate workstream, not modified): `PYTHONHASHSEED=56` still fails, as documented.
- **Integrity:**
  - model sha256 `d1de9cb6…` and live DB `f7c83c2dafd7ea24` unchanged;
  - no real `storage/auth.db`;
  - `local_only.py` unchanged;
  - no invoice code in HEAD or the working tree (only the guard tests mention it);
  - `main` (95c151b) and the snapshot (d0ea84d) unchanged;
  - the flaky-test worktree (`claude/elastic-shtern-f65dfa`, 95c151b) has no changes.
- **Scans and line endings:**
  - no secrets; only synthetic test addresses in tests and docs;
  - consistent line endings;
  - nothing staged, committed or pushed.
- **No browser run in this pass.** The fixes are backend and test changes plus a one-line tracker change, all covered by the automated tests and the TypeScript check and build. The idle-policy browser run is in the fourth pass.

### Remaining risks and limitations

1. **Logout has no Origin check.** Another page on localhost could force a sign-out: a nuisance, with no access gained. Low.
2. **Cookie-holder limitation** of the activity endpoint (fourth pass): inherent, and bounded by the 12 h limit.
3. **The redactor is pattern-based.** A host name that appears in plain text (not in a URL), and is not one of the configured camera hosts or usernames, is not recognised. IPv4, IPv6, URLs and paths are.
4. **Login-off development mode is unchanged by design**: open to processes on this PC, and the local-only safeguard applies.
5. **Carried forward:**
   - no rate limiting beyond login (Step 5);
   - the admin token is still kept in sessionStorage;
   - 38 skipped tests;
   - the dataset-tools fixture defect.

### Readiness

Step 4 is ready for human review and commit. There are no known open security defects within the approved scope; the risks above are low or deferred by decision. The change set is the same 32 files, now including this pass's fixes.

## Outstanding-risk review (sixth pass)

Checked first: branch `security/step1-local-only`, HEAD `85e365c`, 32 changed files (the same set and content as the last audit), nothing staged, flaky-test worktree unchanged. This pass changed 5 of those files (no new files):

- `backend/api/authz.py`
- `backend/api/routes.py`
- `backend/api/routes_lab.py`
- `backend/api/routes_auth.py`
- `tests/test_authz.py`

### 1. Logout (and login) origin validation: fixed

- **Problem:** `POST /api/auth/logout` had no Origin check. `SameSite=Strict` does not separate ports on `localhost`, so any page on this PC (e.g. `http://localhost:5000`) could sign the user out. The same applied to `POST /api/auth/login` (login CSRF: signing the browser into another account).
- **Fix:** both endpoints now refuse a browser request whose `Origin` is not in `CORS_ORIGINS` or the API's own origin (403, with the session untouched and no cookie set). This is the same rule as every other cookie-authenticated write.
  - Requests without an `Origin` header (non-browser clients, tests) are unaffected.
  - The existing frontend is unaffected: its origin must already be in `CORS_ORIGINS`, or the browser would block every API response.
- **Test:** `test_login_and_logout_refuse_other_origins` (foreign and `null` Origin refused for login and logout and the user stays signed in; the allowed Origin works; logging out really ends the session; no Origin works).

### 2. Hostname and path redaction: replaced with an allowlist for messages

- **Problem:** event messages and error fields were cleaned by patterns (URLs, IPs, paths, configured camera hosts). A host name in plain text that was not a configured camera host, or a new message shape, could still reach viewers.
- **Fix (deny by default), in `backend/api/authz.py`, `MessageFilter`:**
  - **Event messages** (`/api/events` and the live SSE stream): a non-admin sees a message only if it matches one of the templates for messages the application itself writes. All `log_event` call sites in `backend/` were catalogued, and the variable parts are limited to identifiers, numbers and enums.
  - Messages that end in raw exception text keep only their safe prefix, e.g. `camera disconnected; retrying (details visible to administrators)` or `capture skipped: auth_failed (details visible to administrators)`.
  - Any other message, including messages added in future, becomes `<source> <warning|error|event> (details visible to administrators)`. The event `data_json` is withheld (the UI does not use it).
  - **Error fields** (camera `last_error`, capture `error`, schedule `last_error`, job and evaluation `error`): non-admins get `error (details visible to administrators)`, or `<code> (details…)` when the text starts with a machine code such as `auth_failed:`.
  - **Admins, and login-off mode,** still see the original text and `data_json`. Stored data is never changed.
  - The camera notice logic in the UI keeps working, because "disconnected", "error" and "reconnected" are preserved.
  - **Documents** read from disk (dataset README/data.yaml, experiment and evaluation files) keep the pattern redactor (fifth pass). Their content is not a fixed set of messages. This is a residual risk (below).
- **Tests:**
  - `test_free_text_hides_hosts_paths_and_users_from_non_admins` (viewer and operator; realistic messages from each writer plus an unknown one);
  - `test_admins_and_login_off_see_the_original_text`;
  - `test_every_event_message_the_application_writes_is_classified` (every catalogued message shape, plus attempts to smuggle a path, URL or IP through a "shown" template);
  - `test_live_event_stream_hides_infrastructure_from_viewers`.

### 3. Rate limiting: not implemented (Step 5), risk documented

- **Today:**
  - Only login is limited: 5 failures per username and per client address in 15 minutes lock that key for 15 minutes.
  - There is no request-rate limit on any other route, including image/video detection uploads (CPU/GPU heavy), the activity endpoint, exports and streams.
- **Why it is acceptable now:** the local-only safeguard rejects every non-loopback client, and login is off by default. Only processes on this PC can reach the API.
- **If the API were exposed to a network with login on, without Step 5:**
  - any signed-in user could flood detection endpoints and starve live monitoring;
  - an attacker could lock out real users by failing logins for their username (username lock-out is a deliberate trade-off);
  - the per-client lock-out can be evaded by changing address.
- **Step 5 should add:** per-user and per-client request rate limits (stricter on detection/upload, login and activity), a cap on concurrent streams per session, and request-size limits at the proxy.

### 4. Admin-token storage: inspected, no change (needs a decision)

- **Current behaviour:**
  - The legacy `ADMIN_API_TOKEN`, when an admin types it into Settings → Security, is kept in `sessionStorage` (`lm.adminToken`): per tab, cleared when the tab closes.
  - It is sent as a `Bearer` header on admin calls.
  - Any script running in the page can read it.
- **XSS surface:**
  - The UI renders data through React (escaped). The only raw HTML is a fixed theme script in `app/layout.tsx`, and there is no `eval`.
  - No Content-Security-Policy is configured.
  - The exposure therefore depends on there being no script-injection bug.
- **Why it can't simply be removed:** with login **off** (the default), the token is the only way to perform admin actions in the UI. Moving it out of browser-readable storage either loses it on every page reload (re-type the token), or needs a server-side exchange for an HttpOnly admin cookie, which requires session storage that login-off mode deliberately does not create. Both change how admins work today.
- **Options for approval:**
  - **(a)** With login **on**: hide the token form and clear any stored token, because admins have accounts. The API still accepts the header for scripts. This is a small UI change with no effect on login-off mode. **Recommended next.**
  - **(b)** With login **off**: keep the token only in memory, at the cost of re-entering it after a reload.
  - **(c)** Exchange the token for a short-lived HttpOnly cookie (a larger change).
  - **(d)** Add a Content-Security-Policy (independent of the above).
- None of these was implemented in this pass.

### 5. HTTPS and network exposure: verified local-only, requirements documented

- **Verified unchanged:**
  - `backend/api/local_only.py` (outermost middleware: any non-loopback client gets 403, whatever `--host` is used);
  - `deploy/start-ldm.ps1`, which refuses `-Lan` and binds `127.0.0.1`;
  - `tests/test_local_only.py` passes (part of the security suites below).
- No LAN or production setting was changed.
- **Required before any network deployment** (all currently missing, by design):
  1. TLS: HTTPS only, via a reverse proxy (or uvicorn with certificates), with HTTP redirected or disabled, and HSTS.
  2. `LM_AUTH_COOKIE_SECURE=true` (or `auto` behind HTTPS), so the session cookie never travels in clear text.
  3. `CORS_ORIGINS` containing only the exact `https://` origin(s) of the UI. The origin checks for writes, login, logout and activity rely on it.
  4. Login on, with admin/operator/viewer accounts; the admin token decision above; a Content-Security-Policy.
  5. Step 5 rate limits and upload/stream caps.
  6. A deliberate, reviewed change to the local-only safeguard (trusted proxy / allowed networks). The proxy must be the only path to the API, and forwarded headers must be trusted only from it.
  7. Backups and protection of `storage/auth.db`, and of `.env` (`ADMIN_API_TOKEN`, CCTV credentials).

### Results (this pass, all actually run)

- **Security suites** (`test_authz`, `test_auth`, `test_local_only`, `test_app_routes`): **133 passed, 0 failed.**
- **Full backend suite** (isolated temporary storage): **187 passed, 0 failed, 38 skipped.**
- **Skipped (38):**
  - integration 19, scheduler 8, lab_api 5, capture_faults 4, roi_upload 2;
  - "trained model/test images not present" ×37, "no test images" ×1;
  - **not verified**.
- **`tsc --noEmit`:** pass.
- **Production build:** pass (23/23 pages, no invoice route). `tsconfig.json` and `next-env.d.ts` were backed up and restored byte-identical.
- **Integrity:**
  - model sha256 `d1de9cb6…` and live DB `f7c83c2dafd7ea24` unchanged;
  - no real `storage/auth.db`;
  - `local_only.py` and `start-ldm.ps1` unchanged;
  - no invoice code;
  - `main` and the snapshot branch unchanged;
  - flaky-test worktree untouched.
- **Scans and staging:**
  - no secrets; only synthetic test addresses in tests and docs;
  - consistent line endings;
  - nothing staged, committed or pushed.
- **Not done in this pass:** a browser run. The changes are server-side and tests only, and the frontend code is unchanged since the last browser and build checks.

### Remaining risks after this pass

1. **Admin token in `sessionStorage`:** decision needed (options a–d above). Low while local-only.
2. **Documents** (dataset, experiment and evaluation files) still use pattern redaction, so an unusual host name in plain text inside them could reach viewers. Low.
3. **No rate limiting beyond login:** Step 5. Acceptable only while local-only.
4. **No HTTPS:** acceptable only while local-only. It is required before any network exposure (list above).
5. **Cookie-holder limitation** (fourth pass): inherent, and bounded by 12 h.
6. **38 skipped tests**, and the dataset-tools fixture defect (separate workstream).

### Recommendation

Step 4 is ready for human review and a **local** commit once you are satisfied with this pass. Two follow-ups need decisions rather than more Step 4 work:
- the admin-token option (recommended: option a, plus a Content-Security-Policy);
- Step 5 rate limiting.

The application must stay local-only until Step 5 and HTTPS are done.

## Admin-token decision and Content-Security-Policy (seventh pass)

**Approved decisions:**
1. With login **on**, hide the legacy admin-token input and clear any stored token.
2. With login **off**, keep the existing admin-token workflow.
3. Add a Content-Security-Policy suited to the actual frontend and backend.

Pre-flight: branch `security/step1-local-only`, HEAD `85e365c`, 32 changed files, nothing staged, worktree unchanged.

### Admin token (`frontend/api/adminToken.ts`, new)

- **One module owns the legacy token**, and `client.ts` re-exports its getter and setter, so every caller goes through it.
- **The login gate reports the server's mode** (from `GET /api/auth/me`) through `applyAuthMode()`:
  - **Login on:** any stored token is deleted from `sessionStorage`, `getAdminToken()` always returns nothing, and `setAdminToken()` refuses to store one. No API request can carry a stale token.
  - **Login off:** unchanged. The token is stored per tab and sent as a Bearer header on admin calls.
  - **Mode not yet known** (first load, or server unreachable): the token is **not** used (default deny). This is so a stale token can never reach a server whose login is on during that window. A stored token is kept, not deleted, until the mode is known.
- **The store** (`stores/app.tsx`) ignores a token entirely when login is on: the role comes from the account only.
- **The Security tab** (`SettingsPage.tsx`) shows no token form and no "no ADMIN_API_TOKEN configured" warning when login is on. It explains: "The legacy admin token is not used while login is on."
- **Server compatibility unchanged:** the API still accepts `ADMIN_API_TOKEN` in headers from scripts, as before.

### Content-Security-Policy

#### Web UI (`proxy.ts` + `frontend/lib/csp.ts`, new; `app/layout.tsx`)

- **Nonce-based, as documented for this Next.js version** (16.3.3, `node_modules/next/dist/docs/01-app/02-guides/content-security-policy.md`):
  - `proxy.ts` (Next 16's middleware) creates a fresh random nonce per request and sets the policy on the request (so Next.js applies the nonce to its own scripts) and on the response;
  - the root layout reads the nonce (`headers()`) and puts it on the one inline script (the theme script).
- **As a result every page is rendered per request** (shown as ƒ in the build), which is required for nonces. This costs a little server time per page view, which is negligible for this local application.
- **Policy (production):**
  ```
  default-src 'self';
  script-src 'self' 'nonce-<random>' 'strict-dynamic';
  style-src 'self' 'nonce-<random>';
  style-src-attr 'unsafe-inline';
  img-src 'self' data: blob: <API origin>;
  media-src 'self' blob: <API origin>;
  connect-src 'self' <API origin>;
  font-src 'self' data:;
  object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'
  ```
  plus `X-Content-Type-Options: nosniff` and `Referrer-Policy: same-origin`.
- **Why each source is there:**
  - It comes from an inventory of what the UI loads. There are no third-party scripts, fonts, iframes or workers, and nothing injects `<style>` at runtime.
  - The API origin (from `NEXT_PUBLIC_API_URL`) is needed for fetch and the live event stream (`connect-src`), snapshots, frames and MJPEG (`img-src`), and processed videos (`media-src`).
  - `data:` is for live frames (base64 JPEG); `blob:` is for local previews and downloads.
- **Documented exceptions:**
  - `style-src-attr 'unsafe-inline'`: 22 React `style={...}` attributes, which cannot carry a nonce. Only *attributes* are allowed; `<style>` elements still need the nonce. The risk is CSS-only (no script).
  - Development only (`next dev`): `'unsafe-eval'` (required by React dev tooling, per the Next.js docs), inline styles (hot reload) and `ws://localhost:*` / `ws://127.0.0.1:*` (hot-reload websocket). None of these is in the production policy.
  - **Not used:** `'unsafe-eval'` and `'unsafe-inline'` scripts in production; wildcard sources; `upgrade-insecure-requests`. The last one would rewrite the API's `http://` requests to `https://` and break the local application; add it together with HTTPS.

#### API (`backend/api/security_headers.py`, new; registered in `backend/main.py`)

- **The API serves no web pages** (JSON, JPEG, MJPEG, SSE, MP4, CSV/ZIP). Every response gets:
  ```
  Content-Security-Policy: default-src 'none'; img-src 'self'; media-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'
  ```
  and `X-Content-Type-Options: nosniff`.
- The middleware is pure ASGI, so streams are not buffered; both stream tests pass with it.
- The optional API docs (`/docs`, `/redoc`, `/openapi.json`, off by default) are exempt when enabled, because Swagger/ReDoc load from a CDN.
- The local-only middleware stays outermost and is unchanged.

### Tests (new)

- **`frontend/tests/adminToken.test.mjs` (6 tests):**
  - default deny before the mode is known;
  - login off: store, read and clear, and an earlier token kept;
  - login on: a stored token deleted, never returned, and cannot be stored;
  - switching back to off does not restore it;
  - no window or storage never throws.
- **`frontend/tests/csp.test.mjs` (5 tests):**
  - production policy: nonce plus strict-dynamic; no eval, inline scripts or wildcards; `object-src`, `frame-ancestors`, `base-uri` and `form-action` restricted; no `upgrade-insecure-requests`;
  - the API origin allowed only in connect, img and media;
  - dev-only additions;
  - a fresh nonce changes the policy;
  - API origin parsing.
- **Running them:** `npm run test:frontend` (new script in `package.json`), using Node's built-in test runner. Node strips the TypeScript types when importing; there are **no new dependencies**.
- **`tests/test_authz.py::test_api_responses_carry_a_content_security_policy`:**
  - policy and nosniff on API responses, including 401 and 404;
  - the docs pages exempt when enabled.

### Browser verification (production build served by `next start`, isolated backend, synthetic token and accounts, all removed afterwards)

- **Real response headers:**
  - the UI page carries the policy, with a different nonce on every request;
  - all 13 `<script>` tags carry that request's nonce, including the theme script;
  - there are no inline `<style>` tags;
  - API responses carry the API policy.
- **The application works under the policy, with no violations:**
  - the dashboard renders with styles, and API data loads;
  - the theme script ran (a saved "light" theme was applied before paint);
  - the live event stream connected, blob images load, and pages with inline style attributes render;
  - the console showed no policy violations.
- **The policy is enforced:**
  - an injected `<img onerror=…>` handler was **blocked** (`script-src-attr`);
  - a `fetch` to an outside host was **blocked** (`connect-src`).
  - (Code run from the browser's developer tooling is exempt from the policy by design, so that was not used as a test.)
- **Login off:** the token form was visible. The synthetic token was accepted, the chip showed "Admin", and the token was stored (the workflow is unchanged).
- **Switch to login on** (same tab, stale token still stored):
  - on reload the stored token was deleted before sign-in;
  - after signing in as viewer the chip showed "Viewer", the token form and warning were hidden, and the "legacy admin token is not used" note was shown.
  - **Evidence that no header was sent:** the network log shows no CORS preflight except for login itself. A cross-site request carrying an `Authorization` header always needs a preflight.
- No token, password or session cookie appeared in the backend or frontend logs.

### Results (all actually run)

- **Security suites** (`test_authz`, `test_auth`, `test_local_only`, `test_app_routes`): **134 passed, 0 failed.**
- **Full backend suite** (isolated temporary storage): **188 passed, 0 failed, 38 skipped.**
- **Skipped (38):**
  - integration 19, scheduler 8, lab_api 5, capture_faults 4, roi_upload 2;
  - "trained model/test images not present" ×37, "no test images" ×1;
  - **not verified**.
- **Frontend unit tests:** **11 passed, 0 failed.**
- **`tsc --noEmit`:** pass.
- **Production build:** pass, run twice (with the test API URL and with the default). 23/23 pages, all dynamic, proxy active, no invoice route. `tsconfig.json` and `next-env.d.ts` were backed up and restored byte-identical each time.
- **Integrity:**
  - model sha256 `d1de9cb6…`, live DB `f7c83c2dafd7ea24` and detection settings (0 rows) unchanged;
  - no real `storage/auth.db`;
  - `local_only.py` and `start-ldm.ps1` unchanged;
  - previous commits unchanged (HEAD `85e365c`);
  - `main` and the snapshot branch unchanged;
  - worktree untouched.
- **Scans and staging:**
  - no secrets (only synthetic test values); no invoice code;
  - line endings consistent (this report was normalised);
  - nothing staged, committed or pushed.

### Changed files

- **This pass (15 files, all uncommitted):**
  - modified: `app/layout.tsx`, `backend/main.py`, `frontend/api/client.ts`, `frontend/features/auth/AuthGate.tsx`, `frontend/features/system/SettingsPage.tsx`, `frontend/stores/app.tsx`, `package.json`;
  - new: `proxy.ts`, `frontend/lib/csp.ts`, `frontend/api/adminToken.ts`, `backend/api/security_headers.py`, `frontend/tests/adminToken.test.mjs`, `frontend/tests/csp.test.mjs`;
  - updated: `tests/test_authz.py`, `docs/security/STEP4_REPORT.md`.
- **Whole Step 4 working tree: 40 files.** That is the 32 from before plus `app/layout.tsx`, `package.json` and the 6 new files listed above.

### Residual risks

1. **`style-src-attr 'unsafe-inline'`:** injected inline *style attributes* would apply (CSS only, no script). Low. Removing it would mean replacing 22 React style attributes.
2. **The API docs pages**, when explicitly enabled, have no CSP (CDN assets). They are off by default and local-only.
3. **Dynamic rendering of every page:** a small per-request cost, and no static caching. Acceptable for a local application.
4. **Login-off mode still keeps the admin token in `sessionStorage`** (approved decision 2). It is now protected by the CSP (inline-handler XSS blocked).
5. **Carried forward:**
   - no rate limiting beyond login (Step 5);
   - no HTTPS (required before any network use; add `upgrade-insecure-requests` and the secure cookie then);
   - pattern-based redaction of dataset, experiment and evaluation documents;
   - the cookie-holder limitation;
   - 38 skipped tests;
   - the dataset-tools fixture defect.
