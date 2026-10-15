# Test results: demo packaging (2026-10-12)

Machine: Windows 11, Intel 2-core/4-thread CPU, 7.8 GB RAM, no GPU, Python 3.14.8, Node.js 24.21.0.
Model: `v3_store_cctv` (production; sha256 `d1de9cb6…862bc6`, unchanged before and after testing).
Everything below was **executed**. Items not executed are listed at the end.

## Root cause of "Backend offline" (UI on :3010)

| Request origin | Before (backend started with `.env` `CORS_ORIGINS=http://localhost:3000`) | After (`start-ldm`) |
|---|---|---|
| `http://localhost:3010` GET /api/health | no `Access-Control-Allow-Origin`, so the browser blocks it | allowed (`http://localhost:3010`) |
| `http://127.0.0.1:3010` | blocked | allowed |
| `http://localhost:3010` preflight POST /api/detect/image | **HTTP 400** | **HTTP 200** |
| `http://evil.example` (unknown site) | blocked | still blocked |

## Startup script

| Test | Result |
|---|---|
| Stale backend on :8000 that blocks :3010 (the original bug) | ✅ detected: `[FAIL] The running backend does not allow the UI address http://localhost:3010 (CORS)…`, exit 1, no UI started |
| Clean start (backend + first UI build + UI) | ✅ exit 0 in 83 s; health OK, model v3 on CPU, 10 classes, CORS verified, UI 200 |
| Start with output captured (`$o = start-ldm.ps1`) | ✅ returned in 46 s (after the fix: children no longer inherit the caller's console handles; before the fix this case hung) |
| Run twice | ✅ both servers re-used, node+python process count unchanged (6 / 6) |
| Backend stopped while the UI is open, then `start-ldm` again | ✅ only the backend restarted, UI re-used; the open page recovered to "Connected" without a reload |
| `deploy\start-ldm.cmd` wrapper | ✅ exit 0 (re-use path) |
| Wrong `-Python` path | ✅ `[FAIL] Python not found…`, exit 1 |
| Backend port held by another program (`-BackendPort 3000`, a Next dev server) | ✅ `[FAIL] Port 3000 is used by another program: node (PID …)`, exit 1 |
| Same port for UI and API | ✅ `[FAIL] -BackendPort and -FrontendPort must be different`, exit 1 |
| Missing model weights (`LM_MODELS_DIR` → empty folder) | ✅ `[FAIL] No model weights found in …\litter_yolo\<version>\best.pt…`, exit 1 |
| `stop-ldm` | ✅ stopped exactly the 2 recorded process trees; ports 8000/3010 free, 0 leftover uvicorn/next processes; a developer `npm run dev` on :3000 left running |
| UI build does not disturb a running `next dev` | ✅ build goes to `.next-demo`; the dev server on :3000 kept answering; `tsconfig.json` / `next-env.d.ts` restored after the build (`git diff` empty) |
| Cameras not started | ✅ health shows `cameras: configured 0, running 0` and the log line `LM_START_CAMERAS=false for this run` |

## UI connection status (browser, http://localhost:3010)

| Test | Result |
|---|---|
| Backend up | ✅ sidebar "Connected to backend" (green), dashboard data loaded |
| Backend killed | ✅ within ~20 s: sidebar and top bar "Backend offline" (red); tooltip "Backend not reachable at http://localhost:8000…" |
| Backend restarted | ✅ back to "Connected to backend" within ~18 s, no reload |
| Health request timeout | implemented (8 s); not separately tested with a hung server |

## Application (isolated copy of storage; production model, real 640x368 store frames and a 10 s clip)

18/18 checks passed: health; model = v3; no notification provider configured; classes; image upload, which returns
class/confidence/bbox and an annotated image (5.4 s first request, including model warm-up); corrupt image → 400;
`.txt` → 415; video upload → 202 → job done in 4 s, timestamped detections, annotated H.264 video served; activating
an unknown model → 404 with v3 still active; activation without admin → 401; API without model weights starts and
reports it; detection then → 503 with a clear message.

Regression: `pytest` **43 passed, 38 skipped** (the skipped tests need the old v1 model and its images, which are
missing, as before this work). `tsc --noEmit`: 0 errors. `next build`: succeeded.

## Not executed

- `-Lan` mode (needs IT approval; would expose the demo to the local network)
- `-Dev` mode
- `-EnableCameras`
- Image/video upload **through the browser UI** of the demo. It works through the same API in the isolated test, but
  was not clicked through, to avoid adding test alerts to the live database.
- First-time setup on a second, clean PC
