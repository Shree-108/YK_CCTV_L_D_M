# Troubleshooting

Logs: `deploy\logs\backend.err.log` (the API writes its log here), `backend.log`, `frontend.log`,
`frontend.log.build` (UI build). The previous run's logs are kept as `*.prev`.

## "Backend offline" in the sidebar

Hover over the badge: it shows the API address the UI is calling and the error.

| Cause | Fix |
|---|---|
| **The UI's address is not allowed by the API (CORS).** This was the original problem: the UI ran on `:3010` but `.env` allowed only `http://localhost:3000`, so the browser blocked every request although the API was running. | Start with `deploy\start-ldm.ps1`, which adds the UI port automatically and refuses to continue if a running API would block it. If you start things by hand, add the exact address (e.g. `http://localhost:3010`) to `CORS_ORIGINS` in `.env` and restart the API. |
| The API is not running or is still loading the model (up to ~1 min) | wait: the UI re-checks every 15 s; otherwise run `start-ldm` again |
| The UI was opened from another PC | not supported yet: the API answers only requests from this PC ("Remote access … disabled", HTTP 403) until login protection is added. Open the demo on this PC |
| `/docs` returns 404 | API docs are off by default. For local development set `LM_ENABLE_API_DOCS=true` in `.env` and restart the API |
| The API is hung | the health check times out after 8 s and shows offline. Restart: `stop-ldm`, then `start-ldm` |

Quick checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health        # API alive? which model?
(Invoke-WebRequest http://127.0.0.1:8000/api/health -Headers @{Origin='http://localhost:3010'} -UseBasicParsing).Headers['Access-Control-Allow-Origin']   # must print http://localhost:3010
```

## start-ldm messages

| Message | Meaning / fix |
|---|---|
| `Python not found` / `Missing Python packages: …` | install Python 3.10+ / run the printed `pip install -r requirements.txt` |
| `node_modules missing` | `npm install` in the project folder |
| `No model weights found in …\litter_yolo\<version>\best.pt` | restore `models\litter_yolo\v3_store_cctv\` from git or `models\backups\v3_store_cctv_20261012\` (check with `sha256sum -c SHA256SUMS`), or fix `LM_MODELS_DIR` |
| `Port N is used by another program` | another app holds the port: close it or pick another port with `-BackendPort` / `-FrontendPort` |
| `The running backend does not allow the UI address …` | an API started outside the script (with a different `CORS_ORIGINS`) holds port 8000. Stop it (Ctrl+C in its terminal) and run `start-ldm` again |
| `Backend is up but NO MODEL is loaded` | see `model_error` in the message and `backend.err.log`. Image/video detection answers 503 until fixed |
| `UI build failed` | read `frontend.log.build`. Usually a TypeScript error after a code change |
| Script window closes immediately | run the `.ps1` from a PowerShell window to read the message |

## Other

- **Annotated video does not play:** the API writes H.264 when possible and falls back to MPEG-4 Part 2. A
  "Failed to load OpenH264 library" line in `backend.err.log` is harmless if the video plays.
- **Slow detection:** CPU-only laptop. Close other heavy programs, since about 4 GB of free RAM is recommended.
- **Port 3000:** a developer's `npm run dev` may already use it. The demo uses 3010 on purpose.
