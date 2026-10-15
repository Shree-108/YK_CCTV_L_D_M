# Setup (one time per PC)

| Component | Version tested | Notes |
|---|---|---|
| Windows | 11 (PowerShell 5.1) | the start/stop scripts are Windows-only |
| Python | 3.14.8 (3.10+ required) | found via `.venv\Scripts\python.exe`, then the `py` launcher, then `python`; or pass `-Python <path>` |
| Python packages | `requirements.txt` (Ultralytics 8.4.173, torch 2.14.1 CPU, FastAPI 0.142, OpenCV 5.0) | `py -m pip install -r requirements.txt` |
| Node.js | 24.21.0 (20.9+ required by Next.js 16) | |
| JS packages | `package.json` (Next.js 16.3.3, React 19) | `npm install` |
| Model | `models\litter_yolo\v3_store_cctv\best.pt` (sha256 `d1de9cb6…862bc6`) | in git; backup in `models\backups\v3_store_cctv_20261012\` |
| Hardware | 4 GB RAM free recommended, no GPU needed for the demo | detection runs on the CPU |

```powershell
cd C:\SHREE\YATRIKART\PROJECTS\YK_CCTV_L_D_M_main\YK_CCTV_L_D_M
py -m pip install -r requirements.txt
npm install
copy .env.example .env          # then edit .env: set ADMIN_API_TOKEN to a long random value
powershell -ExecutionPolicy Bypass -File deploy\start-ldm.ps1
```

## Configuration (`.env`, never committed)

`.env.example` lists every variable with no real values. The ones that matter for the demo:

| Variable | Purpose |
|---|---|
| `ADMIN_API_TOKEN` | enables admin functions; empty = disabled. Generate a random value; never commit it |
| `CORS_ORIGINS` | every address the UI is opened from (scheme://host:port). `start-ldm` adds its own UI port automatically |
| `LM_MODELS_DIR`, `LM_STORAGE_DIR`, `LM_DB_PATH` | optional: other folders for model weights / data / database (relative paths = relative to the project) |
| `LM_START_CAMERAS` | `start-ldm` forces `false` unless `-EnableCameras` is given |
| `CCTV_*`, `SMTP_*`, `ALERT_WEBHOOK_URL`, `TELEGRAM_*` | leave empty for the demo: no camera access, no notifications |

`NEXT_PUBLIC_API_URL` (the API address baked into the UI) is set by `start-ldm` itself. Do not put it in `.env`.

## start-ldm.ps1 options

| Option | Default | Meaning |
|---|---|---|
| `-FrontendPort` | 3010 | web UI port |
| `-BackendPort` | 8000 | API port |
| `-Dev` | off | `next dev` (hot reload) instead of an optimised production build |
| `-Rebuild` | off | force a fresh UI build (normally automatic when sources or the API address change) |
| `-Lan` | off | **disabled until login protection exists**: the script refuses to start, and the API rejects every non-local request. Later (with login, HTTPS and IT approval) it will make the demo reachable from other PCs |
| `-EnableCameras` | off | keep `LM_START_CAMERAS` from `.env` |
| `-Python` | auto | explicit python.exe |

The production UI build goes to `.next-demo\`, separate from `.next\`, so a developer's `npm run dev` keeps working
side by side. Logs are in `deploy\logs\`, and process records in `deploy\run\` (both git-ignored).

Docker is not used: the project has no container setup, and a single-PC Windows demo does not need one.
