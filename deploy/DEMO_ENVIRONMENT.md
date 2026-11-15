# Separate demo environment

Use this instance for demonstrations and test uploads. Its data never mixes with the normal instance.

```powershell
powershell -ExecutionPolicy Bypass -File deploy\start-ldm-demo.ps1      # API :8100, UI :3110 -> open http://localhost:3110
powershell -ExecutionPolicy Bypass -File deploy\stop-ldm-demo.ps1       # stops it and checks production is unchanged
```

Options: `-DemoRoot <folder outside the project>` (default: `..\LDM_demo_data`, next to the project folder),
`-BackendPort`, `-FrontendPort`, `-Rebuild`.

## What is separated, and how

The launcher uses only settings the application already supports (`backend/core/config.py`). Nothing in the
application code was changed.

| Item | Normal instance | Demo instance |
|---|---|---|
| Database | `storage/litter_monitor.db` | `<DemoRoot>\storage\litter_monitor_demo.db` (`LM_DB_PATH`) |
| Uploads, alerts, snapshots, job outputs | `storage/` | `<DemoRoot>\storage\` (`LM_STORAGE_DIR`) |
| Login database | `storage/auth.db` | `<DemoRoot>\storage\auth_demo.db` (`LM_AUTH_DB_PATH`) |
| Ground-truth labels / datasets | `dataset/` | `<DemoRoot>\dataset\` (`LM_DATASET_DIR`, empty) |
| Model weights | `models/` | `<DemoRoot>\models\` (`LM_MODELS_DIR`): a copy of each model, SHA-256 checked against the original |
| `.env` (SMTP, webhook, Telegram, CCTV credentials, admin token) | read | **not read** (`PYTHON_DOTENV_DISABLED=1`; the launcher refuses an old python-dotenv that ignores it) |
| Notifications | as configured | none; the launcher stops if `/api/health` reports any provider configured |
| Admin token | `ADMIN_API_TOKEN` from `.env` | a random demo token in `<DemoRoot>\demo_admin_token.txt` |
| Cameras | `.env` | off (`LM_START_CAMERAS=false`); CCTV credentials removed |
| Ports | 8000 / 3010 | 8100 / 3110 |
| PID files, logs, UI build | `deploy\run\`, `deploy\logs\`, `.next-demo\` | `deploy\run\demo\`, `deploy\logs\demo\`, `deploy\run\demo\next-ui\` (all git-ignored) |

## Proof of separation

At start-up the launcher fingerprints the production `storage/`, `dataset/` and `models/` folders and `.env`
(`deploy/check_demo_isolation.py`). The fingerprint records file sizes and times, plus SHA-256 hashes of the
databases and model weights. At stop, `stop-ldm-demo.ps1` fingerprints them again and compares. Any difference
is listed and the script exits with code 1.

Verified on 2026-11-13, from an isolated copy of the working tree:

- browser walkthrough with 1 image and 1 video upload, 6 alerts and CSV/MP4 downloads;
- production unchanged;
- the demo database held 1 job, 6 alerts and 1,351 detections.

## Not isolated: do not use these in the demo

- **Live CCTV page → saving a camera password.** The backend writes it to the project `.env` (`backend/cameras/cctv.py`).
  The stop check would report it as `.env: CHANGED`.
- **Experiments page.** It reads training runs from the project's `runs/` folders. This is read-only, so it is harmless.

## Building the UI

The first start builds the UI into `deploy\run\demo\next-ui`, which takes a few minutes. `next build` rewrites
`tsconfig.json` and `next-env.d.ts`; the launcher saves both files first and restores them byte-for-byte, as
`start-ldm.ps1` does.
