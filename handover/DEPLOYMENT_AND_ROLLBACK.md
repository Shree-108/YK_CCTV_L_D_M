# Deployment and rollback: LDM

## 1. Target environment (inspected 2026-10-12)

The only environment available and inspected is this Windows 11 laptop: Intel 2-core/4-thread CPU, 7.8 GB RAM
(about 0.7 GB free with the app and a browser open), no GPU, Python 3.14.8, Node.js 24.21.0. **No company server,
VM or hosting environment has been provided**, so the supported deployment is **local, single PC**. There is no
Docker setup in the project, and none is needed for a single-PC demo.

| Item | Value |
|---|---|
| Web UI | `http://localhost:3010` (Next.js 16 production build in `.next-demo\`) |
| API | `http://localhost:8000`, health check `GET /api/health` (status, model, cameras, notification providers) |
| Bind address | 127.0.0.1: reachable from this PC only |
| Model | `models\litter_yolo\v3_store_cctv\best.pt` (production, active). **No other model is approved** |
| Data | `storage\` (database `litter_monitor.db`, alerts, captures, videos, ingest). Override with `LM_STORAGE_DIR`, `LM_DB_PATH` |
| Models folder | `models\`, or `LM_MODELS_DIR` |
| Logs | `deploy\logs\` (API: `backend.err.log`; UI: `frontend.log`; build: `frontend.log.build`); `storage\ingest\ingest.log` |
| Secrets | `.env` only (git-ignored); template `.env.example` |

## 2. Deploy (same or another Windows PC)

1. Copy the project folder. Do **not** copy `.env` between PCs by e-mail or chat: create it from `.env.example`.
2. Follow `deploy\SETUP.md`: `py -m pip install -r requirements.txt`, `npm install`, `.env` with a new random
   `ADMIN_API_TOKEN`.
3. Check that the model checksum is `d1de9cb6ca7a8fe4348bd9a8c81bc3fdc5d0699965c263639b494486e2862bc6`:

```powershell
Get-FileHash models\litter_yolo\v3_store_cctv\best.pt
```

4. Start the app:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\start-ldm.ps1
```

5. Open `http://localhost:3010` and check that the sidebar badge is green.
6. Stop it with `deploy\stop-ldm.cmd`.

### Access from other PCs (manager / QA)

`localhost` works only on the PC running the app. Options, in order of preference:

1. **Approved internal server or VM** (IT): run the same steps there. Put it behind the company's authenticated
   reverse proxy / SSO with HTTPS. Set `CORS_ORIGINS` to the exact UI URL, and set `NEXT_PUBLIC_API_URL` to the API
   URL as seen by browsers (`start-ldm` does this for `-Lan`).
2. ~~Temporary LAN demo with `start-ldm.ps1 -Lan`~~: **disabled** since API security Step 1 (2026-10-15). There is
   no user login yet (defect D-08), so `-Lan` refuses to start and the API rejects every non-local request. It
   becomes available again only after login protection (security plan Steps 2–4), HTTPS and IT approval.

Never expose the app or the NVR to the internet, and never port-forward RTSP (554) or the NVR web UI.

## 3. Backup

```powershell
powershell -ExecutionPolicy Bypass -File deploy\backup-ldm.ps1 -Destination D:\LDM_backups
```

The script is safe while the app runs. It copies:

- the database (SQLite online backup);
- `storage\alerts`, `captures` and `ingest`;
- all model versions;
- a `SHA256SUMS.txt` file of checksums.

Add `-IncludeVideos` for annotated videos. Add `-IncludeEnv` only for a protected destination, because `.env`
holds the admin token. Tested: the backup database passes `integrity_check`, and the model checksum matches.

Recommended: before every demo or pilot change, and weekly during a pilot.

## 4. Rollback

| What | How |
|---|---|
| **Model** (if a future candidate was activated) | Model Lab → activate `v3_store_cctv` (admin), or `POST /api/models/activate {"version":"v3_store_cctv"}` with the admin token. A failed load keeps the previous model. Check that `GET /api/health` shows `v3_store_cctv` |
| **Model file damaged or missing** | `Copy-Item models\backups\v3_store_cctv_20261012\best.pt models\litter_yolo\v3_store_cctv\best.pt`, then check the checksum. Alternatively `git checkout 574c7e1 -- models/litter_yolo/v3_store_cctv/best.pt` |
| **Thresholds changed by a model's `recommended_settings`** (D-09) | Settings → Detection → set weak/candidate/confirmed back to 0.30 / 0.45 / 0.60 (production defaults), or restore the database from a backup |
| **Data** | `stop-ldm`; copy `db\litter_monitor.db` from a backup to `storage\litter_monitor.db` (and `storage\alerts\…` if needed); `start-ldm` |
| **Code / config** | `git` (all changes since commit `95c151b` are uncommitted at hand-over). Previous versions of the config files changed in Phase 8 are in `deploy\backups\2026-10-12\` |
| **Demo build** | `start-ldm.ps1 -Rebuild`, or delete `.next-demo\` |

## 5. Health and monitoring

- `GET /api/health` reports `status`, `model.version`, `model_error`, cameras and notification providers.
  `start-ldm` waits for it and fails loudly if it does not answer.
- The UI badge turns red ("Backend offline", with the API address) within about 20 s of the API stopping.
- Video jobs: status and error in Video Analysis / `GET /api/jobs`.
- Ingest: exit code 0 = ok, 1 = some recordings failed, 3 = API unreachable or no model (recordings stay pending).
  Details in `storage\ingest\ingest.log` and `ingest.db`.
