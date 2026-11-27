# Yatrikart Litter Detection: demo guide (manager / QA)

## Start

Double-click **`deploy\start-ldm.cmd`**, or in PowerShell from the project folder:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\start-ldm.ps1
```

Wait for **`Open: http://localhost:3010/ldm`**. The first start builds the web UI, which takes 1–3 minutes. Later
starts take seconds to about a minute (loading the AI model). Then open **http://localhost:3010/ldm** in Chrome or
Edge and sign in. Create the admin account once (password asked at a hidden prompt):
`py -m backend.auth add-user admin --role admin`.

The sidebar must show a green **"Connected to backend"**. A red "Backend offline" badge means the UI cannot reach
the API; hover over it for the exact reason, or see `TROUBLESHOOTING.md`.

## Stop

Double-click **`deploy\stop-ldm.cmd`**, or:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\stop-ldm.ps1
```

This stops only what `start-ldm` started. Running `start-ldm` twice is safe: it re-uses the running demo and never
starts a second copy.

## What to demonstrate

| Page | What to show |
|---|---|
| Overview | detections over the last 14 days, top litter types, recent alerts, the active model (`v3_store_cctv`) |
| Image Analysis | upload a CCTV still, get boxes with class and confidence, optionally draw an ROI (area to watch) |
| Video Analysis | upload a short clip, which is processed in the background; then play the annotated video with timestamps |
| Alerts | confirmed litter events with snapshots; acknowledge / resolve / mark false positive |
| Analytics | breakdown by class, hour and day, confidence histogram |

Good demo inputs: short clips from `C:\SHREE\Screen Captures\cctv_litter_poc_3min_h264.mp4`, or frames in
`dataset\ldm_v4_workspace\annotation_batches\real_priority1\images\`.

## What QA should know

- **The model is the current production model `v3_store_cctv`.** No fine-tuned model exists yet, and the dataset
  work for one is still in progress (`docs/LDM_FINETUNE_PLAN.md`). Its known weaknesses (bottles/products and
  vehicles mistaken for litter, small distant litter missed) are the reason for that work. Please record such cases
  rather than treating them as bugs in the app.
- **Uploads in the demo are real data.** Each analysis adds detections and alerts to `storage\` (the app's local
  database). Use the Alerts page to mark test alerts resolved or false positive.
- **No messages are sent.** No email, webhook or Telegram provider is configured.
- **No CCTV cameras are connected.** `start-ldm` starts the API with `LM_START_CAMERAS=false`.
- **This PC by default:** `-Lan` makes the demo reachable from other PCs on the network; everyone signs in.
- **Admin functions** (cameras, settings, model activation, saving labels) need an account with the admin role.
- Typical speed on this laptop (CPU): about 0.13–0.14 s per frame for detection, and about 4 s to process a
  10-second clip.

Backup (safe while running): `powershell -ExecutionPolicy Bypass -File deploy\backup-ldm.ps1 -Destination D:\LDM_backups`.
Process KVMS exports automatically: `py -m ingest.export_folder --export-dir <folder>` (see `docs\CCTV_RECORDING_RETRIEVAL.md`).
Handover package and UAT results: `handover\README.md`.

Files here: `SETUP.md` (installation), `TROUBLESHOOTING.md`, `TEST_RESULTS.md` (what was tested on 2026-10-12),
`logs\` (server logs of the last run).
