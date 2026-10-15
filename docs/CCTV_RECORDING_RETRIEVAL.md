# CCTV recording retrieval: status and export-folder pipeline

NVR: CP PLUS **CP-UNR-104F1** (4-channel). Viewing and export software: **KVMS Pro Lite**. Status 2026-10-12.

## 1. What is verified

| Capability | Status | Evidence |
|---|---|---|
| Live camera stream (RTSP) | **Not verified on this NVR.** Code exists (`backend/cameras/cctv.py`, Live CCTV page) and is tested only against a video file | `.env` has no `CCTV_HOST` / user / password; 0 cameras configured; `docs/LIVE_CCTV_SETUP.md` says "not tested against the real CP Plus NVR". The RTSP path `/cam/realmonitor?...` is a common CP Plus/Dahua pattern, **unconfirmed** for this model |
| Search recorded footage by camera and time | **Not verified** | Only retailer listings claim "ONVIF Profile T/S/G" (Profile G = recording search/replay). No official CP PLUS datasheet or manual found online. Nothing was probed on the device |
| Download/export stored recordings | **Manual only, via KVMS Pro Lite** (done by the user today) | user statement. No sample export was available, so the KVMS file format and naming are unknown |
| Automatic retrieval with Python | **Not supported / not verified.** Not implemented | no confirmed interface, credentials or network path. The NVR is identified by its serial/PSN in the CP Plus app, which suggests cloud/P2P access: this PC may not be on the NVR's LAN at all |

No undocumented endpoints were tried, and no NVR settings were changed. The NVR was not contacted.

## 2. What was implemented instead: an export-folder pipeline (verified method)

`ingest/export_folder.py`: you keep exporting from KVMS into one folder, and everything after that is automatic.

- **Input:** any video in the export folder (sub-folders allowed). One sub-folder per camera (`ch1\`, `ch2\`) gives
  the camera id.
- **Waits** until KVMS has finished writing the file.
- **Each recording is processed once,** identified by its SHA-256, so renaming or copying it does not reprocess it.
  `--reprocess` forces it.
- **Uses the running app** (`POST /api/detect/video`): the same model, thresholds, tracking and alerts as the UI,
  and the results appear in the app (Video Analysis history, Alerts).
- **Output:** `storage/ingest/results/<date>_<camera>_<sha8>/`, containing `summary.json`, `detections.csv` and
  `events.csv` (offset in the video and, when the start time is known, wall-clock time) and snapshots of confirmed
  litter. The annotated video is in `storage/videos/<job_id>/`. Ledger: `storage/ingest/ingest.db`. Log:
  `storage/ingest/ingest.log`.
- **Failures are visible:** the API being down leaves the recording pending for the next run (exit code 3). Corrupt
  files, unsupported containers (`.dav`), files above the 500 MB upload limit and recordings longer than the 3600 s
  analysis limit are reported in the log and the ledger. The export folder is only read.
- **No credentials are involved;** the NVR is not contacted.

### Run it

```powershell
powershell -ExecutionPolicy Bypass -File deploy\start-ldm.ps1                 # the app must be running
py -m ingest.export_folder --export-dir "D:\KVMS_Export" --dry-run           # check what would happen
py -m ingest.export_folder --export-dir "D:\KVMS_Export"                     # process new recordings once
```

If the file names contain the start time, pass a pattern. This example is for names like `ch1_20260925154100.mp4`;
**the real KVMS naming must be checked on a sample export first**:

```powershell
py -m ingest.export_folder --export-dir "D:\KVMS_Export" --name-pattern "ch\d+_(?P<start>\d{14})" --time-format "%Y%m%d%H%M%S"
```

Without a pattern, `--start 2026-09-25T15:41:00` sets the start for a single file. Otherwise the start is reported
as *unknown*, and only offsets inside the video are given. The file modification time is stored, but is never
presented as the recording time.

`--watch 300` keeps checking every 5 minutes. Do not leave it running unattended until the pilot is approved.

Settings can also be given as environment variables in `.env`: `LDM_EXPORT_DIR`, `LDM_API_URL`, `LDM_INGEST_DIR`.

### In KVMS Pro Lite

Export as **MP4 (or AVI)**, not the native `.dav`: the app does not accept `.dav`. Export clips of **≤ 1 hour and
≤ 500 MB**, and choose a file-name template that includes the **channel and start time**.

## 3. Tests (2026-10-12, isolated backend on port 8765 with temporary storage; evidence in `handover/evidence/phase9_export_ingest/`)

| Test | Result |
|---|---|
| API not running | ✅ "LDM API not reachable … nothing was changed", exit 3 |
| Dry run | ✅ listed 3 to process, 1 corrupt, 1 `.dav`. After a fix, a dry run now writes nothing to the ledger (the first version did: bug found and fixed) |
| Real run: 2 clips (10 s each, `ch1`, `ch2`) | ✅ processed. Camera taken from the sub-folder, start time from the file name. `events.csv` wall time = start + offset (15:42:00 + 1 s = 15:42:01). Snapshots saved |
| Renamed copy of a processed clip | ✅ "already processed", not uploaded |
| Corrupt `.mp4` | ✅ reported "cannot open the video …", run exit 1 |
| `.dav` file | ✅ reported "export as MP4/AVI", not uploaded |
| Second run | ✅ 0 uploads (3 duplicates, 2 given up) |
| API killed right after a 60 s clip was submitted | ✅ polling retried, then "became unreachable … stays pending", exit 3 |
| API restarted, run again | ✅ the pending clip was processed (6 events, 31 s); nothing else repeated |
| Logs scanned for "password / token / rtsp://" | ✅ 0 matches |
| `tests/test_ingest_export.py` (fake API) | ✅ 5 passed |

The detections in these tests are **unreviewed model output** on footage the model was probably trained on. They
show that the pipeline works, not how accurate the model is.

Not tested: a real KVMS export (none available), `--watch`, files above 500 MB or longer than 1 hour.

## 4. Questions for CP PLUS technical support / the authorised installer

1. Does the CP-UNR-104F1 (our firmware version: ____) officially support **ONVIF Profile G** (recording search and
   replay)? Please send the official datasheet / ONVIF conformance declaration.
2. Is there an official **HTTP API or SDK** for searching and downloading recordings by channel and time? Which
   document, which firmware, and is its use permitted?
3. Is **RTSP playback** of recordings supported (URL format with channel and start/end time)? Is the live RTSP path
   `/cam/realmonitor?channel=N&subtype=M` correct for this model?
4. Can we get a **view-only / playback-only user** that can search and export recordings but cannot change settings?
5. Is the NVR reachable on the office **LAN or VPN** from the AI PC (IP address, ports 554/80/ONVIF port), or only
   through the CP Plus cloud/P2P service (device serial/PSN)? We will not port-forward the NVR.
6. In **KVMS Pro Lite**: which export formats are available (MP4/AVI/DAV)? What is the exported file-name template
   (does it contain channel and start time)? Can exports go automatically to a fixed folder, or can scheduled backup
   be configured?
7. Are export timestamps local time? Which time zone, and is the NVR clock synchronised (NTP)?
8. Any limits on concurrent playback/RTSP sessions, so that automated retrieval does not disturb recording or viewing?

Once 1–5 are answered in writing (and authorised), a `RecordingSource` adapter can be added that downloads
recordings into an export folder of its own, and this same pipeline processes them.
