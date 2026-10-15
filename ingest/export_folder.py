"""Process CCTV recordings exported from KVMS Pro Lite (or any authorised export) with the running LDM API.

Why an export folder: automatic retrieval of recorded footage from the CP-UNR-104F1 NVR is NOT verified (no
official CP PLUS documentation; RTSP/ONVIF details unconfirmed; see docs/CCTV_RECORDING_RETRIEVAL.md). Exporting
stays a manual step in KVMS; everything after it is automated here.

    python -m ingest.export_folder --export-dir "D:/KVMS_Export" --dry-run     # list what would be processed
    python -m ingest.export_folder --export-dir "D:/KVMS_Export"               # process new recordings once
    python -m ingest.export_folder --export-dir "D:/KVMS_Export" --watch 300   # check every 5 min (unattended)

For each video file in the export folder (recursive; .mp4 .avi .mkv .mov .m4v .webm .dav):
  1. waits until the file is completely written (size and modification time stable for --settle seconds)
  2. identifies it by SHA-256: a recording already processed is skipped (also when renamed or copied);
     --reprocess forces it again
  3. opens it with OpenCV (fps, frames, duration, resolution); undecodable files are recorded as failed
  4. camera id: --camera, else the first sub-folder under the export folder (e.g. export/ch1/...), else
     "nvr_export". Recording start time: from the file name via --name-pattern, else --start, else UNKNOWN
     (then only offsets inside the video are reported; the file's modification time is stored but never
     presented as the recording time)
  5. uploads it to POST /api/detect/video of the running API: the SAME model, thresholds, tracking and
     alerting as the app (nothing is overridden unless --sample-fps is given) and the results show up in the app
  6. waits for the job, then writes storage/ingest/results/<date>_<camera>_<sha8>/:
     summary.json, detections.csv, events.csv (with wall-clock time when the start time is known) and the
     snapshots of confirmed litter; the annotated video stays in storage/videos/<job_id>/
Network problems (API down, timeouts) leave the recording pending for the next run; a failed job is retried up to
--max-attempts times. The export folder is only ever read: originals are never moved, renamed or deleted.
No credentials are needed or stored: the NVR is not contacted by this tool.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import msvcrt
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import httpx

from backend.core.config import PROJECT_ROOT

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm", ".dav"}
API_UPLOAD_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm"}   # what POST /api/detect/video accepts
CAMERA_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")
log = logging.getLogger("ingest")


class ApiDown(Exception):
    """The LDM API cannot be reached: stop this run, keep everything pending."""


# ---------------------------------------------------------------- ledger (duplicate prevention)
SCHEMA = """CREATE TABLE IF NOT EXISTS recordings (
  sha256 TEXT PRIMARY KEY, path TEXT, size INTEGER, camera_id TEXT, recording_start TEXT, start_source TEXT,
  status TEXT, attempts INTEGER DEFAULT 0, job_id TEXT, result_dir TEXT, error TEXT,
  duration_sec REAL, first_seen TEXT, updated TEXT)"""


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute(SCHEMA)
        self.db.commit()

    def get(self, sha: str) -> Optional[sqlite3.Row]:
        return self.db.execute("SELECT * FROM recordings WHERE sha256=?", (sha,)).fetchone()

    def upsert(self, sha: str, **f: Any) -> None:
        f["updated"] = now()
        if self.get(sha) is None:
            f.setdefault("first_seen", f["updated"])
            cols = ["sha256", *f]
            self.db.execute(f"INSERT INTO recordings ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                            (sha, *f.values()))
        else:
            self.db.execute(f"UPDATE recordings SET {','.join(f'{k}=?' for k in f)} WHERE sha256=?", (*f.values(), sha))
        self.db.commit()


# ---------------------------------------------------------------- helpers
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_stable(p: Path, settle: float) -> bool:
    """Size and mtime unchanged for `settle` seconds and the file can be opened (KVMS has finished writing)."""
    try:
        a = p.stat()
        if time.time() - a.st_mtime < settle:
            time.sleep(min(settle, max(0.0, settle - (time.time() - a.st_mtime))))
        b = p.stat()
        with open(p, "rb"):
            pass
        return a.st_size == b.st_size and a.st_mtime == b.st_mtime and b.st_size > 0
    except OSError:
        return False


def probe(p: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise ValueError("cannot open the video (unsupported codec/container or corrupt file)")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("opened, but no frame could be decoded (corrupt file or unsupported codec)")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {"fps": round(fps, 3), "frames": frames, "width": frame.shape[1], "height": frame.shape[0],
                "duration_sec": round(frames / fps, 2) if fps > 0 and frames > 0 else None}
    finally:
        cap.release()


def camera_for(p: Path, export_dir: Path, fixed: Optional[str]) -> str:
    raw = fixed
    if not raw:
        rel = p.relative_to(export_dir)
        raw = rel.parts[0] if len(rel.parts) > 1 else "nvr_export"
    cam = CAMERA_ID_RE.sub("_", raw).strip("_")[:40]
    return cam or "nvr_export"


def start_for(p: Path, pattern: Optional[re.Pattern], fmt: Optional[str], fixed: Optional[str]) -> tuple[Optional[str], str]:
    if pattern is not None:
        m = pattern.search(p.name)
        if m and "start" in m.groupdict():
            try:
                t = dt.datetime.strptime(m.group("start"), fmt) if fmt else dt.datetime.fromisoformat(m.group("start"))
                return t.isoformat(timespec="seconds"), "file name"
            except ValueError:
                pass
    if fixed:
        return dt.datetime.fromisoformat(fixed).isoformat(timespec="seconds"), "--start"
    return None, "unknown"


# ---------------------------------------------------------------- API client
class Api:
    def __init__(self, base: str, timeout: float) -> None:
        self.base = base.rstrip("/")
        self.c = httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))

    def _call(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            return self.c.request(method, self.base + path, **kw)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout,
                httpx.RemoteProtocolError) as exc:
            raise ApiDown(f"{type(exc).__name__} on {path}") from exc

    def health(self) -> dict[str, Any]:
        r = self._call("GET", "/api/health")
        r.raise_for_status()
        return r.json()

    def settings(self) -> dict[str, Any]:
        r = self._call("GET", "/api/settings")
        return r.json() if r.status_code == 200 else {}

    def submit(self, p: Path, camera: str, sample_fps: Optional[float]) -> str:
        data = {"camera_id": camera}
        if sample_fps:
            data["sample_fps"] = str(sample_fps)
        with open(p, "rb") as fh:
            r = self._call("POST", "/api/detect/video", data=data, files={"file": (p.name, fh, "video/mp4")})
        if r.status_code != 202:
            raise RuntimeError(f"upload rejected: HTTP {r.status_code} {r.text[:200]}")
        return r.json()["job_id"]

    def wait(self, job_id: str, poll: float, limit: float) -> dict[str, Any]:
        t0, misses = time.time(), 0
        while time.time() - t0 < limit:
            try:
                r = self._call("GET", f"/api/jobs/{job_id}")
                misses = 0
            except ApiDown:
                misses += 1
                if misses >= 5:
                    raise
                time.sleep(min(30, poll * 2 ** misses))
                continue
            j = r.json()
            if j.get("status") in ("done", "failed"):
                return j
            time.sleep(poll)
        raise TimeoutError(f"job {job_id} not finished after {limit:.0f} s")

    def file(self, job_id: str, kind: str) -> Any:
        r = self._call("GET", f"/api/jobs/{job_id}/files/{kind}")
        r.raise_for_status()
        return r.json()

    def snapshot(self, alert_id: int) -> Optional[bytes]:
        r = self._call("GET", f"/api/alerts/{alert_id}/snapshot", params={"kind": "full"})
        return r.content if r.status_code == 200 else None


# ---------------------------------------------------------------- one recording
def write_results(out: Path, meta: dict[str, Any], job: dict[str, Any], dets: list, events: list, api: Api) -> None:
    out.mkdir(parents=True, exist_ok=True)
    start = dt.datetime.fromisoformat(meta["recording_start"]) if meta.get("recording_start") else None

    def wall(sec: Optional[float]) -> str:
        return (start + dt.timedelta(seconds=sec)).isoformat(timespec="seconds") if start and sec is not None else ""

    with open(out / "detections.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_idx", "video_time_sec", "wall_time", "class", "confidence", "x1", "y1", "x2", "y2",
                    "status", "track_id"])
        for fr in dets:
            for d in fr.get("detections", []):
                w.writerow([fr["frame_idx"], round(fr.get("video_time_sec") or 0, 3), wall(fr.get("video_time_sec")),
                            d["class"], d["confidence"], *[round(v, 1) for v in d["bbox"]], d.get("status"),
                            d.get("track_id")])
    with open(out / "events.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["event", "class", "confidence", "video_time_sec", "wall_time", "frame_idx", "track_id", "alert_id"])
        for e in events:
            w.writerow([e.get("event"), e.get("class"), e.get("confidence"), round(e.get("video_time_sec") or 0, 3),
                        wall(e.get("video_time_sec")), e.get("frame_idx"), e.get("track_id"), e.get("alert_id")])
    snaps = []
    for e in events:
        if e.get("alert_id"):
            img = api.snapshot(e["alert_id"])
            if img:
                name = f"alert_{e['alert_id']}_{round(e.get('video_time_sec') or 0)}s.jpg"
                (out / name).write_bytes(img)
                snaps.append(name)
    stats = (job.get("result") or {}).get("stats", {})
    summary = {**meta, "job_id": job["id"] if "id" in job else meta.get("job_id"), "job_status": job.get("status"),
               "frames_analysed": stats.get("frames_analysed"), "litter": stats.get("litter"),
               "events": len(events), "alert_ids": (job.get("result") or {}).get("alert_ids", []),
               "snapshots": snaps, "annotated_video": f"storage/videos/{meta['job_id']}/annotated.mp4",
               "timestamps": "wall_time = recording_start + video_time_sec" if start else
                             "recording start unknown: only video_time_sec (offset in the file) is reliable"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")


def process(p: Path, a: argparse.Namespace, api: Api, ledger: Ledger, limits: dict[str, Any],
            pattern: Optional[re.Pattern]) -> str:
    if not is_stable(p, a.settle):
        log.info("still being written, later: %s", p.name)
        return "not_ready"
    sha = sha256_file(p)
    row = ledger.get(sha)
    if row and not a.reprocess:
        if row["status"] == "done":
            log.info("already processed (%s): %s", row["result_dir"], p.name)
            return "duplicate"
        if row["status"] in ("failed", "unsupported", "too_large") and row["attempts"] >= a.max_attempts:
            log.info("gave up earlier (%s: %s): %s", row["status"], row["error"], p.name)
            return "skipped"
    camera = camera_for(p, a.export_dir, a.camera)
    start, src = start_for(p, pattern, a.time_format, a.start)
    attempts = (row["attempts"] if row else 0) + 1
    base = dict(path=str(p), size=p.stat().st_size, camera_id=camera, recording_start=start, start_source=src)
    if p.suffix.lower() not in API_UPLOAD_EXTS:
        if not a.dry_run:
            ledger.upsert(sha, **base, status="unsupported", attempts=a.max_attempts,
                          error=f"{p.suffix} is not accepted by the app: export as MP4/AVI in KVMS (or convert)")
        log.warning("unsupported container %s (export as MP4/AVI): %s", p.suffix, p.name)
        return "unsupported"
    size_mb = p.stat().st_size / 2 ** 20
    if size_mb > limits["max_mb"]:
        if not a.dry_run:
            ledger.upsert(sha, **base, status="too_large", attempts=a.max_attempts,
                          error=f"{size_mb:.0f} MB > upload limit {limits['max_mb']} MB: export shorter clips or raise MAX_VIDEO_UPLOAD_MB")
        log.warning("too large (%.0f MB > %s MB): %s", size_mb, limits["max_mb"], p.name)
        return "too_large"
    try:
        info = probe(p)
    except ValueError as exc:
        if not a.dry_run:
            ledger.upsert(sha, **base, status="failed", attempts=a.max_attempts, error=str(exc))
        log.error("cannot decode %s: %s", p.name, exc)
        return "failed"
    if a.dry_run:
        log.info("would process %s (camera %s, start %s, %.0f MB, %ss)", p.name, camera, start or "unknown",
                 size_mb, info["duration_sec"])
        return "would_process"
    ledger.upsert(sha, **base, status="processing", attempts=attempts, duration_sec=info["duration_sec"], error=None)
    t0 = time.time()
    try:
        job_id = api.submit(p, camera, a.sample_fps)
        ledger.upsert(sha, job_id=job_id)
        log.info("submitted %s as job %s (camera %s)", p.name, job_id, camera)
        job = api.wait(job_id, a.poll, a.job_timeout)
        if job.get("status") != "done":
            raise RuntimeError(f"job {job_id} {job.get('status')}: {job.get('error')}")
        dets, events = api.file(job_id, "detections"), api.file(job_id, "events")
    except ApiDown:
        ledger.upsert(sha, status="pending", attempts=attempts - 1, error="API unreachable - retried next run")
        raise
    except (RuntimeError, TimeoutError, httpx.HTTPError) as exc:
        ledger.upsert(sha, status="failed", error=str(exc)[:500])
        log.error("failed %s (attempt %d/%d): %s", p.name, attempts, a.max_attempts, exc)
        return "failed"
    out = a.ingest_dir / "results" / f"{(start or now())[:10].replace('-', '')}_{camera}_{sha[:8]}"
    notes = []
    if info["duration_sec"] and info["duration_sec"] > limits["max_seconds"]:
        notes.append(f"only the first {limits['max_seconds']:.0f} s were analysed (video.max_video_seconds); "
                     f"the recording is {info['duration_sec']:.0f} s - export shorter clips")
    meta = {"source_file": str(p), "sha256": sha, "camera_id": camera, "recording_start": start,
            "recording_start_source": src, "file_modified": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            "video": info, "job_id": job_id, "model_version": limits.get("model"),
            "sample_fps": a.sample_fps or limits.get("sample_fps"), "processing_seconds": round(time.time() - t0, 1),
            "processed_at": now(), "notes": notes}
    write_results(out, meta, job, dets, events, api)
    ledger.upsert(sha, status="done", result_dir=str(out), error="; ".join(notes) or None)
    log.info("done %s: %d confirmed litter event(s), %.0f s -> %s", p.name, len(events), time.time() - t0, out)
    for n in notes:
        log.warning(n)
    return "done"


# ---------------------------------------------------------------- main
def run_once(a: argparse.Namespace, ledger: Ledger, pattern: Optional[re.Pattern]) -> dict[str, int]:
    api = Api(a.api_url, a.http_timeout)
    try:
        h = api.health()
    except ApiDown as exc:
        log.error("LDM API not reachable at %s (%s): start it with deploy\\start-ldm.ps1; nothing was changed",
                  a.api_url, exc)
        return {"api_down": 1}
    if not h.get("model"):
        log.error("LDM API has no model loaded (%s): not processing", h.get("model_error"))
        return {"no_model": 1}
    s = api.settings()
    sv = (s.get("settings") or s).get("video", {}) if isinstance(s, dict) else {}
    limits = {"max_mb": a.max_upload_mb, "max_seconds": float(sv.get("max_video_seconds") or 3600),
              "sample_fps": sv.get("sample_fps"), "model": h["model"]["version"]}
    files = sorted(q for q in a.export_dir.rglob("*") if q.is_file() and q.suffix.lower() in VIDEO_EXTS)
    log.info("%d video file(s) in %s; model %s", len(files), a.export_dir, limits["model"])
    counts: dict[str, int] = {}
    for p in files:
        try:
            r = process(p, a, api, ledger, limits, pattern)
        except ApiDown as exc:
            log.error("LDM API became unreachable (%s): stopping this run; %s stays pending", exc, p.name)
            counts["api_down"] = counts.get("api_down", 0) + 1
            break
        counts[r] = counts.get(r, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", type=Path, default=os.getenv("LDM_EXPORT_DIR") or None,
                    help="folder KVMS Pro Lite exports recordings to (env LDM_EXPORT_DIR)")
    ap.add_argument("--api-url", default=os.getenv("LDM_API_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--ingest-dir", type=Path, default=Path(os.getenv("LDM_INGEST_DIR", PROJECT_ROOT / "storage" / "ingest")))
    ap.add_argument("--camera", help="camera id for every file (default: first sub-folder name, else nvr_export)")
    ap.add_argument("--name-pattern", help=r"regex with a named group 'start' (and optionally 'camera') applied to "
                                           r"the file name, e.g. 'ch(?P<camera>\d+)_(?P<start>\d{14})'")
    ap.add_argument("--time-format", help="strptime format of the 'start' group, e.g. %%Y%%m%%d%%H%%M%%S (default ISO)")
    ap.add_argument("--start", help="recording start (ISO) for files whose name has no time; use with one file only")
    ap.add_argument("--sample-fps", type=float, help="frames per second analysed (default: the app's video.sample_fps)")
    ap.add_argument("--max-upload-mb", type=int, default=int(os.getenv("MAX_VIDEO_UPLOAD_MB", "500")))
    ap.add_argument("--settle", type=float, default=10.0, help="seconds a file must be unchanged before processing")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--reprocess", action="store_true", help="process recordings again even if already done")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be processed")
    ap.add_argument("--watch", type=float, help="keep running, scanning every N seconds (Ctrl+C to stop)")
    ap.add_argument("--poll", type=float, default=3.0)
    ap.add_argument("--job-timeout", type=float, default=4 * 3600)
    ap.add_argument("--http-timeout", type=float, default=600.0)
    a = ap.parse_args(argv)

    a.ingest_dir = a.ingest_dir if a.ingest_dir.is_absolute() else PROJECT_ROOT / a.ingest_dir
    a.ingest_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(a.ingest_dir / "ingest.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not a.export_dir:
        log.error("no export folder: pass --export-dir or set LDM_EXPORT_DIR")
        return 2
    a.export_dir = Path(a.export_dir).resolve()
    if not a.export_dir.is_dir():
        log.error("export folder does not exist: %s", a.export_dir)
        return 2
    pattern = None
    if a.name_pattern:
        try:
            pattern = re.compile(a.name_pattern)
        except re.error as exc:
            log.error("invalid --name-pattern: %s", exc)
            return 2
        if "camera" in pattern.groupindex and not a.camera:
            log.info("note: the 'camera' group is informational; camera ids come from --camera or the sub-folder")
    # one ingester at a time (a second instance would upload the same new file twice)
    lock = open(a.ingest_dir / "ingest.lock", "w")
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        log.error("another ingest run is already active (lock %s)", a.ingest_dir / "ingest.lock")
        return 2
    ledger = Ledger(a.ingest_dir / "ingest.db")
    try:
        while True:
            counts = run_once(a, ledger, pattern)
            log.info("run summary: %s", counts or "nothing to do")
            if not a.watch:
                break
            time.sleep(a.watch)
    except KeyboardInterrupt:
        log.info("stopped by user")
    finally:
        lock.close()
    if counts.get("api_down") or counts.get("no_model"):
        return 3
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
