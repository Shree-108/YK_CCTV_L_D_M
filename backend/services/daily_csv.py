"""Daily store CSV: one file per recording day (India time), ``storage/daily_csv/YYYY-MM-DD.csv``.

ONE ROW = ONE STORE ON ONE DAY (storeKey + date is unique). Columns, in this order::

    storeKey, cartId, storeName, date, entries, totalEntries, sourceFiles, parserVersion, lastSyncedAt, createdAt,
    updatedAt

* ``entries``     JSON array, one object per screenshot: capturedAt, cameraId, imageName, litterClass,
                  confidenceScore, detectionStatus, remark, storeName, sourceFile - sorted by capturedAt.
                  The status / class / confidence / remark come from garbage_export.sampled_checks + describe (the
                  same clean / litter_detected / view_blocked rules and remark text as every other export).
                  imageName = ``<video name without .mp4>_s<n>.jpg``, so a re-scan of the same video is a duplicate.
* ``sourceFiles`` JSON array of {key, etag, lastModified}, one per imported video. key = the video's original file
                  name, etag = MD5 of the stored video (what S3 reports as the ETag of a single-part upload),
                  lastModified = the stored file's modification time (+05:30). A video already listed in a row is
                  not imported into that row again.

Appending: a finished scan adds its screenshots to the day file of each screenshot's recording date (a recording
that runs past midnight is split over two files). An existing store row is updated in place (entries merged,
duplicates by sourceFile + imageName skipped, totalEntries / sourceFiles / updatedAt / lastSyncedAt updated, createdAt
kept); a new store gets its own row. The header is written once. Read-modify-write runs under a lock and the file is
replaced through a temp file. A file whose header is not this one is never overwritten.

A scan whose store cannot be resolved from config/store_config.json goes under storeKey ``UNASSIGNED`` (cartId and
storeName empty) so nothing is lost. A screenshot without a trustworthy recording time has no day and is left out
(counted in the result).

Runner hook: ``safe_append_job`` - never raises, so a CSV problem can never fail a scan.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

PARSER_VERSION = 1
UNASSIGNED = "UNASSIGNED"
DIR_NAME = "daily_csv"
COLUMNS = ["storeKey", "cartId", "storeName", "date", "entries", "totalEntries", "sourceFiles", "parserVersion",
           "lastSyncedAt", "createdAt", "updatedAt"]
ENTRY_FIELDS = ["capturedAt", "cameraId", "imageName", "litterClass", "confidenceScore", "detectionStatus", "remark",
                "storeName", "sourceFile"]
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NO_CAMERA = ("", "upload", "unknown")

_LOCK = threading.Lock()            # one read-modify-write at a time in this process
LOCK_WAIT_SEC = 30.0
LOCK_STALE_SEC = 120.0


class DailyCsvError(RuntimeError):
    pass


# ============================================================================================ small helpers
def daily_dir(storage_dir: Path) -> Path:
    return Path(storage_dir) / DIR_NAME


def file_name(day: str) -> str:
    if not DAY_RE.match(day):
        raise DailyCsvError(f"not a YYYY-MM-DD day: {day!r}")
    return f"{day}.csv"


def iso(t: dt.datetime) -> str:
    """ISO 8601 in India time, seconds precision: 2026-11-10T14:03:05+05:30."""
    if t.tzinfo is None:
        raise DailyCsvError("internal error: naive datetime")
    return t.astimezone(IST).replace(microsecond=0).isoformat(timespec="seconds")


def now_iso() -> str:
    return iso(dt.datetime.now(IST))


def day_of(captured_at: str) -> str:
    """Recording day in India time of an ISO timestamp."""
    return dt.datetime.fromisoformat(captured_at).astimezone(IST).strftime("%Y-%m-%d")


def day_start(day: str) -> str:
    return f"{day}T00:00:00+05:30"


def image_name(video_name: str, idx: int) -> str:
    """``<video name without .mp4>_s<n>.jpg`` - stable across re-scans of the same video."""
    stem = Path(video_name or "video").name
    if stem.lower().endswith(".mp4"):
        stem = stem[:-4]
    return f"{stem}_s{idx}.jpg"


def _dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def source_file_info(video_name: str, path: Optional[Path]) -> dict[str, str]:
    """{key, etag, lastModified} of one imported video. etag / lastModified are empty when the stored copy is gone."""
    etag = last = ""
    if path is not None and Path(path).is_file():
        h = hashlib.md5()                                   # noqa: S324 - an S3-style ETag, not a security hash
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        etag = h.hexdigest()
        last = iso(dt.datetime.fromtimestamp(Path(path).stat().st_mtime, IST))
    return {"key": video_name, "etag": etag, "lastModified": last}


# ============================================================================================ file I/O
class _FileLock:
    """Process lock (threading) + cross-process lock file next to the CSV."""

    def __init__(self, path: Path):
        self.path = path.with_name(path.name + ".lock")
        self.fd: Optional[int] = None

    def __enter__(self) -> "_FileLock":
        _LOCK.acquire()
        deadline = time.monotonic() + LOCK_WAIT_SEC
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            while True:
                try:
                    self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    return self
                except FileExistsError:
                    try:
                        if time.time() - self.path.stat().st_mtime > LOCK_STALE_SEC:
                            self.path.unlink(missing_ok=True)     # left behind by a crashed writer
                            continue
                    except FileNotFoundError:
                        continue
                    if time.monotonic() > deadline:
                        raise DailyCsvError(f"{self.path.name}: another writer holds the lock")
                    time.sleep(0.05)
        except BaseException:
            _LOCK.release()
            raise

    def __exit__(self, *exc: Any) -> None:
        try:
            if self.fd is not None:
                os.close(self.fd)
            self.path.unlink(missing_ok=True)
        finally:
            _LOCK.release()


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Rows of a day file with entries / sourceFiles decoded. A missing file is empty; a foreign header raises."""
    if not path.is_file():
        return []
    text = path.read_bytes().decode("utf-8")
    if not text.strip():
        return []
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != COLUMNS:
        raise DailyCsvError(f"{path.name} has a different header ({reader.fieldnames}); it is left untouched")
    rows = []
    for r in reader:
        r = dict(r)
        r["entries"] = json.loads(r["entries"] or "[]")
        r["sourceFiles"] = json.loads(r["sourceFiles"] or "[]")
        rows.append(r)
    return rows


def to_csv(rows: Iterable[dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="raise", quoting=csv.QUOTE_MINIMAL,
                       lineterminator="\r\n")
    w.writeheader()
    for r in rows:
        w.writerow({**r, "entries": _dumps(r["entries"]), "sourceFiles": _dumps(r["sourceFiles"]),
                    "totalEntries": len(r["entries"])})
    return buf.getvalue()


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(to_csv(rows))
    os.replace(tmp, path)


# ============================================================================================ merge
def merge(rows: list[dict[str, Any]], *, store_key: str, cart_id: str, store_name: str, day: str,
          entries: list[dict[str, Any]], source: dict[str, str], now: str) -> dict[str, int]:
    """Merge one video's entries of one day into ``rows`` (in place). Returns {added, duplicates, skipped_source}."""
    row = next((r for r in rows if r["storeKey"] == store_key and r["date"] == day_start(day)), None)
    if row is None:
        row = {"storeKey": store_key, "cartId": cart_id, "storeName": store_name, "date": day_start(day),
               "entries": [], "sourceFiles": [], "parserVersion": PARSER_VERSION, "lastSyncedAt": now,
               "createdAt": now, "updatedAt": now}
        rows.append(row)
    if any(s.get("key") == source["key"] for s in row["sourceFiles"]):
        row["lastSyncedAt"] = now
        return {"added": 0, "duplicates": len(entries), "skipped_source": 1}
    seen = {(e.get("sourceFile"), e.get("imageName")) for e in row["entries"]}
    added = dups = 0
    for e in entries:
        k = (e["sourceFile"], e["imageName"])
        if k in seen:
            dups += 1
            continue
        seen.add(k)
        row["entries"].append(e)
        added += 1
    row["entries"].sort(key=lambda e: (dt.datetime.fromisoformat(e["capturedAt"]), e["cameraId"], e["imageName"]))
    row["sourceFiles"].append(source)
    row["totalEntries"] = len(row["entries"])
    row["parserVersion"] = PARSER_VERSION
    if cart_id or store_name:                 # store config may have been filled in since the row was created
        row["cartId"], row["storeName"] = cart_id, store_name
    row["updatedAt"] = row["lastSyncedAt"] = now
    return {"added": added, "duplicates": dups, "skipped_source": 0}


def append(storage_dir: Path, *, store_key: str, cart_id: str, store_name: str, entries: list[dict[str, Any]],
           source: dict[str, str], now: Optional[str] = None) -> dict[str, Any]:
    """Add one video's entries to the day file(s) of their recording dates."""
    now = now or now_iso()
    by_day: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_day.setdefault(day_of(e["capturedAt"]), []).append(e)
    out: dict[str, Any] = {"files": [], "added": 0, "duplicates": 0, "skipped_source": 0}
    for day in sorted(by_day):
        path = daily_dir(storage_dir) / file_name(day)
        with _FileLock(path):
            rows = read_rows(path)
            res = merge(rows, store_key=store_key, cart_id=cart_id, store_name=store_name, day=day,
                        entries=by_day[day], source=source, now=now)
            _write(path, rows)
        out["files"].append(str(path))
        for k in ("added", "duplicates", "skipped_source"):
            out[k] += res[k]
    return out


# ============================================================================================ from a finished scan
def _store(settings: Any, store_id: Optional[str], camera_id: str) -> tuple[str, str, str]:
    """(storeKey, cartId, storeName) from config/store_config.json; UNASSIGNED when it cannot be resolved."""
    from backend.services import garbage_export as ge
    try:
        stores = ge.load_store_config(settings.store_config)
    except ge.ExportError:
        return UNASSIGNED, "", ""
    key = ge._known(store_id)
    st = stores.get(key) if key else None
    if st is None and camera_id:
        st = next((s for s in stores.values() if camera_id in s.camera_ids), None)
    if st is None:
        return UNASSIGNED, "", ""
    return st.key, str(st.cart_id), st.name


def job_entries(query: Any, settings: Any, job_id: str, allowed_classes: Optional[set[str]] = None,
                ) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """(store + source info, entries, screenshots left out for having no recording time) of a finished
    four-screenshot job. The checks and their text come from garbage_export, unchanged."""
    from backend.services import garbage_export as ge
    src = ge.sampled_source(query, job_id)
    raw_cam = (src.camera_id or "").strip()
    cam = "" if raw_cam.lower() in NO_CAMERA else raw_cam
    tb, reason = None, ""
    try:
        tb = ge.resolve_time_base(video_path=src.video_path, original_name=src.original_name,
                                  duration_sec=src.duration_sec, manual_start=src.manual_start,
                                  manual_label="operator-entered start",
                                  trust_container_time=getattr(settings, "trust_container_time", False))
    except ge.ExportError as exc:
        reason = ge.one_line(str(exc))
    checks, _ = ge.sampled_checks(query, src, cam, tb, reason)
    store_key, cart_id, store_name = _store(settings, src.store_key, cam)
    video = Path(src.original_name or f"{job_id}.mp4").name
    entries, no_time = [], 0
    for c in checks:
        if c.captured_at is None:
            no_time += 1
            continue
        cls_out, conf_out, remark = ge.describe(c, allowed_classes)
        entries.append({"capturedAt": iso(c.captured_at), "cameraId": cam, "imageName": image_name(video, c.idx),
                        "litterClass": cls_out, "confidenceScore": conf_out, "detectionStatus": c.status,
                        "remark": remark, "storeName": store_name, "sourceFile": video})
    info = {"store_key": store_key, "cart_id": cart_id, "store_name": store_name, "video": video,
            "video_path": src.video_path}
    return info, entries, no_time


def append_job(query: Any, settings: Any, storage_dir: Path, job_id: str,
               allowed_classes: Optional[set[str]] = None, now: Optional[str] = None) -> dict[str, Any]:
    info, entries, no_time = job_entries(query, settings, job_id, allowed_classes)
    res: dict[str, Any] = {"files": [], "added": 0, "duplicates": 0, "skipped_source": 0}
    if entries:
        res = append(storage_dir, store_key=info["store_key"], cart_id=info["cart_id"],
                     store_name=info["store_name"], entries=entries,
                     source=source_file_info(info["video"], info["video_path"]), now=now)
    res.update(ok=True, job_id=job_id, store_key=info["store_key"], no_time=no_time,
               unassigned=res["added"] if info["store_key"] == UNASSIGNED else 0)
    return res


def safe_append_job(state: Any, job_id: str) -> dict[str, Any]:
    """Runner hook: append a finished scan to the daily CSV. Never raises."""
    try:
        from backend.services import garbage_export as ge
        classes = set(getattr(state.engine, "class_names", []) or []) or None
        res = append_job(state.db._query, ge.settings_from_config(state.cfg), state.cfg.storage_dir, job_id, classes)
        log.info("daily CSV: %s", res)
        return res
    except Exception as exc:  # noqa: BLE001 - a CSV problem must never change the finished scan
        log.exception("daily CSV not updated for %s; the scan itself is unaffected", job_id)
        return {"ok": False, "job_id": job_id, "error": f"{type(exc).__name__}: {exc}"}


def days(storage_dir: Path) -> list[str]:
    """Days that have a file, newest first."""
    d = daily_dir(storage_dir)
    if not d.is_dir():
        return []
    return sorted((p.stem for p in d.glob("*.csv") if DAY_RE.match(p.stem)), reverse=True)
