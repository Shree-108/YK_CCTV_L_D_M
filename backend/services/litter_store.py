"""Stores, store/camera identity, analyses and litter events (tables: stores, analyses, litter_events).

Identity rules (owner brief 2026-11-19, Part 7):
  * a REGISTERED camera's store comes from its record (cameras.store_id); a request naming another store for it is
    refused (IdentityError -> HTTP 422) - a detection is never attached to the wrong store;
  * a store that is in the stores table gets its name; a store ID that is not registered is kept as typed (validated
    free text, e.g. on a laptop without store setup) with name UNKNOWN and a warning;
  * anything missing is UNKNOWN with a warning. Nothing is guessed or invented.

Analytics count litter EVENTS (unique objects), never frame observations, as "litter"; observations are reported
separately. Events an operator marked REJECTED (false positive) are excluded from the counts.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any, Optional

from backend.database.db import Database, utcnow
from backend.services.litter_reports import (DETECTED, REJECTED_S, REVIEW_STATUSES, UNKNOWN, Identity,
                                             local_now)

STORE_ID_RE = re.compile(r"^[A-Za-z0-9._ \-]{1,64}$")
CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")
NO_CAMERA = ("", "upload", UNKNOWN)


class IdentityError(ValueError):
    """The store/camera combination cannot be accepted (message is shown to the operator)."""


def _clean(v: Optional[str]) -> str:
    return (v or "").strip()


def resolve_identity(db: Database, store_id: Optional[str], camera_id: Optional[str],
                     recording_id: str = "") -> Identity:
    store_id, camera_id = _clean(store_id), _clean(camera_id)
    if camera_id in NO_CAMERA:
        camera_id = ""
    if store_id.upper() == UNKNOWN:
        store_id = ""
    if store_id and not STORE_ID_RE.match(store_id):
        raise IdentityError("Store ID may contain letters, digits, spaces, '.', '-' and '_' (max 64).")
    if camera_id and not CAMERA_ID_RE.match(camera_id):
        raise IdentityError("Camera ID may contain letters, digits, '-' and '_' (max 40).")
    ident = Identity(recording_id=recording_id)
    cam = db.get_camera(camera_id) if camera_id else None
    if cam:
        ident.camera_id, ident.camera_name = cam["id"], cam.get("name") or UNKNOWN
        cam_store = cam.get("store_id") or ""
        if cam_store and store_id and store_id != cam_store:
            raise IdentityError(f"Camera {camera_id} belongs to store {cam_store}, not {store_id}. "
                                "Choose the camera's own store (or leave the store empty).")
        store_id = cam_store or store_id
        if not cam_store:
            ident.warnings.append(f"Camera {camera_id} is not assigned to a store in Camera Integration.")
    elif camera_id:
        ident.camera_id = camera_id
        ident.warnings.append(f"Camera {camera_id} is not registered: its name is UNKNOWN and it has no store "
                              "mapping.")
    else:
        ident.warnings.append("Camera ID not provided: results are recorded under camera UNKNOWN.")
    if store_id:
        ident.store_id = store_id
        st = get_store(db, store_id)
        if st:
            ident.store_name = st["name"]
        else:
            ident.warnings.append(f"Store {store_id} is not registered: its name is UNKNOWN.")
    else:
        ident.warnings.append("Store ID not provided: results are recorded under store UNKNOWN.")
    return ident


# ----------------------------------------------------------------------------------------------- stores
def list_stores(db: Database) -> list[dict[str, Any]]:
    stores = db._query("SELECT id, name, created_at, updated_at FROM stores ORDER BY id", ())
    cams = db._query("SELECT id, name, store_id FROM cameras ORDER BY id", ())
    for s in stores:
        s["cameras"] = [{"id": c["id"], "name": c["name"]} for c in cams if c["store_id"] == s["id"]]
    return stores


def get_store(db: Database, store_id: str) -> Optional[dict[str, Any]]:
    return db._one("SELECT id, name, created_at, updated_at FROM stores WHERE id=?", (store_id,))


def upsert_store(db: Database, store_id: str, name: str) -> dict[str, Any]:
    now = utcnow()
    db._write("INSERT INTO stores(id, name, created_at, updated_at) VALUES (?,?,?,?) "
              "ON CONFLICT(id) DO UPDATE SET name=excluded.name, updated_at=excluded.updated_at",
              (store_id, name, now, now))
    return get_store(db, store_id)  # type: ignore[return-value]


# ----------------------------------------------------------------------------------------------- analyses
def start_analysis(db: Database, analysis_id: str, kind: str, ident: Identity, *, job_id: Optional[str],
                   model_version: str, output_dir: str, started: Optional[dt.datetime] = None) -> None:
    db._write("INSERT INTO analyses(id, kind, job_id, created_at, status, store_id, store_name, camera_id, camera_name,"
              " recording_id, model_version, output_dir, warnings_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (analysis_id, kind, job_id, (started or local_now()).isoformat(), "RUNNING", ident.store_id,
               ident.store_name, ident.camera_id, ident.camera_name, ident.recording_id, model_version, output_dir,
               json.dumps(ident.warnings)))


def finish_analysis(db: Database, analysis_id: str, status: str, *, unique_events: int = 0,
                    frame_observations: int = 0, frames_processed: int = 0, files: Optional[dict[str, str]] = None,
                    error: Optional[str] = None) -> None:
    db._write("UPDATE analyses SET status=?, finished_at=?, unique_events=?, frame_observations=?, frames_processed=?,"
              " files_json=?, error=? WHERE id=?",
              (status, local_now().isoformat(), unique_events, frame_observations, frames_processed,
               json.dumps(files or {}), error, analysis_id))


def _analysis(r: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if r is None:
        return None
    r["warnings"] = json.loads(r.pop("warnings_json") or "[]")
    r["files"] = json.loads(r.pop("files_json") or "{}")
    return r


def get_analysis(db: Database, analysis_id: str) -> Optional[dict[str, Any]]:
    return _analysis(db._one("SELECT * FROM analyses WHERE id=?", (analysis_id,)))


def analysis_for_job(db: Database, job_id: str) -> Optional[dict[str, Any]]:
    return _analysis(db._one("SELECT * FROM analyses WHERE job_id=? ORDER BY created_at DESC LIMIT 1", (job_id,)))


def list_analyses(db: Database, limit: int = 50) -> list[dict[str, Any]]:
    return [_analysis(r) for r in db._query("SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?",  # type: ignore[misc]
                                            (limit,))]


# ----------------------------------------------------------------------------------------------- events
EVENT_COLS = ("event_id", "analysis_id", "source_type", "job_id", "store_id", "store_name", "camera_id", "camera_name",
              "recording_id", "track_id", "class", "max_confidence", "mean_confidence", "x1", "y1", "x2", "y2",
              "first_seen", "last_seen", "first_seen_sec", "last_seen_sec", "duration_sec", "observation_count",
              "detection_status", "review_status", "timestamp_source", "model_version", "image_path", "created_at")


def insert_events(db: Database, analysis_id: str, source_type: str, ident: Identity, events: list[dict[str, Any]], *,
                  job_id: Optional[str], model_version: str, timestamp_source: str) -> None:
    if not events:
        return
    now = utcnow()
    rows = []
    for e in events:
        x1, y1, x2, y2 = e["bbox"]
        rows.append((e["event_id"], analysis_id, source_type, job_id, ident.store_id, ident.store_name,
                     ident.camera_id, ident.camera_name, ident.recording_id, e.get("track_id"), e["class"],
                     e["max_confidence"], e["mean_confidence"], x1, y1, x2, y2, e["first_seen"], e["last_seen"],
                     e.get("first_seen_sec"), e.get("last_seen_sec"), e.get("duration_sec"), e["observation_count"],
                     DETECTED, DETECTED, timestamp_source, model_version, e.get("image_path"), now))
    q = ",".join("?" * len(EVENT_COLS))
    with db.connect() as c:
        c.executemany(f"INSERT INTO litter_events({','.join(EVENT_COLS)}) VALUES ({q})", rows)


def _event_out(r: dict[str, Any]) -> dict[str, Any]:
    r["bbox"] = [r.pop("x1"), r.pop("y1"), r.pop("x2"), r.pop("y2")]
    r["image_url"] = f"/api/litter/events/{r['event_id']}/image" if r.get("image_path") else None
    r.pop("image_path", None)
    return r


def _filters(start: Optional[str], end: Optional[str], store_id: Optional[str], camera_id: Optional[str],
             cls: Optional[str] = None, status: Optional[str] = None,
             exclude_rejected: bool = False) -> tuple[str, list[Any]]:
    w, p = [], []
    for col, op, v in (("first_seen", ">=", start), ("first_seen", "<", end), ("store_id", "=", store_id),
                       ("camera_id", "=", camera_id), ("class", "=", cls), ("review_status", "=", status)):
        if v:
            w.append(f"{col} {op} ?")
            p.append(v)
    if exclude_rejected:
        w.append("review_status != ?")
        p.append(REJECTED_S)
    return (" WHERE " + " AND ".join(w)) if w else "", p


def list_events(db: Database, *, start: Optional[str] = None, end: Optional[str] = None,
                store_id: Optional[str] = None, camera_id: Optional[str] = None, cls: Optional[str] = None,
                status: Optional[str] = None, limit: int = 100, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    where, p = _filters(start, end, store_id, camera_id, cls, status)
    rows = db._query(f"SELECT * FROM litter_events{where} ORDER BY first_seen DESC, id DESC LIMIT ? OFFSET ?",
                     (*p, limit, offset))
    total = db._one(f"SELECT COUNT(*) AS n FROM litter_events{where}", p)["n"]  # type: ignore[index]
    return [_event_out(r) for r in rows], int(total)


def get_event(db: Database, event_id: str) -> Optional[dict[str, Any]]:
    return db._one("SELECT * FROM litter_events WHERE event_id=?", (event_id,))


def review_event(db: Database, event_id: str, review_status: str, actor: Optional[str], note: Optional[str]) -> bool:
    if review_status not in REVIEW_STATUSES:
        raise ValueError(f"review_status must be one of {', '.join(REVIEW_STATUSES)}")
    cur = db._write("UPDATE litter_events SET review_status=?, reviewed_at=?, reviewed_by=?, review_note=? "
                    "WHERE event_id=?", (review_status, utcnow(), actor, note, event_id))
    return cur.rowcount > 0


# ----------------------------------------------------------------------------------------------- analytics
BUCKET = {"hour": "substr(first_seen, 1, 13)", "day": "substr(first_seen, 1, 10)",
          "week": "strftime('%Y-W%W', first_seen)"}


def analytics(db: Database, *, start: Optional[str] = None, end: Optional[str] = None,
              store_id: Optional[str] = None, camera_id: Optional[str] = None,
              granularity: str = "day", today: Optional[dt.date] = None) -> dict[str, Any]:
    if granularity not in BUCKET:
        raise ValueError("granularity must be hour, day or week")
    where, p = _filters(start, end, store_id, camera_id, exclude_rejected=True)
    q = db._query
    tot = q(f"SELECT COUNT(*) AS events, COALESCE(SUM(observation_count), 0) AS obs, "
            f"AVG(CASE WHEN observation_count > 1 THEN duration_sec END) AS avg_visible FROM litter_events{where}", p)[0]
    by_class = q(f"SELECT class, COUNT(*) AS events, SUM(observation_count) AS observations FROM litter_events{where} "
                 "GROUP BY class ORDER BY events DESC, class", p)
    by_store = q(f"SELECT store_id, MAX(store_name) AS store_name, COUNT(*) AS events, "
                 f"SUM(observation_count) AS observations FROM litter_events{where} GROUP BY store_id "
                 "ORDER BY events DESC, store_id", p)
    by_camera = q(f"SELECT camera_id, MAX(camera_name) AS camera_name, MAX(store_id) AS store_id, COUNT(*) AS events, "
                  f"SUM(observation_count) AS observations FROM litter_events{where} GROUP BY camera_id "
                  "ORDER BY events DESC, camera_id", p)
    b = BUCKET[granularity]
    series_rows = q(f"SELECT {b} AS bucket, class, COUNT(*) AS events, SUM(observation_count) AS observations "
                    f"FROM litter_events{where} GROUP BY bucket, class ORDER BY bucket", p)
    series: dict[str, dict[str, Any]] = {}
    for r in series_rows:
        s = series.setdefault(r["bucket"], {"bucket": r["bucket"], "events": 0, "observations": 0, "by_class": {}})
        s["events"] += r["events"]
        s["observations"] += r["observations"]
        s["by_class"][r["class"]] = r["events"]
    day = (today or local_now().date()).isoformat()
    w2, p2 = _filters(day, None, store_id, camera_id, exclude_rejected=True)
    w2 += " AND substr(first_seen, 1, 10) = ?" if w2 else " WHERE substr(first_seen, 1, 10) = ?"
    hours = {int(r["h"]): r["events"] for r in q(
        f"SELECT CAST(substr(first_seen, 12, 2) AS INTEGER) AS h, COUNT(*) AS events FROM litter_events{w2} "
        "GROUP BY h", (*p2, day))}
    aw, ap = [], []
    for col, v in (("store_id", store_id), ("camera_id", camera_id)):
        if v:
            aw.append(f"{col} = ?")
            ap.append(v)
    awhere = (" WHERE " + " AND ".join(aw)) if aw else ""
    processing = {r["status"]: r["n"] for r in q(f"SELECT status, COUNT(*) AS n FROM analyses{awhere} GROUP BY status",
                                                 ap)}
    return {
        "totals": {"unique_events": tot["events"], "frame_observations": tot["obs"],
                   "events_today": sum(hours.values()),
                   "avg_visible_sec": round(tot["avg_visible"], 1) if tot["avg_visible"] is not None else None,
                   "analyses": sum(processing.values()), "analyses_failed": processing.get("FAILED", 0)},
        "most_frequent_class": by_class[0]["class"] if by_class else None,
        "by_class": by_class, "by_store": by_store, "by_camera": by_camera,
        "series": list(series.values()), "granularity": granularity,
        "hourly_today": [{"hour": h, "events": hours.get(h, 0)} for h in range(24)], "today": day,
        "processing": processing,
    }
