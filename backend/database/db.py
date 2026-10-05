"""SQLite persistence (WAL) with a small repository API.

PostgreSQL migration notes: the schema uses only portable types (INTEGER, REAL,
TEXT); timestamps are ISO-8601 UTC strings; JSON is stored as TEXT (map to
JSONB); ``INTEGER PRIMARY KEY`` maps to ``BIGSERIAL``. Parameter placeholders
(``?``) are the only SQLite-specific syntax and live solely in this module.
Large binaries (videos, snapshots) are never stored here - only file paths.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL,              -- rtsp://..., http(s)://..., file path, webcam index, or env:VAR_NAME
    location TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    roi_json TEXT NOT NULL DEFAULT '[]',
    settings_json TEXT NOT NULL DEFAULT '{}',
    notify_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    source_type TEXT NOT NULL,         -- image | video | live
    job_id TEXT,
    frame_idx INTEGER,
    video_time_sec REAL,
    class TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    status TEXT NOT NULL,              -- rejected | weak | candidate | confirmed
    reason TEXT,
    track_id INTEGER,
    model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_det_ts ON detections(ts);
CREATE INDEX IF NOT EXISTS idx_det_cam_ts ON detections(camera_id, ts);
CREATE INDEX IF NOT EXISTS idx_det_job ON detections(job_id);

CREATE TABLE IF NOT EXISTS confirmed_objects (
    id INTEGER PRIMARY KEY,
    camera_id TEXT NOT NULL,
    job_id TEXT,
    session_id TEXT NOT NULL,          -- pipeline run (track ids are unique per session)
    track_id INTEGER NOT NULL,
    class TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    first_seen_ts TEXT NOT NULL,
    confirmed_ts TEXT NOT NULL,
    last_seen_ts TEXT NOT NULL,
    video_time_sec REAL,
    hits INTEGER NOT NULL DEFAULT 0,
    alert_id INTEGER,
    state TEXT NOT NULL DEFAULT 'active'   -- active | resolved
);
CREATE INDEX IF NOT EXISTS idx_conf_cam ON confirmed_objects(camera_id, confirmed_ts);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    class TEXT NOT NULL,
    confidence REAL NOT NULL,
    track_id INTEGER,
    session_id TEXT,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    location TEXT,
    snapshot_path TEXT,
    crop_path TEXT,
    video_time_sec REAL,
    severity TEXT NOT NULL,            -- LOW | MEDIUM | HIGH
    status TEXT NOT NULL DEFAULT 'new',-- new | acknowledged | resolved | false_positive
    occurrences INTEGER NOT NULL DEFAULT 1,
    last_seen_ts TEXT NOT NULL,
    job_id TEXT,
    model_version TEXT,
    source_type TEXT,
    notified TEXT
);
CREATE INDEX IF NOT EXISTS idx_alert_ts ON alerts(ts);
CREATE INDEX IF NOT EXISTS idx_alert_cam_ts ON alerts(camera_id, ts);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    source TEXT NOT NULL,
    camera_id TEXT,
    message TEXT NOT NULL,
    data_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_sys_ts ON system_events(ts);

CREATE TABLE IF NOT EXISTS models (
    version TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    trained_at TEXT,
    classes_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    active INTEGER NOT NULL DEFAULT 0,
    registered_at TEXT NOT NULL,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS hard_examples (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,                -- false_positive | false_negative | tiny | low_confidence
    source TEXT NOT NULL,              -- evaluation:<id> | alert:<id> | manual
    dataset TEXT, split TEXT, image TEXT,
    class TEXT, confidence REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    model_version TEXT,
    review_status TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | rejected | ignored
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_hard_kind ON hard_examples(kind, review_status);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    dataset TEXT, split TEXT, image TEXT,
    model_version TEXT,
    verdict TEXT NOT NULL,             -- true_positive | false_positive | false_negative | uncertain
    class TEXT, confidence REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS captures (
    id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    trigger TEXT NOT NULL,             -- scheduled | manual | recovery
    task TEXT NOT NULL,                -- litter | uniform
    status TEXT NOT NULL,              -- running | done | skipped | failed | interrupted
    model_version TEXT,
    error TEXT,
    counts_json TEXT,
    detections_json TEXT,
    alert_ids_json TEXT,
    original_path TEXT,
    annotated_path TEXT,
    duration_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_captures_cam_ts ON captures(camera_id, ts);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,              -- queued | running | done | failed
    camera_id TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    input_name TEXT,
    output_dir TEXT,
    progress REAL NOT NULL DEFAULT 0,
    result_json TEXT,
    error TEXT
);
"""

ALERT_FILTERS = {"camera_id", "class", "severity", "status"}


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _row(r: sqlite3.Row | None) -> Optional[dict[str, Any]]:
    return dict(r) if r is not None else None


class Database:
    """Thread-safe SQLite wrapper. One connection per call; writes serialized by a lock."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wlock = threading.RLock()
        with self.connect() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)
            self._migrate(c)

    @staticmethod
    def _migrate(c: sqlite3.Connection) -> None:
        """Additive, idempotent column migrations for databases created by older versions."""
        wanted = {"alerts": {"notes": "TEXT", "updated_at": "TEXT"},
                  "cameras": {"latitude": "REAL", "longitude": "REAL", "rtsp_json": "TEXT"},
                  "detections": {"session_id": "TEXT"}}
        for table, cols in wanted.items():
            have = {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
            for col, typ in cols.items():
                if col not in have:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        c.execute("CREATE INDEX IF NOT EXISTS idx_det_track ON detections(camera_id, session_id, track_id)")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _write(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._wlock, self.connect() as c:
            return c.execute(sql, tuple(params))

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self.connect() as c:
            return [dict(r) for r in c.execute(sql, tuple(params)).fetchall()]

    def _one(self, sql: str, params: Iterable[Any] = ()) -> Optional[dict[str, Any]]:
        with self.connect() as c:
            return _row(c.execute(sql, tuple(params)).fetchone())

    # ------------------------------------------------------------------ settings
    def get_setting(self, key: str, default: Any = None) -> Any:
        r = self._one("SELECT value_json FROM settings WHERE key=?", (key,))
        return json.loads(r["value_json"]) if r else default

    def set_setting(self, key: str, value: Any) -> None:
        self._write("INSERT INTO settings(key, value_json, updated_at) VALUES (?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
                    (key, json.dumps(value), utcnow()))

    # ------------------------------------------------------------------ cameras
    @staticmethod
    def _cam(r: dict[str, Any]) -> dict[str, Any]:
        r["enabled"] = bool(r["enabled"])
        for k in ("roi_json", "settings_json", "notify_json", "rtsp_json"):
            r[k.removesuffix("_json")] = json.loads(r.pop(k, None) or ("[]" if k == "roi_json" else "{}"))
        return r

    def list_cameras(self) -> list[dict[str, Any]]:
        return [self._cam(r) for r in self._query("SELECT * FROM cameras ORDER BY id")]

    def get_camera(self, cam_id: str) -> Optional[dict[str, Any]]:
        r = self._one("SELECT * FROM cameras WHERE id=?", (cam_id,))
        return self._cam(r) if r else None

    def upsert_camera(self, cam: dict[str, Any]) -> dict[str, Any]:
        now = utcnow()
        existing = self.get_camera(cam["id"])
        self._write(
            "INSERT INTO cameras(id,name,source,location,enabled,roi_json,settings_json,notify_json,created_at,updated_at,"
            "latitude,longitude,rtsp_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,"
            " source=excluded.source, location=excluded.location, enabled=excluded.enabled,"
            " roi_json=excluded.roi_json, settings_json=excluded.settings_json, notify_json=excluded.notify_json,"
            " updated_at=excluded.updated_at, latitude=excluded.latitude, longitude=excluded.longitude, rtsp_json=excluded.rtsp_json",
            (cam["id"], cam["name"], cam["source"], cam.get("location", ""), int(cam.get("enabled", True)),
             json.dumps(cam.get("roi", [])), json.dumps(cam.get("settings", {})), json.dumps(cam.get("notify", {})),
             existing["created_at"] if existing else now, now, cam.get("latitude"), cam.get("longitude"),
             json.dumps(cam.get("rtsp") or {})))
        return self.get_camera(cam["id"])  # type: ignore[return-value]

    def delete_camera(self, cam_id: str) -> bool:
        return self._write("DELETE FROM cameras WHERE id=?", (cam_id,)).rowcount > 0

    # ------------------------------------------------------------------ detections
    def insert_detections(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        cols = ["ts", "camera_id", "source_type", "job_id", "frame_idx", "video_time_sec", "class", "confidence",
                "x1", "y1", "x2", "y2", "status", "reason", "track_id", "model_version", "session_id"]
        with self._wlock, self.connect() as c:
            c.executemany(f"INSERT INTO detections({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                          [tuple(r.get(k) for k in cols) for r in rows])

    def list_detections(self, limit: int = 100, offset: int = 0, camera_id: str | None = None,
                        status: str | None = None, job_id: str | None = None, since: str | None = None,
                        until: str | None = None, cls: str | None = None, model_version: str | None = None,
                        min_confidence: float | None = None) -> list[dict[str, Any]]:
        where, p = self._detection_where(camera_id, status, job_id, since, until, cls, model_version, min_confidence)
        return self._query(f"SELECT * FROM detections {where} ORDER BY id DESC LIMIT ? OFFSET ?", [*p, limit, offset])

    def count_detections(self, *args: Any) -> int:
        where, p = self._detection_where(*args)
        return int(self._one(f"SELECT COUNT(*) AS n FROM detections {where}", p)["n"])  # type: ignore[index]

    def _detection_where(self, camera_id: str | None = None, status: str | None = None, job_id: str | None = None,
                         since: str | None = None, until: str | None = None, cls: str | None = None,
                         model_version: str | None = None, min_confidence: float | None = None) -> tuple[str, list[Any]]:
        where, p = self._where(camera_id=camera_id, status=status, job_id=job_id, since=since, until=until,
                               model_version=model_version, **{"class": cls})
        if min_confidence is not None:
            where += (" AND " if where else "WHERE ") + "confidence>=?"
            p.append(min_confidence)
        return where, p

    # ------------------------------------------------------------------ confirmed objects
    def insert_confirmed(self, row: dict[str, Any]) -> int:
        cols = list(row)
        return int(self._write(f"INSERT INTO confirmed_objects({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                               row.values()).lastrowid)

    def update_confirmed(self, obj_id: int, **fields: Any) -> None:
        if fields:
            self._write(f"UPDATE confirmed_objects SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                        [*fields.values(), obj_id])

    # ------------------------------------------------------------------ alerts
    def insert_alert(self, row: dict[str, Any]) -> int:
        cols = list(row)
        return int(self._write(f"INSERT INTO alerts({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                               row.values()).lastrowid)

    def update_alert(self, alert_id: int, **fields: Any) -> bool:
        if not fields:
            return False
        return self._write(f"UPDATE alerts SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                           [*fields.values(), alert_id]).rowcount > 0

    def get_alert(self, alert_id: int) -> Optional[dict[str, Any]]:
        return self._one("SELECT * FROM alerts WHERE id=?", (alert_id,))

    def recent_alerts_for_dedup(self, camera_id: str, since: str) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM alerts WHERE camera_id=? AND last_seen_ts>=? AND status!='false_positive'"
                           " ORDER BY id DESC LIMIT 200", (camera_id, since))

    def list_alerts(self, limit: int = 50, offset: int = 0, since: str | None = None, until: str | None = None,
                    min_confidence: float | None = None, **filters: Any) -> tuple[list[dict[str, Any]], int]:
        eq = {k: v for k, v in filters.items() if k in ALERT_FILTERS and v not in (None, "")}
        where, p = self._where(since=since, until=until, **eq)
        if min_confidence is not None:
            where += (" AND " if where else "WHERE ") + "confidence>=?"
            p.append(min_confidence)
        total = self._one(f"SELECT COUNT(*) AS n FROM alerts {where}", p)["n"]  # type: ignore[index]
        rows = self._query(f"SELECT * FROM alerts {where} ORDER BY id DESC LIMIT ? OFFSET ?", [*p, limit, offset])
        return rows, int(total)

    # ------------------------------------------------------------------ system events
    def log_event(self, level: str, source: str, message: str, camera_id: str | None = None,
                  data: Any = None) -> None:
        self._write("INSERT INTO system_events(ts,level,source,camera_id,message,data_json) VALUES (?,?,?,?,?,?)",
                    (utcnow(), level, source, camera_id, message, json.dumps(data) if data is not None else None))

    def list_events(self, limit: int = 100, camera_id: str | None = None) -> list[dict[str, Any]]:
        where, p = self._where(camera_id=camera_id)
        return self._query(f"SELECT * FROM system_events {where} ORDER BY id DESC LIMIT ?", [*p, limit])

    # ------------------------------------------------------------------ models
    def upsert_model(self, m: dict[str, Any]) -> None:
        existing = self._one("SELECT active, registered_at FROM models WHERE version=?", (m["version"],))
        self._write(
            "INSERT INTO models(version,name,path,trained_at,classes_json,metrics_json,active,registered_at,notes)"
            " VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(version) DO UPDATE SET name=excluded.name, path=excluded.path,"
            " trained_at=excluded.trained_at, classes_json=excluded.classes_json, metrics_json=excluded.metrics_json,"
            " notes=excluded.notes",
            (m["version"], m["name"], m["path"], m.get("trained_at"), json.dumps(m.get("classes", [])),
             json.dumps(m.get("metrics", {})), existing["active"] if existing else 0,
             existing["registered_at"] if existing else utcnow(), m.get("notes")))

    def list_models(self) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM models ORDER BY registered_at")
        for r in rows:
            r["classes"] = json.loads(r.pop("classes_json"))
            r["metrics"] = json.loads(r.pop("metrics_json"))
            r["active"] = bool(r["active"])
        return rows

    def set_active_model(self, version: str) -> bool:
        with self._wlock, self.connect() as c:
            if not c.execute("SELECT 1 FROM models WHERE version=?", (version,)).fetchone():
                return False
            c.execute("UPDATE models SET active=CASE WHEN version=? THEN 1 ELSE 0 END", (version,))
        return True

    def active_model(self) -> Optional[dict[str, Any]]:
        return next((m for m in self.list_models() if m["active"]), None)

    # ------------------------------------------------------------------ jobs
    def create_job(self, job_id: str, kind: str, camera_id: str | None, input_name: str, output_dir: str) -> None:
        self._write("INSERT INTO jobs(id,kind,status,camera_id,created_at,input_name,output_dir) VALUES (?,?,?,?,?,?,?)",
                    (job_id, kind, "queued", camera_id, utcnow(), input_name, output_dir))

    # ------------------------------------------------------------------ scheduled captures
    def insert_capture(self, row: dict[str, Any]) -> None:
        cols = list(row)
        self._write(f"INSERT INTO captures({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", [row[c] for c in cols])

    def update_capture(self, cap_id: str, **fields: Any) -> None:
        for k in ("counts", "detections", "alert_ids"):
            if k in fields:
                fields[f"{k}_json"] = json.dumps(fields.pop(k))
        self._write(f"UPDATE captures SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?", [*fields.values(), cap_id])

    @staticmethod
    def _capture(r: dict[str, Any]) -> dict[str, Any]:
        for k in ("counts_json", "detections_json", "alert_ids_json"):
            r[k.removesuffix("_json")] = json.loads(r.pop(k) or ("{}" if k == "counts_json" else "[]"))
        return r

    def get_capture(self, cap_id: str) -> Optional[dict[str, Any]]:
        r = self._one("SELECT * FROM captures WHERE id=?", (cap_id,))
        return self._capture(r) if r else None

    def list_captures(self, camera_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        where, params = ("WHERE camera_id=?", [camera_id]) if camera_id else ("", [])
        return [self._capture(r) for r in self._query(
            f"SELECT * FROM captures {where} ORDER BY ts DESC LIMIT ? OFFSET ?", [*params, limit, offset])]

    def update_job(self, job_id: str, **fields: Any) -> None:
        if "result" in fields:
            fields["result_json"] = json.dumps(fields.pop("result"))
        self._write(f"UPDATE jobs SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?", [*fields.values(), job_id])

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        r = self._one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if r:
            r["result"] = json.loads(r.pop("result_json") or "null")
        return r

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._query("SELECT id,kind,status,camera_id,created_at,finished_at,input_name,progress,error "
                           "FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
        return rows

    def track_timeline(self, camera_id: str, session_id: str, track_id: int, limit: int = 500) -> list[dict[str, Any]]:
        return self._query("SELECT ts, frame_idx, video_time_sec, class, confidence, status, x1, y1, x2, y2 "
                           "FROM detections WHERE camera_id=? AND session_id=? AND track_id=? ORDER BY id LIMIT ?",
                           (camera_id, session_id, track_id, limit))

    def export_rows(self, table: str, since: str | None, until: str | None, camera_id: str | None,
                    limit: int = 100_000) -> list[dict[str, Any]]:
        if table not in ("alerts", "detections"):
            raise ValueError("unknown table")
        where, p = self._where(since=since, until=until, camera_id=camera_id)
        return self._query(f"SELECT * FROM {table} {where} ORDER BY id DESC LIMIT ?", [*p, limit])

    # ------------------------------------------------------------------ hard examples / reviews
    def insert_hard_examples(self, rows: list[dict[str, Any]]) -> None:
        cols = ["created_at", "kind", "source", "dataset", "split", "image", "class", "confidence",
                "x1", "y1", "x2", "y2", "model_version"]
        with self._wlock, self.connect() as c:
            c.executemany(f"INSERT INTO hard_examples({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                          [tuple(r.get(k) for k in cols) for r in rows])

    def list_hard_examples(self, kind: str | None, status: str | None, limit: int,
                           offset: int) -> tuple[list[dict[str, Any]], int]:
        where, p = self._where(kind=kind, review_status=status)
        total = self._one(f"SELECT COUNT(*) AS n FROM hard_examples {where}", p)["n"]  # type: ignore[index]
        rows = self._query(f"SELECT * FROM hard_examples {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                           [*p, limit, offset])
        return rows, int(total)

    def hard_example_counts(self) -> list[dict[str, Any]]:
        return self._query("SELECT kind, review_status, COUNT(*) AS n FROM hard_examples GROUP BY kind, review_status")

    def get_hard_example(self, hid: int) -> Optional[dict[str, Any]]:
        return self._one("SELECT * FROM hard_examples WHERE id=?", (hid,))

    def update_hard_example(self, hid: int, **fields: Any) -> bool:
        return self._write(f"UPDATE hard_examples SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                           [*fields.values(), hid]).rowcount > 0

    def insert_review(self, row: dict[str, Any]) -> int:
        cols = list(row)
        return int(self._write(f"INSERT INTO reviews({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                               row.values()).lastrowid)

    def list_reviews(self, dataset: str, image: str) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM reviews WHERE dataset=? AND image=? ORDER BY id", (dataset, image))

    def db_size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.parent.glob(self.path.name + "*") if p.is_file())

    # ------------------------------------------------------------------ analytics
    def analytics(self, since: str | None, until: str | None, camera_id: str | None = None,
                  cls: str | None = None, model_version: str | None = None) -> dict[str, Any]:
        eq = {"camera_id": camera_id, "class": cls, "model_version": model_version}
        aw, ap = self._where(since=since, until=until, **eq)
        dw, dp = self._where(since=since, until=until, **eq)
        q = self._query
        cw, cp = self._where(since=since, until=until, camera_id=camera_id, ts_col="confirmed_ts", **{"class": cls})
        confirmed = q(f"SELECT COUNT(*) n FROM confirmed_objects {cw}", cp)[0]["n"]
        return {
            "alerts_total": q(f"SELECT COUNT(*) n FROM alerts {aw}", ap)[0]["n"],
            "confirmed_objects": confirmed,
            "by_camera": q(f"SELECT camera_id, COUNT(*) n FROM alerts {aw} GROUP BY camera_id ORDER BY n DESC", ap),
            "by_class": q(f"SELECT class, COUNT(*) n FROM alerts {aw} GROUP BY class ORDER BY n DESC", ap),
            "by_severity": q(f"SELECT severity, COUNT(*) n FROM alerts {aw} GROUP BY severity", ap),
            "by_status": q(f"SELECT status, COUNT(*) n FROM alerts {aw} GROUP BY status", ap),
            "confirmed_by_day": q(f"SELECT substr(confirmed_ts,1,10) AS day, COUNT(*) n FROM confirmed_objects {cw} "
                                  f"GROUP BY day ORDER BY day", cp),
            "detections_by_day": q(f"SELECT substr(ts,1,10) AS day, status, COUNT(*) n FROM detections {dw} "
                                   f"GROUP BY day, status ORDER BY day", dp),
            "by_hour": q(f"SELECT substr(ts,12,2) AS hour, COUNT(*) n FROM alerts {aw} GROUP BY hour ORDER BY hour", ap),
            "by_day": q(f"SELECT substr(ts,1,10) AS day, COUNT(*) n FROM alerts {aw} GROUP BY day ORDER BY day", ap),
            "detections_by_status": q(f"SELECT status, COUNT(*) n FROM detections {dw} GROUP BY status", dp),
            "rejection_reasons": q(f"SELECT reason, COUNT(*) n FROM detections {dw}{' AND' if dw else ' WHERE'} "
                                   f"status='rejected' GROUP BY reason ORDER BY n DESC", dp),
            "confidence_histogram": q(
                f"SELECT CAST(confidence*10 AS INTEGER)/10.0 AS bin, COUNT(*) n FROM alerts {aw} GROUP BY bin ORDER BY bin",
                ap),
        }

    @staticmethod
    def _where(since: str | None = None, until: str | None = None, ts_col: str = "ts",
               **eq: Any) -> tuple[str, list[Any]]:
        parts, params = [], []
        for k, v in eq.items():
            if v is not None:
                parts.append(f'"{k}"=?')
                params.append(v)
        if since:
            parts.append(f"{ts_col}>=?")
            params.append(since)
        if until:
            parts.append(f"{ts_col}<=?")
            params.append(until)
        return ("WHERE " + " AND ".join(parts)) if parts else "", params
