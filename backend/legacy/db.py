"""
db.py — SQLite persistence for litter detections and alerts.

Improvements over the original:
  * WAL journal mode (backend + dev-server can read/write concurrently).
  * Indexes on (ts), (camera_id, ts) and alerts(acknowledged, ts).
  * `method` column on detections (tracks which engine produced the row:
    "ml-classical-rf" vs legacy "yolo"), added via migration so the existing
    litter_events.db keeps working.
  * `analysis_runs` table — trainlittermaxaccuracy.py logs each training run
    here (engine, class count, val F1, artifact paths).
  * get_stats() also returns per-method breakdown.
"""
import sqlite3
import datetime as dt
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "litter_events.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(detections)").fetchall()}
    if "method" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN method TEXT DEFAULT 'unknown'")


def init_db() -> None:
    """Create tables if they don't exist (idempotent, migrates old DBs)."""
    with _conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                camera_id   TEXT    NOT NULL DEFAULT 'upload',
                filename    TEXT,
                class       TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                priority    TEXT    NOT NULL,
                bin         TEXT    NOT NULL,
                x1          REAL, y1 REAL, x2 REAL, y2 REAL,
                area        REAL,
                method      TEXT    NOT NULL DEFAULT 'unknown'
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                camera_id   TEXT    NOT NULL DEFAULT 'upload',
                class       TEXT    NOT NULL,
                priority    TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                filename    TEXT,
                acknowledged INTEGER NOT NULL DEFAULT 0,
                message     TEXT
            );

            CREATE TABLE IF NOT EXISTS analysis_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                engine      TEXT    NOT NULL,
                n_classes   INTEGER NOT NULL DEFAULT 0,
                train_crops INTEGER NOT NULL DEFAULT 0,
                val_f1      REAL,
                model_path  TEXT,
                config_path TEXT,
                notes       TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_det_ts ON detections(ts);
            CREATE INDEX IF NOT EXISTS idx_det_cam_ts ON detections(camera_id, ts);
            CREATE INDEX IF NOT EXISTS idx_alert_ack_ts ON alerts(acknowledged, ts);
        """
        )
        _migrate(conn)


def insert_detections(camera_id: str, filename: str, dets: list[dict],
                      method: str = "unknown") -> list[int]:
    """Persist a batch of detections and return their IDs."""
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = [
        (
            ts, camera_id, filename,
            d["class"], d["confidence"], d["priority"], d["bin"],
            *d.get("box", [None, None, None, None]),
            d.get("area"), method,
        )
        for d in dets
    ]

    ids: list[int] = []
    with _conn() as conn:
        _migrate(conn)
        for row in rows:
            cur = conn.execute(
                """
                INSERT INTO detections
                  (ts, camera_id, filename, class, confidence, priority, bin,
                   x1, y1, x2, y2, area, method)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
                row,
            )
            ids.append(cur.lastrowid)

        for d in dets:
            if d.get("priority") in ("critical", "high"):
                conn.execute(
                    """
                    INSERT INTO alerts (ts, camera_id, class, priority, confidence, filename, message)
                    VALUES (?,?,?,?,?,?,?)
                """,
                    (
                        ts, camera_id, d["class"], d["priority"], d["confidence"],
                        filename,
                        f"{d['priority'].upper()} litter detected: "
                        f"{d['class'].replace('_', ' ').title()} "
                        f"({d['confidence'] * 100:.0f}% confidence)",
                    ),
                )
    return ids


def log_analysis_run(engine: str, n_classes: int, train_crops: int,
                     val_f1: float | None, model_path: str,
                     config_path: str, notes: str = "") -> int:
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO analysis_runs
                 (ts, engine, n_classes, train_crops, val_f1, model_path, config_path, notes)
               VALUES (?,?,?,?,?,?,?,?)""",
            (ts, engine, n_classes, train_crops, val_f1, model_path, config_path, notes),
        )
        return cur.lastrowid


def get_recent_detections(limit: int = 50, camera_id: str = None) -> list[dict]:
    with _conn() as conn:
        if camera_id:
            rows = conn.execute(
                "SELECT * FROM detections WHERE camera_id=? ORDER BY ts DESC LIMIT ?",
                (camera_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM detections ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_alerts(limit: int = 100, include_acknowledged: bool = False) -> list[dict]:
    with _conn() as conn:
        if include_acknowledged:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE acknowledged=0 ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def acknowledge_alert(alert_id: int) -> bool:
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,)
        )
    return cur.rowcount > 0


def get_stats(hours: int = 24) -> dict[str, Any]:
    """Return aggregated stats for the dashboard."""
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)).isoformat()

    with _conn() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE ts >= ?", (since,)
        ).fetchone()[0]

        avg_conf = conn.execute(
            "SELECT AVG(confidence) FROM detections WHERE ts >= ?", (since,)
        ).fetchone()[0] or 0

        class_dist = conn.execute(
            """SELECT class, COUNT(*) as cnt FROM detections WHERE ts >= ?
               GROUP BY class ORDER BY cnt DESC""",
            (since,),
        ).fetchall()

        priority_dist = conn.execute(
            """SELECT priority, COUNT(*) as cnt FROM detections WHERE ts >= ?
               GROUP BY priority""",
            (since,),
        ).fetchall()

        try:
            method_dist = conn.execute(
                """SELECT method, COUNT(*) as cnt FROM detections WHERE ts >= ?
                   GROUP BY method""",
                (since,),
            ).fetchall()
            method_dist = [dict(r) for r in method_dist]
        except Exception:
            method_dist = []

        open_alerts = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE acknowledged=0"
        ).fetchone()[0]

        hourly = conn.execute(
            """SELECT strftime('%H', ts) as hour, COUNT(*) as cnt
               FROM detections WHERE ts >= ?
               GROUP BY hour ORDER BY hour""",
            (since,),
        ).fetchall()

    return {
        "total_detections": total,
        "avg_confidence": avg_conf,
        "open_alerts": open_alerts,
        "class_distribution": [dict(r) for r in class_dist],
        "priority_distribution": [dict(r) for r in priority_dist],
        "method_distribution": method_dist,
        "hourly_activity": [dict(r) for r in hourly],
        "period_hours": hours,
    }
