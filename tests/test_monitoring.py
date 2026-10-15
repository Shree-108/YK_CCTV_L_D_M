"""Monitoring report (monitoring/ldm_monitor.py) on a synthetic database: read-only, no metrics from unreviewed data."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from backend.database.db import Database
from monitoring import ldm_monitor as mon


def _now(minutes: int = 0) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S.000+00:00")


def _db(tmp: Path, statuses: list[str], cam: str = "cam1") -> Path:
    import sqlite3
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "m.db"
    db = Database(p)                      # creates the app's real schema
    if hasattr(db, "close"):
        db.close()
    c = sqlite3.connect(p)
    for i, st in enumerate(statuses):
        c.execute("INSERT INTO alerts (ts, camera_id, class, confidence, severity, status, last_seen_ts, model_version) "
                  "VALUES (?,?,?,?,?,?,?,?)", (_now(i), cam, "food_wrapper", 0.8, "MEDIUM", st, _now(i), "v3_store_cctv"))
    c.execute("INSERT INTO detections (ts, camera_id, source_type, class, confidence, status, model_version) "
              "VALUES (?,?,?,?,?,?,?)", (_now(), cam, "video", "food_wrapper", 0.91, "confirmed", "v3_store_cctv"))
    stats = {"stats": {"mean_inference_ms": 150.0, "wall_time_sec": 12.0, "processing_fps": 9.0}}
    c.execute("INSERT INTO jobs (id, kind, status, camera_id, created_at, result_json) VALUES (?,?,?,?,?,?)",
              ("j1", "video", "done", cam, _now(), json.dumps(stats)))
    c.execute("INSERT INTO jobs (id, kind, status, camera_id, created_at, error) VALUES (?,?,?,?,?,?)",
              ("j2", "video", "failed", cam, _now(), "Could not open video"))
    c.execute("INSERT INTO reviews (created_at, verdict) VALUES (?, 'false_negative')", (_now(),))
    c.commit()
    c.close()
    return p


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_wilson_interval():
    assert mon.wilson(0, 0) is None
    lo, hi = mon.wilson(18, 20)
    assert lo < 0.9 < hi and 0.65 < lo and hi <= 1.0


def test_unreviewed_alerts_never_produce_precision(tmp_path: Path):
    c = mon.open_ro(_db(tmp_path, ["new"] * 30))
    q = mon.quality(c, "2000-01-01", 20, tmp_path / "evals")
    assert q["alert_review"]["decided"] == 0
    assert q["alert_review"]["overall"]["precision"] is None
    assert "only 0 decided" in q["alert_review"]["overall"]["why_not"]


def test_precision_only_from_decided_alerts_with_minimum_n(tmp_path: Path):
    few = mon.open_ro(_db(tmp_path, ["resolved"] * 5 + ["false_positive"] * 2 + ["acknowledged"] * 10))
    q = mon.quality(few, "2000-01-01", 20, tmp_path / "evals")["alert_review"]
    assert (q["decided"], q["acknowledged_only"], q["overall"]["precision"]) == (7, 10, None)
    enough = mon.open_ro(_db(tmp_path / "b", ["resolved"] * 18 + ["false_positive"] * 6 + ["new"] * 50))
    q = mon.quality(enough, "2000-01-01", 20, tmp_path / "evals")["alert_review"]
    assert q["overall"]["precision"] == 0.75 and q["overall"]["false_positive"] == 6
    assert q["overall"]["ci95"][0] < 0.75 < q["overall"]["ci95"][1]
    assert q["by_camera"]["cam1"]["decided"] == 24


def test_operational_latency_failures_and_missed_litter(tmp_path: Path):
    c = mon.open_ro(_db(tmp_path, []))
    op = mon.operational(c, "2000-01-01", tmp_path / "no_ingest.db")
    assert op["video_jobs"]["by_status"] == {"done": 1, "failed": 1}
    assert op["video_jobs"]["mean_inference_ms"]["median"] == 150.0
    assert op["video_jobs"]["failed"][0]["error"] == "Could not open video"
    assert mon.quality(c, "2000-01-01", 20, tmp_path)["test_lab_reviews"]["missed_litter_reported"] == 1


def test_retraining_rule_needs_reviewed_need_and_data(tmp_path: Path):
    c = mon.open_ro(_db(tmp_path, ["resolved"] * 10 + ["false_positive"] * 30))     # precision 0.25 on 40
    q = mon.quality(c, "2000-01-01", 20, tmp_path)
    d = mon.retrain_decision(q, {"datasets": []}, 20, 0.8)
    assert d["verdict"].startswith("RETRAINING MAY BE JUSTIFIED") and d["need_evidence"] and d["data_evidence"]
    c2 = mon.open_ro(_db(tmp_path / "unreviewed", ["new"] * 100))
    d2 = mon.retrain_decision(mon.quality(c2, "2000-01-01", 20, tmp_path / "none"), {"datasets": []}, 20, 0.8)
    assert d2["verdict"].startswith("NOT ENOUGH REVIEWED EVIDENCE")


def test_full_report_is_read_only(tmp_path: Path, monkeypatch):
    db = _db(tmp_path, ["resolved"] * 3 + ["new"] * 3)
    before = _sha(db)
    out = tmp_path / "report"
    assert mon.main(["--db", str(db), "--out", str(out), "--ingest-dir", str(tmp_path / "ingest")]) == 0
    assert _sha(db) == before                                              # database untouched
    m = json.loads((out / "metrics.json").read_text())
    assert m["activity"]["note"].startswith("UNREVIEWED")
    assert m["quality"]["alert_review"]["overall"]["precision"] is None   # 3 decided < 20
    assert (out / "report.md").read_text().count("not reported") >= 1
    with pytest.raises(Exception):
        mon.open_ro(db).execute("INSERT INTO reviews (created_at, verdict) VALUES ('x','uncertain')")
