"""History reset tool (backend/services/history_reset.py): backup first, history cleared, everything else untouched.

Every test works on a throw-away project root in tmp_path (own database, storage, models, config, ledger and a copy of
deploy/backup-ldm.ps1); the real database and storage are never opened.
"""
from __future__ import annotations

import http.server
import json
import shutil
import sqlite3
import threading
from pathlib import Path

import pytest

from backend.core.config import PROJECT_ROOT
from backend.database.db import Database
from backend.services import history_reset as hr

HAS_PS = shutil.which("powershell") is not None
needs_ps = pytest.mark.skipif(not HAS_PS, reason="deploy/backup-ldm.ps1 needs Windows PowerShell")
CLOSED = ["http://127.0.0.1:9"]          # nothing listens on the discard port


def _project(tmp: Path) -> hr.Paths:
    root = tmp / "proj"
    (root / "deploy").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "deploy" / "backup-ldm.ps1", root / "deploy" / "backup-ldm.ps1")
    weights = root / "models" / "litter_yolo" / "vT" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights" * 100)
    (root / "dataset" / "d1").mkdir(parents=True)
    (root / "dataset" / "d1" / "a.txt").write_text("0 0.5 0.5 0.1 0.1")
    (root / "config").mkdir()
    (root / "config" / "store_config.json").write_text('{"stores": []}')
    (root / "storage" / "exports").mkdir(parents=True)
    (root / "storage" / "exports" / "ai_garbage_detection_ledger.db").write_bytes(b"ledger")
    paths = hr.Paths.for_root(root, tmp / "backups")
    db = Database(paths.db)
    with db.connect() as c:
        c.execute("INSERT INTO models(version,name,path,classes_json,active,registered_at) VALUES (?,?,?,?,1,?)",
                  ("vT", "test", str(weights), '["paper"]', "2026-11-01"))
        c.execute("INSERT INTO settings(key,value_json,updated_at) VALUES ('k','1','2026-11-01')")
        c.execute("INSERT INTO alerts(ts,camera_id,class,confidence,severity,last_seen_ts,snapshot_path) "
                  "VALUES ('t','CAM','paper',0.9,'LOW','t','x')")
        c.execute("INSERT INTO detections(ts,camera_id,source_type,class,confidence,status) "
                  "VALUES ('t','CAM','image','paper',0.9,'confirmed')")
        c.execute("INSERT INTO jobs(id,kind,status,created_at) VALUES ('j1','video','done','t')")
        c.execute("INSERT INTO sampled_jobs(id,media_id,created_at,stage,state,seed,sample_count,roi_mode,output_dir) "
                  "VALUES ('s1','m1','t','done','done',1,4,'none','o')")
        c.execute("INSERT INTO system_events(ts,level,source,message) VALUES ('t','info','x','m')")
        c.execute("INSERT INTO captures(id,camera_id,ts,trigger,task,status,original_path) "
                  "VALUES ('c1','CAM','t','scheduled','litter','done','captures/CAM/c1/original.jpg')")
    for d, f in (("alerts/2026-11-21", "a.jpg"), ("videos/v1", "out.mp4"), ("media", "m.mp4"),
                 ("sampled/s1", "frame_1.jpg"), ("LDM_OUTPUT/S/2026/11", "r.csv"),
                 ("analysis_outputs/analysis_old", "x.json"), ("proposals/roi", "p.jpg"),
                 ("captures/CAM/c1", "original.jpg")):
        (paths.storage / d).mkdir(parents=True, exist_ok=True)
        (paths.storage / d / f).write_bytes(b"data-" + f.encode())
    (paths.storage / "eval").mkdir()
    (paths.storage / "eval" / "keep.json").write_text("{}")
    return paths


def _counts(db: Path) -> dict[str, int]:
    con = sqlite3.connect(db)
    try:
        return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in hr.HISTORY_TABLES + ("models", "settings")}
    finally:
        con.close()


@needs_ps
def test_run_backs_up_then_clears_history_and_keeps_everything_else(tmp_path):
    p = _project(tmp_path)
    before = hr.snapshot(p)
    assert before["history_rows"] == 6
    res = hr.run(p, phrase=hr.PHRASE, health_urls=CLOSED)

    # backup made first and verified: checksums, and the backed-up DB still holds the history
    bdir = Path(res["backup"]["path"])
    assert (bdir / "SHA256SUMS.txt").is_file() and res["backup"]["checksums_verified"] == res["backup"]["files"]
    assert _counts(bdir / "db" / "litter_monitor.db")["alerts"] == 1
    assert (bdir / "storage" / "videos" / "v1" / "out.mp4").is_file()          # -IncludeVideos

    # history gone, protected rows/files untouched
    c = _counts(p.db)
    assert all(c[t] == 0 for t in hr.HISTORY_TABLES)
    assert c["models"] == 1 and c["settings"] == 1
    after = hr.snapshot(p)
    for k in ("active_model", "store_config", "ledger", "models_rows", "protected_tables", "protected_dirs"):
        assert after[k] == before[k], k
    assert (p.storage / "eval" / "keep.json").is_file()

    # files moved (not deleted) with a manifest; empty folders left in place
    q = Path(res["quarantine"])
    man = json.loads((q / "MANIFEST.json").read_text(encoding="utf-8"))
    assert man["file_count"] == 8 and set(man["folders"]) == set(hr.HISTORY_DIRS)
    assert (q / "captures" / "CAM" / "c1" / "original.jpg").is_file()
    assert all((q / f["quarantine"]).is_file() for f in man["files"])
    assert all((p.storage / d).is_dir() and not any((p.storage / d).iterdir()) for d in hr.HISTORY_DIRS)
    assert json.loads((q / "db_export" / "alerts.json").read_text(encoding="utf-8"))["count"] == 1
    assert json.loads((q / "db_export" / "captures.json").read_text(encoding="utf-8"))["count"] == 1
    assert (bdir / "storage" / "captures" / "CAM" / "c1" / "original.jpg").is_file()   # backup-ldm.ps1 copies it
    assert (q / "before.json").is_file() and (q / "RESULT.json").is_file()
    assert hr.verify(p, q)["ok"]


@needs_ps
def test_failed_backup_changes_nothing(tmp_path):
    p = _project(tmp_path)
    p.backup_script.write_text("throw 'disk full'")
    with pytest.raises(hr.ResetError, match="backup failed"):
        hr.run(p, phrase=hr.PHRASE, health_urls=CLOSED)
    assert _counts(p.db)["alerts"] == 1 and (p.storage / "media" / "m.mp4").is_file()
    assert not (p.storage / hr.QUARANTINE).exists()


def test_wrong_phrase_changes_nothing(tmp_path):
    p = _project(tmp_path)
    with pytest.raises(hr.ResetError, match="phrase"):
        hr.run(p, phrase="clear history", health_urls=CLOSED)
    assert _counts(p.db)["alerts"] == 1


def test_refuses_while_app_running(tmp_path):
    p = _project(tmp_path)

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(hr.ResetError, match="still running"):
            hr.run(p, phrase=hr.PHRASE, health_urls=[f"http://127.0.0.1:{srv.server_port}"])
    finally:
        srv.shutdown()
    assert _counts(p.db)["alerts"] == 1


def test_refuses_while_a_job_is_running(tmp_path):
    p = _project(tmp_path)
    with sqlite3.connect(p.db) as c:
        c.execute("UPDATE sampled_jobs SET state='running'")
    assert hr.busy_jobs(p.db) == ["four-screenshot job s1 (running)"]
    with pytest.raises(hr.ResetError, match="queued/running"):
        hr.run(p, phrase=hr.PHRASE, health_urls=CLOSED)
    assert _counts(p.db)["alerts"] == 1


def test_delete_rolls_back_when_a_protected_table_would_change(tmp_path):
    p = _project(tmp_path)
    before = hr.snapshot(p)
    before["protected_tables"]["settings"]["sha256"] = "tampered"
    with pytest.raises(hr.ResetError, match="rolled back"):
        hr.delete_history(p.db, before)
    assert _counts(p.db)["alerts"] == 1 and _counts(p.db)["detections"] == 1


def test_verify_detects_a_changed_config(tmp_path):
    p = _project(tmp_path)
    q = tmp_path / "q"
    q.mkdir()
    (q / "before.json").write_text(json.dumps(hr.snapshot(p)), encoding="utf-8")
    (q / "MANIFEST.json").write_text('{"file_count": 0, "files": []}', encoding="utf-8")
    p.store_config.write_text('{"stores": ["changed"]}')
    res = hr.verify(p, q)
    assert not res["ok"] and any("store_config changed" in f for f in res["failures"])


def test_refuses_before_deleting_when_quarantine_paths_would_be_too_long(tmp_path, monkeypatch):
    p = _project(tmp_path)
    base = p.storage / "sampled"
    deep = base / ("d" * max(1, 240 - len(str(base)) - 1))        # fits now (<=247), too long once quarantined
    deep.mkdir(parents=True)
    monkeypatch.setattr(hr, "_long_paths_enabled", lambda: False)
    with pytest.raises(hr.ResetError, match="too long"):
        hr.check_path_lengths(p.storage, p.storage / hr.QUARANTINE / "history_20261121_000000")
    assert _counts(p.db)["alerts"] == 1


@needs_ps
def test_verify_allows_only_start_up_events_written_after_the_reset(tmp_path):
    p = _project(tmp_path)
    q = Path(hr.run(p, phrase=hr.PHRASE, health_urls=CLOSED)["quarantine"])
    db = Database(p.db)
    db.log_event("info", "system", "backend started")              # the restarted app logs this: allowed
    res = hr.verify(p, q)
    assert res["ok"] and len(res["new_since_reset"]) == 1
    db.log_event("warning", "sampled", "four-screenshot analysis failed")   # anything else is real history: fails
    assert not hr.verify(p, q)["ok"]


def test_captures_are_history_and_listed_in_the_dry_run(tmp_path):
    p = _project(tmp_path)
    st = hr.status(p, CLOSED)
    assert st["history_counts"]["captures"] == 1 and st["history_dirs"]["captures"]["files"] == 1
    assert "captures" not in hr.snapshot(p)["protected_tables"]
    assert {"models", "cameras", "stores", "settings"} <= set(hr.snapshot(p)["protected_tables"])
    hr.delete_history(p.db, hr.snapshot(p))
    assert _counts(p.db)["captures"] == 0 and _counts(p.db)["models"] == 1 and _counts(p.db)["settings"] == 1
