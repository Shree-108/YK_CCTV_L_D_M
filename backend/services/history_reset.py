"""Clear the dashboard history (alerts, jobs, results, events, detections) safely and repeatably.

What it does (``run``), stopping at the first failed check:
  1. preflight: the app must be stopped (no /api/health answers), the paths must exist, the typed phrase must match
  2. "before" snapshot: row counts of every table, a digest of every table that is NOT history (models, settings,
     cameras, stores, ...), SHA-256 of the active best.pt, config/store_config.json and the export ledger (if present),
     file count + size of models/ and dataset/
  3. backup with deploy/backup-ldm.ps1 -IncludeVideos, then verify the backup: every line of SHA256SUMS.txt re-hashed,
     every file listed, the backed-up database passes integrity_check and has the same history counts
  4. export every history table to JSON in the quarantine folder (storage/_quarantine/history_<timestamp>/db_export/)
  5. delete the history tables in ONE transaction (the database file is never replaced; non-history tables are
     re-checked inside the transaction and it rolls back if any changed), then VACUUM
  6. MOVE (never delete) the history folders to the quarantine folder, same layout, with MANIFEST.json listing every
     file; empty folders are left in their place
  7. offline verification (counts 0, digests/checksums unchanged, every moved file present) -> RESULT.json

``verify`` repeats the step-7 checks and, with --health-url, also checks the restarted app: /api/health reports the
same model version and every dashboard history endpoint is empty.

Nothing here writes a settings row, touches models/, dataset/, config/, .env or the export ledger.
Undo: restore storage/litter_monitor.db from the backup (app stopped) and move the quarantine folders back.

CLI:
  py -m backend.services.history_reset status                       # dry run: what would be cleared (default)
  py -m backend.services.history_reset run --confirm                # asks for the phrase CLEAR HISTORY
  py -m backend.services.history_reset verify --quarantine <dir> --health-url http://127.0.0.1:8001
Wrapper (stop app -> run -> start app -> verify): deploy/clear-history.ps1
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

PHRASE = "CLEAR HISTORY"
# Dashboard history: one row per alert / job / analysis / detection / event / scheduled camera capture. Everything
# else (models, settings, cameras, stores, roi_versions, hard_examples, reviews) is protected and must stay
# byte-identical.
HISTORY_TABLES = ("alerts", "detections", "confirmed_objects", "litter_events", "analyses",
                  "jobs", "sampled_media", "sampled_jobs", "sampled_frames", "system_events", "captures")
# Folders under storage/ that hold the files those rows point to (+ ROI proposal images and old analysis folders).
HISTORY_DIRS = ("alerts", "videos", "media", "sampled", "LDM_OUTPUT", "analysis_outputs", "proposals", "captures")
QUARANTINE = "_quarantine"
LIFECYCLE_MESSAGES = ("backend started", "backend stopped")     # written by backend/main.py on every start/stop
# Dashboard endpoints that must be empty after a reset: (path, check(json) -> problem or None)
DASHBOARD_CHECKS: tuple[tuple[str, Any], ...] = (
    ("/api/alerts", lambda j: None if j.get("alerts") == [] else f"{len(j.get('alerts') or [])} alert(s)"),
    ("/api/jobs", lambda j: None if j.get("jobs") == [] else f"{len(j.get('jobs') or [])} job(s)"),
    ("/api/sampled/jobs", lambda j: None if j.get("total") == 0 and j.get("analyses") == []
        else f"total={j.get('total')}"),
    ("/api/detections", lambda j: None if j.get("detections") == [] else "detections present"),
    ("/api/litter/events", lambda j: None if j.get("events") == [] else "litter events present"),
    ("/api/litter/analyses", lambda j: None if j.get("analyses") == [] else "analyses present"),
    ("/api/events", lambda j: None if j.get("events") == [] else "system events present"),
    ("/api/reports/daily", lambda j: None if j.get("days") == [] else f"{len(j.get('days') or [])} daily report(s)"),
    ("/api/summary", lambda j: None if j.get("new_alerts") == 0 and j.get("alerts_today") == 0
        else f"new_alerts={j.get('new_alerts')} alerts_today={j.get('alerts_today')}"),
    ("/api/analytics", lambda j: None if j.get("alerts_total") == 0 and j.get("confirmed_objects") == 0
        else f"alerts_total={j.get('alerts_total')}"),
    ("/api/litter/analytics", lambda j: None if (j.get("totals") or {}).get("unique_events") == 0
        and (j.get("totals") or {}).get("analyses") == 0 else f"totals={j.get('totals')}"),
)


class ResetError(RuntimeError):
    """A check failed: the reset stops here and reports what was (and was not) done."""


@dataclass
class Paths:
    root: Path                 # project root: deploy/backup-ldm.ps1 lives here and backs up <root>/storage/...
    db: Path
    storage: Path
    store_config: Path
    ledger: Path
    models: Path
    dataset: Path
    backup_dest: Path

    @property
    def backup_script(self) -> Path:
        return self.root / "deploy" / "backup-ldm.ps1"

    @classmethod
    def for_root(cls, root: Path, backup_dest: Optional[Path] = None) -> "Paths":
        root = Path(root).resolve()
        return cls(root, root / "storage" / "litter_monitor.db", root / "storage", root / "config" / "store_config.json",
                   root / "storage" / "exports" / "ai_garbage_detection_ledger.db", root / "models", root / "dataset",
                   Path(backup_dest) if backup_dest else Path.home() / "LDM_backups")

    @classmethod
    def from_config(cls, backup_dest: Optional[Path] = None) -> "Paths":
        from backend.core.config import PROJECT_ROOT, AppConfig
        cfg = AppConfig()
        return cls(PROJECT_ROOT, cfg.db_path, cfg.storage_dir, cfg.store_config_path, cfg.garbage_export_ledger,
                   cfg.models_dir, cfg.dataset_dir, Path(backup_dest) if backup_dest else Path.home() / "LDM_backups")


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)


def _tables(con: sqlite3.Connection) -> list[str]:
    return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                      "AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def _counts(con: sqlite3.Connection) -> dict[str, int]:
    return {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in _tables(con)}


def _table_digest(con: sqlite3.Connection, table: str) -> str:
    cur = con.execute(f'SELECT * FROM "{table}" ORDER BY 1')
    h = hashlib.sha256(json.dumps([d[0] for d in cur.description]).encode())
    for row in cur:
        h.update(json.dumps(list(row), default=str).encode())
    return h.hexdigest()


def _dir_stats(d: Path) -> dict[str, Any]:
    files = size = 0
    if d.is_dir():
        for p in d.rglob("*"):
            if p.is_file():
                files += 1
                size += p.stat().st_size
    return {"path": str(d), "exists": d.is_dir(), "files": files, "bytes": size}


def _file_sum(p: Path) -> dict[str, Any]:
    return {"path": str(p), "exists": p.is_file(), "sha256": sha256_file(p) if p.is_file() else None}


def snapshot(paths: Paths) -> dict[str, Any]:
    """Everything the reset must clear (counts) and everything it must not change (digests, checksums)."""
    con = _ro(paths.db)
    try:
        counts = _counts(con)
        missing = [t for t in HISTORY_TABLES if t not in counts]
        if missing:
            raise ResetError(f"history table(s) missing from {paths.db}: {missing}")
        protected = {t: {"rows": counts[t], "sha256": _table_digest(con, t)} for t in counts if t not in HISTORY_TABLES}
        cols = [d[1] for d in con.execute("PRAGMA table_info(models)")]
        models = [dict(zip(cols, r)) for r in con.execute("SELECT * FROM models ORDER BY version")]
    finally:
        con.close()
    active = next((m for m in models if m.get("active")), None)
    weights = Path(active["path"]) if active else None
    return {
        "taken_at": _now(),
        "db": str(paths.db),
        "counts": counts,
        "history_rows": sum(counts[t] for t in HISTORY_TABLES),
        "protected_tables": protected,
        "models_rows": models,
        "active_model": {"version": active["version"] if active else None,
                         **(_file_sum(weights) if weights else {"path": None, "exists": False, "sha256": None})},
        "store_config": _file_sum(paths.store_config),
        "ledger": _file_sum(paths.ledger),
        "protected_dirs": {"models": _dir_stats(paths.models), "dataset": _dir_stats(paths.dataset)},
        "history_dirs": {n: _dir_stats(paths.storage / n) for n in HISTORY_DIRS},
    }


# ================================================================================================ preflight
def default_health_urls(root: Path) -> list[str]:
    ports = {8000, 8001}
    run = root / "deploy" / "run" / "backend.json"
    try:
        ports.add(int(json.loads(run.read_text(encoding="utf-8-sig"))["port"]))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return [f"http://127.0.0.1:{p}" for p in sorted(ports)]


def _get_json(url: str, timeout: float = 3.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:      # noqa: S310 (local URLs only)
        return json.loads(r.read().decode("utf-8"))


def running_backends(urls: Sequence[str]) -> list[str]:
    alive = []
    for u in urls:
        try:
            _get_json(u.rstrip("/") + "/api/health", timeout=2.0)
            alive.append(u)
        except (urllib.error.URLError, OSError, ValueError):
            continue
    return alive


def busy_jobs(db: Path) -> list[str]:
    """Jobs or analyses still queued/running (read from the database, which the app keeps current)."""
    con = _ro(db)
    try:
        rows = [f"job {r[0]} ({r[1]})" for r in con.execute(
            "SELECT id, status FROM jobs WHERE status IN ('queued','running')")]
        rows += [f"four-screenshot job {r[0]} ({r[1]})" for r in con.execute(
            "SELECT id, state FROM sampled_jobs WHERE state IN ('queued','running','awaiting_roi')")]
        rows += [f"analysis {r[0]} (RUNNING)" for r in con.execute("SELECT id FROM analyses WHERE status='RUNNING'")]
    finally:
        con.close()
    return rows


def preflight(paths: Paths, health_urls: Sequence[str]) -> None:
    for label, p in (("database", paths.db), ("backup script", paths.backup_script)):
        if not Path(p).is_file():
            raise ResetError(f"{label} not found: {p}")
    busy = busy_jobs(paths.db)
    if busy:
        raise ResetError(f"work is still queued/running: {busy}")
    if Path(paths.storage).resolve() != (paths.root / "storage").resolve():
        raise ResetError(f"storage ({paths.storage}) is not <root>/storage: deploy/backup-ldm.ps1 would back up the "
                         "wrong folders")
    alive = running_backends(health_urls)
    if alive:
        raise ResetError(f"the app is still running ({', '.join(alive)}): stop it first (deploy/stop-ldm.ps1)")


# ================================================================================================ backup
def run_backup(paths: Paths) -> Path:
    """deploy/backup-ldm.ps1 -IncludeVideos into paths.backup_dest; returns the new ldm_backup_* folder."""
    dest = Path(paths.backup_dest)
    dest.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in dest.glob("ldm_backup_*")}
    env = dict(os.environ, LM_DB_PATH=str(paths.db))
    proc = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(paths.backup_script),
                           "-Destination", str(dest), "-IncludeVideos"],
                          capture_output=True, text=True, env=env, cwd=str(paths.root))
    if proc.returncode != 0:
        raise ResetError(f"backup failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")
    new = sorted(p for p in dest.glob("ldm_backup_*") if p.name not in before)
    if len(new) != 1:
        raise ResetError(f"expected one new backup folder in {dest}, found {[p.name for p in new]}")
    return new[0]


def verify_backup(bdir: Path, paths: Paths, before: dict[str, Any]) -> dict[str, Any]:
    sums = bdir / "SHA256SUMS.txt"
    if not sums.is_file():
        raise ResetError(f"backup has no SHA256SUMS.txt: {bdir}")
    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            listed[rel.strip().replace("\\", "/")] = h.strip().lower()
    on_disk = {p.relative_to(bdir).as_posix() for p in bdir.rglob("*") if p.is_file() and p != sums}
    if set(listed) != on_disk:
        raise ResetError(f"SHA256SUMS.txt does not list exactly the backup's files "
                         f"(unlisted {sorted(on_disk - set(listed))[:5]}, missing {sorted(set(listed) - on_disk)[:5]})")
    bad = [rel for rel, h in listed.items() if sha256_file(bdir / rel) != h]
    if bad:
        raise ResetError(f"{len(bad)} backup file(s) fail their checksum: {bad[:5]}")
    bdb = bdir / "db" / "litter_monitor.db"
    if not bdb.is_file():
        raise ResetError("backup has no db/litter_monitor.db")
    con = _ro(bdb)
    try:
        if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ResetError("backed-up database fails integrity_check")
        bcounts = _counts(con)
    finally:
        con.close()
    diff = {t: (before["counts"][t], bcounts.get(t)) for t in before["counts"] if bcounts.get(t) != before["counts"][t]}
    if diff:
        raise ResetError(f"backed-up database row counts differ from the live database: {diff}")
    for sub in ("storage/alerts", "storage/videos", "models/litter_yolo"):
        src = _dir_stats(paths.root / sub)
        if src["files"] and _dir_stats(bdir / sub)["files"] != src["files"]:
            raise ResetError(f"backup of {sub} is incomplete")
    return {"path": str(bdir), "files": len(listed), "bytes": sum((bdir / r).stat().st_size for r in listed),
            "checksums_verified": len(listed)}


# ================================================================================================ reset steps
def export_tables(db: Path, qdir: Path) -> dict[str, int]:
    out = qdir / "db_export"
    out.mkdir(parents=True, exist_ok=True)
    con = _ro(db)
    written: dict[str, int] = {}
    try:
        for t in HISTORY_TABLES:
            cur = con.execute(f'SELECT * FROM "{t}"')
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur]
            (out / f"{t}.json").write_text(json.dumps({"table": t, "columns": cols, "count": len(rows), "rows": rows},
                                                      ensure_ascii=False, default=str), encoding="utf-8")
            written[t] = len(rows)
    finally:
        con.close()
    for t, n in written.items():        # read back: the export must be complete before anything is deleted
        if json.loads((out / f"{t}.json").read_text(encoding="utf-8"))["count"] != n:
            raise ResetError(f"export of {t} could not be read back")
    return written


def delete_history(db: Path, before: dict[str, Any]) -> dict[str, int]:
    """DELETE every history table in one transaction; roll back unless all are empty and nothing else changed."""
    con = sqlite3.connect(str(db), timeout=30.0, isolation_level=None)
    deleted: dict[str, int] = {}
    try:
        con.execute("BEGIN IMMEDIATE")
        try:
            for t in HISTORY_TABLES:
                deleted[t] = con.execute(f'DELETE FROM "{t}"').rowcount
            left = {t: n for t, n in _counts(con).items() if t in HISTORY_TABLES and n}
            changed = [t for t, v in before["protected_tables"].items() if _table_digest(con, t) != v["sha256"]]
            if left or changed:
                raise ResetError(f"rolled back: rows left {left}, protected tables changed {changed}")
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
        con.execute("VACUUM")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()
    return deleted


def _long_paths_enabled() -> bool:
    if os.name != "nt":
        return True
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1
    except OSError:
        return False


def check_path_lengths(storage: Path, qdir: Path) -> dict[str, int]:
    """Refuse (before anything is deleted) if a moved file or folder would exceed the Windows path limit: Python,
    Explorer and the restore step could not open it. Limits: 259 characters for a file, 247 for a folder."""
    longest = {"file": 0, "dir": 0}
    for name in HISTORY_DIRS:
        src = storage / name
        if not src.is_dir():
            continue
        for p in [src, *src.rglob("*")]:
            n = len(str(qdir.resolve() / p.relative_to(storage)))
            kind = "file" if p.is_file() else "dir"
            longest[kind] = max(longest[kind], n)
    if not _long_paths_enabled() and (longest["file"] > 259 or longest["dir"] > 247):
        raise ResetError(f"quarantine paths would be too long for Windows (longest file {longest['file']}, folder "
                         f"{longest['dir']} characters; limits 259/247): use a shorter storage path")
    return longest


def move_history_dirs(storage: Path, qdir: Path) -> dict[str, Any]:
    """Rename each history folder into qdir/<name> (same volume: no copy, nothing deleted)."""
    dst_root = qdir
    dst_root.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    moved: list[str] = []
    for name in HISTORY_DIRS:
        src = storage / name
        if not src.is_dir():
            continue
        listing = [(p.relative_to(storage).as_posix(), p.stat().st_size) for p in sorted(src.rglob("*")) if p.is_file()]
        os.rename(src, dst_root / name)
        src.mkdir()                                      # the app expects the folder to exist
        moved.append(name)
        files += [{"original": f"storage/{rel}", "quarantine": rel, "bytes": n} for rel, n in listing]
    missing = [f["original"] for f in files
               if not (qdir / f["quarantine"]).is_file() or (qdir / f["quarantine"]).stat().st_size != f["bytes"]]
    manifest = {"created_at": _now(), "storage": str(storage), "quarantine": str(qdir), "folders": moved,
                "file_count": len(files), "total_bytes": sum(f["bytes"] for f in files), "files": files}
    (qdir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    if missing:
        raise ResetError(f"{len(missing)} moved file(s) not found in the quarantine: {missing[:5]}")
    return {k: manifest[k] for k in ("folders", "file_count", "total_bytes")}


# ================================================================================================ verification
def lifecycle_events_since(db: Path, since_iso: str) -> list[dict[str, Any]]:
    """App start/stop events (backend/main.py writes one on every start) logged after the reset snapshot: new
    records, not old history, so they are allowed after a restart (and listed in the report)."""
    since = dt.datetime.fromisoformat(since_iso)
    con = _ro(db)
    try:
        rows = con.execute("SELECT id, ts, source, message FROM system_events").fetchall()
    finally:
        con.close()
    return [{"id": i, "ts": ts, "message": m} for i, ts, src, m in rows
            if src == "system" and m in LIFECYCLE_MESSAGES and dt.datetime.fromisoformat(ts) > since]


def verify(paths: Paths, qdir: Path, health_url: Optional[str] = None) -> dict[str, Any]:
    before = json.loads((qdir / "before.json").read_text(encoding="utf-8"))
    after = snapshot(paths)
    fails: list[str] = []
    new_events = lifecycle_events_since(paths.db, before["taken_at"])
    after["counts"]["system_events"] -= len(new_events)
    nonzero = {t: after["counts"][t] for t in HISTORY_TABLES if after["counts"][t]}
    if nonzero:
        fails.append(f"history rows remain: {nonzero}")
    for t, v in before["protected_tables"].items():
        if t in HISTORY_TABLES:          # a snapshot from before the table became history (captures, 2026-11-21)
            continue
        if after["protected_tables"].get(t) != v:
            fails.append(f"protected table changed: {t}")
    if after["models_rows"] != before["models_rows"]:
        fails.append("models table changed")
    for key in ("active_model", "store_config", "ledger"):
        if after[key] != before[key]:
            fails.append(f"{key} changed: {before[key]} -> {after[key]}")
    for key, v in before["protected_dirs"].items():
        if after["protected_dirs"][key] != v:
            fails.append(f"{key}/ changed: {v} -> {after['protected_dirs'][key]}")
    left = {n: s["files"] for n, s in after["history_dirs"].items() if s["files"]}
    if left:
        fails.append(f"files remain in history folders: {left}")
    man = json.loads((qdir / "MANIFEST.json").read_text(encoding="utf-8"))
    gone = [f["quarantine"] for f in man["files"] if not (qdir / f["quarantine"]).is_file()]
    if gone:
        fails.append(f"{len(gone)} quarantined file(s) missing")
    app: dict[str, Any] = {}
    if health_url:
        base = health_url.rstrip("/")
        try:
            h = _get_json(base + "/api/health", timeout=10)
            app["health_status"] = h.get("status")
            app["model_version"] = (h.get("model") or {}).get("version")
            if h.get("status") != "ok" or app["model_version"] != before["active_model"]["version"]:
                fails.append(f"health: status={h.get('status')} model={app['model_version']} "
                             f"(expected {before['active_model']['version']})")
            allowed = {e["id"] for e in new_events}
            for path, check in DASHBOARD_CHECKS:
                body = _get_json(base + path, timeout=30)
                if path == "/api/events":           # the start-up event(s) listed in new_events are expected
                    body = {**body, "events": [e for e in body.get("events") or [] if e.get("id") not in allowed]}
                problem = check(body)
                app[path] = problem or "empty"
                if problem:
                    fails.append(f"dashboard {path}: {problem}")
        except (urllib.error.URLError, OSError, ValueError) as e:
            fails.append(f"app check failed: {e}")
    return {"checked_at": _now(), "ok": not fails, "failures": fails,
            "history_counts": {t: after["counts"][t] for t in HISTORY_TABLES},
            "active_model": after["active_model"], "store_config": after["store_config"], "ledger": after["ledger"],
            "quarantined_files": man["file_count"], "new_since_reset": new_events, "app": app}


# ================================================================================================ orchestration
def run(paths: Paths, *, phrase: str, health_urls: Sequence[str]) -> dict[str, Any]:
    if phrase != PHRASE:
        raise ResetError(f"confirmation phrase does not match (type exactly: {PHRASE})")
    preflight(paths, health_urls)
    before = snapshot(paths)
    print(f"[1/6] snapshot: {before['history_rows']} history row(s); model {before['active_model']['version']}")
    bdir = run_backup(paths)
    backup = verify_backup(bdir, paths, before)
    print(f"[2/6] backup verified: {bdir} ({backup['files']} files, {backup['bytes'] / 1e6:.1f} MB)")
    again = snapshot(paths)
    if again["counts"] != before["counts"] or again["protected_tables"] != before["protected_tables"]:
        raise ResetError("the database changed during the backup (is something still writing?)")
    preflight(paths, health_urls)
    qdir = paths.storage / QUARANTINE / f"history_{dt.datetime.now():%Y%m%d_%H%M%S}"
    before["quarantine_path_lengths"] = check_path_lengths(paths.storage, qdir)
    qdir.mkdir(parents=True, exist_ok=False)
    before["backup"] = backup
    (qdir / "before.json").write_text(json.dumps(before, indent=1, default=str), encoding="utf-8")
    exported = export_tables(paths.db, qdir)
    print(f"[3/6] exported {sum(exported.values())} row(s) to {qdir / 'db_export'}")
    deleted = delete_history(paths.db, before)
    print(f"[4/6] deleted {sum(deleted.values())} row(s) in one transaction; VACUUM done")
    moved = move_history_dirs(paths.storage, qdir)
    print(f"[5/6] moved {moved['file_count']} file(s) ({moved['total_bytes'] / 1e9:.2f} GB) to {qdir}")
    check = verify(paths, qdir)
    result = {"finished_at": _now(), "quarantine": str(qdir), "backup": backup, "exported": exported,
              "deleted": deleted, "moved": moved, "verification": check}
    (qdir / "RESULT.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    if not check["ok"]:
        raise ResetError(f"post-reset verification failed: {check['failures']} (see {qdir / 'RESULT.json'})")
    print(f"[6/6] verified offline: all history counts 0, model/config/ledger unchanged. Quarantine: {qdir}")
    return result


def status(paths: Paths, health_urls: Sequence[str]) -> dict[str, Any]:
    s = snapshot(paths)
    return {"history_counts": {t: s["counts"][t] for t in HISTORY_TABLES}, "history_rows": s["history_rows"],
            "history_dirs": {n: {"files": v["files"], "MB": round(v["bytes"] / 1e6, 1)}
                             for n, v in s["history_dirs"].items()},
            "active_model": s["active_model"]["version"], "app_running": running_backends(health_urls),
            "busy": busy_jobs(paths.db),
            "backup_dest": str(paths.backup_dest)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.services.history_reset", description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, help="project root to operate on (default: the configured app paths)")
    ap.add_argument("--backup-dest", type=Path, help="backup folder (default: %%USERPROFILE%%\\LDM_backups)")
    ap.add_argument("--health-url", action="append", help="app URL(s) that must NOT answer (default: ports 8000, "
                    "8001 and deploy/run/backend.json); with verify: the restarted app to check")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("status", help="dry run (default)")
    r = sub.add_parser("run", help="back up, then clear the history")
    r.add_argument("--confirm", action="store_true", required=True)
    r.add_argument("--phrase", help=f"the confirmation phrase ({PHRASE}); asked for when omitted")
    v = sub.add_parser("verify", help="re-check a finished reset (and the restarted app with --health-url)")
    v.add_argument("--quarantine", type=Path, required=True)
    a = ap.parse_args(argv)
    paths = Paths.for_root(a.root, a.backup_dest) if a.root else Paths.from_config(a.backup_dest)
    urls = a.health_url or default_health_urls(paths.root)
    try:
        if a.cmd == "run":
            phrase = a.phrase
            if phrase is None:
                if not sys.stdin.isatty():
                    raise ResetError("no terminal to ask for the phrase: pass --phrase")
                phrase = input(f"This clears the dashboard history (backup first). Type {PHRASE} to continue: ")
            res = run(paths, phrase=phrase.strip(), health_urls=urls)
            print(json.dumps({"quarantine": res["quarantine"], "backup": res["backup"]["path"]}, indent=1))
        elif a.cmd == "verify":
            res = verify(paths, a.quarantine, (a.health_url or [None])[0])
            print(json.dumps(res, indent=1, default=str))
            return 0 if res["ok"] else 1
        else:
            st = status(paths, urls)
            print(json.dumps(st, indent=1))
            return 3 if st["busy"] else 0
    except ResetError as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
