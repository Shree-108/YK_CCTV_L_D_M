"""LDM monitoring report: built ONLY from data the application already records. Read-only.

    python -m monitoring.ldm_monitor                 # last 30 days -> storage/monitoring/<timestamp>/
    python -m monitoring.ldm_monitor --days 7 --min-reviewed 20

The database is opened read-only (SQLite mode=ro); nothing in the app, its data or its files is changed or deleted.
Output (git-ignored, under storage/ like the data it summarises): report.md, metrics.json, daily_activity.csv,
camera_summary.csv.

Three kinds of numbers, never mixed:
  OPERATIONAL   recorded facts: jobs, failures, processing time, inference latency, captures, restarts, ingest
  ACTIVITY      unreviewed model output: detections/alerts by day, camera, class, status, confidence. These are
                NOT accuracy figures; a detection is not litter until a person says so.
  QUALITY       only from human review: alert status set by an operator (resolved = real litter handled,
                false_positive = wrong), Test Lab verdicts (true/false positive, false negative = missed litter),
                reviewed hard examples, and evaluations on annotated test sets. Precision is shown only when at
                least --min-reviewed decided items exist, with a 95 % Wilson interval. Missed litter can only come
                from verdicts/evaluations: the absence of an alert is never treated as "no litter".
It also lists the model / dataset / evaluation register, checks the approved-change log, inventories stored CCTV
data against the proposed retention periods (report only), scans logs for leaked secrets, and states whether
reviewed evidence supports retraining. It never retrains, activates models or sends notifications.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import yaml

from backend.core.config import AppConfig, PROJECT_ROOT

# Proposed retention periods in days (docs/LDM_MONITORING.md). Report only: nothing is deleted.
RETENTION_DAYS = {"uploads": 1, "videos": 30, "captures": 30, "alerts": 90, "ingest": 90, "hard_examples": 180,
                  "evaluations": 3650, "monitoring": 365}
CHANGE_LOG = PROJECT_ROOT / "governance" / "model_change_log.csv"
CHANGE_LOG_FIELDS = ["date", "change", "model_version", "dataset_version", "evaluation_id", "decision",
                     "approved_by", "notes"]
DECIDED = ("resolved", "false_positive")      # operator verdicts on alerts; 'acknowledged' = seen, not judged


# ---------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.96) -> Optional[tuple[float, float]]:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:.1f} %"


def open_ro(db: Path) -> sqlite3.Connection:
    c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def rows(c: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(r) for r in c.execute(sql, args)]
    except sqlite3.OperationalError:
        return []


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- sections
def operational(c: sqlite3.Connection, since: str, ingest_db: Path) -> dict[str, Any]:
    jobs = rows(c, "SELECT * FROM jobs WHERE created_at >= ?", (since,))
    vid = [j for j in jobs if j["kind"] == "video"]
    lat, wall, fps, per_cam = [], [], [], defaultdict(Counter)
    for j in vid:
        per_cam[j["camera_id"] or "?"][j["status"]] += 1
        st = (json.loads(j["result_json"] or "{}") or {}).get("stats", {})
        if st.get("mean_inference_ms") is not None:
            lat.append(st["mean_inference_ms"])
        if st.get("wall_time_sec") is not None:
            wall.append(st["wall_time_sec"])
        if st.get("processing_fps") is not None:
            fps.append(st["processing_fps"])
    caps = rows(c, "SELECT camera_id, status, duration_ms, error FROM captures WHERE ts >= ?", (since,))
    ev = rows(c, "SELECT level, source, message FROM system_events WHERE ts >= ?", (since,))
    out = {
        "video_jobs": {"total": len(vid), "by_status": dict(Counter(j["status"] for j in vid)),
                       "failed": [{"id": j["id"], "camera": j["camera_id"], "error": (j["error"] or "")[:200]}
                                  for j in vid if j["status"] == "failed"],
                       "mean_inference_ms": _stats(lat), "wall_time_sec": _stats(wall), "processing_fps": _stats(fps),
                       "by_camera": {k: dict(v) for k, v in per_cam.items()}},
        "evaluation_jobs": dict(Counter(j["status"] for j in jobs if j["kind"] == "evaluation")),
        "captures": {"total": len(caps), "by_status": dict(Counter(x["status"] for x in caps)),
                     "duration_ms": _stats([x["duration_ms"] for x in caps if x["duration_ms"] is not None]),
                     "errors": dict(Counter((x["error"] or "")[:80] for x in caps if x["error"]))},
        "system_events": {"by_level": dict(Counter(e["level"] for e in ev)),
                          "backend_starts": sum(1 for e in ev if e["message"] == "backend started"),
                          "errors": [f"{e['source']}: {e['message'][:150]}" for e in ev if e["level"] in ("error", "warning")][:30]},
        "image_upload_latency": "NOT RECORDED: /api/detect/image returns inference_ms but does not store it (gap G1)",
    }
    if ingest_db.is_file():
        ic = open_ro(ingest_db)
        ing = rows(ic, "SELECT status, error, duration_sec FROM recordings")
        out["export_ingest"] = {"recordings": len(ing), "by_status": dict(Counter(r["status"] for r in ing)),
                                "errors": dict(Counter((r["error"] or "")[:100] for r in ing if r["status"] != "done" and r["error"]))}
        ic.close()
    else:
        out["export_ingest"] = "not used yet (no storage/ingest/ingest.db)"
    return out


def _stats(v: list[float]) -> Optional[dict[str, float]]:
    if not v:
        return None
    s = sorted(v)
    return {"n": len(s), "median": round(s[len(s) // 2], 1), "p90": round(s[min(len(s) - 1, int(len(s) * 0.9))], 1),
            "max": round(s[-1], 1)}


def activity(c: sqlite3.Connection, since: str) -> dict[str, Any]:
    dets = rows(c, "SELECT ts, camera_id, class, confidence, status, reason, model_version FROM detections WHERE ts >= ?", (since,))
    alerts = rows(c, "SELECT ts, camera_id, class, confidence, status, model_version FROM alerts WHERE ts >= ?", (since,))
    hist = Counter(min(9, int((d["confidence"] or 0) * 10)) for d in dets)
    daily: dict[str, Counter] = defaultdict(Counter)
    for d in dets:
        daily[d["ts"][:10]]["detections"] += 1
        daily[d["ts"][:10]][f"det_{d['status']}"] += 1
    for a in alerts:
        daily[a["ts"][:10]]["alerts"] += 1
    cams: dict[str, Counter] = defaultdict(Counter)
    for d in dets:
        cams[d["camera_id"]]["detections"] += 1
        cams[d["camera_id"]][f"det_{d['status']}"] += 1
    for a in alerts:
        cams[a["camera_id"]]["alerts"] += 1
        cams[a["camera_id"]][f"alert_{a['status']}"] += 1
    return {
        "note": "UNREVIEWED model output - not accuracy",
        "detections": len(dets), "alerts": len(alerts),
        "detections_by_status": dict(Counter(d["status"] for d in dets)),
        "rejection_reasons": dict(Counter(d["reason"] for d in dets if d["reason"])),
        "alerts_by_class": dict(Counter(a["class"] for a in alerts)),
        "alerts_by_model": dict(Counter(a["model_version"] for a in alerts)),
        "detection_confidence_histogram": {f"{i / 10:.1f}-{(i + 1) / 10:.1f}": hist.get(i, 0) for i in range(10)},
        "daily": {k: dict(v) for k, v in sorted(daily.items())},
        "cameras": {k: dict(v) for k, v in sorted(cams.items())},
        "timestamps": "UTC as stored by the app",
    }


def quality(c: sqlite3.Connection, since: str, min_n: int, eval_dir: Path) -> dict[str, Any]:
    alerts = rows(c, "SELECT camera_id, class, status, model_version FROM alerts WHERE ts >= ?", (since,))
    total = len(alerts)
    decided = [a for a in alerts if a["status"] in DECIDED]

    def prec(group: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(group)
        tp = sum(1 for a in group if a["status"] == "resolved")
        d = {"decided": n, "real_litter": tp, "false_positive": n - tp}
        if n >= min_n:
            d["precision"] = round(tp / n, 3)
            d["ci95"] = wilson(tp, n)
        else:
            d["precision"] = None
            d["why_not"] = f"only {n} decided alert(s); at least {min_n} needed"
        return d

    by = {k: defaultdict(list) for k in ("camera_id", "class", "model_version")}
    for a in decided:
        for k in by:
            by[k][a[k]].append(a)
    reviews = rows(c, "SELECT verdict, model_version, class FROM reviews WHERE created_at >= ?", (since,))
    rv = Counter(r["verdict"] for r in reviews)
    hard = rows(c, "SELECT kind, review_status, source FROM hard_examples")
    evals = []
    for f in sorted(eval_dir.glob("*.json")) if eval_dir.is_dir() else []:
        try:
            e = json.loads(f.read_text(encoding="utf-8"))
            evals.append({"id": e.get("id", f.stem), "model_version": e.get("model_version"), "dataset": e.get("dataset"),
                          "split": e.get("split"), "overall": e.get("overall")})
        except (OSError, json.JSONDecodeError):
            evals.append({"id": f.stem, "error": "unreadable"})
    out = {
        "alert_review": {"alerts": total, "decided": len(decided),
                         "acknowledged_only": sum(1 for a in alerts if a["status"] == "acknowledged"),
                         "unreviewed": sum(1 for a in alerts if a["status"] == "new"),
                         "review_coverage": round(len(decided) / total, 3) if total else None,
                         "overall": prec(decided),
                         "by_camera": {k: prec(v) for k, v in by["camera_id"].items()},
                         "by_class": {k: prec(v) for k, v in by["class"].items()},
                         "by_model": {k: prec(v) for k, v in by["model_version"].items()}},
        "test_lab_reviews": {"total": len(reviews), "by_verdict": dict(rv),
                             "missed_litter_reported": rv.get("false_negative", 0),
                             "note": "false_negative verdicts are the only record of missed litter outside evaluations"},
        "hard_examples": {"total": len(hard), "by_kind": dict(Counter(h["kind"] for h in hard)),
                          "by_review_status": dict(Counter(h["review_status"] for h in hard))},
        "evaluations": evals,
    }
    return out


def register(c: sqlite3.Connection) -> dict[str, Any]:
    models = []
    for m in rows(c, "SELECT version, path, active, registered_at, trained_at FROM models"):
        p = Path(m["path"])
        models.append({**m, "file_present": p.is_file(), "sha256": sha256(p) if p.is_file() else None})
    on_disk = sorted(d.name for d in (PROJECT_ROOT / "models" / "litter_yolo").glob("*") if d.is_dir())
    backups = []
    for s in (PROJECT_ROOT / "models" / "backups").glob("*/SHA256SUMS"):
        line = s.read_text(encoding="utf-8").split()
        bp = s.parent / "best.pt"
        backups.append({"folder": s.parent.name, "recorded_sha256": line[0] if line else None,
                        "matches_file": bool(line) and bp.is_file() and sha256(bp) == line[0]})
    datasets = []
    for y in sorted((PROJECT_ROOT / "dataset").glob("*/data.yaml")):
        d = y.parent
        names = (yaml.safe_load(y.read_text(encoding="utf-8")) or {}).get("names", {})
        imgs = sum(1 for p in (d / "images").rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if (d / "images").is_dir() else 0
        info = {"name": d.name, "classes": len(names), "images_on_disk": imgs,
                "build_report": (d / "build_report.json").is_file(), "validation_report": (d / "validation_report.json").is_file()}
        mf = d / "manifest.csv"
        if mf.is_file():
            with open(mf, newline="", encoding="utf-8-sig") as fh:
                st = Counter(r.get("review_status") or "unreviewed" for r in csv.DictReader(fh))
            info["workspace_review_status"] = dict(st)
            info["verified"] = st.get("annotated", 0) + st.get("reviewed_empty", 0)
        datasets.append(info)
    log, problems = [], []
    if CHANGE_LOG.is_file():
        with open(CHANGE_LOG, newline="", encoding="utf-8-sig") as fh:
            log = list(csv.DictReader(fh))
        known = {m["version"] for m in models} | set(on_disk)
        for i, r in enumerate(log, 2):
            if r.get("model_version") and r["model_version"] not in known:
                problems.append(f"line {i}: model_version {r['model_version']} not registered")
            if r.get("change") in ("activate", "deploy") and not (r.get("approved_by") or "").strip():
                problems.append(f"line {i}: '{r['change']}' without approved_by")
    active = [m["version"] for m in models if m["active"]]
    if active and log and not any(r.get("model_version") == active[0] and r.get("change") in ("activate", "baseline")
                                  for r in log):
        problems.append(f"active model {active[0]} has no activate/baseline entry in the change log")
    return {"models": models, "model_folders_on_disk": on_disk, "backups": backups, "datasets": datasets,
            "change_log_entries": len(log), "change_log_problems": problems if CHANGE_LOG.is_file() else ["governance/model_change_log.csv missing"]}


def retention(cfg: AppConfig, ingest_dir: Path, now: dt.datetime) -> dict[str, Any]:
    dirs = {"uploads": cfg.storage_dir / "uploads", "videos": cfg.storage_dir / "videos",
            "captures": cfg.storage_dir / "captures", "alerts": cfg.storage_dir / "alerts", "ingest": ingest_dir,
            "hard_examples": cfg.storage_dir / "hard_examples", "evaluations": cfg.storage_dir / "evaluations",
            "monitoring": cfg.storage_dir / "monitoring"}
    out = {}
    for k, d in dirs.items():
        files = [p for p in d.rglob("*") if p.is_file()] if d.is_dir() else []
        ages = [(now - dt.datetime.fromtimestamp(p.stat().st_mtime)).days for p in files]
        limit = RETENTION_DAYS[k]
        out[k] = {"path": str(d), "files": len(files), "size_mb": round(sum(p.stat().st_size for p in files) / 2 ** 20, 1),
                  "oldest_days": max(ages) if ages else None, "proposed_retention_days": limit,
                  "older_than_policy": sum(1 for a in ages if a > limit)}
    db = cfg.db_path
    out["database"] = {"path": str(db), "size_mb": round(db.stat().st_size / 2 ** 20, 1) if db.is_file() else None,
                       "note": "rows (detections/alerts) are not aged out by the app"}
    return out


def security(cfg: AppConfig, ingest_dir: Path) -> dict[str, Any]:
    env = PROJECT_ROOT / ".env"
    secrets = []
    if env.is_file():
        for ln in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in ln and not ln.lstrip().startswith("#"):
                k, v = ln.split("=", 1)
                v = v.split("#")[0].strip()
                if any(s in k for s in ("TOKEN", "PASSWORD", "SECRET", "KEY")) and len(v) >= 8:
                    secrets.append(v)
    logs = [p for d in (PROJECT_ROOT / "deploy" / "logs", ingest_dir, cfg.storage_dir) if d.is_dir()
            for p in d.glob("*.log*") if p.is_file()]
    cred_url = re.compile(r"\w+://[^/\s:@]+:[^@\s/]+@")
    hits = []
    for p in logs:
        t = p.read_text(encoding="utf-8", errors="ignore")
        if any(s in t for s in secrets):
            hits.append(f"{p.name}: contains a secret value from .env")
        if cred_url.search(t):
            hits.append(f"{p.name}: contains a URL with embedded credentials")
    try:
        tracked = bool(subprocess.run(["git", "ls-files", ".env"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip())
    except OSError:
        tracked = None
    return {"logs_scanned": len(logs), "secret_leaks": hits, "env_tracked_by_git": tracked,
            "admin_token_set": bool(cfg.admin_token), "api_bind_note": "deploy/start-ldm.ps1 binds 127.0.0.1 unless -Lan"}


def retrain_decision(q: dict[str, Any], reg: dict[str, Any], min_n: int, target: float) -> dict[str, Any]:
    reasons_need, reasons_data = [], []
    ov = q["alert_review"]["overall"]
    if ov.get("precision") is not None and ov["ci95"] and ov["ci95"][1] < target:
        reasons_need.append(f"reviewed alert precision {pct(ov['precision'])} (95 % CI up to {pct(ov['ci95'][1])}) "
                            f"is below the {pct(target)} target on {ov['decided']} decided alerts")
    fn = q["test_lab_reviews"]["missed_litter_reported"]
    if fn >= min_n:
        reasons_need.append(f"{fn} missed-litter (false_negative) verdicts in Test Lab")
    for e in q["evaluations"]:
        o = e.get("overall") or {}
        if o.get("recall") is not None and o["recall"] < target:
            reasons_need.append(f"evaluation {e['id']} recall {pct(o['recall'])} < {pct(target)}")
    fp_rev = ov.get("false_positive", 0)
    approved_hard = q["hard_examples"]["by_review_status"].get("approved", 0)
    if fp_rev + approved_hard >= 30:
        reasons_data.append(f"{fp_rev} reviewed false-positive alerts + {approved_hard} approved hard examples")
    ws = next((d for d in reg["datasets"] if "verified" in d), None)
    if ws and ws["verified"] >= 300:
        reasons_data.append(f"{ws['verified']} verified images in {ws['name']}")
    if reasons_need and reasons_data:
        verdict = "RETRAINING MAY BE JUSTIFIED: prepare a candidate and evaluate it (needs approval; never automatic)"
    elif reasons_need:
        verdict = "QUALITY PROBLEM SEEN, but not enough reviewed data to retrain yet: collect and annotate first"
    else:
        verdict = "NO RETRAINING RECOMMENDED: reviewed evidence does not show a need"
        if q["alert_review"]["decided"] < min_n and not q["evaluations"]:
            verdict = ("NOT ENOUGH REVIEWED EVIDENCE TO JUDGE: review alerts (resolved / false positive) "
                       "and build an annotated test set")
    return {"verdict": verdict, "need_evidence": reasons_need, "data_evidence": reasons_data}


# ---------------------------------------------------------------- report
def markdown(m: dict[str, Any]) -> str:
    op, ac, q, rg, rt, sc, rd = (m[k] for k in ("operational", "activity", "quality", "register", "retention", "security", "retraining"))
    vj = op["video_jobs"]
    L = [f"# LDM monitoring report", "", f"Generated {m['generated']} · period: last {m['days']} days (since {m['since']} UTC) · "
         f"database opened read-only", "", f"## Retraining decision: **{rd['verdict']}**", ""]
    L += [f"- need: {x}" for x in rd["need_evidence"]] + [f"- data: {x}" for x in rd["data_evidence"]] + [""]
    L += ["## 1. Operational (recorded facts)", "",
          f"- Video jobs: {vj['total']} {vj['by_status']}; failed: {len(vj['failed'])}",
          f"- Inference latency per job (mean ms): {vj['mean_inference_ms']}; wall time (s): {vj['wall_time_sec']}; "
          f"processing fps: {vj['processing_fps']}",
          f"- Scheduled captures: {op['captures']['total']} {op['captures']['by_status']}; errors: {op['captures']['errors'] or 'none'}",
          f"- System events: {op['system_events']['by_level']}; backend starts: {op['system_events']['backend_starts']}",
          f"- Export ingest: {op['export_ingest']}",
          f"- Image uploads: {op['image_upload_latency']}", ""]
    for f in vj["failed"]:
        L.append(f"  - FAILED job {f['id']} ({f['camera']}): {f['error']}")
    L += ["## 2. Activity (UNREVIEWED model output: not accuracy)", "",
          f"- Detections: {ac['detections']} by status {ac['detections_by_status']}; rejection reasons {ac['rejection_reasons']}",
          f"- Alerts: {ac['alerts']} by class {ac['alerts_by_class']}; by model {ac['alerts_by_model']}",
          f"- Detection confidence histogram: {ac['detection_confidence_histogram']}", "",
          "| camera | detections | confirmed | alerts | alerts new | alerts resolved | alerts false positive |", "|---|---|---|---|---|---|---|"]
    for cam, v in ac["cameras"].items():
        L.append(f"| {cam} | {v.get('detections', 0)} | {v.get('det_confirmed', 0)} | {v.get('alerts', 0)} | {v.get('alert_new', 0)} | "
                 f"{v.get('alert_resolved', 0)} | {v.get('alert_false_positive', 0)} |")
    ar = q["alert_review"]
    L += ["", "## 3. Quality (human-reviewed data only)", "",
          f"- Alerts decided by an operator: {ar['decided']} of {ar['alerts']} (coverage {pct(ar['review_coverage'])}); "
          f"acknowledged only: {ar['acknowledged_only']}; unreviewed: {ar['unreviewed']}",
          f"- Alert precision (resolved / decided): " + (f"**{pct(ar['overall']['precision'])}**, 95 % CI {ar['overall']['ci95']}"
                                                         if ar["overall"]["precision"] is not None else f"not reported: {ar['overall']['why_not']}"),
          f"- False positives confirmed by operators: {ar['overall']['false_positive']}",
          f"- Missed litter (Test Lab false_negative verdicts): {q['test_lab_reviews']['missed_litter_reported']} "
          f"(of {q['test_lab_reviews']['total']} verdicts {q['test_lab_reviews']['by_verdict']})",
          f"- Hard examples: {q['hard_examples']}",
          f"- Evaluations on annotated test sets: {len(q['evaluations'])}", ""]
    for scope in ("by_camera", "by_class", "by_model"):
        if ar[scope]:
            L.append(f"  - {scope}: " + "; ".join(f"{k}: {pct(v['precision']) if v['precision'] is not None else 'n<' + str(m['min_reviewed'])} "
                                                   f"({v['decided']} decided)" for k, v in ar[scope].items()))
    for e in q["evaluations"]:
        L.append(f"  - evaluation {e['id']}: {e.get('model_version')} on {e.get('dataset')}/{e.get('split')}: {e.get('overall')}")
    L += ["", "## 4. Model / dataset / evaluation register", "", "| model | active | registered | file | sha256 |", "|---|---|---|---|---|"]
    for mm in rg["models"]:
        L.append(f"| {mm['version']} | {'yes' if mm['active'] else ''} | {mm['registered_at']} | {'ok' if mm['file_present'] else 'MISSING'} | "
                 f"{(mm['sha256'] or '')[:16]} |")
    L += [f"", f"- Model folders on disk: {rg['model_folders_on_disk']}", f"- Backups: {rg['backups']}",
          f"- Datasets: " + "; ".join(f"{d['name']} ({d['images_on_disk']} images" + (f", {d['verified']} verified" if 'verified' in d else "")
                                      + (", built" if d['build_report'] else "") + (", validated" if d['validation_report'] else "") + ")" for d in rg["datasets"]),
          f"- Approved-change log: {rg['change_log_entries']} entries; problems: {rg['change_log_problems'] or 'none'}", "",
          "## 5. Stored data vs proposed retention (report only, nothing deleted)", "",
          "| area | files | MB | oldest (days) | proposed limit (days) | older than limit |", "|---|---|---|---|---|---|"]
    for k, v in rt.items():
        if k != "database":
            L.append(f"| {k} | {v['files']} | {v['size_mb']} | {v['oldest_days']} | {v['proposed_retention_days']} | {v['older_than_policy']} |")
    L += [f"", f"- Database: {rt['database']}", "", "## 6. Security checks", "",
          f"- Logs scanned: {sc['logs_scanned']}; leaks: {sc['secret_leaks'] or 'none'}; .env tracked by git: {sc['env_tracked_by_git']}; "
          f"admin token set: {sc['admin_token_set']}; {sc['api_bind_note']}", ""]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--min-reviewed", type=int, default=20, help="decided alerts needed before precision is reported")
    ap.add_argument("--target", type=float, default=0.80, help="precision/recall target used in the retraining rule")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default storage/monitoring/<timestamp>)")
    ap.add_argument("--db", type=Path, default=None, help="database (default: the app's LM_DB_PATH)")
    ap.add_argument("--ingest-dir", type=Path, default=PROJECT_ROOT / "storage" / "ingest")
    a = ap.parse_args(argv)
    cfg = AppConfig()
    db = a.db or cfg.db_path
    if not db.is_file():
        print(f"ERROR: database not found: {db}", file=sys.stderr)
        return 2
    now = dt.datetime.now()
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=a.days)).strftime("%Y-%m-%dT%H:%M:%S")
    c = open_ro(db)
    try:
        q = quality(c, since, a.min_reviewed, cfg.storage_dir / "evaluations")
        reg = register(c)
        m = {"generated": now.isoformat(timespec="seconds"), "days": a.days, "since": since, "min_reviewed": a.min_reviewed,
             "database": str(db), "operational": operational(c, since, a.ingest_dir), "activity": activity(c, since),
             "quality": q, "register": reg, "retention": retention(cfg, a.ingest_dir, now),
             "security": security(cfg, a.ingest_dir)}
        m["retraining"] = retrain_decision(q, reg, a.min_reviewed, a.target)
    finally:
        c.close()
    out = a.out or (cfg.storage_dir / "monitoring" / now.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(markdown(m), encoding="utf-8")
    with open(out / "daily_activity.csv", "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for v in m["activity"]["daily"].values() for k in v})
        w = csv.writer(f)
        w.writerow(["date_utc", *keys])
        for d, v in m["activity"]["daily"].items():
            w.writerow([d, *[v.get(k, 0) for k in keys]])
    with open(out / "camera_summary.csv", "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for v in m["activity"]["cameras"].values() for k in v})
        w = csv.writer(f)
        w.writerow(["camera", *keys])
        for cam, v in m["activity"]["cameras"].items():
            w.writerow([cam, *[v.get(k, 0) for k in keys]])
    print(f"report: {out / 'report.md'}")
    print(f"retraining decision: {m['retraining']['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
