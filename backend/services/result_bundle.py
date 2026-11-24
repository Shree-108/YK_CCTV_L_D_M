"""Result bundle of one finished four-screenshot job: the 4 annotated images + one import CSV + manifest.json.

    <export_root>/<storeKey>/<YYYY-MM-DD>/<job_id>/
        <job_id>_s1_annotated.jpg ... <job_id>_s4_annotated.jpg    copies (the job folder keeps its own files)
        <storeKey>_<YYYY-MM-DD>_<camera_id>_<job_id>.csv             header and rows: backend/services/garbage_export.py
        manifest.json                                                 job, store, camera, time source, model, counts,
                                                                      validator result, why no CSV (if so)

Nothing about detection is decided here: the rows come from garbage_export.sampled_checks / build_row (the only
definition of the format), the statuses from the job's stored outcomes (view_blocked = the existing frame-quality
check, outcome UNREADABLE), the images from the job's annotated screenshots.

The CSV is written only when ALL of this holds; otherwise the bundle has the images + manifest and the manifest
says why (``csv.reasons``):
  * the job has a store and a camera that config/store_config.json lists (cart_id, storeName from there)
  * a recording start was found (container metadata when trusted, NVR file name, OSD - not available -,
    operator-entered / --recording-start); processing time is never used
  * every one of the 4 screenshots is a camera check (none "processing failed") and has its image
  * no check is already in another CSV of the same store and date (a check is exported at most once)
  * the written bundle passes garbage_export.validate_bundle
Folders when the store or the time is unknown: ``_unassigned`` instead of the storeKey, ``time_unknown`` instead of
the date - neither is a valid store key / date, so an importer reading <storeKey>/<YYYY-MM-DD>/ never picks them up.
A bundle is never overwritten; it is assembled in a hidden temporary folder and renamed into place.

CLI (reads the detection DB read-only; writes only the bundle folder):
    py -m backend.services.result_bundle --sampled-job <id> [--store-key K] [--camera-id C] [--recording-start ISO]
    py -m backend.services.garbage_export validate-bundle <folder>
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import logging
import os
import secrets
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, Optional

from backend.services import garbage_export as ge

log = logging.getLogger(__name__)

SCHEMA = "ldm_result_bundle"
UNASSIGNED, TIME_UNKNOWN = "_unassigned", "time_unknown"
STATUS_FILE = "result_bundle.json"          # written into the job's reports/ folder by the runner hook
ANALYSED = ("LITTER_DETECTED", "NO_LITTER")


image_name = ge.bundle_image_name


def _blocked_picture(frame: Any, idx: int, reason: str) -> Any:
    """The screenshot that the frame-quality check rejected, with a banner saying so (or a plain card when the
    frame cannot be decoded again). Rendering only: the status was decided when the job ran."""
    import cv2
    import numpy as np
    img = frame.copy() if frame is not None else np.full((720, 1280, 3), 48, np.uint8)
    h, w = img.shape[:2]
    scale = max(0.5, w / 1600)
    lines = [f"VIEW BLOCKED - screenshot {idx} - not analysed for litter"]
    lines += textwrap.wrap(reason or "frame not usable", 70)[:3]
    if frame is None:
        lines.append("(the frame could not be decoded again from the stored video)")
    lh = int(42 * scale)
    cv2.rectangle(img, (0, 0), (w, lh * len(lines) + int(20 * scale)), (0, 0, 170), -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (int(16 * scale), lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (255, 255, 255),
                    max(1, int(2 * scale)), cv2.LINE_AA)
    return img


def _checks_in(path: Path) -> set[tuple[str, str]]:
    try:
        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"), newline="")))
    except (OSError, UnicodeDecodeError, csv.Error):
        return set()
    return {(r.get("camera_id") or "", r.get("timestamp") or "") for r in rows}


def _already_exported(day_dir: Path, job_id: str, keys: set[tuple[str, str]]) -> list[str]:
    """CSV files of the same store and date (other bundles, or the flat exporter's files) holding one of ``keys``."""
    if not day_dir.is_dir():
        return []
    files = [p for p in day_dir.glob("*.csv")]
    files += [p for d in day_dir.iterdir() if d.is_dir() and d.name != job_id and not d.name.startswith(".")
              for p in d.glob("*.csv")]
    return sorted(str(p.relative_to(day_dir)) for p in files if _checks_in(p) & keys)


def build_bundle(query: ge.Query, *, job_id: str, settings: ge.Settings, store_key: Optional[str] = None,
                 camera_id: Optional[str] = None, recording_start: Optional[str] = None,
                 allowed_classes: Optional[set[str]] = None,
                 frame_reader: Optional[Callable[[Path, list[int]], dict[int, Any]]] = None) -> dict[str, Any]:
    """Assemble the bundle of a finished job; returns its manifest (with ``folder``). Raises ExportError only when
    no bundle can be made at all (job not found / not finished / bundle already exists)."""
    src = ge.sampled_source(query, job_id)                       # raises unless the job exists and is done
    job = query("SELECT * FROM sampled_jobs WHERE id=?", (job_id,))[0]
    frames = query("SELECT * FROM sampled_frames WHERE job_id=? ORDER BY idx", (job_id,))
    reasons: list[str] = []

    # ---- store + camera (config/store_config.json; nothing defaulted)
    store: Optional[ge.Store] = None
    cam: Optional[str] = None
    store_info: dict[str, Any] = {}
    try:
        cam = ge.pick("camera_id", src.camera_id, camera_id)
        key = ge.pick("store key", src.store_key, store_key)
        store_info["requested_store_key"] = key
        if not cam:
            reasons.append("no camera assigned to this job (camera_id is empty; the screenshots were analysed as "
                           "'upload')")
        if not key:
            reasons.append("no store assigned to this job")
        if cam and key:
            store = ge.store_for(ge.load_store_config(settings.store_config), key, cam)
            store_info.update(storeKey=store.key, storeName=store.name, cart_id=store.cart_id)
    except ge.ExportError as exc:
        reasons.append(ge.one_line(str(exc)))
    if store is None:
        store_info["error"] = "store not resolved from the store config"

    # ---- recording time (approved order; never processing time)
    tb: Optional[ge.TimeBase] = None
    time_reason = ""
    try:
        tb = ge.resolve_time_base(
            video_path=src.video_path, original_name=src.original_name, duration_sec=src.duration_sec,
            manual_start=recording_start or src.manual_start,
            manual_label="recording-start argument" if recording_start else "operator-entered start",
            trust_container_time=settings.trust_container_time)
    except ge.ExportError as exc:
        time_reason = ge.one_line(str(exc))
        reasons.append(time_reason)

    # ---- checks (one per screenshot) from the stored outcomes
    checks: list[ge.Check] = []
    try:
        checks, excluded = ge.sampled_checks(query, src, cam or "", tb, time_reason)
        reasons += excluded
    except ge.ExportError as exc:
        reasons.append(ge.one_line(str(exc)))
    by_idx = {c.idx: c for c in checks}
    if len(frames) != ge.BUNDLE_IMAGES:
        reasons.append(f"the job has {len(frames)} screenshot(s), not {ge.BUNDLE_IMAGES}")

    # ---- folder
    first = min((c for c in checks if c.captured_at), key=lambda c: c.idx or 0, default=None)
    day = ge.format_ts(first.captured_at)[:10] if first else (ge.format_ts(tb.start)[:10] if tb else TIME_UNKNOWN)
    day_dir = Path(settings.export_root) / (store.key if store else UNASSIGNED) / day
    final = day_dir / job_id
    if final.exists():
        raise ge.ExportError(f"a bundle for {job_id} already exists: {final} (bundles are never overwritten)")
    tmp = day_dir / f".{job_id}.tmp-{secrets.token_hex(3)}"
    tmp.mkdir(parents=True)
    try:
        # ---- images: copies of the annotated screenshots; blocked ones re-decoded from the stored video
        blocked = [f for f in frames if f.get("outcome") == "UNREADABLE"]
        pictures: dict[int, Any] = {}
        if blocked:
            reader = frame_reader or ge.read_frames
            try:
                if not src.video_path or not Path(src.video_path).is_file():
                    raise ge.ExportError("the stored video is no longer on disk")
                pictures = reader(Path(src.video_path), [int(f["frame_idx"]) for f in blocked])
            except Exception as exc:  # noqa: BLE001 - a card is drawn instead
                log.warning("bundle %s: blocked screenshots not re-decoded: %s", job_id, exc)
        shots: list[dict[str, Any]] = []
        def weak_count(f: dict[str, Any]) -> int:
            try:
                return sum(1 for d in json.loads(f.get("result_json") or "{}").get("detections", [])
                           if d.get("status") == "weak")
            except ValueError:
                return 0

        for f in frames:
            idx, outcome = f["idx"], f.get("outcome")
            name = image_name(job_id, idx)
            c = by_idx.get(idx)
            ok = False
            if outcome in ANALYSED and f.get("annotated_path") and Path(f["annotated_path"]).is_file():
                shutil.copy2(f["annotated_path"], tmp / name)
                ok, source = True, f["annotated_path"]
            elif outcome == "UNREADABLE":
                import cv2
                img = _blocked_picture(pictures.get(int(f["frame_idx"])), idx, f.get("error") or "")
                ok = bool(cv2.imwrite(str(tmp / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90]))
                source = "re-decoded from the stored video" if pictures.get(int(f["frame_idx"])) is not None \
                    else "frame not decodable: placeholder card"
            else:
                source = f.get("annotated_path") or None
            if not ok:
                reasons.append(f"screenshot {idx}: no annotated image ({outcome or f.get('status')}"
                               f"{', file missing' if f.get('annotated_path') else ''})")
            if c is not None and ok:
                c.image_name = name
            shots.append({"idx": idx, "image": name if ok else None, "source": source, "outcome": outcome,
                          "detection_status": c.status if c else None,
                          "video_time_sec": f.get("video_time_sec"),
                          "timestamp": ge.format_ts(c.captured_at) if c and c.captured_at else None,
                          "litter_objects": len(c.litter) if c else 0,
                          "under_review": len(c.under_review) if c else 0,     # candidate only (CSV remark rule)
                          "weak_not_counted": weak_count(f) if c and c.status != ge.BLOCKED else 0})

        # ---- import CSV (only when every rule holds)
        rows: list[dict[str, str]] = []
        csv_name: Optional[str] = None
        if not reasons and store is not None:
            try:
                rows = [ge.build_row(c, store, allowed_classes) for c in sorted(checks, key=lambda c: c.idx or 0)]
                dups = _already_exported(day_dir, job_id, {(r["camera_id"], r["timestamp"]) for r in rows})
                if dups:
                    reasons.append("check(s) already exported in " + ", ".join(dups)
                                   + " (a camera check is exported at most once)")
                    rows = []
            except ge.ExportError as exc:
                reasons.append(ge.one_line(str(exc)))
                rows = []
        if rows:
            csv_name = f"{store.key}_{day}_{cam}_{job_id}.csv"
            (tmp / csv_name).write_text(ge.rows_to_csv(rows), encoding="utf-8", newline="")

        manifest: dict[str, Any] = {
            "schema": SCHEMA, "version": 1, "job_id": job_id, "import_ready": False,
            "store": store_info, "camera_id": cam,
            "camera_note": None if cam else "no camera assigned",
            "time": ({"source": tb.source, "detail": tb.detail, "recording_start": ge.format_ts(tb.start)} if tb
                     else {"source": None, "reason": time_reason}),
            "model_version": job.get("model_version"),
            "video": {"file": src.original_name, "duration_sec": src.duration_sec},
            "counts": {"screenshots": len(frames), "images": sum(1 for s in shots if s["image"]), "rows": len(rows),
                       **{s: sum(1 for c in checks if c.status == s) for s in ge.STATUSES},
                       "litter_objects": sum(len(c.litter) for c in checks),
                       "under_review_not_counted": sum(len(c.under_review) for c in checks),
                       # weak detections (below the candidate rule): recorded here only, never in the CSV, the
                       # image header or the dashboard
                       "weak_not_counted": sum(s["weak_not_counted"] for s in shots)},
            "csv": {"file": csv_name, "reasons": reasons},
            "screenshots": shots,
            "validator": {"ok": False, "issues": ["not run yet"]},
            "created_at": dt.datetime.now(ge.IST).isoformat(timespec="seconds"),
            "created_at_note": "processing time of the bundle - never used as a recording time",
        }
        man_path = tmp / ge.BUNDLE_MANIFEST
        man_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        issues = ge.validate_bundle(tmp, allowed_classes=allowed_classes,
                                    stores={store.key: store} if store else None,
                                    store_key=store.key if store else None,
                                    folder_date=None if day == TIME_UNKNOWN else day, job_id=job_id)
        if csv_name and issues:                 # never leave a CSV that failed validation
            (tmp / csv_name).unlink()
            reasons.append("bundle failed validation: " + "; ".join(map(str, issues[:5])))
            manifest["csv"]["file"] = None
            manifest["counts"]["rows"] = 0
        manifest["validator"] = {"ok": not issues, "issues": [str(i) for i in issues]}
        manifest["import_ready"] = not issues and manifest["csv"]["file"] is not None
        man_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    manifest["folder"] = str(final)
    return manifest


def auto_bundle(state: Any, job_id: str) -> dict[str, Any]:
    """Runner hook (always on, independent of LDM_GARBAGE_EXPORT). Call through export_guard.safe_result_bundle."""
    classes = set(getattr(state.engine, "class_names", []) or []) or None
    return build_bundle(state.db._query, job_id=job_id, settings=ge.settings_from_config(state.cfg),
                        allowed_classes=classes)


def summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "folder": manifest.get("folder"), "import_ready": manifest.get("import_ready"),
            "csv": manifest["csv"]["file"], "reasons": manifest["csv"]["reasons"], "counts": manifest["counts"],
            "validator": manifest["validator"]}


def main(argv: Optional[list[str]] = None) -> int:
    from backend.core.config import AppConfig
    cfg = AppConfig()
    ap = argparse.ArgumentParser(prog="python -m backend.services.result_bundle", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sampled-job", required=True)
    ap.add_argument("--store-key")
    ap.add_argument("--camera-id")
    ap.add_argument("--recording-start", help="ISO 8601; used only when no stronger recording-time source exists")
    ap.add_argument("--store-config", type=Path, default=cfg.store_config_path)
    ap.add_argument("--out", type=Path, default=cfg.garbage_export_dir)
    ap.add_argument("--db", type=Path, default=cfg.db_path)
    a = ap.parse_args(argv)
    try:
        q = ge.sqlite_query(a.db)
        m = build_bundle(q, job_id=a.sampled_job, settings=ge.Settings(a.store_config, a.out, cfg.garbage_export_ledger,
                                                                       cfg.trust_container_time),
                         store_key=a.store_key, camera_id=a.camera_id, recording_start=a.recording_start,
                         allowed_classes=ge._active_classes(q))
    except ge.ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary(m), indent=2, default=str))
    return 0 if m["import_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
