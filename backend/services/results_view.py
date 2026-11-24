"""Dashboard "Results" view: ONE ROW = ONE CHECK of one camera (one screenshot of a four-screenshot job), with the
nine importer CSV fields - clean and view_blocked checks included, not only litter.

The fields are produced by backend/services/garbage_export.py (sampled_checks + describe + build_row), so the view,
the importer CSV download and the result bundles can never drift apart. Unlike the export, a row is shown even when
it is not import-ready, with the reason:
  * no trustworthy recording time -> timestamp None, "time unknown" warning (processing time is never shown instead)
  * camera_id empty or "upload"   -> "no camera assigned"
  * store / camera not in config/store_config.json -> cart_id and store_name empty, with the config problem
The importer CSV download (``importer_csv``) contains only import-ready rows, built by garbage_export.build_row
exactly as the exporter and the bundles build them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backend.services import garbage_export as ge

NO_CAMERA = "no camera assigned"
BUNDLE_STATUS = "result_bundle.json"


@dataclass
class Filters:
    store: Optional[str] = None          # storeKey (or the store id typed on the job)
    camera: Optional[str] = None
    status: Optional[str] = None         # clean | litter_detected | view_blocked
    litter_class: Optional[str] = None
    min_confidence: Optional[float] = None   # 0-100, as in the CSV
    date_from: Optional[str] = None      # YYYY-MM-DD, recording date (rows without a recording time never match)
    date_to: Optional[str] = None
    job_id: Optional[str] = None

    def match(self, r: dict[str, Any]) -> bool:
        if self.store and self.store not in (r["store_key"], r["job_store_id"]):
            return False
        if self.camera and r["camera_id"] != self.camera:
            return False
        if self.status and r["detection_status"] != self.status:
            return False
        if self.litter_class and r["litter_class"] != self.litter_class:
            return False
        if self.min_confidence is not None and self.min_confidence > 0:
            if not r["confidence_score"] or float(r["confidence_score"]) < self.min_confidence:
                return False
        day = (r["timestamp"] or "")[:10]
        if (self.date_from or self.date_to) and not day:
            return False
        if self.date_from and day < self.date_from:
            return False
        if self.date_to and day > self.date_to:
            return False
        if self.job_id and r["job_id"] != self.job_id:
            return False
        return True


def _bundle_status(output_dir: Optional[str]) -> Optional[dict[str, Any]]:
    if not output_dir:
        return None
    p = Path(output_dir) / "reports" / BUNDLE_STATUS
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect(query: ge.Query, settings: ge.Settings, allowed_classes: Optional[set[str]] = None,
            ) -> tuple[list[dict[str, Any]], list[str]]:
    """All checks of all finished four-screenshot jobs, newest job first. Returns (rows, problems); each row keeps
    its import-ready CSV row (or why there is none) under ``_import``."""
    try:
        stores: Optional[dict[str, ge.Store]] = ge.load_store_config(settings.store_config)
        config_error = ""
    except ge.ExportError as exc:
        stores, config_error = None, ge.one_line(str(exc))
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for job in query("SELECT * FROM sampled_jobs WHERE state='done' ORDER BY created_at DESC, id DESC", ()):
        jid = job["id"]
        try:
            src = ge.sampled_source(query, jid)
        except ge.ExportError as exc:
            problems.append(f"{jid}: {exc}")
            continue
        raw_cam = (src.camera_id or "").strip()
        cam = "" if raw_cam.lower() in ("", "upload", "unknown") else raw_cam
        store: Optional[ge.Store] = None
        store_issue = ""
        if stores is None:
            store_issue = config_error
        elif not cam:
            store_issue = NO_CAMERA
        else:
            try:
                store = ge.store_for(stores, ge._known(src.store_key), cam)
            except ge.ExportError as exc:
                store_issue = ge.one_line(str(exc))
        tb: Optional[ge.TimeBase] = None
        time_reason = ""
        try:
            tb = ge.resolve_time_base(video_path=src.video_path, original_name=src.original_name,
                                      duration_sec=src.duration_sec, manual_start=src.manual_start,
                                      manual_label="operator-entered start",
                                      trust_container_time=settings.trust_container_time)
        except ge.ExportError as exc:
            time_reason = ge.one_line(str(exc))
        try:
            checks, _excluded = ge.sampled_checks(query, src, cam, tb, time_reason)
        except ge.ExportError as exc:
            problems.append(f"{jid}: {exc}")
            continue
        bundle = _bundle_status(job.get("output_dir"))
        for c in checks:
            try:
                cls_out, conf_out, remark = ge.describe(c, allowed_classes)
            except ge.ExportError as exc:
                problems.append(f"{jid} screenshot {c.idx}: {exc}")
                continue
            imp: Optional[dict[str, str]] = None
            imp_issue = store_issue
            if store is not None:
                try:
                    imp = ge.build_row(c, store, allowed_classes)
                except ge.ExportError as exc:
                    imp_issue = ge.one_line(str(exc))
            rows.append({
                "job_id": jid, "idx": c.idx, "job_created_at": job.get("created_at"),
                "timestamp": ge.format_ts(c.captured_at) if c.captured_at else None,
                "time_known": c.captured_at is not None,
                "time_warning": None if c.captured_at else f"time unknown: {time_reason or 'no recording start'}",
                "camera_id": cam, "camera_raw": raw_cam or None,
                "camera_flag": None if cam else NO_CAMERA,
                "image_name": c.image_name,
                "image_url": f"/api/results/checks/{jid}/{c.idx}/image.jpg",
                "litter_class": cls_out, "confidence_score": conf_out, "detection_status": c.status,
                "remark": remark,
                "litter_objects": len(c.litter), "under_review": len(c.under_review),   # candidate only
                "cart_id": str(store.cart_id) if store else "", "store_name": store.name if store else "",
                "store_key": store.key if store else "", "job_store_id": ge._known(src.store_key) or "",
                "store_flag": store_issue or None,
                "import_ready": imp is not None, "import_issue": None if imp is not None else imp_issue,
                "bundle": ({k: bundle.get(k) for k in ("ok", "folder", "import_ready", "csv", "reasons", "error")}
                           if bundle else None),
                "_import": imp,
            })
    return rows, problems


def public(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def view(query: ge.Query, settings: ge.Settings, flt: Filters, *, limit: int = 100, offset: int = 0,
         allowed_classes: Optional[set[str]] = None) -> dict[str, Any]:
    rows, problems = collect(query, settings, allowed_classes)
    hit = [r for r in rows if flt.match(r)]
    summary = {"total": len(hit), **{s: sum(1 for r in hit if r["detection_status"] == s) for s in ge.STATUSES},
               "time_unknown": sum(1 for r in hit if not r["time_known"]),
               "no_camera": sum(1 for r in hit if r["camera_flag"]),
               "import_ready": sum(1 for r in hit if r["import_ready"])}
    return {"rows": [public(r) for r in hit[offset:offset + limit]], "total": len(hit), "summary": summary,
            "columns": ge.COLUMNS, "problems": problems,
            "options": {"stores": sorted({r["store_key"] or r["job_store_id"] for r in rows} - {""}),
                        "cameras": sorted({r["camera_id"] for r in rows} - {""}),
                        "classes": sorted({r["litter_class"] for r in rows} - {""})}}


def importer_csv(query: ge.Query, settings: ge.Settings, flt: Filters,
                 allowed_classes: Optional[set[str]] = None) -> tuple[str, int, int]:
    """(csv text, rows written, rows left out because they are not import-ready). Rows in recording-time order."""
    rows, _ = collect(query, settings, allowed_classes)
    hit = [r for r in rows if flt.match(r)]
    ready = sorted((r for r in hit if r["_import"]), key=lambda r: (r["timestamp"], r["camera_id"], r["job_id"],
                                                                   r["idx"] or 0))
    return ge.rows_to_csv([r["_import"] for r in ready]), len(ready), len(hit) - len(ready)


def image_path(query: ge.Query, job_id: str, idx: int) -> Optional[Path]:
    """The annotated image of a check: the bundle's copy (also exists for view_blocked screenshots), else the
    job's own annotated screenshot."""
    jobs = query("SELECT output_dir FROM sampled_jobs WHERE id=?", (job_id,))
    if not jobs:
        return None
    b = _bundle_status(jobs[0].get("output_dir"))
    if b and b.get("folder"):
        p = Path(b["folder"]) / ge.bundle_image_name(job_id, idx)
        if p.is_file():
            return p
    fr = query("SELECT annotated_path FROM sampled_frames WHERE job_id=? AND idx=?", (job_id, idx))
    if fr and fr[0].get("annotated_path") and Path(fr[0]["annotated_path"]).is_file():
        return Path(fr[0]["annotated_path"])
    return None
