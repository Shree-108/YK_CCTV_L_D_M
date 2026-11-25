"""Upload a finished four-screenshot job to S3 (layout + write rules: backend/core/s3.py).

For one job (cart from config/store_config.json; a store that cannot be resolved goes to ``cart_unassigned``):

1. the source video -> ``<recording day>/cart_<N>/input_data/<safe name>`` (recording day = earliest screenshot time,
   India time; skipped when no screenshot has a trustworthy time). Uploaded once: same bytes already there = done.
2. each screenshot's annotated image -> ``<today>/cart_<N>/output_data/cart<N>_<today>_pic<K>.jpg``, K continuing
   after the highest pic already in the folder or the CSV.
3. one row per screenshot appended to ``<today>/cart_<N>/output_data/cart<N>_<today>.csv`` (CSV_COLUMNS).
   A job already in that CSV is not added again (so a re-run never duplicates rows).

The same clean / litter_detected / view_blocked status, class, confidence and remark as the daily CSV
(backend/services/daily_csv.job_entries -> garbage_export). Runner hook: ``submit_job`` - runs on one background
worker and never raises, so S3 can never fail or slow down a scan.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from backend.core import s3 as s3l
from backend.services import daily_csv as dc

log = logging.getLogger(__name__)

CSV_COLUMNS = ["pic", "imageKey", "capturedAt", "cameraId", "storeKey", "storeName", "cartId", "litterClass",
               "confidenceScore", "detectionStatus", "remark", "sourceFile", "videoKey", "jobId", "uploadedAt"]
UNASSIGNED_CART = "cart_unassigned"
SHOT_RE = re.compile(r"_s(\d+)\.jpg$")

_POOL: Optional[ThreadPoolExecutor] = None


def cart_of(cart_id: str) -> str:
    return s3l.entity_id(cart_id) if str(cart_id or "").strip() else UNASSIGNED_CART


def upload_job(query: Any, settings: Any, job_id: str, *, s3: Any, bucket: str,
               image_path: Callable[[str, int], Optional[Path]], allowed_classes: Optional[set[str]] = None,
               processing_day: Optional[dt.date] = None) -> dict[str, Any]:
    info, entries, no_time = dc.job_entries(query, settings, job_id, allowed_classes)
    eid = cart_of(info["cart_id"])
    day = processing_day or s3l.today()
    csv_key = s3l.daily_csv_key(day, eid)
    res: dict[str, Any] = {"ok": True, "job_id": job_id, "cart": eid, "csv": csv_key, "no_time": no_time,
                           "video": None, "images": [], "appended": 0}
    if not entries:
        res["skipped"] = "no screenshot with a recording time"
        return res

    text, _ = s3l.read_text(s3, bucket, csv_key)
    if any(r.get("jobId") == job_id for r in s3l.csv_rows(text)):
        res["skipped"] = "job already in the CSV"
        return res

    # 1. input video, under the recording day
    video_key = ""
    vpath = info.get("video_path")
    if vpath and Path(vpath).is_file():
        rec_day = min(dt.datetime.fromisoformat(e["capturedAt"]).astimezone(s3l.IST).date() for e in entries)
        video_key = s3l.input_key(rec_day, eid, info["video"])
        res["video"] = s3l.put_new_or_same(s3, bucket, video_key, Path(vpath), "video/mp4")
    else:
        res["video"] = {"skipped": "stored video not found"}

    # 2. images, numbered after the highest pic already used today
    k = s3l.max_pic(s3l.list_keys(s3, bucket, s3l.output_prefix(day, eid)), s3l.csv_rows(text))
    rows = []
    now = dc.now_iso()
    for e in entries:
        m = SHOT_RE.search(e["imageName"])
        p = image_path(job_id, int(m.group(1))) if m else None
        pic = img_key = ""
        if p is not None and Path(p).is_file():
            while True:
                k += 1
                try:
                    s3l.put_new(s3, bucket, s3l.image_key(day, eid, k), Path(p), "image/jpeg")
                    break
                except s3l.Exists:
                    continue                       # another writer took pic<k>; try the next number
            pic, img_key = f"pic{k}", s3l.image_key(day, eid, k)
            res["images"].append(img_key)
        rows.append({"pic": pic, "imageKey": img_key, "capturedAt": e["capturedAt"], "cameraId": e["cameraId"],
                     "storeKey": info["store_key"], "storeName": info["store_name"], "cartId": info["cart_id"],
                     "litterClass": e["litterClass"], "confidenceScore": e["confidenceScore"],
                     "detectionStatus": e["detectionStatus"], "remark": e["remark"], "sourceFile": e["sourceFile"],
                     "videoKey": video_key, "jobId": job_id, "uploadedAt": now})

    # 3. CSV rows (re-checked against the latest file on every attempt)
    def new_rows(existing: list[dict[str, str]]) -> list[dict[str, str]]:
        return [] if any(r.get("jobId") == job_id for r in existing) else rows

    res["appended"] = s3l.append_csv(s3, bucket, csv_key, CSV_COLUMNS, new_rows)["appended"]
    return res


def upload_state_job(state: Any, job_id: str) -> dict[str, Any]:
    """Upload one job using the running app's DB / config. Never raises."""
    try:
        from backend.services import garbage_export as ge
        from backend.services import results_view as rv
        cfg = s3l.S3Config.from_env()
        query = state.db._query
        classes = set(getattr(state.engine, "class_names", []) or []) or None
        res = upload_job(query, ge.settings_from_config(state.cfg), job_id, s3=s3l.client(cfg), bucket=cfg.bucket,
                         image_path=lambda j, i: rv.image_path(query, j, i), allowed_classes=classes)
        log.info("S3 upload: %s", res)
        return res
    except Exception as exc:  # noqa: BLE001 - an upload problem must never change the finished scan
        log.exception("S3 upload failed for %s; the scan itself is unaffected", job_id)
        return {"ok": False, "job_id": job_id, "error": f"{type(exc).__name__}: {exc}"}


def submit_job(state: Any, job_id: str) -> Optional[Future]:
    """Runner hook: queue the upload on one background worker (keeps CSV appends in order). No-op when off."""
    global _POOL
    if not getattr(state.cfg, "s3_upload_enabled", False):
        return None
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="s3-upload")
    return _POOL.submit(upload_state_job, state, job_id)
