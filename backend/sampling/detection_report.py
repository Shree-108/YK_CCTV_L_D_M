"""Detection report of a four-screenshot analysis: ``reports/detection_report.csv`` (+ ``analysis_summary.csv`` and
``analysis_summary.json``) in the job folder. Written for every completed job, before the job is marked completed.

Record types in detection_report.csv (column ``record_type``):
  SCREENSHOT  exactly one row per screenshot (always 4): its status - LITTER_OBSERVED, NO_LITTER_OBSERVED,
              UNREADABLE or FAILED. A screenshot without litter is therefore never
              just "missing" from the file.
  DETECTION   one row per detected object per screenshot: confirmed litter and possible litter (candidate, below the
              confirmed threshold). Weak and filtered detections (size, aspect, class) are not listed here; the
              standard detection CSV (backend/services/litter_reports.py) lists every detection.

Detection runs on the full frame; there is no region of interest (removed 2026-11-19). The ROI / pavement columns of
reports written before that date are not produced any more.

Job-wide totals are NOT repeated on every row (summing them in a spreadsheet would multiply them); they are in
analysis_summary.csv (one row) and analysis_summary.json. Only the per-screenshot and overall *status* texts are on
every row. Original model confidence (confidence_score) is kept separate from the cross-screenshot evidence
(object_sightings, object_max_confidence).

Cross-screenshot linking: the four screenshots of a fixed camera are seconds to minutes apart, so the same piece of
litter appears in several of them at the same place. Detections in different screenshots are linked to one
estimated object with the existing alert de-duplication rule (similar class per class_metadata similarity groups,
and box IoU >= alerts.dedup_iou or centre distance <= alerts.dedup_center_frac of the frame diagonal). This is an
estimate: a moved camera, or two similar objects swapping places, defeats it. It is NOT the tracker's stationary
confirmation, which needs consecutive frames that four screenshots do not provide.
"""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from backend.core.csv_safety import safe_rows

LISTED = ("confirmed", "candidate")          # objects that get an object id
SEVERITIES = ("HIGH", "MEDIUM", "LOW")
REPRESENTATIVE_NOTE = ("Four representative screenshots were analysed (10/35/65/90 % of the video, or as recorded "
                       "per screenshot); the rest of the video was not analysed.")

DETECTION_FIELDS = [
    # A. video and job
    "job_id", "video_filename", "video_duration_seconds", "video_resolution", "video_fps", "total_video_frames",
    "analysis_started_at", "analysis_completed_at", "processing_duration_seconds", "store_id", "camera_id",
    # B. screenshot
    "screenshot_id", "screenshot_number", "screenshot_timestamp_seconds", "frame_number", "screenshot_filename",
    "annotated_screenshot_filename",
    # C. detection
    "record_type", "detection_id", "object_track_id", "litter_class", "confidence_score",
    "bounding_box_x1", "bounding_box_y1", "bounding_box_x2", "bounding_box_y2", "bounding_box_width",
    "bounding_box_height", "object_area_pixels", "object_area_percentage",
    # E. alert and classification
    "detection_status", "severity", "alert_id", "stationary_confirmation_status", "duplicate_detection_status",
    "final_classification", "review_required",
    # F. status (the numeric totals are in analysis_summary.csv)
    "screenshot_analysis_status", "overall_analysis_status",
    # traceability
    "planned_frame_number", "sampling_adjustment", "object_sightings", "object_max_confidence", "rejection_reason",
    "model_version",
]

SUMMARY_FIELDS = [
    "job_id", "video_filename", "video_duration_seconds", "video_resolution", "video_fps", "total_video_frames",
    "analysis_started_at", "analysis_completed_at", "processing_duration_seconds", "store_id", "camera_id",
    "analysis_mode", "sampling_strategy", "screenshots_planned", "screenshots_analysed", "screenshots_failed",
    "sampling_adjustments",
    "total_objects_detected", "unique_objects_estimated", "total_high_severity", "total_medium_severity",
    "total_low_severity", "possible_litter_objects", "alerts_created",
    "overall_analysis_status", "review_required", "model_version", "note",
]

SCREENSHOT_STATUS = {"LITTER_DETECTED": "LITTER_OBSERVED", "NO_LITTER": "NO_LITTER_OBSERVED",
                     "UNREADABLE": "UNREADABLE", "FAILED": "FAILED"}
ANALYSED = ("LITTER_OBSERVED", "NO_LITTER_OBSERVED")


# ---------------------------------------------------------------------- linking
def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link_objects(frames: list[dict[str, Any]], *, iou_min: float, center_frac: float,
                 similar: Callable[[str, str], bool]) -> dict[str, dict[str, Any]]:
    """Give every confirmed/candidate detection an ``object_id`` (written into the detection dicts in place) and
    return the estimated objects. A detection joins an object first seen in an EARLIER screenshot (never one from
    the same screenshot: two boxes in one picture are two objects); best IoU wins, then the smallest distance."""
    objects: dict[str, dict[str, Any]] = {}
    for f in sorted(frames, key=lambda x: x["idx"]):
        res = f.get("result") or {}
        size = res.get("image_size") or [0, 0]
        diag = math.hypot(*size) or 1.0
        used: set[str] = set()
        dets = sorted((d for d in res.get("detections", []) if d.get("status") in LISTED),
                      key=lambda d: -d["confidence"])
        for d in dets:
            cx, cy = (d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2
            best: Optional[tuple[float, float, str]] = None
            for oid, o in objects.items():
                if oid in used or o["last_screenshot"] >= f["idx"] or not similar(d["class"], o["class"]):
                    continue
                ob = o["bbox"]
                iou = _iou(d["bbox"], ob)
                dist = math.hypot(cx - (ob[0] + ob[2]) / 2, cy - (ob[1] + ob[3]) / 2) / diag
                if (iou >= iou_min or dist <= center_frac) and (best is None or (iou, -dist) > (best[0], -best[1])):
                    best = (iou, dist, oid)
            if best is None:
                oid = f"OBJ-{len(objects) + 1:03d}"
                objects[oid] = {"object_id": oid, "class": d["class"], "sightings": [], "confirmed_sightings": 0,
                                "max_confidence": 0.0, "severity": None, "alert_id": None, "last_screenshot": 0}
            else:
                oid = best[2]
            o = objects[oid]
            used.add(oid)
            d["object_id"] = oid
            o["bbox"], o["last_screenshot"] = d["bbox"], f["idx"]
            o["sightings"].append(f["idx"])
            o["max_confidence"] = max(o["max_confidence"], float(d["confidence"]))
            if d.get("status") == "confirmed":
                o["confirmed_sightings"] += 1
                if d.get("severity") and (o["severity"] is None or
                                          SEVERITIES.index(d["severity"]) < SEVERITIES.index(o["severity"])):
                    o["severity"] = d["severity"]
            if d.get("alert_id") and not o["alert_id"]:
                o["alert_id"] = d["alert_id"]
    for o in objects.values():
        o.pop("bbox", None)
        o.pop("last_screenshot", None)
    return objects


# ---------------------------------------------------------------------- rows
def _r(v: Optional[float], n: int = 3) -> Any:
    return "" if v is None else round(float(v), n)


def _name(path: Optional[str]) -> str:
    return Path(path).name if path else ""


def _duration(start: Optional[str], end: Optional[str]) -> Any:
    if not start or not end:
        return ""
    import datetime as dt
    try:
        return round((dt.datetime.fromisoformat(end) - dt.datetime.fromisoformat(start)).total_seconds(), 3)
    except ValueError:
        return ""


def screenshot_status(f: dict[str, Any]) -> str:
    return SCREENSHOT_STATUS.get(f.get("outcome") or "", "FAILED" if f.get("status") == "failed" else "NOT_ANALYSED")


def overall_status(frames: list[dict[str, Any]], failed: bool = False) -> str:
    st = [screenshot_status(f) for f in frames]
    analysed = [s for s in st if s in ANALYSED]
    if failed or not analysed:
        return "FAILED"
    litter = "LITTER_OBSERVED" in analysed
    if len(analysed) < len(st):
        return "INCOMPLETE_LITTER_OBSERVED" if litter else "INCOMPLETE_NO_LITTER_IN_ANALYSED_SCREENSHOTS"
    return "LITTER_OBSERVED" if litter else "NO_LITTER_OBSERVED"


def build(job: dict[str, Any], media: dict[str, Any], frames: list[dict[str, Any]], objects: dict[str, dict[str, Any]],
          *, completed_at: str, failed: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(detection rows, summary) of a four-screenshot job (full-frame detection)."""
    frames = sorted(frames, key=lambda x: x["idx"])
    w, h = media.get("width") or 0, media.get("height") or 0
    frame_area = float(w * h) or 0.0
    overall = overall_status(frames, failed)
    job_cols = {
        "job_id": job["id"], "video_filename": media.get("filename", ""),
        "video_duration_seconds": _r(media.get("duration_sec")), "video_resolution": f"{w}x{h}" if w and h else "",
        "video_fps": _r(media.get("fps")), "total_video_frames": media.get("frames", ""),
        "analysis_started_at": job.get("started_at") or "", "analysis_completed_at": completed_at,
        "processing_duration_seconds": _duration(job.get("started_at"), completed_at),
        "store_id": job.get("store_id") or "", "camera_id": job.get("camera_id") or "",
    }
    rows: list[dict[str, Any]] = []
    for f in frames:
        res = f.get("result") or {}
        sstat = screenshot_status(f)
        shot = {
            "screenshot_id": f"{job['id']}-S{f['idx']}", "screenshot_number": f["idx"],
            "screenshot_timestamp_seconds": _r(f.get("video_time_sec")), "frame_number": f.get("frame_idx", ""),
            "screenshot_filename": _name(f.get("original_path")),
            "annotated_screenshot_filename": _name(f.get("annotated_path")),
            "planned_frame_number": f.get("planned_frame_idx") if f.get("planned_frame_idx") is not None else f.get("frame_idx", ""),
            "sampling_adjustment": f.get("sampling_note") or "",
            "screenshot_analysis_status": sstat, "overall_analysis_status": overall,
            "model_version": res.get("model_version") or job.get("model_version") or "",
        }
        rows.append({**job_cols, **shot, "record_type": "SCREENSHOT",
                     "detection_status": "NONE" if sstat in ANALYSED else "",
                     "final_classification": sstat,
                     "review_required": "TRUE" if sstat not in ANALYSED else "FALSE",
                     "stationary_confirmation_status": "NOT_APPLICABLE", "duplicate_detection_status": "NOT_APPLICABLE"})
        listed = [d for d in res.get("detections", []) if d.get("status") in LISTED]
        for d in sorted(listed, key=lambda d: -d["confidence"]):
            x1, y1, x2, y2 = (float(v) for v in d["bbox"])
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            o = objects.get(d.get("object_id") or "")
            status = d.get("status", "")
            final = "CONFIRMED_LITTER" if status == "confirmed" else "POSSIBLE_LITTER"
            review = "TRUE" if status != "confirmed" else "FALSE"
            n_seen = len(o["sightings"]) if o else 0
            rows.append({
                **job_cols, **shot, "record_type": "DETECTION",
                "detection_id": f"{job['id']}-S{f['idx']}-D{d.get('track_id', 0):02d}",
                "object_track_id": d.get("object_id") or "", "litter_class": d.get("class", ""),
                "confidence_score": round(float(d["confidence"]), 4),
                "bounding_box_x1": round(x1, 1), "bounding_box_y1": round(y1, 1),
                "bounding_box_x2": round(x2, 1), "bounding_box_y2": round(y2, 1),
                "bounding_box_width": round(bw, 1), "bounding_box_height": round(bh, 1),
                "object_area_pixels": round(bw * bh, 1),
                "object_area_percentage": round(100 * bw * bh / frame_area, 4) if frame_area else "",
                "detection_status": status.upper(),
                "severity": d.get("severity") or "",
                "alert_id": (o or {}).get("alert_id") or d.get("alert_id") or "",
                "stationary_confirmation_status": ("NOT_APPLICABLE" if not o else
                                                   "REPEATED_AT_SAME_POSITION" if n_seen > 1 else "SINGLE_OBSERVATION"),
                "duplicate_detection_status": ("NOT_APPLICABLE" if not o else "UNIQUE" if n_seen == 1 else
                                               "FIRST_SIGHTING" if o["sightings"][0] == f["idx"] else "REPEAT_SIGHTING"),
                "final_classification": final, "review_required": review,
                "object_sightings": n_seen or "", "object_max_confidence": round(o["max_confidence"], 4) if o else "",
                "rejection_reason": d.get("reason") or "",
            })

    confirmed_objects = [o for o in objects.values() if o["confirmed_sightings"]]
    sev = {s: sum(1 for o in confirmed_objects if o["severity"] == s) for s in SEVERITIES}
    statuses = [screenshot_status(f) for f in frames]
    all_dets = [d for f in frames for d in (f.get("result") or {}).get("detections", [])]
    summary = {
        **job_cols,
        "analysis_mode": job.get("mode") or "manual", "sampling_strategy": job.get("sampling") or "",
        "screenshots_planned": job.get("sample_count") or len(frames),
        "screenshots_analysed": sum(s in ANALYSED for s in statuses),
        "screenshots_failed": sum(s in ("UNREADABLE", "FAILED") for s in statuses),
        "sampling_adjustments": sum(1 for f in frames if f.get("sampling_note")),
        "total_objects_detected": sum(1 for d in all_dets if d.get("status") == "confirmed"),
        "unique_objects_estimated": len(confirmed_objects),
        "total_high_severity": sev["HIGH"], "total_medium_severity": sev["MEDIUM"], "total_low_severity": sev["LOW"],
        "possible_litter_objects": sum(1 for o in objects.values() if not o["confirmed_sightings"]),
        "alerts_created": len({o["alert_id"] for o in objects.values() if o["alert_id"]}),
        "overall_analysis_status": overall,
        "review_required": "TRUE" if (overall not in ("LITTER_OBSERVED", "NO_LITTER_OBSERVED")
                                      or any(not o["confirmed_sightings"] for o in objects.values())) else "FALSE",
        "model_version": job.get("model_version") or next(
            ((f.get("result") or {}).get("model_version") for f in frames if (f.get("result") or {}).get("model_version")), ""),
        "note": REPRESENTATIVE_NOTE,
    }
    return rows, summary


def to_csv(rows: Iterable[dict[str, Any]], fields: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(safe_rows(rows))          # file names / error text cannot start a spreadsheet formula
    return buf.getvalue()


def summary_json(summary: dict[str, Any], objects: dict[str, dict[str, Any]], frames: list[dict[str, Any]]) -> str:
    return json.dumps({
        "summary": summary,
        "severity_counted_per": "estimated unique object (highest severity of its confirmed sightings)",
        "screenshots": [{"number": f["idx"], "frame_number": f.get("frame_idx"),
                         "planned_frame_number": f.get("planned_frame_idx", f.get("frame_idx")),
                         "timestamp_seconds": f.get("video_time_sec"), "status": screenshot_status(f),
                         "sampling_adjustment": f.get("sampling_note"), "error": f.get("error"),
                         "screenshot_filename": _name(f.get("original_path")),
                         "annotated_screenshot_filename": _name(f.get("annotated_path"))} for f in frames],
        "objects": list(objects.values()),
    }, indent=2)
