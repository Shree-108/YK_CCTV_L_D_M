"""CSV report of a four-screenshot analysis.

Columns follow the existing LDM detection export where they overlap (class, confidence, x1..y2 in original-frame
pixels, status, frame_idx, job_id, model_version) and add the video/store/screenshot context.

Granularity: one row per confirmed litter detection; a screenshot without a confirmed detection (no litter,
uncertain ROI, unreadable, failed) still gets exactly one row, so all four screenshots are always traceable.
Unknown store, camera or recording date stay blank - nothing is invented.
"""
from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from backend.core.csv_safety import safe_cell, safe_rows
from backend.sampling.plan import clock_time, hhmmss, parse_recording_start

FIELDS = ["video_name", "store_id", "camera_id", "recording_date", "video_timestamp", "recording_timestamp",
          "screenshot", "frame_idx", "class", "confidence", "x1", "y1", "x2", "y2", "status", "detection_count",
          "outcome", "roi_status", "remark", "evidence_image", "job_id", "model_version"]

OUTCOMES = ("LITTER_DETECTED", "NO_LITTER", "ROI_UNCERTAIN", "UNREADABLE", "FAILED")


def remark(outcome: Optional[str], n: int, roi_source: Optional[str], error: Optional[str]) -> str:
    where = "the whole frame (no ROI)" if roi_source == "whole_frame" else "the outdoor ROI"
    if outcome == "LITTER_DETECTED":
        return f"{n} litter object{'s' if n != 1 else ''} detected inside {where}."
    if outcome == "NO_LITTER":
        return f"No litter detected inside {where}."
    if outcome == "ROI_UNCERTAIN":
        return "Not analysed: the outdoor area could not be confirmed for this screenshot."
    if outcome == "UNREADABLE":
        return f"Not analysed: the screenshot is unreadable ({error or 'no picture'})."
    if outcome == "FAILED":
        return f"Processing failed: {error or 'unknown error'}"
    return "Not analysed yet."


def _rel(path: Optional[str], storage_dir: Path) -> str:
    if not path:
        return ""
    try:
        return Path(path).resolve().relative_to(storage_dir.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def rows_for_job(job: dict[str, Any], media: dict[str, Any], frames: Iterable[dict[str, Any]],
                 storage_dir: Path) -> list[dict[str, Any]]:
    try:
        start = parse_recording_start(job.get("recording_start"))
    except ValueError:
        start = None
    out: list[dict[str, Any]] = []
    # an automatic job without an approved ROI analysed the whole frame: never call that "the outdoor ROI"
    roi_source = ("whole_frame" if job.get("roi_status") in ("not_configured", "disabled_in_settings")
                  else job.get("roi_source"))
    for f in sorted(frames, key=lambda x: x["idx"]):
        t = f.get("video_time_sec")
        clock = clock_time(start, t)
        res = f.get("result") or {}
        confirmed = [d for d in res.get("detections", []) if d.get("status") == "confirmed"]
        base = {
            "video_name": media.get("filename", ""), "store_id": job.get("store_id") or "",
            "camera_id": job.get("camera_id") or "", "recording_date": clock.date().isoformat() if clock else "",
            "video_timestamp": hhmmss(t, millis=True), "recording_timestamp": clock.isoformat(timespec="milliseconds") if clock else "",
            "screenshot": Path(f["original_path"]).name if f.get("original_path") else "",
            "frame_idx": f.get("frame_idx"), "detection_count": len(confirmed), "outcome": f.get("outcome") or "",
            "roi_status": f.get("roi_status") or job.get("roi_status") or "",
            "remark": remark(f.get("outcome"), len(confirmed), roi_source, f.get("error")),
            "evidence_image": _rel(f.get("annotated_path") or f.get("original_path"), storage_dir),
            "job_id": job["id"], "model_version": res.get("model_version") or job.get("model_version") or "",
        }
        if confirmed:
            for d in sorted(confirmed, key=lambda d: -d["confidence"]):
                x1, y1, x2, y2 = (round(float(v), 1) for v in d["bbox"])
                out.append({**base, "class": d["class"], "confidence": round(float(d["confidence"]), 4),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "status": "confirmed"})
        else:
            out.append({**base, "class": "", "confidence": "", "x1": "", "y1": "", "x2": "", "y2": "", "status": ""})
    return out


def to_csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()                      # the header is written even with no rows
    w.writerows(safe_rows(rows))         # file names / error text cannot start a spreadsheet formula
    return buf.getvalue()


def sanitize_csv_text(text: str) -> str:
    """A stored report CSV (possibly written before formula protection existed) with every data cell made safe.
    The header row is kept as is; a file that is already safe comes back byte-identical."""
    rows = list(csv.reader(io.StringIO(text, newline="")))
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\r\n")         # the csv module's default, as written by to_csv
    for i, row in enumerate(rows):
        w.writerow(row if i == 0 else [safe_cell(c) for c in row])
    return buf.getvalue()


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)
