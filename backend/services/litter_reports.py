"""Standard LDM litter outputs: detection CSV, summary CSV, metadata JSON and the LDM_OUTPUT folder layout.

Used by every analysis kind (single image, full video with tracking, four-screenshot quick check), so a report always
has the same columns whatever produced it. Everything here is derived from the analysis' own observations and events;
nothing re-runs the model or invents a value. Column contracts: docs/ldm_litter_focus/API_CONTRACT.md.

Terms (owner brief 2026-11-19):
  observation  one detection of an object in one processed frame (a row of the detection CSV)
  litter event one DETECTED object (image: confidence rule; video: temporally confirmed track). The same object
               seen in 15 frames is ONE event with observation_count 15 - never 15 events.

Folder layout (nothing is ever overwritten - every file name carries the analysis id):
    <root>/<STORE>/<YYYY>/<MM>/<DD>/<CAMERA>/{input,annotated,detections,reports,metadata}/
    <STORE>_<YYYY_MM_DD>_<CAMERA>_<analysis_id>_detections.csv   (detections/)
    <STORE>_<YYYY_MM_DD>_<CAMERA>_<analysis_id>_summary.csv      (reports/)
    <STORE>_<YYYY_MM_DD>_<CAMERA>_<analysis_id>_metadata.json    (metadata/)
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import re
import secrets
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from backend.core.csv_safety import safe_rows
from backend.detection.types import CANDIDATE, CONFIRMED, REJECTED, WEAK

UNKNOWN = "UNKNOWN"

# ----------------------------------------------------------------------------------------------- vocabulary
DETECTED, TRACKED, UNDER_REVIEW, REJECTED_S, VERIFIED, RESOLVED = (
    "DETECTED", "TRACKED", "UNDER_REVIEW", "REJECTED", "VERIFIED", "RESOLVED")
REVIEW_STATUSES = (DETECTED, VERIFIED, REJECTED_S, RESOLVED)          # what an operator may set on an event

RUNNING, COMPLETED_LITTER, COMPLETED_NONE, FAILED = (
    "RUNNING", "COMPLETED_LITTER_DETECTED", "COMPLETED_NO_LITTER_DETECTED", "FAILED")

# the finalized seven-class schema (ids 0..6). Summary columns count events of exactly these names.
LDM7_CLASSES = ("plastic_packet", "plastic_bottle", "paper", "food_waste", "other_litter", "glass", "cardboard")

DETECTION_COLUMNS = [
    "store_id", "store_name", "camera_id", "camera_name", "recording_id", "timestamp", "frame_timestamp",
    "frame_number", "event_id", "track_id", "litter_class", "confidence_score", "detection_status", "bbox_x",
    "bbox_y", "bbox_width", "bbox_height", "observation_count", "first_seen", "last_seen", "model_version",
    "image_path",
]
SUMMARY_COLUMNS = [
    "store_id", "store_name", "camera_id", "date", "hour", *[f"{c}_count" for c in LDM7_CLASSES],
    "total_unique_litter_events", "total_frame_observations", "processing_status",
]


def detection_status(internal: str, *, event_seen_before: bool = False) -> str:
    """Model/tracker status -> reported status. ``internal`` is the pipeline tier (confirmed / candidate / weak /
    rejected). A confirmed observation of an object that is already an event (an earlier frame recorded it) is
    TRACKED, not a new DETECTED."""
    if internal == CONFIRMED:
        return TRACKED if event_seen_before else DETECTED
    if internal in (CANDIDATE, WEAK):
        return UNDER_REVIEW
    if internal == REJECTED:
        return REJECTED_S
    return UNDER_REVIEW


def processing_status(failed: bool, unique_events: int) -> str:
    if failed:
        return FAILED
    return COMPLETED_LITTER if unique_events else COMPLETED_NONE


def local_now() -> dt.datetime:
    """Wall-clock time of this PC (the store's time zone), second precision, naive."""
    return dt.datetime.now().replace(microsecond=0)


def iso(t: Optional[dt.datetime]) -> str:
    return t.replace(microsecond=0).isoformat() if t else ""


# ----------------------------------------------------------------------------------------------- identity
@dataclass
class Identity:
    store_id: str = UNKNOWN
    store_name: str = UNKNOWN
    camera_id: str = UNKNOWN
    camera_name: str = UNKNOWN
    recording_id: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------------------------- layout
_PART = re.compile(r"[^A-Za-z0-9._\-]+")


def safe_part(value: Optional[str], unknown: str) -> str:
    """A path component from an identifier (spaces and other characters become '_'); ``unknown`` when empty."""
    v = _PART.sub("_", (value or "").strip()).strip("._")
    return v[:64] if v and v != UNKNOWN else unknown


def new_analysis_id(kind: str, now: Optional[dt.datetime] = None) -> str:
    """img_/vid_/qck_<YYYYMMDDTHHMMSS>_<6 hex> - unique, sortable, safe in paths and URLs."""
    return f"{kind}_{(now or local_now()).strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(3)}"


ANALYSIS_ID_RE = re.compile(r"^(img|vid|qck)_\d{8}T\d{6}_[0-9a-f]{6}$")
SUBDIRS = ("input", "annotated", "detections", "reports", "metadata")


@dataclass
class OutputPaths:
    base: Path
    stem: str

    def sub(self, name: str) -> Path:
        return self.base / name

    @property
    def detections_csv(self) -> Path:
        return self.base / "detections" / f"{self.stem}_detections.csv"

    @property
    def summary_csv(self) -> Path:
        return self.base / "reports" / f"{self.stem}_summary.csv"

    @property
    def metadata(self) -> Path:
        return self.base / "metadata" / f"{self.stem}_metadata.json"


def output_paths(root: Path, ident: Identity, when: dt.datetime, analysis_id: str, create: bool = True) -> OutputPaths:
    store, cam = safe_part(ident.store_id, "UNKNOWN_STORE"), safe_part(ident.camera_id, "UNKNOWN_CAMERA")
    base = Path(root) / store / f"{when:%Y}" / f"{when:%m}" / f"{when:%d}" / cam
    if create:
        for s in SUBDIRS:
            (base / s).mkdir(parents=True, exist_ok=True)
    return OutputPaths(base, f"{store}_{when:%Y_%m_%d}_{cam}_{analysis_id}")


# ----------------------------------------------------------------------------------------------- rows
def _r(v: Optional[float], n: int = 1) -> Any:
    return "" if v is None else round(float(v), n)


def detection_rows(observations: Iterable[dict[str, Any]], ident: Identity, model_version: str,
                   events: Optional[dict[str, dict[str, Any]]] = None) -> list[dict[str, Any]]:
    """One CSV row per observation. An observation: ``timestamp`` (ISO), ``frame_timestamp`` (seconds in the video
    or None), ``frame_number``, ``event_id`` (or None), ``track_id``, ``class``, ``confidence``, ``detection_status``,
    ``bbox`` [x1, y1, x2, y2], ``image_path``. Event fields (observation_count / first_seen / last_seen) come from
    ``events[event_id]``; for an observation without an event they describe the track when given, else are empty."""
    events = events or {}
    rows = []
    for o in observations:
        x1, y1, x2, y2 = o["bbox"]
        ev = events.get(o.get("event_id") or "") or {}
        rows.append({
            "store_id": ident.store_id, "store_name": ident.store_name, "camera_id": ident.camera_id,
            "camera_name": ident.camera_name, "recording_id": ident.recording_id,
            "timestamp": o.get("timestamp") or "",
            "frame_timestamp": _r(o.get("frame_timestamp"), 3), "frame_number": o.get("frame_number", ""),
            "event_id": o.get("event_id") or "",
            "track_id": "" if o.get("track_id") is None else o["track_id"],
            "litter_class": o["class"], "confidence_score": round(float(o["confidence"]), 4),
            "detection_status": o["detection_status"],
            "bbox_x": _r(x1), "bbox_y": _r(y1), "bbox_width": _r(x2 - x1), "bbox_height": _r(y2 - y1),
            "observation_count": ev.get("observation_count", o.get("track_observations", "")),
            "first_seen": ev.get("first_seen", o.get("track_first_seen", "")),
            "last_seen": ev.get("last_seen", o.get("track_last_seen", "")),
            "model_version": model_version, "image_path": o.get("image_path") or "",
        })
    return rows


def summary_rows(events: Iterable[dict[str, Any]], observations: Iterable[dict[str, Any]], ident: Identity,
                 status: str, when: dt.datetime) -> list[dict[str, Any]]:
    """One row per (date, hour) with litter: unique events by the hour they were FIRST seen, observations (frames of
    DETECTED/TRACKED rows) by their own timestamp. No litter -> one row for the analysis hour with zeros and the
    processing status (a failed analysis is reported with FAILED, never with zeros as if it were completed)."""
    ev_by: dict[tuple[str, int], Counter] = defaultdict(Counter)
    obs_by: Counter = Counter()
    for e in events:
        t = dt.datetime.fromisoformat(e["first_seen"])
        ev_by[(t.date().isoformat(), t.hour)][e["class"]] += 1
    for o in observations:
        if o["detection_status"] in (DETECTED, TRACKED) and o.get("timestamp"):
            t = dt.datetime.fromisoformat(o["timestamp"])
            obs_by[(t.date().isoformat(), t.hour)] += 1
    keys = sorted(set(ev_by) | set(obs_by)) or [(when.date().isoformat(), when.hour)]
    rows = []
    for d, h in keys:
        c = ev_by.get((d, h), Counter())
        rows.append({"store_id": ident.store_id, "store_name": ident.store_name, "camera_id": ident.camera_id,
                     "date": d, "hour": h, **{f"{k}_count": c.get(k, 0) for k in LDM7_CLASSES},
                     "total_unique_litter_events": sum(c.values()), "total_frame_observations": obs_by.get((d, h), 0),
                     "processing_status": status})
    return rows


def csv_text(rows: Iterable[dict[str, Any]], columns: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    w.writeheader()
    w.writerows(safe_rows(rows))
    return buf.getvalue()


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file + rename: a crash never leaves a half-written report under the final name. An
    existing file is never replaced (analysis ids are unique; a clash is a bug)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8-sig" if path.suffix == ".csv" else "utf-8", newline="")
    os.replace(tmp, path)


def write_reports(paths: OutputPaths, *, observations: list[dict[str, Any]], events: list[dict[str, Any]],
                  ident: Identity, model_version: str, status: str, when: dt.datetime,
                  metadata: dict[str, Any]) -> dict[str, str]:
    """Detection CSV + summary CSV + metadata JSON for a COMPLETED analysis. Returns {kind: path}."""
    by_id = {e["event_id"]: e for e in events}
    write_atomic(paths.detections_csv, csv_text(detection_rows(observations, ident, model_version, by_id),
                                                DETECTION_COLUMNS))
    write_atomic(paths.summary_csv, csv_text(summary_rows(events, observations, ident, status, when),
                                             SUMMARY_COLUMNS))
    classes = Counter(e["class"] for e in events)
    meta = {**metadata, "processing_status": status, "identity": ident.as_dict(), "model_version": model_version,
            "unique_events": len(events), "frame_observations": count_observations(observations),
            "events_by_class": dict(classes),
            "classes_outside_ldm7_schema": sorted(c for c in classes if c not in LDM7_CLASSES),
            "files": {"detections_csv": str(paths.detections_csv), "summary_csv": str(paths.summary_csv)}}
    write_atomic(paths.metadata, json.dumps(meta, indent=2, default=str))
    return {"detections_csv": str(paths.detections_csv), "summary_csv": str(paths.summary_csv),
            "metadata": str(paths.metadata)}


def write_failure(paths: OutputPaths, *, ident: Identity, model_version: str, error: str,
                  metadata: dict[str, Any]) -> dict[str, str]:
    """A failed analysis gets ONLY a metadata file saying FAILED: no detection or summary CSV that could be read as
    a completed (or clean) result."""
    write_atomic(paths.metadata, json.dumps({**metadata, "processing_status": FAILED, "error": error,
                                             "identity": ident.as_dict(), "model_version": model_version},
                                            indent=2, default=str))
    return {"metadata": str(paths.metadata)}


def count_observations(observations: Iterable[dict[str, Any]]) -> int:
    """Frame observations of litter events (DETECTED + TRACKED rows)."""
    return sum(1 for o in observations if o["detection_status"] in (DETECTED, TRACKED))
