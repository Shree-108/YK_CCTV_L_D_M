"""Alert creation for CONFIRMED litter only: severity, dedup, snapshots, notification.

Pilot mode (phase 2, ``alerts.pilot_mode``): an alert reaches the CCTV room (status ``new``) only for a CONFIRMED
detection INSIDE the applied monitored region (``require_roi``) with severity >= ``min_alert_severity``; lower severity
goes to the review list (status ``review``); one CCTV-room alert per location (camera + grid cell) per
``cooldown_sec``; owner notifications only when ``notify_owner``. The video/live stationary rule lives in
backend/services/pipeline.py (StreamPipeline).

Dedup layers (one physical object -> one logical alert):
  1. track: a confirmed track creates at most one alert per pipeline session;
  2. time + space + class: a new confirmation on the same camera within
     ``dedup_window_sec`` whose box overlaps (IoU >= dedup_iou) or whose centre
     is within ``dedup_center_frac`` of the frame diagonal of an existing alert
     of a similar class is merged into that alert (occurrences += 1) instead of
     creating a new one - e.g. the same packet re-tracked after an occlusion.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from backend.core.settings import DetectionSettings, SeverityRule
from backend.database.db import Database, utcnow
from backend.detection.tiling import iou_matrix

log = logging.getLogger(__name__)


class ClassMetadata:
    def __init__(self, path: Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else {}
        self.classes: dict[str, dict[str, str]] = data.get("classes", {})
        self.colors: dict[str, str] = data.get("priority_colors", {})
        self.groups: dict[str, list[str]] = data.get("similarity_groups", {})
        self.report_groups: dict[str, list[str]] = data.get("report_groups", {})
        self._group_of = {c: g for g, cs in self.groups.items() for c in cs}

    def priority(self, cls: str) -> str:
        return self.classes.get(cls, {}).get("priority", "medium")

    def bin(self, cls: str) -> str:
        return self.classes.get(cls, {}).get("bin", "general")

    def similar(self, a: str, b: str, mode: str) -> bool:
        if mode == "any" or a == b:
            return True
        if mode == "exact":
            return False
        ga = self._group_of.get(a)
        return ga is not None and ga == self._group_of.get(b)


def _rule_matches(rule: SeverityRule, cls: str, priority: str, conf: float, active: int) -> bool:
    if rule.classes and cls not in rule.classes:
        return False
    if rule.priorities and priority not in rule.priorities:
        return False
    return conf >= rule.min_confidence and active >= rule.min_active_objects


def compute_severity(cls: str, conf: float, active_objects: int, s: DetectionSettings, meta: ClassMetadata) -> str:
    p = meta.priority(cls)
    if any(_rule_matches(r, cls, p, conf, active_objects) for r in s.severity.high):
        return "HIGH"
    if any(_rule_matches(r, cls, p, conf, active_objects) for r in s.severity.medium):
        return "MEDIUM"
    return "LOW"


@dataclass
class ConfirmedEvent:
    camera_id: str
    session_id: str
    track_id: int
    cls_name: str
    confidence: float
    bbox: list[float]
    frame: np.ndarray                # clean BGR frame
    annotated: Optional[np.ndarray]  # annotated frame (for the full snapshot)
    video_time_sec: Optional[float]
    first_seen_ts: str
    hits: int
    active_objects: int
    location: str = ""
    job_id: Optional[str] = None
    source_type: str = "live"
    model_version: str = ""
    live_track_ids: frozenset = frozenset()   # other tracks visible right now in this session
    in_roi: Optional[bool] = None             # True: inside an applied monitored region; False: region not
                                              # determined (full-frame fallback); None: ROI feature switched off


SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def location_cell(bbox: list[float], frame_shape: tuple[int, ...], grid: int) -> str:
    """Pilot cooldown location: the grid cell (row, col) of the box's bottom-centre (where it lies on the ground)."""
    h, w = frame_shape[:2]
    g = max(1, int(grid))
    cx, by = (bbox[0] + bbox[2]) / 2, bbox[3]
    col = min(g - 1, max(0, int(cx / max(1, w) * g)))
    row = min(g - 1, max(0, int(by / max(1, h) * g)))
    return f"{g}:{row}:{col}"


class AlertManager:
    def __init__(self, db: Database, alerts_dir: Path, meta: ClassMetadata, notifier: Any = None) -> None:
        self.db, self.alerts_dir, self.meta, self.notifier = db, Path(alerts_dir), meta, notifier
        self._lock = threading.Lock()
        self._track_alerts: dict[tuple[str, int], int] = {}   # (session, track) -> alert id

    # ------------------------------------------------------------------ dedup
    def find_duplicate(self, ev: ConfirmedEvent, s: DetectionSettings, frame_shape: tuple[int, ...]) -> Optional[dict]:
        a = s.alerts
        if a.dedup_window_sec <= 0:
            return None
        since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=a.dedup_window_sec)).isoformat()
        diag = math.hypot(frame_shape[1], frame_shape[0])
        cx, cy = (ev.bbox[0] + ev.bbox[2]) / 2, (ev.bbox[1] + ev.bbox[3]) / 2
        best: tuple[float, float, Optional[dict]] = (-1.0, 1e9, None)
        for r in self.db.recent_alerts_for_dedup(ev.camera_id, since):
            # uploads to the same camera from different jobs are different recordings unless same job
            # and live streams never merge with uploads (and vice versa)
            if r.get("job_id") != ev.job_id:
                continue
            # an alert whose track is STILL visible in this session is a different, simultaneous object
            if r.get("session_id") == ev.session_id and r.get("track_id") in ev.live_track_ids:
                continue
            if not self.meta.similar(ev.cls_name, r["class"], a.class_match):
                continue
            rb = [r["x1"], r["y1"], r["x2"], r["y2"]]
            iou = float(iou_matrix(np.array([ev.bbox], np.float32), np.array([rb], np.float32))[0, 0])
            dist = math.hypot(cx - (rb[0] + rb[2]) / 2, cy - (rb[1] + rb[3]) / 2) / diag
            if (iou >= a.dedup_iou or dist <= a.dedup_center_frac) and (iou, -dist) > (best[0], -best[1]):
                best = (iou, dist, r)
        return best[2]

    # ------------------------------------------------------------------ snapshots
    def _save_snapshots(self, alert_id: int, ev: ConfirmedEvent, s: DetectionSettings) -> tuple[str, str]:
        day = dt.datetime.now().strftime("%Y-%m-%d")
        safe_cam = "".join(ch for ch in ev.camera_id if ch.isalnum() or ch in "-_")[:40] or "camera"
        folder = self.alerts_dir / day / safe_cam
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"alert_{alert_id:05d}"
        q = [cv2.IMWRITE_JPEG_QUALITY, s.alerts.snapshot_jpeg_quality]
        full = ev.annotated if ev.annotated is not None else ev.frame
        full = full.copy()
        x1, y1, x2, y2 = (int(round(v)) for v in ev.bbox)
        cv2.rectangle(full, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), 3)
        full_path, crop_path = folder / f"{stem}.jpg", folder / f"{stem}_crop.jpg"
        cv2.imwrite(str(full_path), full, q)
        h, w = ev.frame.shape[:2]
        pad = s.alerts.crop_padding_frac
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
        cx2, cy2 = min(w, int(x2 + pad * bw)), min(h, int(y2 + pad * bh))
        crop = ev.frame[cy1:cy2, cx1:cx2]
        if crop.size:
            cv2.imwrite(str(crop_path), crop, q)
        meta = {"alert_id": alert_id, "camera_id": ev.camera_id, "timestamp": utcnow(), "class": ev.cls_name,
                "confidence": round(ev.confidence, 4), "track_id": ev.track_id, "bbox": ev.bbox,
                "video_time_sec": ev.video_time_sec, "model_version": ev.model_version}
        (folder / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return str(full_path), str(crop_path)

    # ------------------------------------------------------------------ main entry
    def on_confirmed(self, ev: ConfirmedEvent, s: DetectionSettings) -> Optional[dict[str, Any]]:
        """Handle a newly confirmed object. Returns the NEW alert row, or None if deduplicated/disabled."""
        with self._lock:
            if s.alerts.one_alert_per_track and (ev.session_id, ev.track_id) in self._track_alerts:
                return None  # this physical object was already handled (one logical alert per track)
            now = utcnow()
            obj_id = self.db.insert_confirmed({
                "camera_id": ev.camera_id, "job_id": ev.job_id, "session_id": ev.session_id,
                "track_id": ev.track_id, "class": ev.cls_name, "confidence": ev.confidence,
                "x1": ev.bbox[0], "y1": ev.bbox[1], "x2": ev.bbox[2], "y2": ev.bbox[3],
                "first_seen_ts": ev.first_seen_ts, "confirmed_ts": now, "last_seen_ts": now,
                "video_time_sec": ev.video_time_sec, "hits": ev.hits})
            if not s.alerts.enabled:
                return None
            a = s.alerts
            if a.pilot_mode and a.require_roi and ev.in_roi is False:
                # the monitored region was not determined (full-frame fallback): recorded, never a CCTV-room alert
                self.db.log_event("info", "alerts", f"{ev.cls_name} confirmed outside a determined monitored region: "
                                  "no alert (pilot: alerts only inside the ROI)", ev.camera_id)
                return None
            dup = self.find_duplicate(ev, s, ev.frame.shape)
            if dup is not None:
                self.db.update_alert(dup["id"], occurrences=dup["occurrences"] + 1, last_seen_ts=now)
                self.db.update_confirmed(obj_id, alert_id=dup["id"])
                self._track_alerts[(ev.session_id, ev.track_id)] = dup["id"]
                return None
            severity = compute_severity(ev.cls_name, ev.confidence, ev.active_objects, s, self.meta)
            cell = location_cell(ev.bbox, ev.frame.shape, a.cooldown_grid)
            status = "new"
            if a.pilot_mode and SEVERITY_RANK.get(severity, 0) < SEVERITY_RANK[a.min_alert_severity]:
                status = "review"                            # low severity: review list, no CCTV-room alert
            elif a.pilot_mode and a.cooldown_sec > 0:
                held = self._cooldown_holder(ev, cell, a.cooldown_sec)
                if held is not None:                         # same location within 10 min: counted on that alert
                    self.db.update_alert(held["id"], occurrences=held["occurrences"] + 1, last_seen_ts=now)
                    self.db.update_confirmed(obj_id, alert_id=held["id"])
                    self._track_alerts[(ev.session_id, ev.track_id)] = held["id"]
                    return None
            alert_id = self.db.insert_alert({
                "ts": now, "camera_id": ev.camera_id, "class": ev.cls_name, "confidence": round(ev.confidence, 4),
                "track_id": ev.track_id, "session_id": ev.session_id,
                "x1": ev.bbox[0], "y1": ev.bbox[1], "x2": ev.bbox[2], "y2": ev.bbox[3],
                "location": ev.location, "video_time_sec": ev.video_time_sec, "severity": severity, "status": status,
                "occurrences": 1, "last_seen_ts": now, "job_id": ev.job_id, "model_version": ev.model_version,
                "source_type": ev.source_type, "location_cell": cell})
            try:
                snap, crop = self._save_snapshots(alert_id, ev, s)
                self.db.update_alert(alert_id, snapshot_path=snap, crop_path=crop)
            except OSError as exc:
                log.error("Snapshot save failed for alert %s: %s", alert_id, exc)
                self.db.log_event("error", "alerts", f"snapshot save failed: {exc}", ev.camera_id)
            self.db.update_confirmed(obj_id, alert_id=alert_id)
            self._track_alerts[(ev.session_id, ev.track_id)] = alert_id
            alert = self.db.get_alert(alert_id)
        owner = s.alerts.notify_owner or not s.alerts.pilot_mode
        if self.notifier is not None and alert is not None and owner and alert.get("status") == "new":
            self.notifier.dispatch(alert)
        return alert

    def _cooldown_holder(self, ev: ConfirmedEvent, cell: str, cooldown: float) -> Optional[dict]:
        """The CCTV-room alert (not a review-list one) of the same camera and location cell within ``cooldown`` s.
        Same video job: measured in video time; otherwise (live, scheduled captures) in wall-clock time."""
        since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=cooldown)).isoformat()
        for r in self.db.recent_alerts_for_dedup(ev.camera_id, since if ev.video_time_sec is None else "0"):
            if r.get("status") == "review" or r.get("location_cell") != cell or r.get("job_id") != ev.job_id:
                continue
            if ev.video_time_sec is not None and r.get("video_time_sec") is not None:
                if abs(ev.video_time_sec - r["video_time_sec"]) < cooldown:
                    return r
            elif r.get("last_seen_ts", "") >= since:
                return r
        return None

    # ------------------------------------------------------------------ CCTV room decisions (pilot)
    def dismiss(self, alert_id: int, reason: str, note: str = "", actor: str = "",
                dismissals_dir: Optional[Path] = None) -> Optional[dict[str, Any]]:
        """Dismiss an alert and log it (crop, full frame, reason, time). The crop and full frame are COPIED under
        ``dismissals_dir`` so the log survives alert clean-ups (and serves as hard-negative material)."""
        import shutil
        a = self.db.get_alert(alert_id)
        if a is None:
            return None
        now = utcnow()
        crop_copy = full_copy = None
        if dismissals_dir is not None:
            folder = Path(dismissals_dir) / now[:10]
            folder.mkdir(parents=True, exist_ok=True)
            for key, suffix in (("crop_path", "crop"), ("snapshot_path", "full")):
                src = a.get(key)
                if src and Path(src).is_file():
                    dst = folder / f"alert_{alert_id:05d}_{suffix}.jpg"
                    try:
                        shutil.copyfile(src, dst)
                        if suffix == "crop":
                            crop_copy = str(dst)
                        else:
                            full_copy = str(dst)
                    except OSError as exc:
                        log.warning("could not copy %s of alert %s: %s", key, alert_id, exc)
        self.db.insert_dismissal({
            "alert_id": alert_id, "ts": now, "actor": actor or None, "reason": reason, "note": note or None,
            "camera_id": a["camera_id"], "class": a["class"], "confidence": a["confidence"], "severity": a["severity"],
            "x1": a["x1"], "y1": a["y1"], "x2": a["x2"], "y2": a["y2"], "alert_ts": a["ts"], "job_id": a.get("job_id"),
            "source_type": a.get("source_type"), "video_time_sec": a.get("video_time_sec"),
            "model_version": a.get("model_version"), "crop_path": crop_copy or a.get("crop_path"),
            "full_path": full_copy or a.get("snapshot_path")})
        self.db.update_alert(alert_id, status="dismissed", decided_at=now, decided_by=actor or None,
                             decision_reason=reason, updated_at=now)
        return self.db.get_alert(alert_id)

    def confirm(self, alert_id: int, actor: str = "", note: str = "") -> Optional[dict[str, Any]]:
        if self.db.get_alert(alert_id) is None:
            return None
        now = utcnow()
        fields: dict[str, Any] = {"status": "confirmed", "decided_at": now, "decided_by": actor or None,
                                  "decision_reason": None, "updated_at": now}
        if note:
            fields["notes"] = note
        self.db.update_alert(alert_id, **fields)
        return self.db.get_alert(alert_id)

    def linked_alert(self, session_id: str, track_id: int) -> Optional[int]:
        """The alert this object was counted on (its own, or the one a dedup / cooldown merge added it to)."""
        return self._track_alerts.get((session_id, track_id))

    def on_lost(self, session_id: str, track_id: int) -> None:
        """A confirmed object disappeared (e.g. picked up). Alerts are NOT auto-resolved:
        an operator decides; we only record it."""
        self._track_alerts.pop((session_id, track_id), None)
