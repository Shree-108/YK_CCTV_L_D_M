"""Alert creation for CONFIRMED litter only: severity, dedup, snapshots, notification.

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
            dup = self.find_duplicate(ev, s, ev.frame.shape)
            if dup is not None:
                self.db.update_alert(dup["id"], occurrences=dup["occurrences"] + 1, last_seen_ts=now)
                self.db.update_confirmed(obj_id, alert_id=dup["id"])
                self._track_alerts[(ev.session_id, ev.track_id)] = dup["id"]
                return None
            severity = compute_severity(ev.cls_name, ev.confidence, ev.active_objects, s, self.meta)
            alert_id = self.db.insert_alert({
                "ts": now, "camera_id": ev.camera_id, "class": ev.cls_name, "confidence": round(ev.confidence, 4),
                "track_id": ev.track_id, "session_id": ev.session_id,
                "x1": ev.bbox[0], "y1": ev.bbox[1], "x2": ev.bbox[2], "y2": ev.bbox[3],
                "location": ev.location, "video_time_sec": ev.video_time_sec, "severity": severity, "status": "new",
                "occurrences": 1, "last_seen_ts": now, "job_id": ev.job_id, "model_version": ev.model_version,
                "source_type": ev.source_type})
            try:
                snap, crop = self._save_snapshots(alert_id, ev, s)
                self.db.update_alert(alert_id, snapshot_path=snap, crop_path=crop)
            except OSError as exc:
                log.error("Snapshot save failed for alert %s: %s", alert_id, exc)
                self.db.log_event("error", "alerts", f"snapshot save failed: {exc}", ev.camera_id)
            self.db.update_confirmed(obj_id, alert_id=alert_id)
            self._track_alerts[(ev.session_id, ev.track_id)] = alert_id
            alert = self.db.get_alert(alert_id)
        if self.notifier is not None and alert is not None:
            self.notifier.dispatch(alert)
        return alert

    def on_lost(self, session_id: str, track_id: int) -> None:
        """A confirmed object disappeared (e.g. picked up). Alerts are NOT auto-resolved:
        an operator decides; we only record it."""
        self._track_alerts.pop((session_id, track_id), None)
