"""Litter events of a live camera (one StreamPipeline session).

A track the tracker confirms becomes ONE litter event (litter_events row) at that moment; while it stays visible
nothing new is created. When the tracker loses it, the event's last_seen / observation_count / duration are updated.
Timestamps are this PC's wall-clock time (live = real time). The store comes from the camera record.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Optional

from backend.database.db import Database
from backend.services import litter_store as ls
from backend.services.litter_reports import iso, local_now

log = logging.getLogger(__name__)


def _local(utc_iso: Optional[str]) -> str:
    if not utc_iso:
        return iso(local_now())
    return iso(dt.datetime.fromisoformat(utc_iso).astimezone().replace(tzinfo=None))


class LiveEventRecorder:
    def __init__(self, db: Database, camera_id: str) -> None:
        self.db, self.camera_id = db, camera_id
        self._events: dict[tuple[str, int], str] = {}

    def __call__(self, kind: str, track: Any, pipe: Any, frame: Any, dets: list) -> None:
        key = (pipe.session_id, track.track_id)
        if kind == "confirmed" and key not in self._events:
            ident = ls.resolve_identity(self.db, None, self.camera_id, recording_id=f"live:{pipe.session_id}")
            eid = f"live_{self.camera_id}_{pipe.session_id}-T{track.track_id}"
            first = _local(pipe._first_seen.get(track.track_id))
            now = iso(local_now())
            ls.insert_events(self.db, f"live:{self.camera_id}:{pipe.session_id}", "live", ident, [{
                "event_id": eid, "track_id": track.track_id, "class": track.cls_name,
                "max_confidence": round(track.best_conf, 4), "mean_confidence": round(track.mean_hit_conf, 4),
                "bbox": [round(v, 1) for v in track.bbox], "first_seen": first, "last_seen": now,
                "observation_count": track.total_hits,
                "duration_sec": (dt.datetime.fromisoformat(now) - dt.datetime.fromisoformat(first)).total_seconds()}],
                job_id=None, model_version=pipe.engine.version, timestamp_source="PROCESSING_TIME")
            self._events[key] = eid
        elif kind == "lost" and key in self._events:
            eid = self._events.pop(key)
            row = ls.get_event(self.db, eid)
            if row:
                now = iso(local_now())
                dur = (dt.datetime.fromisoformat(now) - dt.datetime.fromisoformat(row["first_seen"])).total_seconds()
                self.db._write("UPDATE litter_events SET last_seen=?, observation_count=?, duration_sec=?, "
                               "max_confidence=MAX(max_confidence, ?) WHERE event_id=?",
                               (now, track.total_hits, round(dur, 1), round(track.best_conf, 4), eid))
