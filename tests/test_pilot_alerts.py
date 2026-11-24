"""Pilot-mode alerts (phase 2, 2026-11-24): CCTV room Confirm / Dismiss, owner notifications off, alerts only for
CONFIRMED detections inside the monitored region with severity >= MEDIUM, a 10-minute cooldown per location, a 30 s
stationary rule for video / live, LOW severity to the review list, and the dismissal log.

No model: AlertManager on a temporary database; StreamPipeline with a fake engine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.alerts.manager import AlertManager, ClassMetadata, ConfirmedEvent, location_cell
from backend.core.config import AppConfig
from backend.core.settings import DetectionSettings
from backend.database.db import Database, utcnow
from backend.detection.types import Detection, InferenceInfo
from backend.services.pipeline import StreamPipeline

W, H = 1280, 720
FRAME = np.full((H, W, 3), 120, np.uint8)


class Notifier:
    def __init__(self):
        self.sent = []

    def dispatch(self, alert):
        self.sent.append(alert)


@pytest.fixture
def am(tmp_path):
    return AlertManager(Database(tmp_path / "a.db"), tmp_path / "alerts", ClassMetadata(AppConfig().class_metadata_path),
                        Notifier())


def ev(cls="plastic_bag", track=1, box=(100, 500, 160, 560), conf=0.7, in_roi=True, vt=1.0, job="job1"):
    return ConfirmedEvent(camera_id="CAM_STREET_01", session_id="s1", track_id=track, cls_name=cls, confidence=conf,
                          bbox=list(box), frame=FRAME, annotated=None, video_time_sec=vt, first_seen_ts=utcnow(),
                          hits=3, active_objects=1, job_id=job, in_roi=in_roi)


def S(**alerts) -> DetectionSettings:
    s = DetectionSettings()
    return s.model_copy(update={"alerts": s.alerts.model_copy(update=alerts)})


def test_pilot_defaults():
    a = DetectionSettings().alerts
    assert a.pilot_mode and not a.notify_owner and a.min_alert_severity == "MEDIUM"
    assert a.cooldown_sec == 600 and a.stationary_sec == 30 and a.require_roi


def test_medium_inside_roi_reaches_the_cctv_room_without_owner_notification(am):
    a = am.on_confirmed(ev("plastic_bag"), S())                       # critical class -> MEDIUM at conf 0.7
    assert a and a["status"] == "new" and a["severity"] == "MEDIUM"
    assert am.notifier.sent == []                                     # owner notifications OFF
    b = am.on_confirmed(ev("plastic_bag", track=2, box=(900, 100, 960, 160)), S(notify_owner=True))
    assert b and am.notifier.sent and am.notifier.sent[0]["id"] == b["id"]


def test_low_severity_goes_to_the_review_list(am):
    a = am.on_confirmed(ev("paper"), S())                             # medium priority class -> LOW severity
    assert a and a["severity"] == "LOW" and a["status"] == "review"
    assert am.notifier.sent == []


def test_outside_a_determined_roi_never_alerts(am):
    assert am.on_confirmed(ev(in_roi=False), S()) is None
    assert am.db.list_alerts()[1] == 0
    assert am.on_confirmed(ev(track=2, in_roi=None), S())             # ROI feature off: unchanged behaviour


def test_cooldown_one_alert_per_location_per_ten_minutes(am):
    a = am.on_confirmed(ev(track=1, vt=10.0), S())
    assert a
    # another object, same location cell, 5 minutes later in the same video: counted on the first alert
    assert am.on_confirmed(ev(track=2, box=(200, 520, 250, 580), vt=310.0), S(dedup_window_sec=0)) is None
    assert am.db.get_alert(a["id"])["occurrences"] == 2
    # same cell 11 minutes later: a new alert
    assert am.on_confirmed(ev(track=3, box=(200, 520, 250, 580), vt=700.0), S(dedup_window_sec=0))
    # other cell at once: a new alert
    assert am.on_confirmed(ev(track=4, box=(1100, 100, 1150, 150), vt=12.0), S(dedup_window_sec=0))


def test_location_cell_uses_the_bottom_centre():
    assert location_cell([0, 0, 10, 10], (720, 1280), 3) == "3:0:0"
    assert location_cell([1200, 600, 1270, 715], (720, 1280), 3) == "3:2:2"


def test_dismiss_logs_crop_full_frame_reason_and_time(am, tmp_path):
    a = am.on_confirmed(ev(), S())
    out = am.dismiss(a["id"], "shadow", "tree shadow at noon", "operator1", tmp_path / "dismissals")
    assert out["status"] == "dismissed" and out["decision_reason"] == "shadow" and out["decided_by"] == "operator1"
    rows, total = am.db.list_dismissals()
    assert total == 1 and rows[0]["reason"] == "shadow" and rows[0]["note"] == "tree shadow at noon" and rows[0]["ts"]
    assert Path(rows[0]["crop_path"]).is_file() and Path(rows[0]["full_path"]).is_file()
    assert str(tmp_path / "dismissals") in rows[0]["crop_path"]
    c = am.on_confirmed(ev(track=9, box=(1000, 100, 1060, 160)), S())
    assert am.confirm(c["id"], "operator1")["status"] == "confirmed"


class StillEngine:
    """A plastic bag at a fixed (or moving) place in every frame."""
    version, device = "fake", "cpu"

    def __init__(self, move=0.0):
        self.move, self.t = move, 0

    def detect_many(self, images, s=None, force_tiling=None):
        out = []
        for _ in images:
            dx = self.move * self.t
            self.t += 1
            out.append(([Detection([400 + dx, 500, 460 + dx, 560], 0, "plastic_bag", 0.9)], InferenceInfo()))
        return out

    def detect(self, image, s=None, force_tiling=None):
        return self.detect_many([image], s, force_tiling)[0]


def _run(engine, am, seconds, fps=2.0, **alerts):
    s = S(**alerts)
    pipe = StreamPipeline(engine, s, am, None, "CAM_STREET_01", "video", "job1")
    first = None
    for i in range(int(seconds * fps)):
        r = pipe.process(FRAME, i, video_time=i / fps)
        if r.new_alerts and first is None:
            first = i / fps
    return first


def test_video_alert_waits_until_the_object_is_stationary_for_30_s(am):
    t = _run(StillEngine(), am, 40)
    assert t is not None and 30.0 <= t <= 31.0


def test_moving_object_never_alerts(am):
    assert _run(StillEngine(move=60), am, 40) is None


def test_stationary_rule_off_alerts_at_confirmation(am):
    t = _run(StillEngine(), am, 10, stationary_sec=0)
    assert t is not None and t < 5
