"""Alert severity and deduplication for the v3 classes, including cup / newspaper / plastic_piece.

No model, no API: AlertManager on a temporary database with the project's class_metadata.json (read only).
Dedup rule under test (backend/alerts/manager.py find_duplicate, defaults in backend/core/settings.py):
same camera AND same job, existing alert last seen within 900 s, not a false positive, track not still visible,
class equal or in the same similarity group, and IoU >= 0.30 OR centre distance <= 4 % of the frame diagonal.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from backend.alerts.manager import AlertManager, ClassMetadata, ConfirmedEvent, compute_severity
from backend.core.config import AppConfig
from backend.core.settings import DetectionSettings
from backend.database.db import Database, utcnow

W, H = 1280, 720
DIAG_4PCT = 0.04 * float(np.hypot(W, H))        # 58.7 px on a 1280x720 frame
BOX = (100.0, 100.0, 150.0, 140.0)              # 50 x 40 px
NEW_CLASSES = ("cup", "newspaper", "plastic_piece")


@pytest.fixture
def am(tmp_path):
    db = Database(tmp_path / "alerts.db")
    return AlertManager(db, tmp_path / "alerts", ClassMetadata(AppConfig().class_metadata_path))


_FRAME = np.full((H, W, 3), 120, np.uint8)


def ev(cls, track, box=BOX, camera="CAM01", job="job1", session="s1", conf=0.8, active=1, live=frozenset()):
    return ConfirmedEvent(camera_id=camera, session_id=session, track_id=track, cls_name=cls, confidence=conf,
                          bbox=list(box), frame=_FRAME, annotated=None, video_time_sec=1.0, first_seen_ts=utcnow(),
                          hits=3, active_objects=active, job_id=job, live_track_ids=live)


def shifted(dx, dy=0.0, box=BOX):
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def first_then(am, existing, new, **kw_new):
    """Create an alert for `existing`, then confirm `new` (a different track). Returns (first alert, second result)."""
    s = DetectionSettings()
    a = am.on_confirmed(ev(existing, track=1), s)
    assert a is not None
    b = am.on_confirmed(ev(new, track=2, **kw_new), s)
    return a, b


# ------------------------------------------------------------------ severity: explicit 'medium' == previous default
def _meta_without_new_entries(tmp_path) -> ClassMetadata:
    """class_metadata.json as it was before the three entries were added (entries and group memberships removed)."""
    d = json.loads(Path(AppConfig().class_metadata_path).read_text(encoding="utf-8"))
    for c in NEW_CLASSES:
        d["classes"].pop(c, None)
        for members in d["similarity_groups"].values():
            if c in members:
                members.remove(c)
    p = tmp_path / "old_meta.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return ClassMetadata(p)


@pytest.mark.parametrize("cls", NEW_CLASSES)
def test_new_entries_do_not_change_severity(cls, tmp_path):
    new, old, s = ClassMetadata(AppConfig().class_metadata_path), _meta_without_new_entries(tmp_path), DetectionSettings()
    assert old.priority(cls) == new.priority(cls) == "medium"
    for conf in (0.30, 0.60, 0.75, 0.95):
        for active in (0, 1, 4, 5, 9):
            sev = compute_severity(cls, conf, active, s, new)
            assert sev == compute_severity(cls, conf, active, s, old)          # identical to before the change
            assert sev == ("HIGH" if active >= 5 else "LOW")                    # medium priority: LOW unless busy


def test_severity_is_stored_and_not_recomputed(am):
    a = am.on_confirmed(ev("newspaper", track=1), DetectionSettings())
    assert a["severity"] == "LOW"
    am.meta.classes["newspaper"] = {"priority": "critical", "bin": "paper"}   # later metadata change (in memory)
    assert am.db.get_alert(a["id"])["severity"] == "LOW"                     # existing alert keeps its severity


# ------------------------------------------------------------------ same group, same place -> one alert
@pytest.mark.parametrize("existing,new", [
    ("paper", "newspaper"), ("cardboard", "newspaper"), ("newspaper", "paper"), ("newspaper", "cardboard"),
    ("food_wrapper", "plastic_piece"), ("plastic_bag", "plastic_piece"), ("plastic_packaging", "plastic_piece"),
    ("plastic_piece", "food_wrapper"), ("plastic_piece", "plastic_bag"),
    ("cup", "cup"),
])
def test_same_group_same_place_merges(am, existing, new):
    a, b = first_then(am, existing, new, box=shifted(3, 2))                   # re-detected on the same spot
    assert b is None
    rows, total = am.db.list_alerts()
    assert total == 1 and rows[0]["occurrences"] == 2 and rows[0]["class"] == existing


# ------------------------------------------------------------------ different groups never merge
@pytest.mark.parametrize("existing,new", [
    ("plastic_bottle", "cup"), ("cup", "plastic_bottle"),
    ("paper", "plastic_piece"), ("newspaper", "plastic_piece"), ("food_wrapper", "newspaper"),
    ("cup", "newspaper"), ("cup", "plastic_piece"), ("other_litter", "plastic_piece"),
])
def test_different_groups_same_place_do_not_merge(am, existing, new):
    a, b = first_then(am, existing, new, box=shifted(3, 2))
    assert b is not None and b["class"] == new
    assert am.db.list_alerts()[1] == 2


# ------------------------------------------------------------------ camera / job / live-track separation
def test_different_cameras_never_merge(am):
    a, b = first_then(am, "paper", "newspaper", camera="CAM02")
    assert b is not None and b["camera_id"] == "CAM02"


@pytest.mark.parametrize("new_job", ["job2", None])        # another upload job, or a live stream (job None)
def test_different_jobs_never_merge(am, new_job):
    a, b = first_then(am, "plastic_bag", "plastic_piece", job=new_job)
    assert b is not None


def test_track_still_visible_is_a_separate_object(am):
    a, b = first_then(am, "cup", "cup", live=frozenset({1}))                    # track 1 is still on screen
    assert b is not None


# ------------------------------------------------------------------ spatial condition
@pytest.mark.parametrize("dx,merge", [
    (0.8 * DIAG_4PCT, True),        # no overlap (IoU 0) but centres within 4 % of the diagonal
    (1.2 * DIAG_4PCT, False),       # no overlap and centres beyond 4 %
    (500.0, False),                 # far away
])
def test_same_group_merges_only_when_close(am, dx, merge):
    a, b = first_then(am, "newspaper", "paper", box=shifted(dx))
    assert (b is None) == merge


def test_distant_objects_in_same_group_stay_separate(am):
    s = DetectionSettings()
    assert am.on_confirmed(ev("plastic_bag", 1, box=(100, 100, 150, 140)), s)
    assert am.on_confirmed(ev("plastic_piece", 2, box=(700, 500, 750, 540)), s)
    assert am.on_confirmed(ev("food_wrapper", 3, box=(1100, 150, 1150, 190)), s)
    assert am.db.list_alerts()[1] == 3


# ------------------------------------------------------------------ time window and false positives
def _age_alert(am, alert_id, seconds):
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=seconds)).isoformat()
    am.db.update_alert(alert_id, last_seen_ts=old)       # temporary test database only


@pytest.mark.parametrize("age,merge", [(880, True), (920, False)])
def test_900_second_window(am, age, merge):
    s = DetectionSettings()
    a = am.on_confirmed(ev("paper", 1), s)
    _age_alert(am, a["id"], age)
    b = am.on_confirmed(ev("newspaper", 2, box=shifted(3, 2)), s)
    assert (b is None) == merge


def test_false_positive_alerts_are_not_merged_into(am):
    s = DetectionSettings()
    a = am.on_confirmed(ev("plastic_bag", 1), s)
    am.db.update_alert(a["id"], status="false_positive")
    assert am.on_confirmed(ev("plastic_piece", 2, box=shifted(3, 2)), s) is not None


def test_same_track_never_alerts_twice(am):
    s = DetectionSettings()
    assert am.on_confirmed(ev("cup", 1), s)
    assert am.on_confirmed(ev("cup", 1, box=shifted(400)), s) is None           # one alert per track
    assert am.db.list_alerts()[1] == 1
