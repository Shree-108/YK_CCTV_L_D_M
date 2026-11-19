"""Standard litter outputs (backend/services/litter_reports.py) and identity/events/analytics
(backend/services/litter_store.py)."""
from __future__ import annotations

import csv
import datetime as dt
import io
import json

import pytest

from backend.database.db import Database
from backend.services import litter_reports as lr
from backend.services import litter_store as ls


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "t.db")


def _read(path):
    return list(csv.DictReader(io.StringIO(open(path, encoding="utf-8-sig").read())))


def test_columns_are_exactly_the_brief():
    assert lr.DETECTION_COLUMNS == [
        "store_id", "store_name", "camera_id", "camera_name", "recording_id", "timestamp", "frame_timestamp",
        "frame_number", "event_id", "track_id", "litter_class", "confidence_score", "detection_status", "bbox_x",
        "bbox_y", "bbox_width", "bbox_height", "observation_count", "first_seen", "last_seen", "model_version",
        "image_path"]
    assert lr.SUMMARY_COLUMNS == [
        "store_id", "store_name", "camera_id", "date", "hour", "plastic_packet_count", "plastic_bottle_count",
        "paper_count", "food_waste_count", "other_litter_count", "glass_count", "cardboard_count",
        "total_unique_litter_events", "total_frame_observations", "processing_status"]
    assert lr.LDM7_CLASSES == ("plastic_packet", "plastic_bottle", "paper", "food_waste", "other_litter", "glass",
                               "cardboard")


def test_status_mapping():
    assert lr.detection_status("confirmed") == "DETECTED"
    assert lr.detection_status("confirmed", event_seen_before=True) == "TRACKED"
    assert lr.detection_status("candidate") == lr.detection_status("weak") == "UNDER_REVIEW"
    assert lr.detection_status("rejected") == "REJECTED"
    assert lr.processing_status(False, 0) == "COMPLETED_NO_LITTER_DETECTED"      # never "CLEAN"
    assert lr.processing_status(True, 3) == "FAILED"


def test_layout_and_never_overwrite(tmp_path):
    ident = lr.Identity("STORE 001", "Main", "CAMERA_001", "Door")
    when = dt.datetime(2026, 11, 19, 14, 5)
    p = lr.output_paths(tmp_path, ident, when, "vid_20261119T140500_abcdef")
    assert p.base == tmp_path / "STORE_001" / "2026" / "11" / "19" / "CAMERA_001"
    assert all((p.base / s).is_dir() for s in lr.SUBDIRS)
    assert p.detections_csv.name == "STORE_001_2026_11_19_CAMERA_001_vid_20261119T140500_abcdef_detections.csv"
    lr.write_atomic(p.summary_csv, "x")
    with pytest.raises(FileExistsError):
        lr.write_atomic(p.summary_csv, "y")
    unk = lr.output_paths(tmp_path, lr.Identity(), when, "img_20261119T140500_000000")
    assert "UNKNOWN_STORE" in unk.base.parts and "UNKNOWN_CAMERA" in unk.base.parts


def _obs(t, status, ev, cls="plastic_packet", conf=0.8, frame=0):
    return {"timestamp": t, "frame_timestamp": frame / 2, "frame_number": frame, "event_id": ev, "track_id": 1 if ev else None,
            "class": cls, "confidence": conf, "detection_status": status, "bbox": [10, 20, 40, 50], "image_path": ""}


def test_one_object_in_many_frames_is_one_event(tmp_path):
    """TEST 10: 15 observations of one packet -> 1 unique event, 15 frame observations (both reported)."""
    ident = lr.Identity("S1", "Store", "C1", "Cam", "rec1")
    obs = [_obs(f"2026-11-19T14:00:{i:02d}", "DETECTED" if i == 0 else "TRACKED", "E1", frame=i) for i in range(15)]
    obs.append(_obs("2026-11-19T14:00:20", "UNDER_REVIEW", None, cls="paper", conf=0.35))
    events = [{"event_id": "E1", "class": "plastic_packet", "first_seen": "2026-11-19T14:00:00",
               "last_seen": "2026-11-19T14:00:14", "observation_count": 15}]
    when = dt.datetime(2026, 11, 19, 14)
    p = lr.output_paths(tmp_path, ident, when, "vid_20261119T140000_aaaaaa")
    files = lr.write_reports(p, observations=obs, events=events, ident=ident, model_version="m1",
                             status=lr.processing_status(False, 1), when=when, metadata={"analysis_id": "x"})
    det = _read(files["detections_csv"])
    assert len(det) == 16 and list(det[0]) == lr.DETECTION_COLUMNS
    assert {r["observation_count"] for r in det if r["event_id"] == "E1"} == {"15"}
    assert det[-1]["detection_status"] == "UNDER_REVIEW" and det[-1]["event_id"] == ""
    summ = _read(files["summary_csv"])
    assert len(summ) == 1
    assert summ[0]["plastic_packet_count"] == "1" and summ[0]["total_unique_litter_events"] == "1"
    assert summ[0]["total_frame_observations"] == "15"          # UNDER_REVIEW is not litter
    meta = json.loads(open(files["metadata"]).read())
    assert meta["unique_events"] == 1 and meta["frame_observations"] == 15


def test_no_litter_is_not_clean_and_failure_has_no_csv(tmp_path):
    ident = lr.Identity()
    when = dt.datetime(2026, 11, 19, 9)
    p = lr.output_paths(tmp_path, ident, when, "img_20261119T090000_bbbbbb")
    files = lr.write_reports(p, observations=[], events=[], ident=ident, model_version="m", when=when,
                             status=lr.processing_status(False, 0), metadata={})
    s = _read(files["summary_csv"])
    assert s[0]["processing_status"] == "COMPLETED_NO_LITTER_DETECTED" and s[0]["hour"] == "9"
    p2 = lr.output_paths(tmp_path, ident, when, "vid_20261119T090000_cccccc")
    out = lr.write_failure(p2, ident=ident, model_version="m", error="codec", metadata={})
    assert set(out) == {"metadata"} and not p2.detections_csv.exists() and not p2.summary_csv.exists()
    assert json.loads(p2.metadata.read_text())["processing_status"] == "FAILED"


def test_csv_formula_injection_is_neutralised(tmp_path):
    ident = lr.Identity("=HYPERLINK(1)", "x", "C", "@cmd")
    text = lr.csv_text(lr.detection_rows([_obs("2026-11-19T10:00:00", "DETECTED", "E")], ident, "m"),
                       lr.DETECTION_COLUMNS)
    assert "'=HYPERLINK" in text and "'@cmd" in text


# ----------------------------------------------------------------------------------------------- identity
def test_identity_unknown_with_warnings(db):
    """TEST 12: missing metadata -> UNKNOWN + warning, nothing invented."""
    i = ls.resolve_identity(db, None, "upload")
    assert (i.store_id, i.store_name, i.camera_id, i.camera_name) == ("UNKNOWN",) * 4
    assert len(i.warnings) == 2


def test_identity_from_registered_camera_and_wrong_store_refused(db):
    """TEST 11: each camera reports its own store; a mismatching store is refused, never silently attached."""
    ls.upsert_store(db, "STORE_001", "Main Street")
    ls.upsert_store(db, "STORE_002", "Station Road")
    db.upsert_camera({"id": "CAMERA_001", "name": "Door", "source": "0", "store_id": "STORE_001"})
    db.upsert_camera({"id": "CAMERA_003", "name": "Yard", "source": "1", "store_id": "STORE_002"})
    a = ls.resolve_identity(db, None, "CAMERA_001")
    b = ls.resolve_identity(db, "STORE_002", "CAMERA_003")
    assert (a.store_id, a.store_name, a.camera_name) == ("STORE_001", "Main Street", "Door") and not a.warnings
    assert (b.store_id, b.store_name, b.camera_name) == ("STORE_002", "Station Road", "Yard")
    with pytest.raises(ls.IdentityError):
        ls.resolve_identity(db, "STORE_002", "CAMERA_001")
    c = ls.resolve_identity(db, "STORE_X", None)
    assert c.store_id == "STORE_X" and c.store_name == "UNKNOWN" and any("not registered" in w for w in c.warnings)
    with pytest.raises(ls.IdentityError):
        ls.resolve_identity(db, "bad/store", None)


def test_stores_list_cameras(db):
    ls.upsert_store(db, "S1", "One")
    db.upsert_camera({"id": "C1", "name": "Cam", "source": "0", "store_id": "S1"})
    assert ls.list_stores(db)[0]["cameras"] == [{"id": "C1", "name": "Cam"}]


# ----------------------------------------------------------------------------------------------- analytics
def _event(eid, cls, first, obs=1, dur=0.0):
    return {"event_id": eid, "class": cls, "max_confidence": 0.9, "mean_confidence": 0.8, "bbox": [0, 0, 5, 5],
            "first_seen": first, "last_seen": first, "observation_count": obs, "duration_sec": dur, "track_id": 1}


def test_analytics_counts_events_not_observations(db):
    s1 = lr.Identity("S1", "One", "C1", "Cam1", "r1")
    s2 = lr.Identity("S2", "Two", "C3", "Cam3", "r2")
    ls.start_analysis(db, "vid_a", "video", s1, job_id="j1", model_version="m", output_dir="x")
    ls.finish_analysis(db, "vid_a", "COMPLETED_LITTER_DETECTED", unique_events=3, frame_observations=21)
    ls.start_analysis(db, "vid_b", "video", s2, job_id="j2", model_version="m", output_dir="x")
    ls.finish_analysis(db, "vid_b", "FAILED", error="boom")
    ls.insert_events(db, "vid_a", "video", s1, [_event("a1", "paper", "2026-11-19T14:05:00", 15, 30.0),
                                                _event("a2", "paper", "2026-11-19T14:40:00", 5, 10.0),
                                                _event("a3", "glass", "2026-11-19T15:01:00")],
                     job_id="j1", model_version="m", timestamp_source="RECORDING")
    ls.insert_events(db, "vid_b", "video", s2, [_event("b1", "paper", "2026-11-18T09:00:00", 2, 4.0)],
                     job_id="j2", model_version="m", timestamp_source="RECORDING")
    a = ls.analytics(db, granularity="hour", today=dt.date(2026, 11, 19))
    assert a["totals"]["unique_events"] == 4 and a["totals"]["frame_observations"] == 23
    assert a["totals"]["events_today"] == 3 and a["most_frequent_class"] == "paper"
    assert a["by_store"][0] == {"store_id": "S1", "store_name": "One", "events": 3, "observations": 21}
    assert {s["bucket"]: s["events"] for s in a["series"]} == {"2026-11-18T09": 1, "2026-11-19T14": 2,
                                                             "2026-11-19T15": 1}
    assert a["hourly_today"][14]["events"] == 2 and a["processing"] == {"COMPLETED_LITTER_DETECTED": 1, "FAILED": 1}
    # an operator marks a false positive: it leaves the counts but stays in the event list
    assert ls.review_event(db, "a1", "REJECTED", "op", "shadow")
    a2 = ls.analytics(db, store_id="S1")
    assert a2["totals"]["unique_events"] == 2
    rows, total = ls.list_events(db, store_id="S1")
    assert total == 3 and {r["review_status"] for r in rows} == {"DETECTED", "REJECTED"}
    with pytest.raises(ValueError):
        ls.review_event(db, "a1", "ROI_APPROVED", None, None)
    assert ls.analytics(db, granularity="week")["series"][0]["bucket"].startswith("2026-W")
