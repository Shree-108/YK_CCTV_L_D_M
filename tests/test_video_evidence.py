"""Four evidence screenshots + six-column litter report of a full-video analysis (backend/services/video_evidence.py).

Unit tests need no model. API tests run the real v3_store_cctv model (read-only, CPU) on short clips built from the
verified fixture image, through the normal /api/detect/video job - nothing is simulated.
"""
from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.services import litter_reports as lr
from backend.services import video_evidence as ve

from .conftest import V3_VERSION, needs_v3

AUTH = {"Authorization": "Bearer test-admin-token"}
SIX = ["store_id", "camera_id", "timestamp", "litter_class", "confidence_score", "detection_status"]


# ---------------------------------------------------------------------- selection (unit)
def _analysed(frames: int, step: int, fps: float = 25.0):
    return [(i, i / fps) for i in range(0, frames, step)]


def test_four_distinct_frames_one_per_quarter_and_reproducible():
    fps, n = 25.0, 3000                                   # 120 s video, analysed twice a second
    a = _analysed(n, 12, fps)
    picks = ve.select_frames(a, n, fps, seed=7)
    assert len(picks) == 4 and len({i for i, _ in picks}) == 4
    assert all(flag for _, flag in picks)                 # all are analysed frames
    quarters = [int((i / fps) // 30) for i, _ in picks]
    assert quarters == [0, 1, 2, 3]                       # spread over the recording, not just the start
    assert all(i < n for i, _ in picks)
    assert ve.select_frames(a, n, fps, seed=7) == picks   # reproducible with the same seed
    seeds = {tuple(ve.select_frames(a, n, fps, seed=s)) for s in range(10)}
    assert len(seeds) > 1                                  # actually random across seeds


def test_long_interval_fills_from_other_analysed_frames():
    fps, n = 25.0, 2500                                    # 100 s, analysed at 0, 60 s only + 3 more early
    a = [(0, 0.0), (25, 1.0), (50, 2.0), (75, 3.0), (1500, 60.0)]
    picks = ve.select_frames(a, n, fps, seed=1)
    assert len({i for i, _ in picks}) == 4 and all(f for _, f in picks)
    assert {i for i, _ in picks} <= {i for i, _ in a}


def test_fewer_than_four_analysed_frames_uses_other_decoded_frames():
    picks = ve.select_frames([(0, 0.0), (100, 4.0)], 200, 25.0, seed=3)
    assert len({i for i, _ in picks}) == 4
    assert sum(f for _, f in picks) == 2                   # the two analysed + two flagged non-analysed
    assert all(0 <= i < 200 for i, _ in picks)


def test_a_video_with_fewer_than_four_frames_is_refused():
    with pytest.raises(ValueError, match="4"):
        ve.select_frames([(0, 0.0)], 3, 25.0, seed=0)


# ---------------------------------------------------------------------- report (unit)
def test_review_candidates_need_the_candidate_threshold_and_repeated_sightings():
    t = lambda ev, conf, n: {"is_event": ev, "max_confidence": conf, "observation_count": n}   # noqa: E731
    lit = {"tracks": {1: t(True, 0.9, 9), 2: t(False, 0.7, 1), 3: t(False, 0.7, 3), 4: t(False, 0.40, 8)}}
    assert set(ve.review_tracks(lit, 0.45, 3)) == {3}      # event, one-frame flicker and weak-only are excluded
    assert set(ve.review_tracks(lit, 0.45)) == {2, 3}



def test_report_rows_statuses_unknown_identity_and_format():
    events = [
        {"class": "food_wrapper", "max_confidence": 0.8912, "first_seen": "2026-11-20T10:15:20", "review_status": "DETECTED"},
        {"class": "plastic_bottle", "max_confidence": 0.93, "first_seen": "2026-11-20T10:18:42", "review_status": "VERIFIED"},
        {"class": "paper", "max_confidence": 0.7, "first_seen": "2026-11-20T10:19:00", "review_status": "REJECTED"},
        {"class": "cup", "max_confidence": 0.66, "first_seen": "2026-11-20T10:20:00", "review_status": "RESOLVED"},
    ]
    review = [{"class": "plastic_bag", "confidence": 0.5, "first_seen": "2026-11-20T10:22:10"}]
    rows = ve.report_rows("", "CAM_01", events, review)
    text = ve.report_csv(rows)
    header = text.splitlines()[0].split(",")
    assert header == SIX == ve.REPORT_COLUMNS
    got = list(csv.DictReader(io.StringIO(text)))
    assert [r["detection_status"] for r in got] == ["DETECTED", "VERIFIED", "VERIFIED", "UNDER_REVIEW"]
    assert "paper" not in {r["litter_class"] for r in got}          # operator-rejected false positive left out
    assert {r["store_id"] for r in got} == {"UNKNOWN"} and {r["camera_id"] for r in got} == {"CAM_01"}
    assert got[0]["timestamp"] == "2026-11-20 10:15:20" and got[0]["confidence_score"] == "0.89"
    assert all(len(r["confidence_score"].split(".")[1]) == 2 for r in got)


def test_report_of_no_litter_is_headers_only():
    assert ve.report_csv(ve.report_rows("S1", "C1", [], [])) == ",".join(SIX) + "\n"


# ---------------------------------------------------------------------- daily consolidated CSV (unit)
def _row(cam: str, ts: str, cls: str = "plastic_bag", status: str = "DETECTED") -> dict:
    return {"store_id": "S1", "camera_id": cam, "timestamp": ts, "litter_class": cls, "confidence_score": "0.80",
            "detection_status": status}


def test_daily_csv_is_created_then_appended_per_video(tmp_path):
    import datetime as dt
    from backend.services import daily_report as dr
    day = dt.date(2026, 11, 8)
    first = [_row("CAM_A", f"2026-11-08 09:00:0{i}") for i in range(6)]
    a = dr.append(tmp_path, day, first)
    p = tmp_path / "analysis_outputs" / "daily_reports" / "litter_report_2026-11-08.csv"
    assert a["path"] == str(p) and a["rows_appended"] == 6 and a["total_rows"] == 6
    b = dr.append(tmp_path, day, [_row("CAM_B", "2026-11-08 11:00:00"), _row("CAM_B", "2026-11-08 11:00:05")])
    assert b["rows_appended"] == 2 and b["total_rows"] == 8
    dr.append(tmp_path, day, [])                                   # a video without rows changes nothing
    text = p.read_text(encoding="utf-8-sig")
    assert text.count("store_id") == 1                             # one header, not one per video
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [r["camera_id"] for r in rows] == ["CAM_A"] * 6 + ["CAM_B"] * 2   # completion order kept
    assert next(csv.reader(io.StringIO(text))) == SIX
    dr.append(tmp_path, dt.date(2026, 11, 9), [_row("CAM_A", "2026-11-09 08:00:00")])
    assert [d["date"] for d in dr.list_days(tmp_path)] == ["2026-11-09", "2026-11-08"]
    assert [d["rows"] for d in dr.list_days(tmp_path)] == [1, 8]


def test_daily_csv_with_a_foreign_header_is_never_overwritten(tmp_path):
    import datetime as dt
    from backend.services import daily_report as dr
    p = dr.path_for(tmp_path, "2026-11-08")
    p.parent.mkdir(parents=True)
    p.write_text("something,else\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        dr.append(tmp_path, dt.date(2026, 11, 8), [_row("CAM_A", "2026-11-08 09:00:00")])
    assert p.read_text(encoding="utf-8") == "something,else\n1,2\n"
    assert dr.find(tmp_path, "../etc") is None and dr.find(tmp_path, "2026-13-40") is None


# ---------------------------------------------------------------------- API (real model)
@pytest.fixture
def client(cfg):
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        assert c.post("/api/models/activate", json={"version": V3_VERSION}, headers=AUTH).status_code == 200
        yield c


def _wait(client, job_id: str, timeout: float = 400) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.3)
    raise AssertionError("job did not finish")


def _csv(client, url: str):
    r = client.get(url)
    assert r.status_code == 200, (url, r.status_code)
    assert r.headers["content-type"].startswith("text/csv")
    text = r.content.decode("utf-8-sig")
    return next(csv.reader(io.StringIO(text))), list(csv.DictReader(io.StringIO(text)))


def _run(client, video: Path, **form) -> dict:
    data = {"sample_interval_sec": "1", "tiling": "off", **form}
    r = client.post("/api/detect/video", files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")}, data=data)
    assert r.status_code == 202, r.text
    j = _wait(client, r.json()["job_id"])
    assert j["status"] == "done", j.get("error")
    return j


def _check_screenshots(client, cfg, res: dict, duration: float) -> list[dict]:
    shots = res["screenshots"]
    assert [s["file"] for s in shots] == [f"screenshot_{i:02d}.png" for i in range(1, 5)]
    assert len({s["frame_idx"] for s in shots}) == 4
    assert [s["video_time_sec"] for s in shots] == sorted(s["video_time_sec"] for s in shots)
    assert all(0 <= s["video_time_sec"] < duration for s in shots)
    folder = cfg.storage_dir / "analysis_outputs" / res["analysis"]["analysis_id"] / "screenshots"
    assert sorted(p.name for p in folder.glob("*.png")) == [s["file"] for s in shots]
    meta = json.loads((folder / "screenshots.json").read_text())
    assert [s["frame_idx"] for s in meta["screenshots"]] == [s["frame_idx"] for s in shots]
    for s in shots:
        r = client.get(s["url"])
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape[:2] == (360, 640)
    return shots


@needs_v3
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_video_with_litter_four_screenshots_and_six_column_csv_matching_db(client, cfg, litter_image, make_video):
    frame = cv2.resize(litter_image, (640, 360))
    video = make_video([frame], repeat=40, fps=5.0)                  # 8 s, analysed once per second
    j = _run(client, video, camera_id="CAM_EV1", store_id="STORE_EV", recording_start="2026-11-20T10:00:00",
             screenshot_seed="42")
    res = j["result"]
    shots = _check_screenshots(client, cfg, res, 8.0)
    assert res["screenshot_selection"]["seed"] == 42
    assert all(s["analysed_frame"] for s in shots)
    assert all(s["timestamp"].startswith("2026-11-20 10:00:0") for s in shots)   # recording time of each frame
    lit = res["litter"]
    assert lit["unique_events"] >= 1
    # a screenshot's litter boxes are the analysis' own events - never something the CSV/DB lacks
    event_ids = {e["event_id"] for e in lit["events"]}
    for s in shots:
        for d in s["detections"]:
            if d["detection_status"] == "DETECTED":
                assert d["event_id"] in event_ids
    assert any(s["litter_detected"] for s in shots)

    header, rows = _csv(client, j["files"]["report_csv"])
    assert header == SIX
    db = client.get("/api/litter/events", params={"camera_id": "CAM_EV1"}).json()["events"]
    detected = [r for r in rows if r["detection_status"] == "DETECTED"]
    assert len(detected) == len(db) == lit["unique_events"]          # one row per unique event, not per frame
    assert sorted((r["litter_class"], r["confidence_score"], r["timestamp"]) for r in detected) == sorted(
        (e["class"], f"{e['max_confidence']:.2f}", e["first_seen"].replace("T", " ")) for e in db)
    assert {(r["store_id"], r["camera_id"]) for r in rows} == {("STORE_EV", "CAM_EV1")}
    assert {r["detection_status"] for r in rows} <= {"DETECTED", "UNDER_REVIEW", "VERIFIED"}
    assert res["report"]["rows"] == rows                              # snapshot == live while nothing was reviewed

    # the rows were appended to today's consolidated CSV; a second video the same day appends to the same file
    daily = res["report"]["daily"]
    assert daily["rows_appended"] == len(rows)
    assert not list((cfg.storage_dir / "analysis_outputs" / res["analysis"]["analysis_id"]).glob("*.csv"))
    dh, drows = _csv(client, daily["url"])
    assert dh == SIX
    assert drows[len(drows) - len(rows):] == rows
    j2 = _run(client, video, camera_id="CAM_EV2", store_id="STORE_EV", recording_start="2026-11-20T12:00:00")
    d2 = j2["result"]["report"]["daily"]
    assert d2["url"] == daily["url"] and d2["total_rows"] == daily["total_rows"] + d2["rows_appended"]
    _, drows2 = _csv(client, daily["url"])
    assert drows2[:len(drows)] == drows and len(drows2) == d2["total_rows"]
    assert {r["camera_id"] for r in drows2[len(drows):]} <= {"CAM_EV2"}
    listed = client.get("/api/reports/daily").json()["days"]
    assert listed[0]["date"] == daily["date"] and listed[0]["rows"] == d2["total_rows"]

    # operator review is reflected on the next download: VERIFIED stays, REJECTED disappears
    ids = [e["event_id"] for e in db]
    assert client.patch(f"/api/litter/events/{ids[0]}", headers=AUTH, json={"review_status": "VERIFIED"}).status_code == 200
    if len(ids) > 1:
        assert client.patch(f"/api/litter/events/{ids[1]}", headers=AUTH,
                            json={"review_status": "REJECTED"}).status_code == 200
    _, rows2 = _csv(client, j["files"]["report_csv"])
    st = [r["detection_status"] for r in rows2]
    assert st.count("VERIFIED") == 1
    assert st.count("DETECTED") == len(ids) - 1 - (1 if len(ids) > 1 else 0)


@needs_v3
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_video_without_litter_still_four_screenshots_and_headers_only_csv(client, cfg, make_video):
    yy, xx = np.mgrid[0:360, 0:640]
    clean = np.dstack([(90 + xx // 8) % 256, (100 + yy // 6) % 256, np.full_like(xx, 110)]).astype(np.uint8)
    video = make_video([clean], repeat=30, fps=5.0)                  # 6 s of plain "pavement"
    j = _run(client, video)                                          # no store, no camera
    res = j["result"]
    shots = _check_screenshots(client, cfg, res, 6.0)
    assert res["litter"]["unique_events"] == 0
    assert all(s["litter_detected"] == 0 for s in shots)            # nothing fabricated
    assert res["analysis"]["processing_status"] == "COMPLETED_NO_LITTER_DETECTED"
    header, rows = _csv(client, j["files"]["report_csv"])
    assert header == SIX
    assert all(r["detection_status"] == "UNDER_REVIEW" for r in rows)
    assert all(r["store_id"] == "UNKNOWN" and r["camera_id"] == "UNKNOWN" for r in rows)
    warnings = " ".join(res["identity"]["warnings"])
    assert "Store ID not provided" in warnings and "Camera ID not provided" in warnings
    if not rows:                                                     # the expected case for a plain frame
        assert client.get(j["files"]["report_csv"]).content.decode("utf-8-sig") == ",".join(SIX) + "\n"
