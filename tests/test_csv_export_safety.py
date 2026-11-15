"""CSV formula-injection protection: the alerts/detections export and the sampled-analysis report CSVs.

A cell that starts with = + - @ (or tab / carriage return) is run as a formula by spreadsheet programs. Free text in
an export (alert notes, uploaded file names, error text) must therefore come out prefixed with a single quote, while
numbers, plain-number text and every other value keep their exact existing format."""
from __future__ import annotations

import csv
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.alerts.manager import ConfirmedEvent
from backend.core.csv_safety import safe_cell, safe_row
from backend.core.settings import DetectionSettings
from backend.database.db import utcnow
from backend.main import create_app
from backend.sampling import report

from .test_sampled import GROUND, _needs_engine, _upload, _wait, write_video

RISKY = ["=1+1", "+SUM(A1:A2)", "-2+3", "@cmd", '=HYPERLINK("http://example.invalid","x")', "\t=1", "\r=1"]


# ------------------------------------------------------------------ the shared rule
@pytest.mark.parametrize("value", RISKY)
def test_formula_text_is_prefixed_with_a_quote(value):
    assert safe_cell(value) == "'" + value
    assert safe_cell(safe_cell(value)) == "'" + value            # applying it twice changes nothing more


@pytest.mark.parametrize("value", [-0.5, 3, 0.0, None, "", "-0.5", "+3", "-12", "1e-3", "-1.5E+2", "plastic_bag",
                                   "2026-11-05T10:15:00", "Store 12", "it's fine", "a=b", "x-y", "'=already safe"])
def test_normal_values_keep_their_exact_format(value):
    assert safe_cell(value) == value


def test_safe_row_changes_only_risky_text():
    row = {"id": 7, "class": "cup", "confidence": 0.8123, "x1": -1.5, "notes": "=1+1", "location": "@home"}
    assert safe_row(row) == {**row, "notes": "'=1+1", "location": "'@home"}


# ------------------------------------------------------------------ export path 1: /api/export/{alerts,detections}.csv
@pytest.fixture
def client(cfg):
    with TestClient(create_app(cfg)) as c:
        yield c


def test_alert_export_neutralises_formula_notes_and_keeps_other_columns(client):
    st = client.app.state.app_state
    frame = np.full((720, 1280, 3), 120, np.uint8)
    alert = st.alerts.on_confirmed(ConfirmedEvent(
        camera_id="CAM01", session_id="s1", track_id=1, cls_name="cup", confidence=0.8123, bbox=[100.0, 100.0, 150.0, 140.0],
        frame=frame, annotated=None, video_time_sec=1.0, first_seen_ts=utcnow(), hits=3, active_objects=1,
        job_id="job1", live_track_ids=frozenset()), DetectionSettings())
    assert alert is not None
    payload = '=HYPERLINK("http://example.invalid","click")'
    assert client.patch(f"/api/alerts/{alert['id']}", json={"notes": payload}).status_code == 200

    raw = client.get("/api/export/alerts.csv")
    assert raw.status_code == 200 and raw.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(raw.text)))
    row = next(r for r in rows if int(r["id"]) == alert["id"])
    assert row["notes"] == "'" + payload                                       # shown as text, never run
    db_row = next(r for r in st.db.export_rows("alerts", None, None, None) if r["id"] == alert["id"])
    for k, v in db_row.items():                                                # every other column unchanged
        if k in ("notes", "snapshot_path", "crop_path"):
            continue
        assert row[k] == ("" if v is None else str(v)), k

    # an ordinary note is written exactly as typed
    assert client.patch(f"/api/alerts/{alert['id']}", json={"notes": "bag near bin, cleaned at 10:30"}).status_code == 200
    row = next(r for r in csv.DictReader(io.StringIO(client.get("/api/export/alerts.csv").text)) if int(r["id"]) == alert["id"])
    assert row["notes"] == "bag near bin, cleaned at 10:30"


# ------------------------------------------------------------------ export path 2: sampled-analysis report CSVs
def _report_row(**over):
    row = {f: "" for f in report.FIELDS}
    row.update(video_name="store.mp4", store_id="Store 12", camera_id="CAM-07", screenshot=1, frame_idx=120,
               **{"class": "cup"}, confidence=0.8123, x1=-1.5, y1=2.0, x2=30.25, y2=40.0, status="confirmed",
               detection_count=1, outcome="LITTER_DETECTED", remark="1 litter object", job_id="abc", model_version="v3")
    row.update(over)
    return row


def test_report_csv_neutralises_formula_text_and_keeps_numbers():
    rows = list(csv.DictReader(io.StringIO(report.to_csv([_report_row(video_name="-x.mp4", remark="=1+1")]))))
    assert rows[0]["video_name"] == "'-x.mp4" and rows[0]["remark"] == "'=1+1"
    assert rows[0]["x1"] == "-1.5" and rows[0]["confidence"] == "0.8123" and rows[0]["frame_idx"] == "120"


def test_report_csv_without_risky_text_is_unchanged():
    """The protection must not alter an ordinary report: same bytes as the plain DictWriter output."""
    rows = [_report_row(), _report_row(screenshot=2, **{"class": ""}, confidence="", x1="", y1="", x2="", y2="",
                                       status="", outcome="NO_LITTER")]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=report.FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    assert report.to_csv(rows) == buf.getvalue()


def test_stored_report_written_before_the_fix_is_made_safe_when_served():
    old = report.to_csv([_report_row()]).replace("store.mp4", "=cmd|' /C calc'!A0", 1)   # as an unprotected file
    served = report.sanitize_csv_text(old)
    rows = list(csv.DictReader(io.StringIO(served)))
    assert rows[0]["video_name"] == "'=cmd|' /C calc'!A0" and rows[0]["x1"] == "-1.5"
    safe = report.to_csv([_report_row(), _report_row(screenshot=2)])
    assert report.sanitize_csv_text(safe) == safe                            # an already-safe file: byte-identical


def test_sampled_job_month_and_all_csv_exports_are_protected(cfg, tmp_path):
    cfg.roi_segmenter = "off"
    with TestClient(create_app(cfg)) as client:
        _needs_engine(client)
        m = _upload(client, write_video(tmp_path / "v.mp4", 150, fps=10), name="-x.mp4").json()
        assert m["filename"].startswith("-")                                  # the sanitised display name keeps '-'
        jid = client.post("/api/sampled/jobs", json={"media_id": m["id"], "roi_mode": "manual", "seed": 5}).json()["job_id"]
        _wait(client, jid, ("awaiting_roi", "failed"))
        assert client.put(f"/api/sampled/jobs/{jid}/roi", json={"zones": GROUND}).status_code == 200
        j = _wait(client, jid, ("done", "failed"))
        assert j["state"] == "done", j["error"]
        month = client.get("/api/sampled/reports/overview").json()["months"]["completion"][0]
        for url in (j["csv_url"], f"/api/sampled/reports/months/{month}.csv", "/api/sampled/reports/all.csv"):
            r = client.get(url)
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv"), url
            assert 'filename="' in r.headers["content-disposition"], url
            rows = list(csv.DictReader(io.StringIO(r.text)))
            assert rows and all(x["video_name"] == "'" + m["filename"] for x in rows), url
            assert all(x["confidence"] == "" or float(x["confidence"]) >= 0 for x in rows), url

        # a report file stored by an earlier version (no protection) is still served safely
        st = client.app.state.app_state
        path = st.sampled.repo.job(jid)["csv_path"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(report.to_csv([_report_row()]).replace("store.mp4", "=1+1", 1))
        rows = list(csv.DictReader(io.StringIO(client.get(j["csv_url"]).text)))
        assert rows[0]["video_name"] == "'=1+1"
