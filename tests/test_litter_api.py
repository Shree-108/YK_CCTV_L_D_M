"""API-level acceptance tests of the ROI-free litter workflow (docs/ldm_litter_focus/API_CONTRACT.md), through the
FastAPI TestClient with the real v3_store_cctv model (read-only) and temporary storage + DB (cfg fixture).

Numbering follows the litter-focus acceptance list: 10 tracking, 11 store/camera identity, 12 missing metadata,
13 CSV-vs-DB consistency, 14 corrupt / failed videos, 15 ROI-free image analysis. Inputs are kept small (one image
per analysis, a 6-second clip analysed once per second) because the model runs on CPU.
"""
from __future__ import annotations

import csv
import io
import time
from collections import Counter
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from backend.services import litter_reports as lr

from .conftest import V3_VERSION, needs_v3

pytestmark = [needs_v3, pytest.mark.filterwarnings("ignore::DeprecationWarning")]
AUTH = {"Authorization": "Bearer test-admin-token"}
PROCESSING = ("COMPLETED_LITTER_DETECTED", "COMPLETED_NO_LITTER_DETECTED")
DETECTION_STATUSES = {"DETECTED", "TRACKED", "UNDER_REVIEW", "REJECTED"}


# ---------------------------------------------------------------------- helpers
@pytest.fixture
def client(cfg):
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        assert c.post("/api/models/activate", json={"version": V3_VERSION}, headers=AUTH).status_code == 200
        yield c


def _jpg(img) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


def _image(client, img, **form):
    data = {"return_image": "false", "create_alerts": "false", **form}
    return client.post("/api/detect/image", files={"file": ("scene.jpg", _jpg(img), "image/jpeg")}, data=data)


def _csv(client, url: str) -> tuple[list[str], list[dict[str, str]]]:
    r = client.get(url)
    assert r.status_code == 200, (url, r.status_code)
    text = r.content.decode("utf-8-sig")                  # the CSVs carry a BOM for Excel
    header = next(csv.reader(io.StringIO(text)))
    return header, list(csv.DictReader(io.StringIO(text)))


def _keys(obj, path="$"):
    """Every key of a JSON document, recursively, with its path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}", str(k)
            yield from _keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _keys(v, f"{path}[{i}]")


def _assert_no_roi(doc) -> None:
    bad = [p for p, k in _keys(doc) if "roi" in k.lower() or "pavement" in k.lower()]
    assert not bad, bad


def _wait_job(client, job_id: str, timeout: float = 300) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.3)
    raise AssertionError(f"job {job_id} did not finish: {j['status']}")


def _setup_store(client, store_id: str, name: str, cam_id: str, cam_name: str) -> None:
    assert client.post("/api/stores", headers=AUTH, json={"id": store_id, "name": name}).status_code == 201
    r = client.post("/api/cameras", headers=AUTH, json={"id": cam_id, "name": cam_name, "source": "env:NO_SUCH_CAM",
                                                        "enabled": False, "store_id": store_id})
    assert r.status_code == 201, r.text


# ---------------------------------------------------------------------- TEST 15: ROI-free image analysis
def test_15_image_without_camera_or_store_is_analysed_on_the_full_frame(client, litter_image, cfg):
    r = _image(client, litter_image)
    assert r.status_code == 200, r.text
    body = r.json()
    _assert_no_roi(body)                                   # no roi_* / pavement* key anywhere in the response
    dets = body["detections"]
    assert dets and all(d["detection_status"] in DETECTION_STATUSES - {"TRACKED"} for d in dets)
    n_detected = sum(d["detection_status"] == "DETECTED" for d in dets)
    assert n_detected == body["count"] == body["counts"]["confirmed"] >= 1
    assert all(d.get("event_id") for d in dets if d["detection_status"] == "DETECTED")
    ident = body["identity"]
    assert (ident["store_id"], ident["store_name"], ident["camera_id"], ident["camera_name"]) == (lr.UNKNOWN,) * 4
    assert any("Store ID not provided" in w for w in ident["warnings"])
    assert any("Camera ID not provided" in w for w in ident["warnings"])

    a = body["analysis"]
    assert a["processing_status"] == "COMPLETED_LITTER_DETECTED" and a["unique_events"] == n_detected
    assert set(a["urls"]) >= {"original", "annotated", "detections_csv", "summary_csv", "metadata"}
    for kind, url in a["urls"].items():
        assert client.get(url).status_code == 200, kind
    header, rows = _csv(client, a["urls"]["detections_csv"])
    assert header == lr.DETECTION_COLUMNS
    assert len(rows) == len(dets)                          # every detection (any status) is a CSV row
    assert Counter(r["detection_status"] for r in rows) == Counter(d["detection_status"] for d in dets)
    assert all(r["store_id"] == lr.UNKNOWN and r["camera_id"] == lr.UNKNOWN for r in rows)
    sheader, srows = _csv(client, a["urls"]["summary_csv"])
    assert sheader == lr.SUMMARY_COLUMNS
    assert sum(int(r["total_unique_litter_events"]) for r in srows) == n_detected
    meta = client.get(a["urls"]["metadata"]).json()
    _assert_no_roi(meta)
    # stored under UNKNOWN_STORE / UNKNOWN_CAMERA
    assert list((cfg.storage_dir / "LDM_OUTPUT" / "UNKNOWN_STORE").rglob("UNKNOWN_CAMERA"))
    # the analysis and its events are in the DB
    stored = client.get(f"/api/analyses/{a['analysis_id']}").json()
    assert stored["status"] == "COMPLETED_LITTER_DETECTED" and stored["unique_events"] == n_detected
    _assert_no_roi(stored)
    ev = client.get("/api/litter/events", params={"store_id": lr.UNKNOWN}).json()
    assert ev["total"] == n_detected and all(e["analysis_id"] == a["analysis_id"] for e in ev["events"])
    assert client.get(ev["events"][0]["image_url"]).headers["content-type"] == "image/jpeg"
    _assert_no_roi(ev)
    listed = client.get("/api/litter/analyses").json()["analyses"]
    assert [x["analysis_id"] for x in listed] == [a["analysis_id"]]
    _assert_no_roi(listed)

    # save_outputs=false: analysed, nothing stored
    r2 = _image(client, litter_image, save_outputs="false").json()
    assert r2["analysis"]["analysis_id"] is None and r2["analysis"]["urls"] == {}
    assert client.get("/api/litter/events", params={"store_id": lr.UNKNOWN}).json()["total"] == n_detected


# ---------------------------------------------------------------------- TEST 10 + 13: tracking, CSV vs DB
def test_10_13_repeated_litter_is_one_event_per_track_and_csv_matches_db(client, litter_image, make_video):
    frame = cv2.resize(litter_image, (640, 360))
    video = make_video([frame], repeat=30, fps=5.0)             # 6 s of the same scene
    r = client.post("/api/detect/video", files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")},
                    data={"camera_id": "CAM_T10", "sample_interval_sec": "1", "tiling": "off",
                          "recording_start": "2026-11-19T14:05:00"})
    assert r.status_code == 202, r.text
    j = _wait_job(client, r.json()["job_id"])
    assert j["status"] == "done", j.get("error")
    res = j["result"]
    _assert_no_roi(res)
    lit, stats = res["litter"], res["stats"]
    assert stats["frames_analysed"] == 6
    events = lit["events"]
    # one event per distinct confirmed track - not one per frame
    assert lit["unique_events"] == len(events) == len({e["track_id"] for e in events}) >= 1
    assert lit["unique_events"] == stats["unique_confirmed_objects"]
    assert lit["frame_observations"] == sum(e["observation_count"] for e in events)
    assert lit["frame_observations"] >= 3 * lit["unique_events"]          # seen in many frames, counted once
    assert all(e["detection_status"] == "DETECTED" and e["observation_count"] >= 3 for e in events)
    assert all(e["first_seen"].startswith("2026-11-19T14:05") for e in events)   # recording time, not wall clock
    assert res["analysis"]["processing_status"] == "COMPLETED_LITTER_DETECTED"
    assert res["identity"]["camera_id"] == "CAM_T10" and res["identity"]["store_id"] == lr.UNKNOWN
    assert client.get(events[0]["evidence_url"]).headers["content-type"] == "image/jpeg"

    # TEST 13: detection CSV rows per event == observation_count; exactly one DETECTED row per event
    header, rows = _csv(client, j["files"]["detections_csv"])
    assert header == lr.DETECTION_COLUMNS
    per_event = Counter(r["event_id"] for r in rows if r["event_id"])
    assert per_event == {e["event_id"]: e["observation_count"] for e in events}
    detected = Counter(r["event_id"] for r in rows if r["detection_status"] == "DETECTED")
    assert detected == {e["event_id"]: 1 for e in events}
    assert all(r["detection_status"] == "TRACKED" for r in rows
               if r["event_id"] and r["detection_status"] != "DETECTED")
    assert {r["detection_status"] for r in rows} <= DETECTION_STATUSES
    # summary CSV totals == the result == the DB
    sheader, srows = _csv(client, j["files"]["summary_csv"])
    assert sheader == lr.SUMMARY_COLUMNS
    assert sum(int(r["total_unique_litter_events"]) for r in srows) == lit["unique_events"]
    assert sum(int(r["total_frame_observations"]) for r in srows) == lit["frame_observations"]
    class_cols = sum(int(r[f"{c}_count"]) for r in srows for c in lr.LDM7_CLASSES)
    assert class_cols <= lit["unique_events"]                      # v3 classes are never relabelled
    assert {r["processing_status"] for r in srows} == {"COMPLETED_LITTER_DETECTED"}
    db = client.get("/api/litter/events", params={"camera_id": "CAM_T10"}).json()
    assert db["total"] == lit["unique_events"]
    assert {e["event_id"]: e["observation_count"] for e in db["events"]} == dict(per_event)
    assert all(e["source_type"] == "video" and e["review_status"] == "DETECTED" for e in db["events"])
    an = client.get("/api/litter/analytics", params={"camera_id": "CAM_T10"}).json()
    assert an["totals"]["unique_events"] == lit["unique_events"]
    assert an["totals"]["frame_observations"] == lit["frame_observations"]
    assert an["processing"] == {"COMPLETED_LITTER_DETECTED": 1}


# ---------------------------------------------------------------------- TEST 11: stores and cameras
def test_11_two_stores_keep_their_own_identity_and_a_wrong_store_is_refused(client, litter_image):
    _setup_store(client, "STORE_A", "Alpha Street", "CAM_A", "Alpha door")
    _setup_store(client, "STORE_B", "Beta Road", "CAM_B", "Beta forecourt")
    stores = {s["id"]: s for s in client.get("/api/stores").json()["stores"]}
    assert stores["STORE_A"]["cameras"] == [{"id": "CAM_A", "name": "Alpha door"}]
    cam = client.get("/api/cameras/CAM_B").json()
    assert (cam["store_id"], cam["store_name"]) == ("STORE_B", "Beta Road")
    _assert_no_roi(cam)

    img = cv2.resize(litter_image, (640, 360))
    a = _image(client, img, camera_id="CAM_A").json()                         # store from the camera record
    b = _image(client, img, camera_id="CAM_B", store_id="STORE_B").json()     # matching store: accepted
    for body, sid, sname, cid, cname in ((a, "STORE_A", "Alpha Street", "CAM_A", "Alpha door"),
                                         (b, "STORE_B", "Beta Road", "CAM_B", "Beta forecourt")):
        ident = body["identity"]
        assert (ident["store_id"], ident["store_name"], ident["camera_id"], ident["camera_name"]) == \
            (sid, sname, cid, cname) and not ident["warnings"]
        _, rows = _csv(client, body["analysis"]["urls"]["detections_csv"])
        assert rows and {(r["store_id"], r["store_name"], r["camera_id"], r["camera_name"]) for r in rows} == \
            {(sid, sname, cid, cname)}
        ev = client.get("/api/litter/events", params={"store_id": sid}).json()
        assert ev["total"] == body["analysis"]["unique_events"] >= 1
        assert {(e["store_name"], e["camera_id"], e["camera_name"]) for e in ev["events"]} == {(sname, cid, cname)}
    by_store = {s["store_id"]: s for s in client.get("/api/litter/analytics").json()["by_store"]}
    assert set(by_store) == {"STORE_A", "STORE_B"} and by_store["STORE_A"]["store_name"] == "Alpha Street"

    # a registered camera with another store: refused, nothing analysed or stored
    before = client.get("/api/litter/analyses").json()["analyses"]
    bad = _image(client, img, camera_id="CAM_A", store_id="STORE_B")
    assert bad.status_code == 422 and "belongs to store STORE_A" in bad.json()["detail"]
    vid = client.post("/api/detect/video", files={"file": ("v.mp4", b"x", "video/mp4")},
                      data={"camera_id": "CAM_B", "store_id": "STORE_A"})
    assert vid.status_code == 422
    assert client.get("/api/litter/analyses").json()["analyses"] == before


# ---------------------------------------------------------------------- TEST 12: missing / unregistered metadata
def test_12_unregistered_store_and_camera_are_kept_as_given_with_unknown_names(client, litter_image):
    body = _image(client, cv2.resize(litter_image, (640, 360)), store_id="Shop 9", camera_id="CAM_X").json()
    ident = body["identity"]
    assert (ident["store_id"], ident["store_name"], ident["camera_id"], ident["camera_name"]) == \
        ("Shop 9", lr.UNKNOWN, "CAM_X", lr.UNKNOWN)
    assert any("Store Shop 9 is not registered" in w for w in ident["warnings"])
    assert any("Camera CAM_X is not registered" in w for w in ident["warnings"])
    _, rows = _csv(client, body["analysis"]["urls"]["detections_csv"])
    assert {(r["store_id"], r["store_name"], r["camera_name"]) for r in rows} == {("Shop 9", lr.UNKNOWN, lr.UNKNOWN)}
    assert client.get(f"/api/analyses/{body['analysis']['analysis_id']}").json()["warnings"] == ident["warnings"]
    # an invalid free-text store is refused (never stored)
    assert _image(client, litter_image, store_id="../../etc").status_code == 422


# ---------------------------------------------------------------------- TEST 14: corrupt / failing videos
def test_14_corrupt_and_truncated_videos_are_refused_at_upload(client, litter_image, make_video, cfg):
    garbage = client.post("/api/detect/video", files={"file": ("cctv.mp4", b"\x00garbage" * 500, "video/mp4")})
    assert garbage.status_code == 400
    good = make_video([cv2.resize(litter_image, (320, 180))], repeat=20, fps=5.0).read_bytes()
    cut = client.post("/api/detect/video", files={"file": ("cut.mp4", good[: len(good) // 3], "video/mp4")})
    assert cut.status_code == 400            # mp4v writes the index (moov) last: a cut file cannot be opened
    assert client.get("/api/jobs").json()["jobs"] == []                       # no job, no analysis
    assert client.get("/api/litter/analyses").json()["analyses"] == []
    assert not list(cfg.uploads_dir.iterdir())                                 # nothing left on disk


def test_14_a_video_that_fails_during_processing_is_failed_with_no_csv(client, litter_image, make_video, cfg,
                                                                      monkeypatch):
    import backend.services.state as state_mod

    def broken(*a, **kw):
        raise RuntimeError("decoder crashed half way")
    monkeypatch.setattr(state_mod, "detect_video", broken)
    video = make_video([cv2.resize(litter_image, (320, 180))], repeat=10, fps=5.0)
    r = client.post("/api/detect/video", files={"file": ("v.mp4", video.read_bytes(), "video/mp4")},
                    data={"camera_id": "CAM_F"})
    assert r.status_code == 202, r.text
    j = _wait_job(client, r.json()["job_id"])
    assert j["status"] == "failed" and "decoder crashed" in j["error"]
    a = j["result"]["analysis"]
    assert a["processing_status"] == "FAILED" and a["analysis_id"]
    assert "files" not in j
    stored = client.get(f"/api/analyses/{a['analysis_id']}").json()
    assert stored["status"] == "FAILED" and stored["error"]
    assert not {"detections_csv", "summary_csv"} & set(stored["urls"])
    for kind in ("detections_csv", "summary_csv"):
        assert client.get(f"/api/analyses/{a['analysis_id']}/files/{kind}").status_code == 404
    out = cfg.storage_dir / "LDM_OUTPUT"
    assert not list(out.rglob("*_detections.csv")) and not list(out.rglob("*_summary.csv"))
    an = client.get("/api/litter/analytics").json()
    assert an["processing"] == {"FAILED": 1} and an["totals"]["analyses_failed"] == 1
    assert an["totals"]["unique_events"] == 0


# ---------------------------------------------------------------------- operator review
def test_rejected_event_leaves_the_analytics_and_invalid_reviews_are_refused(client, litter_image):
    body = _image(client, cv2.resize(litter_image, (640, 360)), camera_id="CAM_R").json()
    n = body["analysis"]["unique_events"]
    assert n >= 1
    before = client.get("/api/litter/analytics", params={"camera_id": "CAM_R"}).json()
    assert before["totals"]["unique_events"] == n
    ev = client.get("/api/litter/events", params={"camera_id": "CAM_R"}).json()["events"][0]
    r = client.patch(f"/api/litter/events/{ev['event_id']}", json={"review_status": "REJECTED", "note": "shadow"})
    assert r.status_code == 200 and r.json()["review_status"] == "REJECTED"
    after = client.get("/api/litter/analytics", params={"camera_id": "CAM_R"}).json()
    assert after["totals"]["unique_events"] == n - 1
    assert after["totals"]["frame_observations"] == before["totals"]["frame_observations"] - ev["observation_count"]
    cls = {c["class"]: c["events"] for c in after["by_class"]}
    assert cls.get(ev["class"], 0) == {c["class"]: c["events"] for c in before["by_class"]}[ev["class"]] - 1
    rejected = client.get("/api/litter/events", params={"camera_id": "CAM_R", "status": "REJECTED"}).json()
    assert [e["event_id"] for e in rejected["events"]] == [ev["event_id"]]       # still listed for audit
    assert client.patch(f"/api/litter/events/{ev['event_id']}", json={"review_status": "VERIFIED"}).status_code == 200
    assert client.get("/api/litter/analytics", params={"camera_id": "CAM_R"}).json()["totals"]["unique_events"] == n
    for bad in ({"review_status": "BOGUS"}, {"review_status": "TRACKED"}, {}, {"review_status": "VERIFIED", "x": 1}):
        assert client.patch(f"/api/litter/events/{ev['event_id']}", json=bad).status_code == 422, bad
    assert client.patch("/api/litter/events/img_20261119T000000_abcdef-E999",
                        json={"review_status": "VERIFIED"}).status_code == 404
    assert client.get("/api/litter/events", params={"status": "BOGUS"}).status_code == 422
    assert client.get("/api/litter/analytics", params={"start": "yesterday"}).status_code == 422


# ---------------------------------------------------------------------- cameras: no ROI, registered stores only
def test_cameras_refuse_roi_fields_and_unregistered_stores(client):
    cam = {"id": "CAM_N", "name": "N", "source": "env:NO_SUCH_CAM", "enabled": False}
    for extra in ({"roi": []}, {"roi": [{"polygon": [[0, 0], [1, 0], [1, 1]]}]}, {"roi_reference_size": [640, 360]}):
        assert client.post("/api/cameras", headers=AUTH, json={**cam, **extra}).status_code == 422, extra
    r = client.post("/api/cameras", headers=AUTH, json={**cam, "store_id": "NOT_A_STORE"})
    assert r.status_code == 422 and "not registered" in r.json()["detail"]
    assert client.get("/api/cameras").json()["cameras"] == []                 # nothing was created
    assert client.post("/api/cameras", headers=AUTH, json=cam).status_code == 201
    assert client.put("/api/cameras/CAM_N", headers=AUTH, json={"store_id": "NOT_A_STORE"}).status_code == 422
    assert client.put("/api/cameras/CAM_N", headers=AUTH, json={"roi": []}).status_code == 422
    assert client.post("/api/stores", headers=AUTH, json={"id": "S_N", "name": "N"}).status_code == 201
    out = client.put("/api/cameras/CAM_N", headers=AUTH, json={"store_id": "S_N"}).json()
    assert (out["store_id"], out["store_name"]) == ("S_N", "N")
    _assert_no_roi(out)
    assert client.post("/api/stores", headers=AUTH, json={"id": "bad/id", "name": "x"}).status_code == 422
    assert client.post("/api/stores", headers=AUTH, json={"id": "S_N", "name": "again"}).status_code == 409
    assert client.put("/api/stores/NOPE", headers=AUTH, json={"name": "x"}).status_code == 404
