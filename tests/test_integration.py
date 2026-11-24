"""Integration tests with the REAL production model (v3_store_cctv, read-only) and a real image fixture.

Model-dependent tests are marked `needs_v3`; tests of alerts, database, camera CRUD and settings do not need a model
and always run. The fixture frame was probably part of v3's training data (tests/fixtures/v3/README.md), so the
detection checks here are regression/smoke checks, not accuracy measurements.
v1-only assertions live in test_v1_legacy.py.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.alerts.manager import AlertManager, ClassMetadata, ConfirmedEvent
from backend.cameras.manager import CameraWorker, mask_source
from backend.core.settings import DetectionSettings, merged_settings
from backend.database.db import Database, utcnow
from backend.services.pipeline import StreamPipeline, detect_image, detect_video

from .conftest import CLEAN_FRAMES, V3_CLASSES, V3_VERSION, fixture_boxes, iou, needs_v3

AUTH = {"Authorization": "Bearer test-admin-token"}


def _frame() -> np.ndarray:
    """A plain frame for tests that only need pixels (alerts / snapshots), no model."""
    rng = np.random.default_rng(0)
    return (np.full((720, 1280, 3), 120, np.uint8) + rng.integers(0, 20, (720, 1280, 3), dtype=np.uint8))


# ------------------------------------------------------------------ model + image detection
@needs_v3
def test_model_loads_with_v3_classes(engine):
    assert engine.class_names == V3_CLASSES and len(engine.class_names) == 10
    assert engine.class_names[0] == "plastic_bag"
    assert engine.device in ("cpu", "cuda:0", "mps")


@needs_v3
def test_detect_image_finds_verified_litter_as_separate_boxes(engine, litter_image):
    """Regression floor on the fixture: most verified boxes are found, each as its own (small) box."""
    res = detect_image(engine, litter_image, DetectionSettings())
    confirmed = [d for d in res["detections"] if d["status"] == "confirmed"]
    gt = fixture_boxes(litter_image)
    matched = sum(any(iou(g, d["bbox"]) >= 0.3 for d in confirmed) for g in gt)
    assert matched / len(gt) >= 0.70, (matched, len(gt))           # measured 17/19 on 2026-11-13
    h, w = litter_image.shape[:2]
    for d in confirmed:
        x1, y1, x2, y2 = d["bbox"]
        assert (x2 - x1) * (y2 - y1) < 0.1 * w * h                # no giant box covering several items
        assert set(d) >= {"class", "confidence", "bbox", "status", "detection_status"}
        assert d["detection_status"] == "DETECTED"
        assert d["class"] in V3_CLASSES
    assert res["count"] == len(confirmed)


@needs_v3
def test_featureless_frames_produce_no_detections(engine):
    """Sanity check (not a clean-scene test): flat, noisy and gradient frames must not produce litter."""
    flat = np.full((720, 1280, 3), 128, np.uint8)
    noise = (flat + np.random.default_rng(0).normal(0, 6, flat.shape)).clip(0, 255).astype(np.uint8)
    grad = np.dstack([np.tile(np.linspace(60, 200, 1280).astype(np.uint8), (720, 1))] * 3)
    for f in (flat, noise, grad):
        assert detect_image(engine, f, DetectionSettings())["counts"]["confirmed"] == 0


@needs_v3
def test_tiled_inference_runs_and_maps_coordinates(engine, litter_image):
    big = cv2.resize(litter_image, None, fx=2.5, fy=2.5)  # 3200x1800 -> multiple tiles
    dets, info = engine.detect(big, force_tiling=True)
    assert info.tiled and info.tiles > 1
    h, w = big.shape[:2]
    assert all(0 <= d.bbox[0] < d.bbox[2] <= w + 1 and 0 <= d.bbox[1] < d.bbox[3] <= h + 1 for d in dets)
    assert info.raw_before_merge >= len(dets)


@needs_v3
def test_invalid_image_input(engine):
    with pytest.raises(ValueError):
        engine.detect(np.zeros((0, 0, 3), np.uint8))
    with pytest.raises(ValueError):
        engine.detect(None)  # type: ignore[arg-type]


# ------------------------------------------------------------------ alerts / dedup / snapshots (no model needed)
def _alert_mgr(tmp_path) -> tuple[Database, AlertManager]:
    from backend.core.config import AppConfig
    db = Database(tmp_path / "a.db")
    return db, AlertManager(db, tmp_path / "alerts", ClassMetadata(AppConfig().class_metadata_path))


def _event(frame, track=1, session="s1", box=(100, 100, 150, 140), cls="plastic_bag", job=None):
    return ConfirmedEvent(camera_id="CAM01", session_id=session, track_id=track, cls_name=cls, confidence=0.91,
                          bbox=list(map(float, box)), frame=frame, annotated=None, video_time_sec=3.2,
                          first_seen_ts=utcnow(), hits=3, active_objects=1, job_id=job)


def test_alert_created_with_snapshots_and_severity(tmp_path):
    db, am = _alert_mgr(tmp_path)
    a = am.on_confirmed(_event(_frame()), DetectionSettings())
    assert a and a["severity"] == "HIGH" and a["status"] == "new"   # plastic_bag is 'critical', conf 0.91 >= 0.75
    assert Path(a["snapshot_path"]).is_file() and Path(a["crop_path"]).is_file()
    assert "CAM01" in a["snapshot_path"] and Path(a["snapshot_path"]).name == f"alert_{a['id']:05d}.jpg"
    meta = json.loads(Path(a["snapshot_path"]).with_suffix(".json").read_text())
    assert meta["class"] == "plastic_bag" and meta["camera_id"] == "CAM01"


def test_duplicate_prevention(tmp_path):
    db, am = _alert_mgr(tmp_path)
    s, f = DetectionSettings(), _frame()
    assert am.on_confirmed(_event(f), s)
    assert am.on_confirmed(_event(f), s) is None                                         # same track again
    assert am.on_confirmed(_event(f, track=9, session="s2", box=(103, 101, 152, 141)), s) is None  # re-tracked, same place
    assert am.on_confirmed(_event(f, track=4, box=(101, 100, 151, 140), cls="food_wrapper"), s) is None  # same group (soft packaging)
    assert am.on_confirmed(_event(f, track=2, box=(600, 400, 650, 440)), s)              # different place
    assert am.on_confirmed(_event(f, track=3, box=(100, 100, 150, 140), cls="plastic_bottle"), s)  # different class group
    rows, total = db.list_alerts()
    assert total == 3 and max(r["occurrences"] for r in rows) == 3


def test_every_v3_class_has_explicit_metadata():
    """All 10 v3 classes have an explicit priority, bin and similarity group (no silent defaults)."""
    from backend.core.config import AppConfig
    meta = ClassMetadata(AppConfig().class_metadata_path)
    assert len(V3_CLASSES) == 10
    for c in V3_CLASSES:
        assert c in meta.classes, f"{c} missing from class_metadata.json"
        assert meta.classes[c]["priority"] in ("critical", "high", "medium", "low")
        assert meta.classes[c]["bin"] in ("general", "plastic", "paper", "metal", "glass")
        assert c in meta._group_of, f"{c} has no similarity group"
    # the three entries added for v3 (approved 2026-11-13): priority medium, existing groups only
    expected = {"cup": ("general", "cups_containers"), "newspaper": ("paper", "paper_like"),
                "plastic_piece": ("plastic", "soft_packaging")}
    for c, (bin_, group) in expected.items():
        assert meta.priority(c) == "medium" and meta.bin(c) == bin_ and meta._group_of[c] == group
    # 2026-11-19: the seven-class LDM schema added plastic_packet, food_waste (own 'organic' group) and glass
    assert set(meta.groups) == {"soft_packaging", "cups_containers", "bottles_cans", "paper_like", "small_items",
                                "organic"}
    assert meta._group_of["food_waste"] == "organic" and meta.bin("food_waste") == "organic"
    assert set(meta.report_groups) == {"chips_packet", "wrapper", "plastic_packaging"}           # unchanged


def test_severity_rules_configurable(tmp_path):
    db, am = _alert_mgr(tmp_path)
    s = merged_settings({"severity": {"high": [], "medium": []}})
    assert am.on_confirmed(_event(_frame()), s)["severity"] == "LOW"


# ------------------------------------------------------------------ video pipeline (real video, real model)
@needs_v3
def test_video_confirms_static_litter_once_and_outputs_files(engine, litter_image, make_video, tmp_path):
    video = make_video([litter_image], repeat=15, fps=5.0)        # 3 s static scene
    db, am = _alert_mgr(tmp_path)
    # this test is about temporal confirmation + dedup (one alert per object), not the pilot rules (30 s stationary,
    # 10-minute cooldown per location: tests/test_pilot_alerts.py), so pilot mode is off here
    s = merged_settings({"alerts": {"pilot_mode": False}})
    res = detect_video(engine, video, tmp_path / "out", s, db, am, "CAM01", "job1")
    st = res["stats"]
    gt = len(fixture_boxes(litter_image))
    assert 1 <= st["unique_confirmed_objects"] <= gt + 5           # roughly one object per real item, not per frame
    assert st["new_alerts"] == st["unique_confirmed_objects"]      # one alert per physical object, not per frame
    for f in res["files"].values():
        assert Path(f).is_file() and Path(f).stat().st_size > 0
    cap = cv2.VideoCapture(res["files"]["video"])
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 15
    cap.release()
    assert len(db.list_alerts()[0]) == st["new_alerts"]
    # one litter event per confirmed track; its later sightings are observations, not new events
    lit = res["litter"]
    assert lit["unique_events"] == st["unique_confirmed_objects"]
    assert lit["frame_observations"] >= lit["unique_events"]


@needs_v3
@pytest.mark.skipif(not CLEAN_FRAMES, reason="no human-verified clean (litter-free) CCTV frames available yet; add them "
                                             "to tests/fixtures/clean/ once the store footage is reviewed")
def test_clean_frames_create_no_alerts(engine, make_video, tmp_path):
    frames = [cv2.imread(p) for p in CLEAN_FRAMES[:5]]
    video = make_video(frames, repeat=5)
    db, am = _alert_mgr(tmp_path)
    res = detect_video(engine, video, tmp_path / "neg", DetectionSettings(), db, am, "CAM01", "jobn")
    print("clean-video stats:", res["stats"]["detections_by_status"], "alerts:", res["stats"]["new_alerts"])
    assert res["stats"]["new_alerts"] == 0


@needs_v3
def test_featureless_video_creates_no_alerts(engine, make_video, tmp_path):
    flat = np.full((720, 1280, 3), 128, np.uint8)
    video = make_video([flat], repeat=15)
    db, am = _alert_mgr(tmp_path)
    res = detect_video(engine, video, tmp_path / "flat", DetectionSettings(), db, am, "CAM01", "jobf")
    assert res["stats"]["new_alerts"] == 0


# ------------------------------------------------------------------ camera reconnect
class FlakyCapture:
    """Fake cv2.VideoCapture: fails to open the first `fail_opens` times, then serves frames and drops."""
    opens = 0

    def __init__(self, frame, fail_opens=2, frames_before_drop=5):
        FlakyCapture.opens += 1
        self.ok = FlakyCapture.opens > fail_opens
        self.frame, self.left = frame, frames_before_drop

    def isOpened(self):
        return self.ok

    def read(self):
        if self.left <= 0:
            return False, None
        self.left -= 1
        time.sleep(0.02)
        return True, self.frame.copy()

    def get(self, _):
        return 25.0

    def set(self, *_):
        return True

    def release(self):
        pass


@needs_v3
def test_camera_reconnects_and_keeps_processing(engine, litter_image, tmp_path):
    FlakyCapture.opens = 0
    events = []
    db = Database(tmp_path / "c.db")
    cam = {"id": "CAM01", "name": "Test", "source": "rtsp://user:secret@10.0.0.1/stream"}
    w = CameraWorker(cam, lambda c: StreamPipeline(engine, DetectionSettings(), None, db, "CAM01"),
                     process_fps=20, on_event=lambda lvl, cid, msg: events.append(msg),
                     capture_factory=lambda src: FlakyCapture(litter_image), max_backoff=0.05)
    w.start()
    deadline = time.time() + 30
    while time.time() < deadline and w.status.frames_processed < 2:
        time.sleep(0.1)
    w.stop()
    assert w.status.frames_processed >= 2
    assert w.status.reconnects >= 2
    assert any("disconnected" in e for e in events) and any("reconnected" in e for e in events)
    assert w.latest_jpeg is not None
    assert mask_source(cam["source"]) == "rtsp://user:***@10.0.0.1/stream"


@needs_v3
def test_camera_failure_isolated(engine, litter_image, tmp_path):
    """A permanently broken camera must not affect a healthy one."""
    FlakyCapture.opens = 10  # healthy factory opens immediately
    db = Database(tmp_path / "d.db")
    mk = lambda cid: (lambda c: StreamPipeline(engine, DetectionSettings(), None, db, cid))  # noqa: E731

    class Dead:
        def isOpened(self): return False
        def release(self): pass

    good = CameraWorker({"id": "A", "source": "x"}, mk("A"), 20, lambda *a: None,
                        lambda s: FlakyCapture(litter_image, 0, 10**6), 0.05)
    bad = CameraWorker({"id": "B", "source": "y"}, mk("B"), 20, lambda *a: None, lambda s: Dead(), 0.05)
    good.start(); bad.start()
    deadline = time.time() + 30
    while time.time() < deadline and good.status.frames_processed < 3:
        time.sleep(0.1)
    good.stop(); bad.stop()
    assert good.status.frames_processed >= 3 and bad.status.frames_processed == 0
    assert bad.status.reconnects >= 1


# ------------------------------------------------------------------ API (temporary storage + DB from the cfg fixture)
@pytest.fixture
def client(cfg):
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        r = c.post("/api/models/activate", json={"version": V3_VERSION}, headers=AUTH)
        assert r.status_code == 200, r.text
        yield c


@pytest.fixture
def client_nomodel(cfg):
    """API without needing the model to be active: for routes that never run inference."""
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        yield c


def _jpg(img) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


@needs_v3
def test_api_health_and_models(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["model"]["version"] == V3_VERSION and h["admin_enabled"]
    assert h["model"]["classes"] == V3_CLASSES
    m = client.get("/api/models").json()
    assert m["active"] == V3_VERSION and any(x["active"] for x in m["models"])


@needs_v3
def test_api_end_to_end_image_to_alert(client, litter_image):
    """Image -> YOLO -> detections -> confirmation -> alert -> DB -> API (full frame; no ROI, camera not registered)."""
    r = client.post("/api/detect/image", files={"file": ("scene.jpg", _jpg(litter_image), "image/jpeg")},
                    data={"camera_id": "CAM01"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] >= 1 and body["annotated_image"] and body["model_version"] == V3_VERSION
    assert 1 <= len(body["alerts"]) <= body["count"]        # similar boxes close together may merge into one alert
    alerts = client.get("/api/alerts", params={"camera_id": "CAM01"}).json()
    assert alerts["total"] == len(body["alerts"])
    a = alerts["alerts"][0]
    assert set(a) >= {"id", "camera_id", "ts", "class", "confidence", "track_id", "bbox", "severity", "status",
                      "snapshot_url", "crop_url", "video_time_sec", "location"}
    assert a["class"] in V3_CLASSES
    assert client.get(a["snapshot_url"]).headers["content-type"] == "image/jpeg"
    assert client.get(a["crop_url"]).status_code == 200
    assert client.patch(f"/api/alerts/{a['id']}", json={"status": "acknowledged"}).json()["status"] == "acknowledged"
    dets = client.get("/api/detections", params={"camera_id": "CAM01"}).json()
    assert dets["count"] >= body["counts"]["raw"]
    an = client.get("/api/analytics").json()
    assert an["alerts_total"] == len(body["alerts"]) and an["by_class"]


@needs_v3
def test_api_video_job(client, make_video, litter_image):
    video = make_video([litter_image], repeat=10)
    r = client.post("/api/detect/video", files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")},
                    data={"camera_id": "CAM02"})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    deadline = time.time() + 180
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job.get("error")
    assert job["result"]["stats"]["unique_confirmed_objects"] >= 1
    for kind in ("video", "detections", "events", "stats", "detections_csv", "summary_csv", "metadata"):
        assert client.get(job["files"][kind]).status_code == 200, kind
    assert job["result"]["litter"]["unique_events"] >= 1


@needs_v3
def test_api_invalid_inputs(client, cfg):
    bad = client.post("/api/detect/image", files={"file": ("x.exe", b"MZ....", "application/octet-stream")})
    assert bad.status_code == 415
    corrupt = client.post("/api/detect/image", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert corrupt.status_code == 400
    empty = client.post("/api/detect/image", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert empty.status_code == 400
    badvid = client.post("/api/detect/video", files={"file": ("v.mp4", b"garbage", "video/mp4")})
    assert badvid.status_code == 400
    assert client.post("/api/detect/image", files={"file": ("a.jpg", b"x", "image/jpeg")},
                       data={"camera_id": "../etc"}).status_code == 422
    assert client.get("/api/alerts", params={"severity": "EXTREME"}).status_code == 422
    assert client.get("/api/analytics", params={"start": "yesterday"}).status_code == 422
    # size limits: see test_upload_limits.py


def test_api_admin_auth_and_camera_crud(client_nomodel):
    client = client_nomodel
    cam = {"id": "CAM01", "name": "Main Entrance", "source": "env:CAM01_RTSP_URL", "location": "Front",
           "enabled": False}
    assert client.post("/api/cameras", json=cam).status_code == 401
    assert client.post("/api/cameras", json=cam, headers={"Authorization": "Bearer wrong"}).status_code == 401
    r = client.post("/api/cameras", json=cam, headers=AUTH)
    assert r.status_code == 201 and r.json()["store_id"] is None
    assert not any("roi" in k for k in r.json())                     # ROI removed: no roi / roi_status in the camera
    assert client.post("/api/cameras", json=cam, headers=AUTH).status_code == 409
    for gone in ({"roi": []}, {"roi_reference_size": [640, 360]}):    # removed fields are refused (extra forbidden)
        assert client.post("/api/cameras", json={**cam, "id": "CAM09", **gone}, headers=AUTH).status_code == 422
        assert client.put("/api/cameras/CAM01", json=gone, headers=AUTH).status_code == 422
    r = client.put("/api/cameras/CAM01", json={"name": "Entrance", "settings": {"tracking": {"min_confirmations": 4}}},
                   headers=AUTH)
    assert r.json()["name"] == "Entrance"
    bad = client.put("/api/cameras/CAM01", json={"settings": {"tracking": {"min_confirmations": 99}}}, headers=AUTH)
    assert bad.status_code == 422
    assert [c["id"] for c in client.get("/api/cameras").json()["cameras"]] == ["CAM01"]
    assert client.delete("/api/cameras/CAM01", headers=AUTH).json() == {"deleted": "CAM01"}
    assert client.delete("/api/cameras/CAM01", headers=AUTH).status_code == 404


def test_api_settings(client_nomodel):
    client = client_nomodel
    s = client.get("/api/settings").json()["settings"]
    assert s["confidence"]["confirmed_threshold"] == 0.6
    assert client.put("/api/settings", json={"tracking": {"temporal_window": 7}}).status_code == 401
    r = client.put("/api/settings", json={"tracking": {"temporal_window": 7, "min_confirmations": 4}}, headers=AUTH)
    assert r.status_code == 200 and r.json()["settings"]["tracking"]["temporal_window"] == 7
    bad = client.put("/api/settings", json={"confidence": {"weak_threshold": 0.9}}, headers=AUTH)
    assert bad.status_code == 422
    assert client.get("/api/settings").json()["settings"]["tracking"]["min_confirmations"] == 4


def test_api_model_activation_errors(client_nomodel):
    assert client_nomodel.post("/api/models/activate", json={"version": "nope"}, headers=AUTH).status_code == 404


# ------------------------------------------------------------------ database (no model needed)
def test_database_operations(tmp_path):
    db = Database(tmp_path / "x.db")
    db.set_setting("k", {"a": 1})
    assert db.get_setting("k") == {"a": 1}
    db.insert_detections([{"ts": utcnow(), "camera_id": "C", "source_type": "live", "class": "paper",
                           "confidence": 0.5, "status": "candidate"}])
    assert db.list_detections(camera_id="C")[0]["class"] == "paper"
    db.log_event("warning", "camera", "down", "C")
    assert db.list_events(camera_id="C")[0]["message"] == "down"
    errs = []

    def writer(i):
        try:
            db.insert_detections([{"ts": utcnow(), "camera_id": f"T{i}", "source_type": "live", "class": "x",
                                   "confidence": 0.5, "status": "weak"}] * 20)
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len(db.list_detections(limit=1000)) == 161


@needs_v3
def test_upload_without_a_camera_detects_litter_on_the_whole_image(client, litter_image):
    """Owner decision 2026-11-19: ROI removed - an image (here: no camera at all) is analysed on the whole frame; the
    existing confidence rule confirms litter and the operator's alert choice applies. No ROI field is returned."""
    before = client.get("/api/alerts").json()["total"]
    r = client.post("/api/detect/image", files={"file": ("scene.jpg", _jpg(litter_image), "image/jpeg")},
                    data={"create_alerts": "true", "return_image": "false"}).json()
    assert r["count"] > 0 and r["alerts"] and not any("roi" in k for k in r)
    assert r["counts"]["confirmed"] == sum(d["status"] == "confirmed" for d in r["detections"])
    status_of = {"confirmed": "DETECTED", "candidate": "UNDER_REVIEW", "weak": "UNDER_REVIEW", "rejected": "REJECTED"}
    assert all(d["detection_status"] == status_of[d["status"]] for d in r["detections"])
    assert client.get("/api/alerts").json()["total"] > before
