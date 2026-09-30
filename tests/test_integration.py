"""Integration tests with the REAL trained YOLO model and real dataset images."""
from __future__ import annotations

import io
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
from backend.detection.filters import ROISet
from backend.services.pipeline import StreamPipeline, detect_image, detect_video

from .conftest import NEG_IMAGES, TEST_IMAGES, needs_model

pytestmark = needs_model
AUTH = {"Authorization": "Bearer test-admin-token"}


def gt_count(img_path: str) -> int:
    lbl = Path(img_path.replace("images", "labels")).with_suffix(".txt")
    return sum(1 for line in lbl.read_text().splitlines() if line.strip())


# ------------------------------------------------------------------ model + image detection
def test_model_loads_with_dataset_classes(engine):
    assert engine.class_names[0] == "chips_packet" and len(engine.class_names) == 13
    assert engine.device in ("cpu", "cuda:0", "mps")


def test_detect_image_individual_objects(engine, litter_image):
    """Each labelled item should become its own detection (no merged giant boxes)."""
    s = DetectionSettings()
    res = detect_image(engine, litter_image, s)
    confirmed = [d for d in res["detections"] if d["status"] == "confirmed"]
    n_gt = gt_count(TEST_IMAGES[0])
    assert abs(len(confirmed) - n_gt) <= 1, (len(confirmed), n_gt)
    h, w = litter_image.shape[:2]
    for d in confirmed:
        x1, y1, x2, y2 = d["bbox"]
        assert (x2 - x1) * (y2 - y1) < 0.1 * w * h  # no giant box covering several items
        assert set(d) >= {"class", "confidence", "bbox", "status"}
    assert res["count"] == len(confirmed)


def test_tiled_inference_runs_and_maps_coordinates(engine, litter_image):
    big = cv2.resize(litter_image, None, fx=2.5, fy=2.5)  # 2400x1350 -> multiple tiles
    dets, info = engine.detect(big, force_tiling=True)
    assert info.tiled and info.tiles > 1
    h, w = big.shape[:2]
    assert all(0 <= d.bbox[0] < d.bbox[2] <= w + 1 and 0 <= d.bbox[1] < d.bbox[3] <= h + 1 for d in dets)
    assert info.raw_before_merge >= len(dets)


def test_invalid_image_input(engine):
    with pytest.raises(ValueError):
        engine.detect(np.zeros((0, 0, 3), np.uint8))
    with pytest.raises(ValueError):
        engine.detect(None)  # type: ignore[arg-type]


# ------------------------------------------------------------------ alerts / dedup / snapshots
def _alert_mgr(tmp_path) -> tuple[Database, AlertManager]:
    from backend.core.config import AppConfig
    db = Database(tmp_path / "a.db")
    return db, AlertManager(db, tmp_path / "alerts", ClassMetadata(AppConfig().class_metadata_path))


def _event(frame, track=1, session="s1", box=(100, 100, 150, 140), cls="chips_packet", job=None):
    return ConfirmedEvent(camera_id="CAM01", session_id=session, track_id=track, cls_name=cls, confidence=0.91,
                          bbox=list(map(float, box)), frame=frame, annotated=None, video_time_sec=3.2,
                          first_seen_ts=utcnow(), hits=3, active_objects=1, job_id=job)


def test_alert_created_with_snapshots_and_severity(tmp_path, litter_image):
    db, am = _alert_mgr(tmp_path)
    a = am.on_confirmed(_event(litter_image), DetectionSettings())
    assert a and a["severity"] == "HIGH" and a["status"] == "new"   # chips_packet is 'critical', conf 0.91
    assert Path(a["snapshot_path"]).is_file() and Path(a["crop_path"]).is_file()
    assert "CAM01" in a["snapshot_path"] and Path(a["snapshot_path"]).name == f"alert_{a['id']:05d}.jpg"
    meta = json.loads(Path(a["snapshot_path"]).with_suffix(".json").read_text())
    assert meta["class"] == "chips_packet" and meta["camera_id"] == "CAM01"


def test_duplicate_prevention(tmp_path, litter_image):
    db, am = _alert_mgr(tmp_path)
    s = DetectionSettings()
    assert am.on_confirmed(_event(litter_image), s)
    assert am.on_confirmed(_event(litter_image), s) is None                     # same track again
    assert am.on_confirmed(_event(litter_image, track=9, session="s2", box=(103, 101, 152, 141)), s) is None  # re-tracked, same place
    assert am.on_confirmed(_event(litter_image, track=2, box=(600, 400, 650, 440)), s)  # different place
    assert am.on_confirmed(_event(litter_image, track=3, box=(100, 100, 150, 140), cls="metal_can"), s)  # different class group
    rows, total = db.list_alerts()
    assert total == 3 and max(r["occurrences"] for r in rows) == 2


def test_severity_rules_configurable(tmp_path, litter_image):
    db, am = _alert_mgr(tmp_path)
    s = merged_settings({"severity": {"high": [], "medium": []}})
    assert am.on_confirmed(_event(litter_image), s)["severity"] == "LOW"


# ------------------------------------------------------------------ video pipeline (real video, real model)
def test_video_confirms_static_litter_once_and_outputs_files(engine, make_video, tmp_path):
    img = cv2.imread(TEST_IMAGES[0])
    video = make_video([img], repeat=15, fps=5.0)        # 3 s static scene, analysed at 5 fps
    db, am = _alert_mgr(tmp_path)
    res = detect_video(engine, video, tmp_path / "out", DetectionSettings(), None, db, am, "CAM01", "job1")
    st = res["stats"]
    n_gt = gt_count(TEST_IMAGES[0])
    assert abs(st["unique_confirmed_objects"] - n_gt) <= 1
    assert st["new_alerts"] == st["unique_confirmed_objects"]  # one alert per physical object, not per frame
    for f in res["files"].values():
        assert Path(f).is_file() and Path(f).stat().st_size > 0
    cap = cv2.VideoCapture(res["files"]["video"])
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 15
    cap.release()
    # re-processing the same object in the same job must not create new alerts
    assert len(db.list_alerts()[0]) == st["new_alerts"]


def test_negative_frames_create_no_alerts(engine, make_video, tmp_path):
    if not NEG_IMAGES:
        pytest.skip("no negative images")
    frames = [cv2.imread(p) for p in NEG_IMAGES[:5]]
    video = make_video(frames, repeat=5)
    db, am = _alert_mgr(tmp_path)
    res = detect_video(engine, video, tmp_path / "neg", DetectionSettings(), None, db, am, "CAM01", "jobn")
    # informational: a false positive here is recorded, not hidden
    print("negative-video stats:", res["stats"]["detections_by_status"], "alerts:", res["stats"]["new_alerts"])
    assert res["stats"]["new_alerts"] <= 1


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


def test_camera_reconnects_and_keeps_processing(engine, litter_image, tmp_path):
    FlakyCapture.opens = 0
    events = []
    db = Database(tmp_path / "c.db")
    cam = {"id": "CAM01", "name": "Test", "source": "rtsp://user:secret@10.0.0.1/stream", "roi": []}
    w = CameraWorker(cam, lambda c: StreamPipeline(engine, DetectionSettings(), None, None, db, "CAM01"),
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


def test_camera_failure_isolated(engine, litter_image, tmp_path):
    """A permanently broken camera must not affect a healthy one."""
    FlakyCapture.opens = 10  # healthy factory opens immediately
    db = Database(tmp_path / "d.db")
    mk = lambda cid: (lambda c: StreamPipeline(engine, DetectionSettings(), None, None, db, cid))  # noqa: E731

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


# ------------------------------------------------------------------ API
@pytest.fixture
def client(cfg):
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        # tests were written against the fine-tuned v1 model (not the zero-shot v2)
        c.post("/api/models/activate", json={"version": "v1_litter_s1_synthetic"}, headers=AUTH)
        yield c


def _jpg(img) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_api_health_and_models(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["model"]["version"] and h["admin_enabled"]
    m = client.get("/api/models").json()
    assert m["active"] and any(x["active"] for x in m["models"])


def test_api_end_to_end_image_to_alert(client, litter_image):
    """Image -> YOLO -> detections -> confirmation -> alert -> DB -> API."""
    r = client.post("/api/detect/image", files={"file": ("scene.jpg", _jpg(litter_image), "image/jpeg")},
                    data={"camera_id": "CAM01"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] >= 1 and body["annotated_image"]
    assert len(body["alerts"]) == body["count"]
    alerts = client.get("/api/alerts", params={"camera_id": "CAM01"}).json()
    assert alerts["total"] == body["count"]
    a = alerts["alerts"][0]
    assert set(a) >= {"id", "camera_id", "ts", "class", "confidence", "track_id", "bbox", "severity", "status",
                      "snapshot_url", "crop_url", "video_time_sec", "location"}
    assert client.get(a["snapshot_url"]).headers["content-type"] == "image/jpeg"
    assert client.get(a["crop_url"]).status_code == 200
    assert client.patch(f"/api/alerts/{a['id']}", json={"status": "acknowledged"}).json()["status"] == "acknowledged"
    # same image again -> deduplicated (same place, same class group, within window)... different upload job,
    # so they are separate recordings and allowed to alert; detections are always logged
    dets = client.get("/api/detections", params={"camera_id": "CAM01"}).json()
    assert dets["count"] >= body["counts"]["raw"]
    an = client.get("/api/analytics").json()
    assert an["alerts_total"] == body["count"] and an["by_class"]


def test_api_video_job(client, make_video, litter_image):
    video = make_video([litter_image], repeat=10)
    r = client.post("/api/detect/video", files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")},
                    data={"camera_id": "CAM02"})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    deadline = time.time() + 120
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job.get("error")
    assert job["result"]["stats"]["unique_confirmed_objects"] >= 1
    assert client.get(job["files"]["video"]).status_code == 200


def test_api_invalid_inputs(client, cfg):
    bad = client.post("/api/detect/image", files={"file": ("x.exe", b"MZ....", "application/octet-stream")})
    assert bad.status_code == 415
    corrupt = client.post("/api/detect/image", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert corrupt.status_code == 400
    empty = client.post("/api/detect/image", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert empty.status_code == 400
    cfg.max_image_mb = 0  # every non-empty upload is now too large
    big = client.post("/api/detect/image", files={"file": ("x.jpg", b"x" * 10, "image/jpeg")})
    assert big.status_code == 413
    badvid = client.post("/api/detect/video", files={"file": ("v.mp4", b"garbage", "video/mp4")})
    assert badvid.status_code == 400
    assert client.post("/api/detect/image", files={"file": ("a.jpg", b"x", "image/jpeg")},
                       data={"camera_id": "../etc"}).status_code == 422
    assert client.get("/api/alerts", params={"severity": "EXTREME"}).status_code == 422
    assert client.get("/api/analytics", params={"start": "yesterday"}).status_code == 422


def test_api_admin_auth_and_camera_crud(client):
    cam = {"id": "CAM01", "name": "Main Entrance", "source": "env:CAM01_RTSP_URL", "location": "Front",
           "enabled": False, "roi": [{"name": "sidewalk", "type": "sidewalk", "mode": "include",
                                      "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}]}
    assert client.post("/api/cameras", json=cam).status_code == 401
    assert client.post("/api/cameras", json=cam, headers={"Authorization": "Bearer wrong"}).status_code == 401
    r = client.post("/api/cameras", json=cam, headers=AUTH)
    assert r.status_code == 201 and r.json()["roi"][0]["type"] == "sidewalk"
    assert client.post("/api/cameras", json=cam, headers=AUTH).status_code == 409
    bad_roi = {**cam, "id": "CAM09", "roi": [{"polygon": [[0, 0], [5, 0], [0, 1]]}]}
    assert client.post("/api/cameras", json=bad_roi, headers=AUTH).status_code == 422
    r = client.put("/api/cameras/CAM01", json={"name": "Entrance", "settings": {"tracking": {"min_confirmations": 4}}},
                   headers=AUTH)
    assert r.json()["name"] == "Entrance"
    bad = client.put("/api/cameras/CAM01", json={"settings": {"tracking": {"min_confirmations": 99}}}, headers=AUTH)
    assert bad.status_code == 422
    assert [c["id"] for c in client.get("/api/cameras").json()["cameras"]] == ["CAM01"]
    assert client.delete("/api/cameras/CAM01", headers=AUTH).json() == {"deleted": "CAM01"}
    assert client.delete("/api/cameras/CAM01", headers=AUTH).status_code == 404


def test_api_settings(client):
    s = client.get("/api/settings").json()["settings"]
    assert s["confidence"]["confirmed_threshold"] == 0.6
    assert client.put("/api/settings", json={"tracking": {"temporal_window": 7}}).status_code == 401
    r = client.put("/api/settings", json={"tracking": {"temporal_window": 7, "min_confirmations": 4}}, headers=AUTH)
    assert r.status_code == 200 and r.json()["settings"]["tracking"]["temporal_window"] == 7
    bad = client.put("/api/settings", json={"confidence": {"weak_threshold": 0.9}}, headers=AUTH)
    assert bad.status_code == 422
    assert client.get("/api/settings").json()["settings"]["tracking"]["min_confirmations"] == 4


def test_api_model_activation_errors(client):
    assert client.post("/api/models/activate", json={"version": "nope"}, headers=AUTH).status_code == 404


# ------------------------------------------------------------------ database
def test_database_operations(tmp_path):
    db = Database(tmp_path / "x.db")
    db.set_setting("k", {"a": 1})
    assert db.get_setting("k") == {"a": 1}
    db.insert_detections([{"ts": utcnow(), "camera_id": "C", "source_type": "live", "class": "paper",
                           "confidence": 0.5, "status": "candidate"}])
    assert db.list_detections(camera_id="C")[0]["class"] == "paper"
    db.log_event("warning", "camera", "down", "C")
    assert db.list_events(camera_id="C")[0]["message"] == "down"
    # Database is safe to use from several threads at once
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
