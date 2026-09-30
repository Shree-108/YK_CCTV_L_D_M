"""ROI for uploads: crop-window inference, ROI persistence, litter summary, clean overlay."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.core.settings import DetectionSettings, merged_settings
from backend.detection.filters import ROISet
from backend.detection.types import CONFIRMED, Detection
from backend.services.pipeline import detect_in_roi, litter_summary, roi_window
from backend.services.visualization import annotate_litter, display_name

from .conftest import needs_model

AUTH = {"Authorization": "Bearer test-admin-token"}
LEFT_HALF = [{"name": "outside-store", "type": "sidewalk", "mode": "include",
              "polygon": [[0, 0.3], [0.5, 0.3], [0.5, 1], [0, 1]]}]


def test_crop_box_covers_include_zones_with_padding():
    box = ROISet.from_json(LEFT_HALF).crop_box(1000, 500, 0.1)
    assert box == (0, 50, 600, 500)          # x: 0..0.5+0.1, y: 0.3-0.2..1
    assert ROISet.from_json([]).crop_box(1000, 500, 0.1) is None
    exclude_only = [{**LEFT_HALF[0], "mode": "exclude"}]
    assert ROISet.from_json(exclude_only).crop_box(1000, 500, 0.1) is None


def test_roi_window_skips_full_frame_and_disabled_roi():
    s = DetectionSettings()
    roi = ROISet.from_json(LEFT_HALF)
    assert roi_window((500, 1000, 3), s, roi) is not None
    assert roi_window((500, 1000, 3), merged_settings({"filters": {"roi_crop_inference": False}}), roi) is None
    assert roi_window((500, 1000, 3), merged_settings({"filters": {"roi_enabled": False}}), roi) is None
    whole = [{**LEFT_HALF[0], "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    assert roi_window((500, 1000, 3), s, ROISet.from_json(whole)) is None


class _FakeEngine:
    """Records what the model was shown and reports one box in crop coordinates."""
    def __init__(self):
        self.shapes = []

    def detect_many(self, images, settings=None, force_tiling=None):
        self.shapes = [i.shape for i in images]
        return [([Detection([10.0, 20.0, 30.0, 40.0], 0, "paper", 0.9)], None) for _ in images]


def test_detect_in_roi_runs_on_crop_and_maps_boxes_back():
    eng = _FakeEngine()
    frame = np.zeros((500, 1000, 3), np.uint8)
    (dets, _), = detect_in_roi(eng, [frame], DetectionSettings(), ROISet.from_json(LEFT_HALF))
    x1, y1, x2, y2 = roi_window(frame.shape, DetectionSettings(), ROISet.from_json(LEFT_HALF))
    assert eng.shapes == [(y2 - y1, x2 - x1, 3)]
    assert dets[0].bbox == [10.0 + x1, 20.0 + y1, 30.0 + x1, 40.0 + y1]
    (dets, _), = detect_in_roi(eng, [frame], DetectionSettings(), None)   # no ROI: full frame
    assert eng.shapes == [(500, 1000, 3)] and dets[0].bbox == [10.0, 20.0, 30.0, 40.0]


def test_litter_summary_per_class():
    events = [{"event": "confirmed", "track_id": 1, "class": "paper", "video_time_sec": 2.0, "confidence": 0.7},
              {"event": "confirmed", "track_id": 2, "class": "paper", "video_time_sec": 9.0, "confidence": 0.8},
              {"event": "confirmed", "track_id": 3, "class": "plastic_bag", "video_time_sec": 5.0, "confidence": 0.6},
              {"event": "lost", "track_id": 1, "class": "paper", "video_time_sec": 12.0, "confidence": 0.7}]
    frames = [{"detections": [{"status": "confirmed", "track_id": 1, "confidence": 0.9},
                              {"status": "confirmed", "track_id": 1, "confidence": 0.7},
                              {"status": "rejected", "reason": "outside_roi", "confidence": 0.8}]}]
    out = litter_summary(events, frames, ROISet.from_json(LEFT_HALF), DetectionSettings())
    assert out["total"] == 3 and out["roi_active"] and out["outside_roi_ignored"] == 1
    paper = next(c for c in out["classes"] if c["class"] == "paper")
    assert paper["count"] == 2 and paper["max_confidence"] == 0.9 and paper["timestamps_sec"] == [2.0, 9.0]
    assert out["highest_confidence"] == 0.9
    assert [o["track_id"] for o in out["objects"]] == [1, 3, 2]   # ordered by first confirmation


def test_annotate_litter_draws_only_confirmed():
    frame = np.zeros((360, 640, 3), np.uint8)
    conf = Detection([100, 200, 140, 240], 0, "plastic_bag", 0.78)
    conf.status = CONFIRMED
    weak = Detection([400, 200, 440, 240], 0, "paper", 0.3)   # status weak/candidate: not drawn
    weak.status = "candidate"
    out = annotate_litter(frame, [conf, weak], ROISet.from_json(LEFT_HALF), roi_active=True)
    assert out[200:240, 100:140].any()           # confirmed box drawn
    assert not out[205:235, 405:435].any()       # candidate area untouched
    assert display_name("plastic_bag") == "Plastic Bag"


@pytest.fixture
def client(cfg):
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        c.post("/api/models/activate", json={"version": "v1_litter_s1_synthetic"}, headers=AUTH)
        yield c


@needs_model
def test_upload_roi_is_saved_reset_and_applied(client, litter_image):
    assert client.get("/api/roi/upload").json() == {"camera_id": "upload", "zones": [], "active": False}
    r = client.put("/api/roi/upload", json={"zones": LEFT_HALF})
    assert r.status_code == 200 and r.json()["active"]
    assert client.get("/api/roi/upload").json()["zones"][0]["name"] == "outside-store"

    jpg = cv2.imencode(".jpg", litter_image)[1].tobytes()
    res = client.post("/api/detect/image", files={"file": ("x.jpg", jpg, "image/jpeg")},
                      data={"create_alerts": "false", "return_image": "false"}).json()
    assert res["roi_active"]
    h, w = res["image_size"][1], res["image_size"][0]
    for d in res["detections"]:
        foot_x, foot_y = (d["bbox"][0] + d["bbox"][2]) / 2 / w, d["bbox"][3] / h
        outside = foot_x > 0.5 + 0.02 or foot_y < 0.3 - 0.02
        assert d["status"] == "rejected" if outside else True, d

    assert client.put("/api/roi/upload", json={"zones": []}).json()["active"] is False
    assert client.put("/api/roi/bad id!", json={"zones": []}).status_code in (404, 422)


@needs_model
def test_camera_roi_needs_admin(client):
    cam = {"id": "cam1", "name": "Front", "source": "0", "enabled": False}
    assert client.post("/api/cameras", json=cam, headers=AUTH).status_code in (200, 201)
    assert client.put("/api/roi/cam1", json={"zones": LEFT_HALF}).status_code == 401
    assert client.put("/api/roi/cam1", json={"zones": LEFT_HALF}, headers=AUTH).json()["active"]
    assert client.get("/api/cameras/cam1").json()["roi"][0]["name"] == "outside-store"
