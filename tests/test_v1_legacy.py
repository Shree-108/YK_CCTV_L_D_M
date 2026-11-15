"""Historical tests that only make sense with the v1 artifacts (kept verbatim in intent, NOT run here).

They need models/litter_yolo/v1_litter_s1_synthetic/best.pt and dataset/cctv_litter_dataset_v1/images/**, which were
never committed and are not on this machine (v1 was trained on another PC; its metadata has no hash). They are skipped
with that reason; they must not be "fixed" by substituting other weights or images. The equivalent checks for the
production model v3 are in test_integration.py and test_lab_api.py.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from backend.core.settings import DetectionSettings
from backend.services.pipeline import detect_image, detect_video

from .conftest import V1_MODEL, V1_NEG_IMAGES, V1_TEST_IMAGES, needs_v1

pytestmark = needs_v1
AUTH = {"Authorization": "Bearer test-admin-token"}
DS = "cctv_litter_dataset_v1"


def gt_count(img_path: str) -> int:
    lbl = Path(img_path.replace("images", "labels")).with_suffix(".txt")
    return sum(1 for line in lbl.read_text().splitlines() if line.strip())


@pytest.fixture(scope="module")
def engine_v1():
    from backend.detection.engine import YoloEngine
    return YoloEngine(V1_MODEL, DetectionSettings(), "test-v1")


def test_v1_model_loads_with_dataset_classes(engine_v1):
    assert engine_v1.class_names[0] == "chips_packet" and len(engine_v1.class_names) == 13


def test_v1_detect_image_matches_ground_truth_count(engine_v1):
    img = cv2.imread(V1_TEST_IMAGES[0])
    res = detect_image(engine_v1, img, DetectionSettings())
    confirmed = [d for d in res["detections"] if d["status"] == "confirmed"]
    assert abs(len(confirmed) - gt_count(V1_TEST_IMAGES[0])) <= 1


def test_v1_video_confirms_each_object_once(engine_v1, make_video, tmp_path):
    from .test_integration import _alert_mgr
    video = make_video([cv2.imread(V1_TEST_IMAGES[0])], repeat=15, fps=5.0)
    db, am = _alert_mgr(tmp_path)
    st = detect_video(engine_v1, video, tmp_path / "out", DetectionSettings(), None, db, am, "CAM01", "job1")["stats"]
    assert abs(st["unique_confirmed_objects"] - gt_count(V1_TEST_IMAGES[0])) <= 1


@pytest.mark.skipif(not V1_NEG_IMAGES, reason="v1 negative images unavailable")
def test_v1_negative_frames_create_at_most_one_alert(engine_v1, make_video, tmp_path):
    from .test_integration import _alert_mgr
    video = make_video([cv2.imread(p) for p in V1_NEG_IMAGES[:5]], repeat=5)
    db, am = _alert_mgr(tmp_path)
    res = detect_video(engine_v1, video, tmp_path / "neg", DetectionSettings(), None, db, am, "CAM01", "jobn")
    assert res["stats"]["new_alerts"] <= 1


@pytest.fixture
def client_v1(cfg):
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        c.post("/api/models/activate", json={"version": "v1_litter_s1_synthetic"}, headers=AUTH)
        yield c


def test_v1_dataset_stats(client_v1):
    s = client_v1.get(f"/api/datasets/{DS}").json()
    assert s["totals"]["images"] == 210 and s["totals"]["negatives"] == 60
    assert client_v1.get(f"/api/datasets/{DS}/images", params={"split": "test"}).json()["total"] == 10


def test_v1_evaluation_has_13_classes(client_v1):
    import time
    r = client_v1.post("/api/evaluations", json={"dataset": DS, "split": "test"})
    job_id = r.json()["job_id"]
    deadline = time.time() + 300
    while time.time() < deadline and client_v1.get(f"/api/jobs/{job_id}").json()["status"] not in ("done", "failed"):
        time.sleep(0.5)
    ev = client_v1.get(f"/api/evaluations/{job_id}").json()
    assert {"chips_packet", "wrapper", "plastic_packaging"} <= set(ev["report_groups"]) and len(ev["per_class"]) == 13
