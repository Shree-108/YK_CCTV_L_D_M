"""Auto ROI (backend/services/auto_roi.py, backend/api/routes_auto_roi.py).

The segmentation model is replaced by a deterministic fake whose output is derived from the picture's colours, so
these tests check the cleaning / polygon / storage / precedence / API logic, not the ADE20K model itself.
Scene used throughout (640x360): grey ground in the bottom 60 %, a brown "carton" standing on the ground, a blue
"person", dark wall at the top.
"""
from __future__ import annotations

import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.services import auto_roi as ar

AUTH = {"Authorization": "Bearer test-admin-token"}
W, H = 640, 360
GROUND = (128, 128, 128)
CARTON = (40, 90, 160)        # BGR brown
PERSON = (200, 60, 20)        # BGR blue
CARTON_BOX = (440, 200, 540, 320)   # x1, y1, x2, y2 - inside the ground
PERSON_BOX = (100, 180, 140, 300)


def scene(person: bool = True) -> np.ndarray:
    img = np.full((H, W, 3), (30, 30, 30), np.uint8)
    img[int(H * 0.4):] = GROUND
    x1, y1, x2, y2 = CARTON_BOX
    img[y1:y2, x1:x2] = CARTON
    if person:
        x1, y1, x2, y2 = PERSON_BOX
        img[y1:y2, x1:x2] = PERSON
    return img


class FakeSeg:
    """ground = grey pixels; remove = carton/person colours. Works at a smaller 'model' resolution like UperNet."""
    name = "fake-seg"

    def __init__(self, work_width: int = 320) -> None:
        self.work_width = work_width
        self.calls = 0

    def segment(self, bgr: np.ndarray) -> ar.SegResult:
        self.calls += 1
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (self.work_width, round(h * self.work_width / w)), interpolation=cv2.INTER_NEAREST)
        d = lambda c: np.abs(small.astype(int) - np.array(c)).sum(2) < 30   # noqa: E731
        ground = d(GROUND).astype(np.float32)
        remove = (d(CARTON) | d(PERSON)).astype(np.float32)
        return ar.SegResult(ground, remove, {"sidewalk": float(ground.mean()), "box": float(remove.mean())})


@pytest.fixture(autouse=True)
def fake_segmenter():
    seg = FakeSeg()
    ar.set_segmenter(seg)
    yield seg
    ar.set_segmenter(None)


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / (W - 1), (y1 + y2) / 2 / (H - 1))


# ---------------------------------------------------------------------- pipeline
def test_ground_polygon_excludes_carton_and_person():
    res = ar.compute_auto_roi(scene())
    rec = res.record
    assert rec and rec["source"] == "auto" and rec["needs_review"] is True
    assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in rec["polygon"])
    assert 3 <= len(rec["polygon"]) <= ar.MAX_POINTS
    assert ar.point_in_roi(0.5, 0.9, rec)                      # open ground
    assert not ar.point_in_roi(0.5, 0.1, rec)                  # wall
    assert not ar.point_in_roi(*_center(CARTON_BOX), rec)      # carton removed
    assert not ar.point_in_roi(*_center(PERSON_BOX), rec)      # person removed
    assert rec["frame_size"] == [W, H] and rec["frames_used"] == 1
    assert 0.3 < rec["ground_share"] < 0.6


def test_carton_inside_ground_becomes_a_hole():
    img = np.full((H, W, 3), GROUND, np.uint8)                 # ground everywhere
    img[150:230, 280:380] = CARTON                             # carton in the middle, not touching an edge
    rec = ar.compute_auto_roi(img).record
    assert rec["holes"], "a large removed object inside the ground must be kept out as a hole"
    assert not ar.point_in_roi(330 / (W - 1), 190 / (H - 1), rec)
    assert ar.point_in_roi(0.05, 0.95, rec)


def test_small_holes_are_filled_large_kept():
    m = np.ones((200, 400), bool)
    m[100:103, 200:203] = False                                # crack / litter: tiny hole
    region = ar.clean_mask(m.astype(np.float32), np.zeros_like(m, np.float32))
    assert region[101, 201]
    rm = np.zeros((200, 400), np.float32)
    rm[60:140, 150:250] = 1                                    # big removed object
    region = ar.clean_mask(m.astype(np.float32), rm)
    assert not region[100, 200] and region[10, 10]


def test_largest_component_only():
    g = np.zeros((200, 400), np.float32)
    g[120:200, 0:300] = 1          # big
    g[0:40, 350:400] = 1           # small, separate
    region = ar.clean_mask(g, np.zeros_like(g))
    assert region[180, 100] and not region[20, 375]


def test_no_ground_gives_no_record():
    res = ar.compute_auto_roi(np.full((H, W, 3), (30, 30, 30), np.uint8))
    assert res.record is None and "manually" in res.note


def test_median_of_frames_removes_a_passer_by():
    frames = [scene(person=False) for _ in range(5)]
    frames[2] = scene(person=True)                             # person visible in one frame only
    med = ar.median_frame(frames)
    x1, y1, x2, y2 = PERSON_BOX
    assert np.abs(med[y1:y2, x1:x2].astype(int) - GROUND).max() == 0
    rec = ar.compute_auto_roi(frames).record
    assert ar.point_in_roi(*_center(PERSON_BOX), rec) and rec["frames_used"] == 5


def test_median_resizes_mismatched_frames():
    a, b = scene(), cv2.resize(scene(), (320, 180))
    assert ar.median_frame([a, b]).shape == a.shape


class FakeCap:
    def __init__(self, n: int, count: bool) -> None:
        self.frames = [np.full((8, 8, 3), i, np.uint8) for i in range(n)]
        self.i, self.count = -1, count

    def get(self, prop):
        return len(self.frames) if (prop == cv2.CAP_PROP_FRAME_COUNT and self.count) else 0

    def grab(self):
        self.i += 1
        return self.i < len(self.frames)

    def retrieve(self):
        return True, self.frames[self.i]


def test_sample_frames_spreads_over_a_file_and_steps_a_live_stream():
    got = ar.sample_frames(FakeCap(100, True), 5)
    assert [int(f[0, 0, 0]) for f in got] == [0, 24, 49, 74, 99]
    got = ar.sample_frames(FakeCap(100, False), 3, live_gap_frames=10)
    assert [int(f[0, 0, 0]) for f in got] == [0, 10, 20]


def test_detection_in_roi_uses_the_ground_contact_point():
    roi = {"polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]], "holes": []}
    assert ar.detection_in_roi([100, 100, 140, 200], W, H, roi)          # top above the ROI, bottom inside
    assert not ar.detection_in_roi([100, 20, 140, 100], W, H, roi)
    assert ar.detection_in_roi([100, 20, 140, 100], W, H, None)          # no ROI = full frame


def test_validate_polygon_rejects_bad_input():
    for bad in ([[0, 0], [1, 1]], [[0, 0], [2, 0], [1, 1]], [[0, 0], [0.5, 0.5], [1, 1]], "x", [[0, 0, 1]] * 3):
        with pytest.raises(ValueError):
            ar.validate_polygon(bad)


def test_overlay_tints_the_roi():
    rec = ar.compute_auto_roi(scene()).record
    img = scene()
    ov = ar.render_overlay(img, rec)
    assert ov.shape == img.shape
    assert ov[330, 300, 1] > img[330, 300, 1]                  # ground got greener
    assert (ov[5:20, 300:600] == img[5:20, 300:600]).all()     # outside untouched (right of the banner)


# ---------------------------------------------------------------------- storage + precedence
def test_store_lifecycle_and_precedence(tmp_path):
    store = ar.RoiStore(tmp_path / "camera_roi.json")
    assert store.resolve(None, "CAM1") == (None, "none")
    rec = store.save_auto("CAM1", ar.compute_auto_roi(scene()))
    assert rec["needs_review"] and store.reference_path("CAM1")
    saved = json.loads((tmp_path / "camera_roi.json").read_text())
    assert saved["cameras"]["CAM1"]["source"] == "auto"

    glob = store.edit(ar.GLOBAL_KEY, [[0, 0.6], [1, 0.6], [1, 1], [0, 1]])
    assert glob["source"] == "manual"
    req = [[0, 0], [1, 0], [1, 1]]
    assert store.resolve(req, "CAM1")[1] == "request"                    # request beats camera
    assert store.resolve(None, "CAM1")[1] == "camera"                    # camera beats global
    assert store.resolve(None, "CAM2") == (glob, "global")               # no camera ROI -> global

    ok = store.approve("CAM1", "qa")
    assert ok["needs_review"] is False and ok["approved_by"] == "qa" and ok["approved_at"]
    ed = store.edit("CAM1", [[0, 0.5], [1, 0.5], [1, 1], [0, 1]])
    assert ed["source"] == "auto_edited" and ed["needs_review"] is False
    assert store.clear("CAM1") and not store.reference_path("CAM1")
    assert store.resolve(None, "CAM1")[1] == "global"
    assert store.clear(ar.GLOBAL_KEY) and store.resolve(None, "CAM1") == (None, "none")
    with pytest.raises(KeyError):
        store.approve("CAM1")


def test_store_survives_a_corrupt_file(tmp_path):
    p = tmp_path / "camera_roi.json"
    p.write_text("{not json")
    assert ar.RoiStore(p).resolve(None, "X") == (None, "none")


# The /api/auto-roi routes now use backend/services/roi_service.py (config/camera_roi.json): see tests/test_roi_service.py.


@pytest.mark.skipif(not (ar.SEGMENTER_DIR / "model.safetensors").is_file(), reason="segmenter weights not present")
def test_real_segmenter_labels_cover_ground_and_cartons():
    """The real ADE20K model has the labels the ROI depends on (no inference: loading only)."""
    seg = ar.UperNetSegmenter()
    names = {seg.labels[i] for i in seg.ground_idx} | {seg.labels[i] for i in seg.remove_idx}
    assert {"floor", "road", "sidewalk", "person", "box", "shelf"} <= names
    assert not ({"bag", "bottle"} & names)          # possible litter is never removed from the ground
