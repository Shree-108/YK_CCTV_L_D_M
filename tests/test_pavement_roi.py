"""Pavement-only mode: camera-specific ROI enforcement + significant-litter size filter (backend/detection/pavement.py).

Geometry note: CAM_A_LAYOUT below is the 2026-11-16 *proposed* (NOT approved, NOT active) polygon set for the indoor
counter-view camera, copied here as plain TEST DATA so the rules are exercised on the real scene layout. It is not a
camera configuration and nothing in the application reads it. The REAL_* boxes are detections the production v3 model
made on that camera's 2026-11-10 recording (job sv_20261115_232628_e603d5), in original 2560x1440 pixels.

The model is replaced by a stub that "detects" magenta rectangles and returns boxes in the coordinates of the image
it was given (exactly like Ultralytics), so the engine's tile / ROI-crop coordinate mapping is checked against known
ground truth. These tests say nothing about detection accuracy.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.alerts.manager import AlertManager, ClassMetadata
from backend.core.config import PROJECT_ROOT
from backend.core.settings import DetectionSettings, PavementSettings, SizeLimit, merged_settings
from backend.database.db import Database
from backend.detection import pavement as pv
from backend.detection.engine import YoloEngine
from backend.detection.filters import ROISet
from backend.detection.types import CANDIDATE, CONFIRMED, REJECTED, WEAK, Detection
from backend.sampling import auto_roi, detection_report
from backend.services.pipeline import StreamPipeline, detect_image, detect_in_roi, filter_detections

W, H = 2560, 1440


def norm(pts, w=W, h=H):
    return [(round(x / w, 6), round(y / h, 6)) for x, y in pts]


CAM_A_LAYOUT = [
    {"name": "outdoor_pavement", "type": "ground", "mode": "include", "polygon": norm([
        (648, 38), (830, 25), (878, 80), (922, 25), (926, 25), (926, 146), (1100, 147), (1280, 172), (1450, 202),
        (1640, 229), (1820, 254), (2000, 284), (2085, 305), (2150, 342), (2180, 360), (2160, 500), (2153, 560),
        (2143, 625), (2124, 688), (2100, 790), (1900, 752), (1900, 684), (1868, 655), (1775, 655), (1745, 684),
        (1745, 733), (1716, 726), (1695, 772), (1285, 676), (1102, 632), (1122, 566), (785, 505), (700, 541),
        (668, 531), (686, 432), (648, 414)])},
    {"name": "shop_left_rack_and_storage_box", "type": "custom", "mode": "exclude", "polygon": norm([
        (0, 0), (628, 0), (628, 432), (668, 447), (652, 553), (0, 720)])},
    {"name": "counter_trays_devices_glass", "type": "custom", "mode": "exclude", "polygon": norm([
        (0, 720), (652, 553), (705, 564), (785, 527), (1102, 584), (1086, 652), (1285, 710), (1700, 796), (1712, 748),
        (1752, 756), (1752, 697), (1779, 672), (1858, 672), (1884, 697), (1884, 768), (2112, 810), (2125, 810),
        (2560, 810), (2560, 1440), (0, 1440)])},
    {"name": "shop_right_racks", "type": "custom", "mode": "exclude", "polygon": norm([
        (2203, 0), (2560, 0), (2560, 810), (2125, 810), (2123, 790), (2143, 690), (2163, 630), (2178, 500),
        (2203, 345)])},
    {"name": "awning_and_hanging_sign", "type": "custom", "mode": "exclude", "polygon": norm([
        (834, 0), (2203, 0), (2203, 345), (2160, 334), (2090, 296), (2000, 274), (1820, 244), (1640, 219),
        (1450, 192), (1280, 162), (1100, 137), (938, 134), (938, 12), (908, 12), (880, 64), (852, 12)])},
]

# real v3 detections on CAM_A (2026-11-10), original pixels
REAL_COUNTER_PIECE = [731.0, 987.0, 743.0, 991.0]        # plastic_piece 0.72 "confirmed" today: inside the counter
REAL_BOX_LID = [607.0, 443.0, 648.0, 454.0]              # plastic_packaging 0.55: storage-box lid edge/reflection
REAL_BOX_LID_2 = [334.0, 452.0, 353.0, 466.0]            # plastic_packaging 0.54: storage-box lid
REAL_SIDE_GLASS = [14.0, 824.0, 33.0, 838.0]             # food_wrapper 0.56: side-glass sliver, bottom-left
REAL_PAVEMENT_MARK = [1550.0, 416.0, 1639.0, 431.0]      # plastic_bag 0.40 (weak): faint mark on the pavers

# a simple second camera ("camera B", 1920x1080): pavement = right half below the kerb, OSD clock bottom-left
CAM_B_LAYOUT = [
    {"name": "pavement", "type": "ground", "mode": "include",
     "polygon": norm([(1120, 545), (1920, 525), (1920, 1080), (1112, 1080)], 1920, 1080)},
    {"name": "osd_timestamp", "type": "custom", "mode": "exclude",
     "polygon": norm([(60, 1020), (560, 1020), (560, 1080), (60, 1080)], 1920, 1080)},
    {"name": "lamp_pole", "type": "custom", "mode": "exclude",
     "polygon": norm([(1500, 300), (1540, 300), (1540, 540), (1500, 540)], 1920, 1080)},   # pole body, base excluded
]


def P(**kw) -> PavementSettings:
    return PavementSettings(enabled=True, **kw)


def S(**pav) -> DetectionSettings:
    return merged_settings({"pavement": {"enabled": True, **pav}})


def det(bbox, conf=0.8, cls="food_wrapper", status=CANDIDATE) -> Detection:
    return Detection(list(bbox), 0, cls, conf, status=status)


def run_gate(dets, roi_zones, shape=(H, W, 3), approved=True, p=None, s=None, camera="CAM_T", registered=True):
    """Exactly what the pipeline does in pavement mode for a named camera: decide the mode, gate, confirm, finalize."""
    s = s or DetectionSettings()
    p = p or P()
    roi = ROISet.from_json(roi_zones) if roi_zones is not None else None
    status, _ = pv.decide_mode(camera, registered, roi, approved, shape, p)
    pv.gate(dets, shape, roi, status, p)
    # still-image confirmation exactly as detect_image does it
    for d in dets:
        if d.status == CANDIDATE and d.confidence >= s.confidence.confirmed_threshold:
            d.status = CONFIRMED
    pv.finalize(dets)
    return status


# ---------------------------------------------------------------------- 1, 5. inside the pavement ROI
def test_large_wrapper_inside_pavement_roi_is_confirmed():
    d = det([1300, 330, 1420, 400], 0.82)                         # on the pavers, 120x70 px
    assert run_gate([d], CAM_A_LAYOUT) == pv.ROI_ACTIVE
    assert d.outcome == pv.CONFIRMED_LITTER and d.status == CONFIRMED
    assert d.geometry["membership"] == pv.INSIDE and d.geometry["zone"] == "outdoor_pavement"
    assert d.geometry["size_ok"] is True


def test_inside_roi_but_below_confirmed_confidence_is_a_candidate_not_confirmed():
    d = det([1300, 330, 1420, 400], 0.50)                         # candidate tier, < 0.60 confirmed threshold
    run_gate([d], CAM_A_LAYOUT)
    assert d.outcome == pv.SMALL_OR_WEAK_CANDIDATE and d.status == CANDIDATE
    assert d.outcome_reason == "below_confirmation_rule"


def test_weak_tier_inside_roi_is_candidate_with_low_confidence_reason():
    d = det([1300, 330, 1420, 400], 0.35, status=WEAK)
    run_gate([d], CAM_A_LAYOUT)
    assert d.outcome == pv.SMALL_OR_WEAK_CANDIDATE and d.status == WEAK and d.outcome_reason == "low_confidence"


# ---------------------------------------------------------------------- 2. outside
def test_detection_outside_pavement_roi_is_rejected_and_kept():
    car = det([1300, 200, 1500, 300], 0.9)                        # camera B: on the road above the kerb
    run_gate([car], CAM_B_LAYOUT, shape=(1080, 1920, 3))
    assert car.status == REJECTED and car.reason == "outside_roi"
    assert car.outcome == pv.REJECTED_OUTSIDE_ROI and car.outcome_reason == "outside_pavement_roi"
    assert car.bbox == [1300, 200, 1500, 300] and car.confidence == 0.9      # nothing about the raw box is lost


# ---------------------------------------------------------------------- 3. exclusion zones
def test_detection_inside_exclusion_zone_is_rejected_even_when_in_pavement_tolerance():
    d = det(REAL_COUNTER_PIECE, 0.72, "plastic_piece")
    run_gate([d], CAM_A_LAYOUT)
    assert d.outcome == pv.REJECTED_OUTSIDE_ROI
    assert d.outcome_reason in ("exclusion_zone:counter_trays_devices_glass", "outside_pavement_roi")


def test_exclusion_applies_after_membership_foot_point_inside_pavement_but_box_on_pole():
    pole = det([1495, 320, 1545, 560], 0.7)                       # tall thin box: foot on pavement, body is the pole
    run_gate([pole], CAM_B_LAYOUT, shape=(1080, 1920, 3))
    assert pole.geometry["exclusion_hit"] == "area"
    assert pole.outcome == pv.REJECTED_OUTSIDE_ROI and pole.outcome_reason == "exclusion_zone:lamp_pole"


def test_osd_timestamp_is_never_litter():
    clock = det([348, 1030, 399, 1068], 0.80)                     # EZVIZ: "food_wrapper 0.80" on the clock digits
    run_gate([clock], CAM_B_LAYOUT, shape=(1080, 1920, 3))
    assert clock.outcome == pv.REJECTED_OUTSIDE_ROI


# ---------------------------------------------------------------------- 4. crossing the boundary
def test_box_crossing_roi_boundary_with_foot_inside_is_kept():
    d = det([1080, 480, 1200, 600], 0.8)                          # camera B: left part over the awning edge
    run_gate([d], CAM_B_LAYOUT, shape=(1080, 1920, 3))
    assert d.outcome == pv.CONFIRMED_LITTER and d.geometry["membership"] == pv.INSIDE


def test_foot_point_just_outside_within_edge_tolerance_is_kept():
    # camera B's left pavement edge is at x≈1119 for y=600; tolerance = 0.01 * 1080 = 10.8 px
    d = det([1050, 540, 1174, 600], 0.8)                          # foot (1112, 600): ~7 px outside, 44 % of box inside
    run_gate([d], CAM_B_LAYOUT, shape=(1080, 1920, 3))
    assert -10.8 <= d.geometry["foot_distance_px"] < 0
    assert d.geometry["membership"] == pv.EDGE and d.outcome == pv.CONFIRMED_LITTER
    far = det([1030, 540, 1150, 600], 0.8)                        # foot (1090, 600): ~29 px outside, 26 % inside
    run_gate([far], CAM_B_LAYOUT, shape=(1080, 1920, 3))
    assert far.outcome == pv.REJECTED_OUTSIDE_ROI


def test_foot_hidden_below_boundary_but_most_of_box_inside_is_kept_by_area_overlap():
    roi = [{"name": "pave", "type": "ground", "mode": "include", "polygon": [(0.2, 0.2), (0.8, 0.2), (0.8, 0.6), (0.2, 0.6)]}]
    d = det([1000, 600, 1200, 900], 0.8)                          # y 600..900, boundary at 0.6*1440 = 864
    run_gate([d], roi)
    assert d.geometry["membership"] == pv.OVERLAP and d.outcome == pv.CONFIRMED_LITTER
    only_foot = det([1000, 600, 1200, 900], 0.8)
    run_gate([only_foot], roi, p=P(membership="footpoint"))
    assert only_foot.outcome == pv.REJECTED_OUTSIDE_ROI


def test_box_mostly_outside_with_foot_outside_is_rejected():
    roi = [{"name": "pave", "type": "ground", "mode": "include", "polygon": [(0.2, 0.2), (0.8, 0.2), (0.8, 0.6), (0.2, 0.6)]}]
    d = det([1000, 820, 1200, 1100], 0.8)                         # only the top 44 px of 280 inside
    run_gate([d], roi)
    assert d.outcome == pv.REJECTED_OUTSIDE_ROI


# ---------------------------------------------------------------------- 6. tiny mark
def test_tiny_plastic_mark_inside_roi_is_small_candidate_never_confirmed():
    d = det([1500, 330, 1510, 334], 0.90, "plastic_piece")        # 10x4 px with very high confidence
    run_gate([d], CAM_A_LAYOUT)
    assert d.outcome == pv.SMALL_OR_WEAK_CANDIDATE and d.status == WEAK
    assert "below_min_size" in d.outcome_reason and d.geometry["size_ok"] is False


def test_size_rule_needs_both_side_and_area():
    thin = det([1300, 330, 1500, 335], 0.9)                       # 200x5: big area? 1000 px, side 5 px
    run_gate([thin], CAM_A_LAYOUT)
    assert thin.outcome == pv.SMALL_OR_WEAK_CANDIDATE             # fails min side
    square = det([1300, 330, 1318, 348], 0.9)                     # 18x18 = 324 px < 0.00015*W*H = 553
    run_gate([square], CAM_A_LAYOUT)
    assert square.geometry["min_side_frac"] >= 0.012 and square.outcome == pv.SMALL_OR_WEAK_CANDIDATE


def test_per_class_size_override():
    p = P(class_size={"plastic_piece": SizeLimit(min_side_frac=0.002, min_area_frac=0.00001)})
    a, b = det([1500, 330, 1510, 334], 0.9, "plastic_piece"), det([1500, 330, 1510, 334], 0.9, "paper")
    run_gate([a, b], CAM_A_LAYOUT, p=p)
    assert a.outcome == pv.CONFIRMED_LITTER and b.outcome == pv.SMALL_OR_WEAK_CANDIDATE


# ---------------------------------------------------------------------- 7, 8, 9. real CAM_A false positives
@pytest.mark.parametrize("box,cls,zone", [
    (REAL_BOX_LID, "plastic_packaging", "shop_left_rack_and_storage_box"),       # 8. reflection on the box lid
    (REAL_BOX_LID_2, "plastic_packaging", "shop_left_rack_and_storage_box"),
    ([150, 120, 260, 300], "food_wrapper", "shop_left_rack_and_storage_box"),     # 7. chips packet on the rack
    ([900, 560, 960, 640], "plastic_bottle", "counter_trays_devices_glass"),      # 7. bottle in the front tray
    (REAL_SIDE_GLASS, "food_wrapper", "counter_trays_devices_glass"),
    ([1400, 100, 1500, 180], "paper", "awning_and_hanging_sign"),                 # 9. white awning flap
    ([2250, 300, 2400, 420], "plastic_packaging", "shop_right_racks"),           # 7. hanging merchandise
])
def test_real_cam_a_non_ground_detections_are_rejected(box, cls, zone):
    d = det(box, 0.75, cls)
    run_gate([d], CAM_A_LAYOUT)
    assert d.outcome == pv.REJECTED_OUTSIDE_ROI, d.geometry
    assert d.outcome_reason in (f"exclusion_zone:{zone}", "outside_pavement_roi")
    assert d.status == REJECTED


def test_real_pavement_mark_stays_a_weak_candidate():
    d = det(REAL_PAVEMENT_MARK, 0.3953, "plastic_bag", status=WEAK)
    run_gate([d], CAM_A_LAYOUT)
    assert d.outcome == pv.SMALL_OR_WEAK_CANDIDATE and d.status == WEAK


# ---------------------------------------------------------------------- 10. resolutions
@pytest.mark.parametrize("w,h", [(2560, 1440), (1920, 1080), (1280, 720), (640, 360)])
def test_same_scene_same_outcomes_at_every_resolution(w, h):
    base = [([1300, 330, 1420, 400], 0.82), ([1500, 330, 1510, 334], 0.9), (REAL_BOX_LID, 0.55),
            (REAL_COUNTER_PIECE, 0.72), ([1400, 100, 1500, 180], 0.7)]
    k = w / W
    dets = [det([v * k for v in b], c) for b, c in base]
    run_gate(dets, CAM_A_LAYOUT, shape=(h, w, 3))
    assert [d.outcome for d in dets] == [pv.CONFIRMED_LITTER, pv.SMALL_OR_WEAK_CANDIDATE, pv.REJECTED_OUTSIDE_ROI,
                                         pv.REJECTED_OUTSIDE_ROI, pv.REJECTED_OUTSIDE_ROI]
    assert dets[0].geometry["min_side_frac"] == pytest.approx(70 / 1440, abs=1e-4)
    assert dets[0].geometry["area_frac"] == pytest.approx(120 * 70 / (W * H), rel=1e-3)


# ---------------------------------------------------------------------- 11. engine coordinate mapping (stub model)
MAGENTA = (255, 0, 255)


class _A:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _A(np.asarray(xyxy, np.float32).reshape(-1, 4)), _A(conf), _A(cls)

    def __len__(self):
        return len(self.xyxy.a)


class _Res:
    def __init__(self, boxes):
        self.boxes = boxes


class StubModel:
    """'Detects' every magenta rectangle in the image it is given; boxes in that image's own pixel coordinates.
    Confidence is read from the rectangle's green channel (g/255) so a test can set it per object."""
    task = "detect"
    names = {0: "food_wrapper"}

    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def predict(self, images, **kw):
        out = []
        for img in images if isinstance(images, list) else [images]:
            self.calls.append(img.shape[:2])
            # tolerant to video compression: strong blue AND red, cleaned with a small closing
            m = ((img[:, :, 0] > 200) & (img[:, :, 2] > 200)).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
            xyxy, conf = [], []
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area < 12:
                    continue
                xyxy.append([x, y, x + w, y + h])
                conf.append(float(np.median(img[:, :, 1][lab == i])) / 255)
            out.append(_Res(_Boxes(xyxy, np.float32(conf), np.zeros(len(conf), int))))
        return out


def stub_engine(settings: DetectionSettings | None = None) -> YoloEngine:
    e = YoloEngine.__new__(YoloEngine)
    import threading
    e._lock, e.settings, e.version, e.weights = threading.Lock(), settings or DetectionSettings(), "stub", "stub"
    e.model, e.names, e.vocab, e._raw_to_cls, e.device, e.train_imgsz = StubModel(), {0: "food_wrapper"}, None, None, "cpu", 640
    return e


def scene(w, h, objects):
    img = np.full((h, w, 3), 90, np.uint8)
    for (x1, y1, x2, y2), conf in objects:
        img[y1:y2, x1:x2] = (255, int(round(conf * 255)), 255)
    return img


def test_tiled_inference_maps_boxes_back_to_original_frame():
    truth = [(1300, 330, 1420, 400), (900, 980, 1150, 1100), (2400, 1300, 2470, 1350)]   # one crosses a tile border
    img = scene(W, H, [(b, 0.8) for b in truth])
    eng = stub_engine()
    dets, info = eng.detect(img, force_tiling=True)
    assert info.tiled and info.tiles > 1 and info.raw_before_merge > len(truth)          # partial tile boxes existed
    got = sorted(tuple(round(v) for v in d.bbox) for d in dets)
    assert got == sorted(truth)                                                       # merged + exact full-frame coords
    assert all(shape != (H, W) for shape in eng.model.calls[1:])                      # tiles really were crops


def test_roi_crop_inference_maps_boxes_back_to_original_frame():
    roi = ROISet.from_json(CAM_A_LAYOUT)
    truth = (1300, 330, 1420, 400)
    img = scene(W, H, [(truth, 0.8)])
    eng = stub_engine()
    s = merged_settings({"filters": {"roi_crop_inference": True}})          # experimental setting, explicitly on
    (dets, _), = detect_in_roi(eng, [img], s, roi, force_tiling=False)
    assert eng.model.calls[0] != (H, W)                                               # the model saw the crop only
    assert [tuple(round(v) for v in d.bbox) for d in dets] == [truth]


# ---------------------------------------------------------------------- 12. camera A ROI never used for camera B
def test_roi_saved_for_camera_a_is_not_applied_to_camera_b(tmp_path):
    db = Database(tmp_path / "t.db")
    db.set_setting("roi:CAM_A", CAM_A_LAYOUT)
    a = auto_roi.decide("CAM_A", None, db.get_setting("roi:CAM_A"))
    b = auto_roi.decide("CAM_B", None, db.get_setting("roi:CAM_B"))
    assert a.status == auto_roi.APPROVED and b.status == auto_roi.NOT_CONFIGURED and b.zones == []
    # and the pavement gate, given camera B's (missing) ROI, confirms nothing
    d = det([1300, 330, 1420, 400], 0.9)
    assert run_gate([d], b.zones or None) == pv.ROI_NOT_CONFIGURED and d.outcome == pv.UNVERIFIED


def test_roi_drawn_on_a_16x9_camera_is_refused_on_a_4x3_frame():
    d = det([500, 300, 600, 380], 0.9)
    status = run_gate([d], CAM_A_LAYOUT, shape=(960, 1280, 3), p=P(roi_reference_size=(2560, 1440)))
    assert status == pv.ROI_FRAME_MISMATCH and d.outcome == pv.UNVERIFIED and d.status == WEAK
    ok = det([1300 / 2, 330 / 2, 1420 / 2, 400 / 2], 0.9)                             # same camera, 1280x720 substream
    assert run_gate([ok], CAM_A_LAYOUT, shape=(720, 1280, 3), p=P(roi_reference_size=(2560, 1440))) == pv.ROI_ACTIVE
    assert ok.outcome == pv.CONFIRMED_LITTER


# ---------------------------------------------------------------------- 13. missing ROI
@pytest.mark.parametrize("zones,approved", [(None, True), ([z for z in CAM_A_LAYOUT if z["mode"] == "exclude"], True),
                                            (CAM_A_LAYOUT, False)])
def test_missing_or_unapproved_roi_gives_roi_not_configured_and_nothing_confirmed(zones, approved):
    dets = [det([1300, 330, 1420, 400], 0.95), det(REAL_COUNTER_PIECE, 0.72), det([0, 0, 5, 5], 0.9, status=REJECTED)]
    dets[2].reason = "too_small"
    assert run_gate(dets, zones, approved=approved) == pv.ROI_NOT_CONFIGURED
    assert [d.outcome for d in dets] == [pv.UNVERIFIED, pv.UNVERIFIED, pv.REJECTED_FILTER]
    assert not any(d.status == CONFIRMED for d in dets)


# ---------------------------------------------------------------------- 14. no confirmed alert from rejected
@pytest.fixture
def alert_env(tmp_path):
    db = Database(tmp_path / "a.db")
    meta = ClassMetadata(PROJECT_ROOT / "backend" / "core" / "class_metadata.json")
    return db, AlertManager(db, tmp_path / "alerts", meta), meta


def n_alerts(db) -> int:
    with db.connect() as c:
        return c.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]


def test_image_only_significant_in_roi_litter_creates_an_alert(alert_env):
    db, alerts, meta = alert_env
    objs = [((1300, 330, 1420, 400), 0.85),            # significant, on the pavement -> the only alert
            ((1500, 330, 1510, 334), 0.95),            # tiny mark on the pavement
            ((607, 443, 648, 454), 0.90),              # storage-box lid
            ((900, 560, 960, 640), 0.90),              # bottle in the front counter tray
            ((1400, 100, 1500, 180), 0.90)]            # awning flap
    img = scene(W, H, objs)
    res = detect_image(stub_engine(), img, S(), ROISet.from_json(CAM_A_LAYOUT), meta, db, alerts, "CAM_A",
                       create_alerts=True, want_diagnostic=True, camera_registered=True)
    assert res["pavement"]["status"] == pv.ROI_ACTIVE
    assert res["count"] == 1 and len(res["alerts"]) == 1 and n_alerts(db) == 1
    c = res["pavement"]["counts"]
    assert (c["confirmed_litter"], c["small_or_weak_candidate"], c["rejected_outside_roi"]) == (1, 1, 3)
    assert isinstance(res.pop("diagnostic"), np.ndarray) and isinstance(res.pop("annotated"), np.ndarray)
    with db.connect() as cn:                                               # every raw detection logged with its status
        assert cn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 5


def test_roi_crop_inference_never_sees_objects_far_outside_the_pavement(alert_env):
    """EXPERIMENTAL filters.roi_crop_inference (OFF by default): the model only sees the ROI's bounding rectangle +
    padding, so an object deep in the counter is not even a raw detection. Default full frame keeps it (rejected)."""
    db, alerts, meta = alert_env
    img = scene(W, H, [((1300, 330, 1420, 400), 0.85), ((728, 980, 760, 1000), 0.90)])
    roi = ROISet.from_json(CAM_A_LAYOUT)
    on = detect_image(stub_engine(), img, merged_settings({"pavement": {"enabled": True},
                                                           "filters": {"roi_crop_inference": True}}),
                      roi, meta, None, None, "CAM_A", camera_registered=True)
    off = detect_image(stub_engine(), img, S(), roi, meta, None, None, "CAM_A", camera_registered=True)   # default
    assert on["counts"]["raw"] == 1 and off["counts"]["raw"] == 2
    assert off["pavement"]["counts"]["rejected_outside_roi"] == 1 and off["count"] == on["count"] == 1


def test_image_without_roi_creates_no_alert_in_pavement_mode(alert_env):
    db, alerts, meta = alert_env
    img = scene(W, H, [((1300, 330, 1420, 400), 0.95)])
    res = detect_image(stub_engine(), img, S(), None, meta, db, alerts, "CAM_X", create_alerts=True)
    assert res["pavement"]["status"] == pv.ROI_NOT_CONFIGURED and res["count"] == 0 and n_alerts(db) == 0
    assert res["detections"][0]["outcome"] == pv.UNVERIFIED


def test_video_pipeline_confirms_only_significant_pavement_litter(alert_env):
    db, alerts, meta = alert_env
    objs = [((1300, 330, 1420, 400), 0.85), ((1500, 330, 1510, 334), 0.95), ((607, 443, 648, 454), 0.90)]
    img = scene(W, H, objs)
    s = S()
    pipe = StreamPipeline(stub_engine(s), s, ROISet.from_json(CAM_A_LAYOUT), alerts, db, "CAM_A", "video", "job1",
                          camera_registered=True)
    events = []
    for i in range(8):
        r = pipe.process(img, i, i * 0.5)
        events += r.events
        assert r.pavement_status == pv.ROI_ACTIVE
    confirmed = [e for e in events if e["event"] == "confirmed"]
    assert len(confirmed) == 1 and n_alerts(db) == 1
    assert {d.outcome for d in r.detections} == {pv.CONFIRMED_LITTER, pv.SMALL_OR_WEAK_CANDIDATE, pv.REJECTED_OUTSIDE_ROI}


def test_video_pipeline_without_roi_never_confirms(alert_env):
    db, alerts, meta = alert_env
    img = scene(W, H, [((1300, 330, 1420, 400), 0.95)])
    s = S()
    pipe = StreamPipeline(stub_engine(s), s, None, alerts, db, "CAM_X", "video", "job2")
    events = [e for i in range(8) for e in pipe.process(img, i, i * 0.5).events]
    assert not [e for e in events if e["event"] == "confirmed"] and n_alerts(db) == 0


# ---------------------------------------------------------------------- pavement mode OFF = old behaviour
def test_pavement_mode_off_is_unchanged_behaviour():
    img = scene(W, H, [((1300, 330, 1420, 400), 0.85), ((607, 443, 648, 454), 0.90)])
    roi = ROISet.from_json(CAM_A_LAYOUT)
    res = detect_image(stub_engine(), img, DetectionSettings(), roi, None, None, None, "CAM_A")
    assert "pavement" not in res and all("outcome" not in d for d in res["detections"])
    dets = [det([1300, 330, 1420, 400]), det(REAL_BOX_LID)]
    assert filter_detections(dets, (H, W, 3), DetectionSettings(), roi) is None
    assert [d.reason for d in dets] == [None, "outside_roi"]          # the existing ROI filter still runs
    assert DetectionSettings().pavement.enabled is False


# ---------------------------------------------------------------------- 15. CSV and JSON
def test_csv_and_json_distinguish_confirmed_candidates_and_rejected(alert_env):
    db, alerts, meta = alert_env
    img = scene(W, H, [((1300, 330, 1420, 400), 0.85), ((1500, 330, 1510, 334), 0.95), ((607, 443, 648, 454), 0.9)])
    res = detect_image(stub_engine(), img, S(), ROISet.from_json(CAM_A_LAYOUT), meta, None, None, "CAM_A",
                       camera_registered=True)
    res.pop("annotated")
    frames = [{"idx": 1, "frame_idx": 10, "video_time_sec": 1.0, "status": "saved", "outcome": "LITTER_DETECTED",
               "original_path": "x/s1.jpg", "result": res}]
    objects = detection_report.link_objects(frames, iou_min=0.3, center_frac=0.04, similar=lambda a, b: a == b)
    job = {"id": "sv_t", "store_id": "", "camera_id": "CAM_A", "mode": "auto", "sample_count": 1}
    roi = {"status": "APPROVED", "source": "SAVED_CAMERA_ROI", "zones": CAM_A_LAYOUT, "approved": True, "name": "p"}
    rows, summary = detection_report.build(job, {"width": W, "height": H}, frames, objects, roi, completed_at="2026-11-16T00:00:00")
    by = {r["pavement_outcome"]: r for r in rows if r["record_type"] == "DETECTION"}
    assert set(by) == {pv.CONFIRMED_LITTER, pv.SMALL_OR_WEAK_CANDIDATE, pv.REJECTED_OUTSIDE_ROI}
    assert by[pv.CONFIRMED_LITTER]["review_required"] == "FALSE" and by[pv.CONFIRMED_LITTER]["size_check"] == "SIGNIFICANT"
    assert by[pv.SMALL_OR_WEAK_CANDIDATE]["review_required"] == "TRUE"
    assert by[pv.SMALL_OR_WEAK_CANDIDATE]["size_check"] == "BELOW_MIN_SIZE"
    assert by[pv.REJECTED_OUTSIDE_ROI]["roi_membership"] == "OUTSIDE"            # lid: foot point not on the pavement
    assert by[pv.REJECTED_OUTSIDE_ROI]["pavement_reason"] == "outside_pavement_roi"
    assert by[pv.REJECTED_OUTSIDE_ROI]["confidence_score"] == 0.902
    assert (summary["confirmed_litter_detections"], summary["small_or_weak_candidates"],
            summary["rejected_outside_roi"], summary["total_objects_detected"]) == (1, 1, 1, 1)
    csv_text = detection_report.to_csv(rows, detection_report.DETECTION_FIELDS)
    assert csv_text.splitlines()[0].endswith("box_area_fraction,size_check")
    import json
    js = json.loads(detection_report.summary_json(summary, objects, frames, roi))
    assert js["pavement"]["status"] == pv.CAMERA_ROI_MODE and js["pavement"]["pavement_validated"] is True and js["pavement"]["per_screenshot"][0]["confirmed_litter"] == 1


def test_report_says_roi_not_configured_instead_of_no_litter():
    frames = [{"idx": i, "frame_idx": i, "status": "saved", "outcome": "ROI_NOT_CONFIGURED",
               "result": {"pavement": {"status": pv.ROI_NOT_CONFIGURED, "counts": {}}, "detections": [
                   {"class": "food_wrapper", "confidence": 0.8, "bbox": [1, 1, 50, 50], "status": "weak",
                    "outcome": pv.UNVERIFIED, "outcome_reason": "roi_not_configured",
                    "geometry": {"membership": "NOT_EVALUATED", "size_ok": True}}]}} for i in (1, 2)]
    job = {"id": "sv_t", "mode": "auto", "sample_count": 2}
    roi = {"status": "NOT_CONFIGURED", "source": "FULL_FRAME_FALLBACK", "zones": [], "approved": False}
    rows, summary = detection_report.build(job, {"width": W, "height": H}, frames, {}, roi, completed_at="t")
    assert summary["overall_analysis_status"] == "PAVEMENT_ROI_NOT_CONFIGURED"
    assert summary["analysis_mode"] == pv.ROI_NOT_CONFIGURED and summary["review_required"] == "TRUE"
    assert summary["pavement_validated"] == "FALSE"
    assert summary["unverified_roi_not_configured"] == 2 and summary["total_objects_detected"] == 0
    dets = [r for r in rows if r["record_type"] == "DETECTION"]
    assert len(dets) == 2 and all(r["final_classification"] == pv.UNVERIFIED and r["review_required"] == "TRUE" for r in dets)


# ---------------------------------------------------------------------- display
def test_annotation_draws_roi_only_when_approved_and_states_status():
    from backend.services.visualization import PAVEMENT_GREEN, annotate_pavement
    img = scene(W, H, [])
    roi = ROISet.from_json(CAM_A_LAYOUT)
    active = annotate_pavement(img, [], roi, pv.ROI_ACTIVE, "CAM_A")
    missing = annotate_pavement(img, [], roi, pv.ROI_NOT_CONFIGURED, "CAM_A")
    green = lambda a: int(np.all(a == PAVEMENT_GREEN, axis=2).sum())     # noqa: E731
    assert green(active) > 1000 and green(missing) < green(active) // 10


ADMIN = {"Authorization": "Bearer test-admin-token"}


# ---------------------------------------------------------------------- zero-click workflow, real v3 model
@pytest.fixture
def pav_client(cfg):
    from fastapi.testclient import TestClient
    from backend.main import create_app
    cfg.roi_segmenter = "off"
    with TestClient(create_app(cfg)) as c:
        if c.app.state.app_state.engine is None:
            pytest.skip("no detection model loaded")
        c.app.state.app_state.save_global_settings({"pavement": {"enabled": True}})   # this test's own temp DB
        yield c


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_auto_job_without_roi_reports_roi_not_configured_and_no_alerts(pav_client, tmp_path):
    from tests.test_auto_analysis import _drop, _wait, litter_frame, video
    c = pav_client
    j, _ = _wait(c, _drop(c, video(tmp_path / "litter.mp4", 80, frame=litter_frame), camera_id="CAM-NEW").json()["job_id"])
    assert j["lifecycle"] == "COMPLETED", j["error"]
    assert [f["outcome"] for f in j["frames"]] == ["ROI_NOT_CONFIGURED"] * 4          # never NO_LITTER, never litter
    assert j["summary"]["overall_analysis_status"] == "PAVEMENT_ROI_NOT_CONFIGURED"
    assert j["summary"]["total_objects_detected"] == 0 and j["summary"]["review_required"] == "TRUE"
    assert c.get("/api/alerts", params={"limit": 1}).json()["total"] == 0
    f = j["frames"][0]
    assert f["pavement"]["status"] == "ROI_NOT_CONFIGURED" and f["pavement"]["counts"]["raw"] > 0
    assert all(d["outcome"] == "UNVERIFIED_ROI_NOT_CONFIGURED" for d in f["detections"] if d["status"] != "rejected")
    assert c.get(f["diagnostic_url"]).status_code == 200 and c.get(f["annotated_url"]).status_code == 200
    rows = list(csv.DictReader(io.StringIO(c.get(j["reports"]["detection_csv"]).text)))
    assert sum(r["record_type"] == "DETECTION" for r in rows) == sum(len(x["detections"]) for x in j["frames"])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_auto_job_with_saved_roi_confirms_only_significant_members(pav_client, tmp_path):
    from tests.test_auto_analysis import _drop, _wait, litter_frame, video
    c = pav_client
    bottom = {"name": "pavement", "type": "ground", "mode": "include", "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}
    r = c.post("/api/cameras", headers=ADMIN, json={"id": "CAM-P", "name": "test camera", "source": "env:LM_TEST_NONE",
                                                     "enabled": False, "roi": [bottom]})   # this test's temp DB only
    assert r.status_code == 201, r.text
    j, _ = _wait(c, _drop(c, video(tmp_path / "litter.mp4", 80, frame=litter_frame), camera_id="CAM-P").json()["job_id"])
    assert j["lifecycle"] == "COMPLETED", j["error"]
    dets = [d for f in j["frames"] for d in f["detections"]]
    assert {f["pavement"]["status"] for f in j["frames"]} == {"CAMERA_ROI_MODE"}
    for d in dets:
        if d["outcome"] == "CONFIRMED_LITTER":
            assert d["geometry"]["size_ok"] and d["geometry"]["membership"] != "OUTSIDE" and d["status"] == "confirmed"
            assert d["bbox"][3] >= 0.5 * j["frames"][0]["image_size"][1] - 0.01 * j["frames"][0]["image_size"][1]
        else:
            assert d["status"] != "confirmed"
    assert any(d["outcome"] == "REJECTED_OUTSIDE_ROI" for d in dets)       # the fixture has litter in the top half too
    assert sum(f["detection_count"] for f in j["frames"]) == sum(d["outcome"] == "CONFIRMED_LITTER" for d in dets)



# ====================================================================== review corrections (2026-11-16)
ADMIN_H = {"Authorization": "Bearer test-admin-token"}


# 1, 2. full-frame inference is the default; ROI-crop inference is experimental and off
def test_full_frame_inference_is_the_default_and_roi_crop_is_off():
    assert DetectionSettings().filters.roi_crop_inference is False
    eng = stub_engine()
    img = scene(W, H, [((1300, 330, 1420, 400), 0.85)])
    res = detect_image(eng, img, S(), ROISet.from_json(CAM_A_LAYOUT), None, None, None, "CAM_A", camera_registered=True)
    assert eng.model.calls[0] == (H, W)                        # the model saw the whole frame (first pass)
    assert res["count"] == 1
    from backend.services.pipeline import roi_window
    assert roi_window((H, W, 3), S(), ROISet.from_json(CAM_A_LAYOUT)) is None


# 3. the ROI is a post-detection filter: coordinates identical to plain inference
def test_roi_post_filtering_does_not_alter_inference_coordinates():
    objs = [((1300, 330, 1420, 400), 0.85), ((607, 443, 648, 454), 0.9), ((1500, 330, 1510, 334), 0.95),
            ((900, 980, 1150, 1100), 0.8)]
    img = scene(W, H, objs)
    plain = detect_image(stub_engine(), img, DetectionSettings(), None, None, None, None, "upload")
    gated = detect_image(stub_engine(), img, S(), ROISet.from_json(CAM_A_LAYOUT), None, None, None, "CAM_A",
                         camera_registered=True)
    key = lambda r: sorted((tuple(d["bbox"]), d["confidence"]) for d in r["detections"])   # noqa: E731
    assert key(plain) == key(gated) and len(gated["detections"]) == 4
    assert {d["outcome"] for d in gated["detections"]} >= {pv.CONFIRMED_LITTER, pv.REJECTED_OUTSIDE_ROI}


# general image mode = the existing behaviour, labelled as not validated
def test_general_image_mode_keeps_existing_results_and_alerts(alert_env):
    db, alerts, meta = alert_env
    objs = [((1300, 330, 1420, 400), 0.85), ((607, 443, 648, 454), 0.90), ((1500, 330, 1510, 334), 0.95)]
    img = scene(W, H, objs)
    old = detect_image(stub_engine(), img, DetectionSettings(), None, meta, None, None, "upload")
    new = detect_image(stub_engine(), img, S(), None, meta, db, alerts, "upload", create_alerts=True)
    assert new["pavement"]["status"] == pv.GENERAL_IMAGE_MODE and new["pavement"]["pavement_validated"] is False
    assert [(d["status"], d["bbox"]) for d in old["detections"]] == [(d["status"], d["bbox"]) for d in new["detections"]]
    assert new["count"] == old["count"] == 3 and len(new["alerts"]) == 3          # alerts exactly as before
    assert {d["outcome"] for d in new["detections"]} == {pv.GENERAL_CONFIRMED}
    assert all(d["geometry"]["membership"] == "NOT_EVALUATED" and "size_ok" in d["geometry"] for d in new["detections"])


# 4-7 through the real API (real v3 model, this test's temp DB)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_manual_upload_without_camera_registration_still_works(pav_client):
    from tests.conftest import FIXTURE_IMAGE
    c = pav_client
    body = lambda: {"file": ("street.jpg", FIXTURE_IMAGE.read_bytes(), "image/jpeg")}      # noqa: E731
    on = c.post("/api/detect/image", files=body(), data={"create_alerts": "false"}).json()
    c.app.state.app_state.save_global_settings({"pavement": {"enabled": False}})
    off = c.post("/api/detect/image", files=body(), data={"create_alerts": "false"}).json()
    assert on["pavement"]["status"] == "GENERAL_IMAGE_MODE" and on["count"] > 0
    assert on["count"] == off["count"] and [d["bbox"] for d in on["detections"]] == [d["bbox"] for d in off["detections"]]
    assert "pavement" not in off


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_unknown_camera_cannot_inherit_another_cameras_roi_and_registered_camera_without_roi_confirms_nothing(pav_client):
    from tests.conftest import FIXTURE_IMAGE
    c = pav_client
    full = {"name": "all", "type": "ground", "mode": "include", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    assert c.post("/api/cameras", headers=ADMIN_H, json={"id": "CAM-A", "name": "a", "source": "env:LM_TEST_NONE",
                                                         "enabled": False, "roi": [full]}).status_code == 201
    assert c.post("/api/cameras", headers=ADMIN_H, json={"id": "CAM-NOROI", "name": "n", "source": "env:LM_TEST_NONE",
                                                         "enabled": False}).status_code == 201
    st = c.app.state.app_state
    assert st.roi_for(st.db.get_camera("CAM-Z"), "CAM-Z") is None                     # unknown camera: nothing
    body = lambda: {"file": ("street.jpg", FIXTURE_IMAGE.read_bytes(), "image/jpeg")}  # noqa: E731
    before = c.get("/api/alerts", params={"limit": 1}).json()["total"]
    for cam in ("CAM-Z", "CAM-NOROI"):
        r = c.post("/api/detect/image", files=body(), data={"camera_id": cam, "create_alerts": "true"}).json()
        assert r["pavement"]["status"] == "ROI_NOT_CONFIGURED" and r["count"] == 0 and r["alerts"] == []
        assert all(d["outcome"] in ("UNVERIFIED_ROI_NOT_CONFIGURED", "REJECTED_FILTER") for d in r["detections"])
    assert c.get("/api/alerts", params={"limit": 1}).json()["total"] == before
    a = c.post("/api/detect/image", files=body(), data={"camera_id": "CAM-A", "create_alerts": "false"}).json()
    assert a["pavement"]["status"] == "CAMERA_ROI_MODE" and a["count"] > 0            # its own ROI applies


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_two_cameras_keep_separate_roi_configurations(pav_client):
    c = pav_client
    for cid, zones in (("INDOOR-T", CAM_A_LAYOUT), ("EZVIZ-T", CAM_B_LAYOUT)):
        r = c.post("/api/cameras", headers=ADMIN_H, json={"id": cid, "name": cid, "source": "env:LM_TEST_NONE",
                                                          "enabled": False, "roi": zones})
        assert r.status_code == 201, r.text
    st = c.app.state.app_state
    a, b = st.roi_for(st.db.get_camera("INDOOR-T")), st.roi_for(st.db.get_camera("EZVIZ-T"))
    assert [z.name for z in a.zones] == [z["name"] for z in CAM_A_LAYOUT]
    assert [z.name for z in b.zones] == [z["name"] for z in CAM_B_LAYOUT]
    assert not {tuple(map(tuple, z.polygon)) for z in a.zones} & {tuple(map(tuple, z.polygon)) for z in b.zones}
    # the same normalised box: CAM_A pavement (centre of the pavers), but on camera B's road above the kerb
    box = (1300, 330, 1420, 400)
    img = scene(W, H, [(box, 0.85)])
    ra = detect_image(stub_engine(), img, S(), a, None, None, None, "INDOOR-T", camera_registered=True)
    rb = detect_image(stub_engine(), img, S(), b, None, None, None, "EZVIZ-T", camera_registered=True)
    assert ra["detections"][0]["outcome"] == pv.CONFIRMED_LITTER
    assert rb["detections"][0]["outcome"] == pv.REJECTED_OUTSIDE_ROI
    # changing one camera's ROI never changes the other
    assert c.put("/api/cameras/EZVIZ-T", headers=ADMIN_H, json={"roi": CAM_B_LAYOUT[:1]}).status_code == 200
    assert [z.name for z in st.roi_for(st.db.get_camera("INDOOR-T")).zones] == [z["name"] for z in CAM_A_LAYOUT]


# 8. full-video workflow still works (pavement off and on)
def test_full_video_workflow_works_with_and_without_pavement_mode(tmp_path, make_video, alert_env):
    from backend.services.pipeline import detect_video
    db, alerts, meta = alert_env
    w, h = 640, 360
    k = w / W
    frame = scene(w, h, [((int(1300 * k), int(330 * k), int(1420 * k), int(400 * k)), 0.85),
                         ((int(607 * k), int(443 * k), int(660 * k), int(470 * k)), 0.9)])
    vid = make_video([frame], repeat=20, fps=4.0)
    roi = ROISet.from_json(CAM_A_LAYOUT)
    off = detect_video(stub_engine(), vid, tmp_path / "off", DetectionSettings(), roi, None, None, "CAM_A")
    on = detect_video(stub_engine(S()), vid, tmp_path / "on", S(), roi, db, alerts, "CAM_A", "vjob",
                      camera_registered=True)
    for r in (off, on):
        assert all(Path(f).is_file() for f in r["files"].values())
    assert "pavement" not in off["stats"] and on["stats"]["pavement"]["status"] == pv.CAMERA_ROI_MODE
    assert on["stats"]["unique_confirmed_objects"] == 1 and len(on["alerts"]) == 1


# 9. CSV / JSON: general detections vs ROI-validated pavement detections
def test_csv_and_json_distinguish_general_from_roi_validated_detections():
    img = scene(W, H, [((1300, 330, 1420, 400), 0.85)])
    gen = detect_image(stub_engine(), img, S(), None, None, None, None, "upload")
    cam = detect_image(stub_engine(), img, S(), ROISet.from_json(CAM_A_LAYOUT), None, None, None, "CAM_A",
                       camera_registered=True)
    out = {}
    for name, res, roi in (("general", gen, {"status": "NOT_CONFIGURED", "zones": [], "approved": False}),
                           ("camera", cam, {"status": "APPROVED", "zones": CAM_A_LAYOUT, "approved": True})):
        res.pop("annotated")
        frames = [{"idx": 1, "frame_idx": 1, "status": "saved", "outcome": "LITTER_DETECTED", "result": res}]
        objects = detection_report.link_objects(frames, iou_min=0.3, center_frac=0.04, similar=lambda a, b: a == b)
        rows, summary = detection_report.build({"id": "j", "mode": "auto", "sample_count": 1}, {"width": W, "height": H},
                                               frames, objects, roi, completed_at="t")
        import json
        out[name] = ([r for r in rows if r["record_type"] == "DETECTION"][0], summary,
                     json.loads(detection_report.summary_json(summary, objects, frames, roi)))
    g, cm = out["general"], out["camera"]
    assert (g[0]["analysis_mode"], g[0]["pavement_validated"], g[0]["pavement_outcome"]) == (
        "GENERAL_IMAGE_MODE", "FALSE", pv.GENERAL_CONFIRMED)
    assert (cm[0]["analysis_mode"], cm[0]["pavement_validated"], cm[0]["pavement_outcome"]) == (
        "CAMERA_ROI_MODE", "TRUE", pv.CONFIRMED_LITTER)
    assert g[0]["roi_membership"] == "NOT_EVALUATED_GENERAL_MODE" and cm[0]["roi_membership"] == "INSIDE"
    assert cm[0]["pavement_reason"].startswith("accepted:INSIDE+significant_size")
    assert g[1]["general_confirmed_detections"] == 1 and g[1]["confirmed_litter_detections"] == 0
    assert cm[1]["confirmed_litter_detections"] == 1 and cm[1]["general_confirmed_detections"] == 0
    assert g[2]["pavement"]["pavement_validated"] is False and cm[2]["pavement"]["pavement_validated"] is True
    assert "General image mode" in g[1]["note"]


def test_scheduled_non_litter_task_is_not_affected_by_pavement_mode():
    """The scheduler disables pavement mode for non-litter models (e.g. the indoor uniform task)."""
    import inspect
    from backend.cameras import scheduler
    src = inspect.getsource(scheduler)
    assert 'if cfg["task"] != "litter":' in src and '"enabled": False' in src
