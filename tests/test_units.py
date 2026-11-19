"""Unit tests that don't need the model: tiling, filters, confidence tiers, tracker, settings."""
import numpy as np
import pytest
from pydantic import ValidationError

from backend.core.settings import DetectionSettings, merged_settings
from backend.detection.filters import apply_filters, confidence_tier, confirm_single_image
from backend.detection.tiling import merge_detections, tile_grid
from backend.detection.types import CANDIDATE, CONFIRMED, REJECTED, WEAK, Detection
from backend.tracking.tracker import LitterTracker


def D(box, conf=0.9, cls=0, name="chips_packet", source="full"):
    return Detection(list(map(float, box)), cls, name, conf, source=source)


# ------------------------------------------------------------------ tiling
def test_tile_grid_covers_frame_with_overlap():
    tiles = tile_grid(3000, 2000, 1024, 0.2)
    assert tiles[0] == (0, 0, 1024, 1024)
    assert max(t[2] for t in tiles) == 3000 and max(t[3] for t in tiles) == 2000
    xs = sorted({t[0] for t in tiles})
    assert all(b - a <= 1024 * 0.8 + 1 for a, b in zip(xs, xs[1:]))  # overlap >= 20%


def test_small_frame_single_tile():
    assert tile_grid(800, 600, 1024, 0.2) == [(0, 0, 800, 600)]


def test_merge_keeps_adjacent_objects_separate():
    # two touching packets from the same pass must remain two boxes
    dets = [D([100, 100, 150, 150], 0.9), D([150, 100, 200, 150], 0.85)]
    assert len(merge_detections(dets, 0.5, 0.75)) == 2


def test_merge_removes_duplicates_and_cut_off_tile_boxes():
    full = D([100, 100, 200, 200], 0.9, source="tile:0")
    dup = D([102, 101, 199, 201], 0.8, source="tile:1")           # same object seen by the overlap tile
    partial = D([150, 100, 200, 200], 0.7, source="tile:2")       # cut by a tile border, inside `full`
    other_class = D([100, 100, 200, 200], 0.6, cls=3, name="plastic_cup", source="tile:1")
    out = merge_detections([full, dup, partial, other_class], 0.5, 0.75)
    assert [d.confidence for d in out] == [0.9, 0.6]


# ------------------------------------------------------------------ confidence tiers
def test_confidence_tiers():
    s = DetectionSettings()
    assert confidence_tier(0.20, s) == REJECTED
    assert confidence_tier(0.30, s) == WEAK
    assert confidence_tier(0.50, s) == CANDIDATE


def test_low_confidence_image_detection_is_never_confirmed():
    s = DetectionSettings()
    dets = [D([10, 10, 40, 40], 0.25), D([50, 50, 80, 80], 0.50), D([90, 90, 120, 120], 0.91)]
    confirm_single_image(apply_filters(dets, (200, 200, 3), s), s)
    assert [d.status for d in dets] == [REJECTED, CANDIDATE, CONFIRMED]


def test_threshold_order_validated():
    with pytest.raises(ValidationError):
        merged_settings({"confidence": {"weak_threshold": 0.7, "candidate_threshold": 0.5}})
    with pytest.raises(ValidationError):
        merged_settings({"tracking": {"temporal_window": 3, "min_confirmations": 4}})


# ------------------------------------------------------------------ size/class filters (no spatial filter)
def test_apply_filters_never_rejects_by_position():
    """ROI was removed: a detection anywhere in the frame (corners, top 'sky' area, edges) is never rejected
    because of WHERE it is - only confidence/class/size/aspect may reject it."""
    s = DetectionSettings()
    boxes = [[0, 0, 20, 20], [80, 0, 100, 20], [0, 80, 20, 100], [80, 80, 100, 100], [40, 40, 60, 60],
             [0, 45, 15, 60], [85, 40, 100, 55]]
    dets = apply_filters([D(b) for b in boxes], (100, 100, 3), s)
    assert all(d.status == CANDIDATE and d.reason is None for d in dets)
    assert not hasattr(s.filters, "roi_enabled")


def test_settings_strip_legacy_roi_and_pavement_keys():
    """Stored settings from before ROI removal still load: legacy keys are dropped, never validated or used."""
    legacy = {"pavement": {"mode": "on", "min_area": 10},
              "filters": {"roi_enabled": True, "roi_anchor": "footpoint", "roi_margin_frac": 0.02,
                          "roi_crop_inference": True, "roi_crop_pad_frac": 0.1, "min_box_area_px": 77}}
    s = merged_settings(legacy, {"pavement": {"mode": "off"}, "filters": {"roi_enabled": False}})
    dumped = s.model_dump()
    assert "pavement" not in dumped
    assert not any(k.startswith("roi") for k in dumped["filters"])
    assert s.filters.min_box_area_px == 77            # real keys next to legacy ones still apply
    assert "pavement" not in repr(dumped) and "roi_" not in repr(dumped)
    with pytest.raises(ValidationError):              # anything else unknown is still rejected (extra=forbid)
        merged_settings({"filters": {"not_a_setting": 1}})


def test_size_and_class_filters():
    s = merged_settings({"filters": {"class_blocklist": ["cardboard"]}})
    dets = [D([0, 0, 3, 3]), D([0, 0, 90, 90]), D([10, 10, 30, 30], name="cardboard")]
    apply_filters(dets, (100, 100, 3), s)
    assert [d.reason for d in dets] == ["too_small", "too_large", "class_blocked"]



# ------------------------------------------------------------------ tracker + temporal confirmation
def _frame(tr, boxes_confs, idx):
    dets = [D(b, c) for b, c in boxes_confs]
    for d in dets:
        d.status = CANDIDATE if d.confidence >= 0.45 else WEAK
    ev = tr.update(dets, idx, (1000, 1000, 3))
    return dets, ev


def test_same_object_keeps_one_id_across_frames():
    tr = LitterTracker(DetectionSettings())
    ids = set()
    for i in range(6):
        dets, _ = _frame(tr, [([100 + i, 100, 140 + i, 130], 0.9)], i)
        ids.add(dets[0].track_id)
    assert len(ids) == 1


def test_temporal_confirmation_with_gap():
    # frames 1,2,3 detected, 4 missed, 5 detected -> confirmed exactly once
    tr = LitterTracker(DetectionSettings())
    box = [100, 100, 140, 130]
    confirmations = []
    for i, seen in enumerate([True, True, True, False, True]):
        _, ev = _frame(tr, [(box, 0.8)] if seen else [], i)
        confirmations += [e for e in ev if e.kind == "confirmed"]
    assert len(confirmations) == 1 and confirmations[0].track.track_id == 1


def test_single_frame_flash_not_confirmed():
    tr = LitterTracker(DetectionSettings())
    _, ev = _frame(tr, [([100, 100, 140, 130], 0.95)], 0)
    for i in range(1, 6):
        _, e2 = _frame(tr, [], i)
        ev += e2
    assert not [e for e in ev if e.kind == "confirmed"]


def test_weak_detections_never_confirm():
    tr = LitterTracker(DetectionSettings())
    ev = []
    for i in range(10):
        _, e = _frame(tr, [([100, 100, 140, 130], 0.35)], i)
        ev += e
    assert not ev and not tr.tracks  # weak dets never start tracks


def test_carried_object_not_confirmed():
    tr = LitterTracker(DetectionSettings())
    ev = []
    for i in range(8):  # moves 40 px per frame (person carrying a bottle)
        x = 100 + 40 * i
        _, e = _frame(tr, [([x, 100, x + 40, 130], 0.9)], i)
        ev += e
    assert not [e for e in ev if e.kind == "confirmed"]


def test_two_adjacent_objects_two_tracks():
    tr = LitterTracker(DetectionSettings())
    for i in range(4):
        dets, _ = _frame(tr, [([100, 100, 140, 130], 0.9), ([141, 100, 181, 130], 0.9)], i)
    assert len({d.track_id for d in dets}) == 2 and len(tr.confirmed_tracks) == 2


def test_lost_track_emits_event():
    s = merged_settings({"tracking": {"max_missed_frames": 2}})
    tr = LitterTracker(s)
    for i in range(3):
        _frame(tr, [([100, 100, 140, 130], 0.9)], i)
    kinds = []
    for i in range(3, 7):
        _, ev = _frame(tr, [], i)
        kinds += [e.kind for e in ev]
    assert kinds == ["lost"] and not tr.tracks
