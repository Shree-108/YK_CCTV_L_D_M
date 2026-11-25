"""Phase 2 item 6: low-confidence mining helpers, keyboard review store, tiled YOLO export (synthetic data, no model)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.detection.tiling import tile_grid
from backend.services import roi_service as rs
from backend.services.labeling import LabelError, ReviewStore
from backend.services.negative_mining import read_csv, write_csv
from training.negative_mining.build_dataset import DatasetError
from training.negative_mining.export_tiles import export, holdout_videos, tiles_for
from training.negative_mining.mine_tiles import COLUMNS, assign_groups, crop_window, select_capped

CLASSES = ["plastic_bag", "plastic_packaging", "food_wrapper", "paper", "cardboard", "plastic_bottle", "cup",
           "newspaper", "plastic_piece", "other_litter"]
W, H = 2560, 1440
POLY = [{"points": [[0.0, 0.3], [0.7, 0.3], [0.7, 1.0], [0.0, 1.0]], "holes": []}]


# ------------------------------------------------------------------ mining helpers
def test_crop_window_is_native_scale_and_inside_the_frame():
    x, y, cw, ch = crop_window([10, 10, 30, 30], W, H, 256)
    assert (x, y, cw, ch) == (0, 0, 256, 256)                      # shifted inside, never resized
    x, y, cw, ch = crop_window([1000, 700, 1400, 900], W, H, 256)
    assert cw == ch == 640 and x <= 1000 and x + cw >= 1400         # 1.6 x the box when larger than 256


def test_groups_chain_the_same_box_through_frames():
    g1, n = assign_groups([], [[0, 0, 10, 10], [100, 100, 120, 120]], 1, "v")
    g2, n = assign_groups(list(zip(g1, [[0, 0, 10, 10], [100, 100, 120, 120]])), [[1, 0, 11, 10], [300, 0, 310, 9]], n, "v")
    assert g2[0] == g1[0] and g2[1] not in g1 and n == 4


def test_cap_keeps_one_crop_per_group_first():
    rows = [{"det_id": f"d{i}", "class": "paper", "conf": 0.1 + i / 1000, "group_id": f"g{i // 10}"} for i in range(100)]
    rows += [{"det_id": f"c{i}", "class": "cup", "conf": 0.5, "group_id": f"c{i}"} for i in range(5)]
    keep = select_capped(rows, 8)
    paper = [r for r in rows if r["det_id"] in keep and r["class"] == "paper"]
    assert len(paper) == 8 and len({r["group_id"] for r in paper}) == 8     # 10 groups -> 8 representatives
    assert {f"c{i}" for i in range(5)} <= keep                              # under the cap: all kept


# ------------------------------------------------------------------ review folders
def _review(root: Path, vid: str, video_file: str, labels: dict[str, str], n_frames: int = 2) -> Path:
    d = root / vid
    (d / "frames").mkdir(parents=True)
    (d / "crops").mkdir()
    rows = []
    frame = np.full((H, W, 3), 128, np.uint8)
    for f in range(n_frames):
        ff = f"frames/f{f:07d}.jpg"
        cv2.imwrite(str(d / ff), frame)
        for k, (x1, y1) in enumerate([(200, 800), (1500, 1200)]):
            det_id = f"{vid}_f{f:07d}_d{k:02d}"
            cv2.imwrite(str(d / "crops" / f"{det_id}.jpg"), frame[:256, :256])
            rows.append({"det_id": det_id, "video_id": vid, "frame_idx": f, "timestamp_sec": f, "frame_file": ff,
                         "crop_file": f"crops/{det_id}.jpg", "class": "food_wrapper", "conf": 0.2 + k / 10,
                         "x1": x1, "y1": y1, "x2": x1 + 40, "y2": y1 + 30, "frame_w": W, "frame_h": H,
                         "status": "weak", "filter_reason": "", "in_roi": 1, "label": labels.get(f"d{k}", ""),
                         "true_class": "", "notes": "", "below_030": 1, "group_id": f"{vid}_g{k}", "crop_x": 0,
                         "crop_y": 0, "crop_w": 256, "crop_h": 256, "capped": 0})
    write_csv(d / "detections.csv", COLUMNS, rows)
    (d / "manifest.json").write_text(json.dumps({
        "video_id": vid, "video_file": video_file, "class_names": CLASSES, "frame_size": [W, H],
        "tiling": {"tile_size": 1024, "overlap": 0.2}, "roi": {"applied": True, "polygons": POLY}}), encoding="utf-8")
    return d


def test_review_store_labels_a_whole_group_with_undo_and_missed_boxes(tmp_path):
    _review(tmp_path, "vidA", "a.mp4", {})
    st = ReviewStore(tmp_path)
    q = st.items("vidA")
    assert len(q["items"]) == 2 and q["items"][0]["members"] == 2          # 2 groups x 2 frames
    g = q["items"][0]["group_id"]
    assert len(st.set_label("vidA", g, "FP_shadow")["changed"]) == 2
    assert {r["label"] for r in read_csv(tmp_path / "vidA" / "detections.csv") if r["group_id"] == g} == {"FP_shadow"}
    st.set_label("vidA", g, "")                                             # undo
    assert st.items("vidA")["items"][0]["label"] == ""
    with pytest.raises(LabelError):
        st.set_label("vidA", g, "maybe")
    m = st.add_missed("vidA", "frames/f0000000.jpg", [10, 10, 60, 50], "plastic_bag")
    assert st.items("vidA")["missed"][0]["missed_id"] == m["missed_id"]
    assert st.delete_missed("vidA", m["missed_id"]) and not st.items("vidA")["missed"]
    assert st.image_path("vidA", "../../etc/passwd") is None
    with pytest.raises(LabelError):
        st.add_missed("vidA", "frames/f0000000.jpg", [10, 10, 60, 50], "not_a_class")


# ------------------------------------------------------------------ tiled export
def test_export_uses_the_inference_tiles_stable_ids_and_a_video_split(tmp_path):
    root = tmp_path / "review"
    _review(root, "street", "20261119_street.mp4", {"d0": "TP", "d1": "FP_shadow"})
    _review(root, "street2", "20261121_street.mp4", {"d0": "FP_carton", "d1": "TP"})
    _review(root, "shop", "20261110_shop.mp4", {"d0": "TP", "d1": "FP_other"})
    st = ReviewStore(root)
    st.add_missed("street", "frames/f0000000.jpg", [300, 900, 360, 950], "plastic_bag")
    m = export([root], tmp_path / "ds", val_videos=["street2"])
    assert m["holdout_test"] == ["shop"]                                     # the 2026-11-10 video
    assert m["splits"]["test"]["videos"] == ["shop"] and m["splits"]["val"]["videos"] == ["street2"]
    assert m["classes"][:10] == CLASSES                                      # litter ids unchanged
    assert m["classes"][10:] == ["fp_shadow", "fp_carton", "fp_stock", "fp_puddle"]
    yaml = (tmp_path / "ds" / "data.yaml").read_text(encoding="utf-8")
    assert "  2: food_wrapper" in yaml and "  10: fp_shadow" in yaml and "test: images/test" in yaml
    # geometry = roi_crop + tile_grid of the inference path
    frame = cv2.imread(str(root / "street" / "frames" / "f0000000.jpg"))
    crop, x0, y0, tiles = tiles_for(frame, POLY, 1024, 0.2)
    mask = rs.polygons_mask(POLY, frame.shape)
    c2, a, b = rs.roi_crop(frame, mask)
    assert (x0, y0) == (a, b) and crop.shape == c2.shape and tiles == tile_grid(c2.shape[1], c2.shape[0], 1024, 0.2)
    imgs = sorted((tmp_path / "ds" / "images" / "train").glob("*.jpg"))
    assert imgs and all(cv2.imread(str(p)).shape[0] <= 1024 for p in imgs)
    ids = {int(l.split()[0]) for p in (tmp_path / "ds" / "labels" / "train").glob("*.txt")
           for l in p.read_text().splitlines()}
    assert ids <= {2, 0, 10}                    # food_wrapper TP, the drawn plastic_bag, fp_shadow
    assert 0 in ids and 10 in ids
    assert all(p.name.startswith("street__") for p in imgs)


def test_export_refuses_a_split_without_validation_video(tmp_path):
    root = tmp_path / "review"
    _review(root, "street", "20261119_street.mp4", {"d0": "TP", "d1": "TP"})
    _review(root, "shop", "20261110_shop.mp4", {"d0": "TP", "d1": "TP"})
    with pytest.raises(DatasetError):
        export([root], tmp_path / "ds")                      # shop held out -> no video left for val


def test_unlabelled_detection_makes_its_tile_unusable(tmp_path):
    root = tmp_path / "review"
    _review(root, "a", "a.mp4", {"d0": "TP"})                # d1 unlabelled
    _review(root, "b", "b.mp4", {"d0": "TP", "d1": "TP"})
    m = export([root], tmp_path / "ds", val_videos=["b"], test_videos=[])
    assert m["skipped"].get("tile_with_unlabelled_detection", 0) >= 1


def test_holdout_picks_the_first_2026_11_10_video():
    assert holdout_videos({"x": {"video_file": "20261110_b.mp4"}, "y": {"video_file": "20261110_a.mp4"},
                           "z": {"video_file": "20261119.mp4"}}) == ["x"]
