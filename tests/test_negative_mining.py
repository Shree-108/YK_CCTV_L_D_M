"""Negative mining: review export (backend/services/negative_mining.py), YOLO dataset builder and before/after metrics
(training/negative_mining/), and the engine's handling of explicit negative classes (fp_*).

The detector is a deterministic fake (a red square = "litter", a dark patch = "shadow"), so these tests check the
bookkeeping exactly; no model is run.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.detection.engine import drop_negative_classes
from backend.detection.types import Detection
from backend.services import negative_mining as nm
from training.negative_mining import build_dataset as bd
from training.negative_mining import report_metrics as rm

CLASSES = ["plastic_bag", "food_wrapper", "other_litter"]
W, H = 320, 180


def frame(litter: bool, shadow: bool, shade: int = 120) -> np.ndarray:
    img = np.full((H, W, 3), shade, np.uint8)
    if litter:
        img[120:140, 60:80] = (0, 0, 255)
    if shadow:
        img[100:150, 200:260] = 20
    return img


def fake_detect(img: np.ndarray) -> list[Detection]:
    out = []                                          # tolerant: the video codec shifts colours a little
    b, g, r = (int(v) for v in img[130, 70])
    if r > 180 and b < 80 and g < 80:
        out.append(Detection([60, 120, 80, 140], 1, "food_wrapper", 0.82))
    if int(img[125, 230].mean()) < 50:
        out.append(Detection([200, 100, 260, 150], 0, "plastic_bag", 0.71))
    return out


def video(make_video, name: str, pattern: list[tuple[bool, bool]], shade: int = 120) -> Path:
    """One pattern entry = one second (fps 5, repeat 5)."""
    return make_video([frame(a, b, shade) for a, b in pattern], repeat=5, fps=5.0, name=name)


def mine(make_video, root: Path, vid: str, pattern, roi=None, **kw) -> nm.MiningResult:
    p = video(make_video, f"{vid}.mp4", pattern, shade=100 + len(vid))
    return nm.mine_video(p, fake_detect, root, video_id=vid, sample_every_sec=1.0, roi=roi,
                         meta={"class_names": CLASSES, "thresholds": {"confirmed_threshold": 0.6}}, **kw)


def label(video_dir: Path, fn) -> None:
    """Fill the label/true_class columns with fn(row) -> (label, true_class)."""
    p = video_dir / "detections.csv"
    rows = nm.read_csv(p)
    for r in rows:
        r["label"], r["true_class"] = fn(r)
    nm.write_csv(p, nm.DET_COLUMNS, rows)


# ---------------------------------------------------------------------- mining
def test_mining_saves_every_detection_with_crop_frame_and_timestamp(tmp_path, make_video):
    pattern = [(True, False), (False, False), (True, True), (False, False), (False, True), (False, False)]
    res = mine(make_video, tmp_path / "review", "vidA", pattern, max_clean_frames=2)
    out = res.out_dir
    rows = nm.read_csv(out / "detections.csv")
    assert list(rows[0].keys()) == nm.DET_COLUMNS
    assert res.sampled_frames == 6 and res.detections == 4 and res.frames_with_detections == 3
    assert [(r["class"], float(r["timestamp_sec"])) for r in rows] == [
        ("food_wrapper", 0.0), ("food_wrapper", 2.0), ("plastic_bag", 2.0), ("plastic_bag", 4.0)]
    for r in rows:
        assert (out / r["crop_file"]).is_file() and (out / r["frame_file"]).is_file()
        assert r["label"] == "" and r["conf"] and r["in_roi"] == ""
    crop = cv2.imread(str(out / rows[0]["crop_file"]))
    assert crop.shape[0] >= 20 and crop.shape[1] >= 20                  # padded context around the 20x20 box
    frames = nm.read_csv(out / "frames.csv")
    assert len(frames) == 6 and sum(f["clean"] == "1" for f in frames) == 3
    assert 1 <= res.clean_frames_saved <= 2                              # spread out, never above max_clean_frames
    assert sum(f["saved"] == "1" and f["clean"] == "1" for f in frames) == res.clean_frames_saved
    m = json.loads((out / "manifest.json").read_text())
    assert m["duration_sec"] == pytest.approx(6.0) and m["labels"] == list(nm.LABELS)
    assert m["class_names"] == CLASSES


def test_mining_records_in_roi_from_the_ground_contact_point(tmp_path, make_video):
    roi = {"polygon": [[0, 0.75], [1, 0.75], [1, 1], [0, 1]], "holes": []}    # bottom quarter only
    res = mine(make_video, tmp_path, "vidR", [(True, True)], roi=roi)
    rows = {r["class"]: r["in_roi"] for r in nm.read_csv(res.out_dir / "detections.csv")}
    assert rows == {"food_wrapper": "1", "plastic_bag": "1"}                 # both boxes end below y=0.75*H
    roi = {"polygon": [[0, 0.8], [1, 0.8], [1, 1], [0, 1]], "holes": []}
    res = mine(make_video, tmp_path, "vidR2", [(True, True)], roi=roi)
    rows = {r["class"]: r["in_roi"] for r in nm.read_csv(res.out_dir / "detections.csv")}
    assert rows == {"food_wrapper": "0", "plastic_bag": "1"}                 # wrapper bottom y=140 < 0.8*180


def test_labelled_review_is_never_overwritten_by_accident(tmp_path, make_video):
    res = mine(make_video, tmp_path, "vidL", [(True, False)])
    label(res.out_dir, lambda r: ("TP", ""))
    with pytest.raises(FileExistsError):
        mine(make_video, tmp_path, "vidL", [(True, False)])
    mine(make_video, tmp_path, "vidL", [(True, False)], force=True)


def test_label_parsing():
    assert nm.normalise_label(" fp shadow ") == "FP_shadow"
    assert nm.normalise_label("tp") == "TP" and nm.normalise_label("FP-carton") == "FP_carton"
    assert nm.normalise_label("") is None
    with pytest.raises(ValueError):
        nm.normalise_label("maybe")


# ---------------------------------------------------------------------- dataset
def _three_labelled_videos(tmp_path, make_video) -> Path:
    root = tmp_path / "review"
    pat = [(True, True), (False, False), (True, False), (False, False), (False, True), (False, False)]
    for vid in ("vid1", "vid2", "vid3"):
        res = mine(make_video, root, vid, pat)
        label(res.out_dir, lambda r: ("TP", "other_litter" if r["timestamp_sec"] == "2.0" else "")
              if r["class"] == "food_wrapper" else ("FP_shadow" if r["timestamp_sec"] == "0.0" else "FP_other", ""))
    return root


def _labels(out: Path, split: str) -> dict[str, list[str]]:
    return {p.stem: p.read_text().split("\n")[:-1] if p.read_text() else []
            for p in sorted((out / "labels" / split).glob("*.txt"))}


def test_dataset_tp_labels_negative_classes_background_and_split_by_video(tmp_path, make_video):
    root = _three_labelled_videos(tmp_path, make_video)
    out = tmp_path / "ds"
    m = bd.build_dataset([root], out, val_videos=["vid2"], test_videos=["vid3"], background_ratio=1.0)
    assert m["classes"] == CLASSES + ["fp_shadow", "fp_carton", "fp_stock", "fp_puddle"]
    assert {s: m["splits"][s]["videos"] for s in ("train", "val", "test")} == {
        "train": ["vid1"], "val": ["vid2"], "test": ["vid3"]}
    data = (out / "data.yaml").read_text()
    assert "train: images/train" in data and "val: images/val" in data and "  3: fp_shadow" in data
    lab = _labels(out, "train")
    assert set(lab) == {"vid1__f0000000", "vid1__f0000010", "vid1__f0000020"} | {
        k for k in lab if k.startswith("vid1__") and not lab[k]}
    cls = lambda k: sorted(int(x.split()[0]) for x in lab[k])           # noqa: E731
    assert cls("vid1__f0000000") == [1, 3]           # TP food_wrapper + FP_shadow -> fp_shadow (id 3)
    assert cls("vid1__f0000010") == [2]              # TP with true_class other_litter
    assert lab["vid1__f0000020"] == []               # FP_other only -> image kept, no box
    bg = [k for k, v in lab.items() if not v and k != "vid1__f0000020"]
    assert len(bg) == 3                              # clean frames as background
    x, y, w, h = map(float, lab["vid1__f0000000"][0].split()[1:])
    assert (x, y, w, h) == pytest.approx((70 / W, 130 / H, 20 / W, 20 / H), abs=1e-5)
    for s in ("train", "val", "test"):               # no video in two splits
        assert {p.name.split("__")[0] for p in (out / "images" / s).glob("*.jpg")} == set(m["splits"][s]["videos"])
    assert m["label_counts"] == {"TP": 6, "FP_shadow": 3, "FP_other": 3}


def test_dataset_skips_frames_with_unlabelled_detections(tmp_path, make_video):
    root = _three_labelled_videos(tmp_path, make_video)
    p = root / "vid1" / "detections.csv"
    rows = nm.read_csv(p)
    rows[0]["label"] = ""                             # frame 0 now has one unlabelled detection
    nm.write_csv(p, nm.DET_COLUMNS, rows)
    m = bd.build_dataset([root], tmp_path / "ds", val_videos=["vid2"], test_videos=["vid3"])
    assert m["skipped"] == {"frame_with_unlabelled_detection": 1}
    assert not (tmp_path / "ds" / "labels" / "train" / "vid1__f0000000.txt").exists()


def test_dataset_refuses_a_single_video_and_bad_labels(tmp_path, make_video):
    root = tmp_path / "review"
    res = mine(make_video, root, "only", [(True, False)])
    label(res.out_dir, lambda r: ("TP", ""))
    with pytest.raises(bd.DatasetError, match="at least 2"):
        bd.build_dataset([root], tmp_path / "ds")
    res2 = mine(make_video, root, "other", [(True, False)])
    label(res2.out_dir, lambda r: ("TP", "unicorn"))
    with pytest.raises(bd.DatasetError, match="unicorn"):
        bd.build_dataset([root], tmp_path / "ds2", val_videos=["other"])
    label(res2.out_dir, lambda r: ("bogus", ""))
    with pytest.raises(bd.DatasetError, match="unknown label"):
        bd.build_dataset([root], tmp_path / "ds3", val_videos=["other"])


def test_split_assignment_is_stable_and_exclusive():
    a = bd.assign_splits(["v1", "v2", "v3", "v4", "v5"], val_ratio=0.2, test_ratio=0.2)
    assert a == bd.assign_splits(["v5", "v4", "v3", "v2", "v1"], val_ratio=0.2, test_ratio=0.2)
    assert sorted(a.values()).count("val") == 1 and sorted(a.values()).count("test") == 1
    with pytest.raises(bd.DatasetError):
        bd.assign_splits(["v1", "v2"], val_videos=["v1"], test_videos=["v1"])


def test_dataset_cli_reports_errors(tmp_path, capsys):
    assert bd.main([str(tmp_path), "--out", str(tmp_path / "ds")]) == 2
    assert "no review folders" in capsys.readouterr().err


# ---------------------------------------------------------------------- engine: explicit negative classes
def test_engine_drops_negative_classes_and_litter_inside_a_stronger_negative():
    xyxy = np.array([[0, 0, 100, 100], [10, 10, 50, 50], [200, 200, 220, 220], [300, 300, 320, 320]], float)
    conf = np.array([0.9, 0.6, 0.7, 0.95])
    cls = np.array([3, 1, 1, 3])                     # 3 = fp_shadow
    b, c, k = drop_negative_classes(xyxy, conf, cls, {3})
    assert k.tolist() == [1] and c.tolist() == [0.7]  # litter inside the stronger shadow box dropped; shadow dropped
    b, c, k = drop_negative_classes(xyxy[1:3], conf[1:3], cls[1:3], {3})
    assert k.tolist() == [1, 1]                       # no negative box: unchanged


# ---------------------------------------------------------------------- metrics
def _labelled_run(tmp_path, make_video, name: str, pattern, labeller, roi=None) -> Path:
    root = tmp_path / name
    res = mine(make_video, root, name, pattern, roi=roi)
    label(res.out_dir, labeller)
    return root


def test_metrics_precision_events_false_alerts_per_hour(tmp_path, make_video):
    # 6 s video: wrapper visible 0-2 s (one event, 3 detections), shadow visible 3-5 s (one FP event)
    pat = [(True, False)] * 3 + [(False, True)] * 3
    root = _labelled_run(tmp_path, make_video, "m1", pat,
                         lambda r: ("TP", "") if r["class"] == "food_wrapper" else ("FP_shadow", ""))
    run = rm.load_run([root], "before")
    assert run.alert_conf == 0.6 and run.hours == pytest.approx(6 / 3600)
    m = rm.metrics(run, gap_sec=3.0)
    assert (m["tp"], m["fp"], m["precision_det"]) == (3, 3, 0.5)
    assert (m["tp_events"], m["fp_events"], m["precision_event"]) == (1, 1, 0.5)
    assert m["false_alerts_per_hour"] == 600.0 and m["fp_by_type"] == {"FP_shadow": 3}
    m = rm.metrics(rm.load_run([root], "high", alert_conf=0.75), gap_sec=3.0)
    assert (m["tp"], m["fp"]) == (3, 0)                                   # shadows are 0.71 < 0.75


def test_metrics_after_roi_and_recall(tmp_path, make_video):
    roi = {"polygon": [[0, 0.8], [1, 0.8], [1, 1], [0, 1]], "holes": []}    # keeps the shadow, drops the wrapper
    pat = [(True, True)] * 2
    root = _labelled_run(tmp_path, make_video, "m2", pat,
                         lambda r: ("TP", "") if r["class"] == "food_wrapper" else ("FP_puddle", ""), roi=roi)
    before = rm.load_run([root], "before")
    rep = rm.compare(before, rm.restrict_to_roi(before), gap_sec=3.0)
    assert rep["before"]["recall"] == 1.0 and rep["after"]["recall"] == 0.0     # relative: the ROI lost the TP
    assert rep["after"]["tp"] == 0 and rep["after"]["fp"] == 2 and rep["complete"]
    assert "relative" in rep["before"]["recall_kind"]
    rep = rm.compare(before, before, gap_sec=3.0, gt={"m2": 2})
    assert rep["before"]["recall"] == 0.5 and "ground truth" in rep["before"]["recall_kind"]
    md = rm.to_markdown(rep)
    assert "| Precision (events) |" in md and "False alerts per hour" in md


def test_metrics_flag_unlabelled_detections(tmp_path, make_video):
    root = _labelled_run(tmp_path, make_video, "m3", [(True, True)],
                         lambda r: ("TP", "") if r["class"] == "food_wrapper" else ("", ""))
    rep = rm.compare(rm.load_run([root], "b"), rm.load_run([root], "a"), gap_sec=3.0)
    assert rep["before"]["unlabelled"] == 1 and not rep["complete"]
    assert "INCOMPLETE" in rm.to_markdown(rep)
    assert rm.main(["--before", str(root), "--after-roi"]) == 2            # mined without ROI -> clear error


def test_event_linking_respects_gap_and_iou():
    d = lambda t, box: rm.Det("v", t, "x", 0.9, box, "FP_other", None)    # noqa: E731
    ev = rm.events([d(0, (0, 0, 10, 10)), d(1, (1, 1, 11, 11)), d(9, (1, 1, 11, 11)), d(10, (50, 50, 60, 60))], 3.0)
    assert len(ev) == 3
