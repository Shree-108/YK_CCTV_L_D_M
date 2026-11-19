"""Fast, deterministic tests for training/ldm7 on tiny synthetic data (no model, no download, no hash() seeding)."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from training.ldm7 import schema
from training.ldm7.compare_models import compare, evaluate_model, sufficiency
from training.ldm7.evalcore import average_precision, greedy_match, load_gt, predict_setting
from training.ldm7.small_object_experiment import make_settings, run_experiment, write_outputs
from training.ldm7.split_by_group import SplitError, assign_splits, build
from training.ldm7.train_yolo11s import TrainError, build_package, check_base, check_staging
from training.ldm7.verify_dataset import verify, write_reports

W, H = 96, 64


def _img(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (H, W, 3), dtype=np.uint8)


def _write(p: Path, img: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)


def make_dataset(root: Path, per_split: dict[str, int] | None = None, manifest: bool = True) -> Path:
    per_split = per_split or {"train": 4, "val": 2, "test": 2}
    rows, seed = [], 1
    for s, n in per_split.items():
        for k in range(n):
            name = f"{s}_{k}.png"
            _write(root / "images" / s / name, _img(seed))
            lab = root / "labels" / s / f"{s}_{k}.txt"
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_text("" if k == n - 1 else f"{k % 7} 0.5 0.5 0.25 0.25\n", encoding="utf-8")
            rows.append({"image": f"images/{s}/{name}", "camera_id": f"CAM_{s}", "session_id": f"S_{s}",
                         "date": "2026-11-19"})
            seed += 1
    schema.write_data_yaml(root)
    if manifest:
        with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image", "camera_id", "session_id", "date"])
            w.writeheader()
            w.writerows(rows)
    return root


def _types(rep: dict, sev: str = "ERROR") -> set[str]:
    return {i["type"] for i in rep["issues"] if i["severity"] == sev}


# ---------------------------------------------------------------- schema
def test_schema_fixed_ids_and_names():
    assert schema.CLASSES == ["plastic_packet", "plastic_bottle", "paper", "food_waste", "other_litter", "glass",
                              "cardboard"]
    assert all(schema.NAME_TO_ID[n] == i and schema.ID_TO_NAME[i] == n for i, n in enumerate(schema.CLASSES))


def test_v3_mapping_invariants():
    assert set(schema.V3_TO_LDM7) == set(schema.V3_CLASSES)
    assert all(t is None or t in schema.CLASSES for t in schema.V3_TO_LDM7.values())
    assert schema.V3_TO_LDM7["cup"] is None and schema.V3_TO_LDM7["newspaper"] == "paper"
    for k, (t, q, why) in schema.V3_MAPPING_TABLE.items():
        assert (t is None) == (q == "unmapped") and why
    assert set(schema.LDM7_WITHOUT_V3_SOURCE) == {"food_waste", "glass"}
    kind, m = schema.class_mapper_for(schema.V3_CLASSES)
    assert kind == "v3" and m[schema.V3_CLASSES.index("paper")] == schema.NAME_TO_ID["paper"]
    assert schema.class_mapper_for(schema.CLASSES) == ("ldm7", list(range(7)))
    with pytest.raises(ValueError):
        schema.class_mapper_for(["litter"])


def test_v3_classes_match_installed_model_if_present():
    p = Path(__file__).resolve().parents[1] / "models" / "litter_yolo" / "v3_store_cctv" / "class_names.json"
    if not p.is_file():
        pytest.skip("v3 class_names.json not present")
    assert json.loads(p.read_text()) == schema.V3_CLASSES


def test_data_yaml_roundtrip(tmp_path):
    y = schema.write_data_yaml(tmp_path)
    assert schema.names_from_yaml(y) == schema.CLASSES


# ---------------------------------------------------------------- verify_dataset
def test_verify_clean_dataset_passes(tmp_path):
    rep = verify(make_dataset(tmp_path / "ds"), min_instances=1)
    assert rep["ok"], [i for i in rep["issues"] if i["severity"] == "ERROR"]
    assert rep["splits"]["train"]["background_images"] == 1
    jp, mp = write_reports(rep, tmp_path / "rep")
    assert json.loads(jp.read_text())["ok"] and "PASS" in mp.read_text()


def test_verify_bad_class_id(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    (ds / "labels" / "train" / "train_0.txt").write_text("7 0.5 0.5 0.1 0.1\n")
    rep = verify(ds)
    assert not rep["ok"] and "class_id" in _types(rep)


def test_verify_bad_coordinates_and_syntax(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    (ds / "labels" / "train" / "train_0.txt").write_text("0 1.5 0.5 0.1 0.1\n0 0.5 0.5 0 0.1\n0 0.5 0.5\n")
    t = _types(verify(ds))
    assert {"not_normalized", "box_size", "label_syntax"} <= t


def test_verify_missing_label_file_is_error(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    (ds / "labels" / "val" / "val_1.txt").unlink()          # was an empty (background) label
    rep = verify(ds)
    assert not rep["ok"] and "missing_label" in _types(rep)


def test_verify_cross_split_duplicate(tmp_path):
    ds = make_dataset(tmp_path / "ds", manifest=False)
    shutil.copy2(ds / "images" / "train" / "train_0.png", ds / "images" / "test" / "dup.png")
    (ds / "labels" / "test" / "dup.txt").write_text("")
    rep = verify(ds)
    assert "cross_split_duplicate" in _types(rep)
    assert "no_manifest" in _types(rep, "WARNING")


def test_verify_cross_split_near_duplicate(tmp_path):
    ds = make_dataset(tmp_path / "ds", manifest=False)
    img = cv2.imread(str(ds / "images" / "train" / "train_0.png")).astype(np.int16)
    img[0, 0] = (img[0, 0] + 1) % 256                         # 1-pixel change: different sha, same dHash
    _write(ds / "images" / "val" / "near.png", img.astype(np.uint8))
    (ds / "labels" / "val" / "near.txt").write_text("")
    t = _types(verify(ds))
    assert "cross_split_near_duplicate" in t and "cross_split_duplicate" not in t


def test_verify_group_leakage(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    rows = list(csv.DictReader(open(ds / "manifest.csv", encoding="utf-8")))
    rows[-1]["session_id"] = "S_train"                        # a test image from the train session
    with open(ds / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rep = verify(ds)
    assert "group_leakage" in _types(rep)


def test_verify_wrong_data_yaml_names(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    (ds / "data.yaml").write_text("names: {0: plastic_bag, 1: paper}\n")
    assert "class_names" in _types(verify(ds))


def test_verify_warns_few_instances_and_missing_test(tmp_path):
    ds = make_dataset(tmp_path / "ds", per_split={"train": 3, "val": 2})
    rep = verify(ds, min_instances=50)
    w = _types(rep, "WARNING")
    assert "few_instances" in w and "no_test_split" in w
    assert sum(rep["splits"]["train"]["size_buckets"]["plastic_packet"].values()) == 1


# ---------------------------------------------------------------- split_by_group
def make_pool(root: Path) -> Path:
    rows, seed = [], 100
    for cam in ("CAM_A", "CAM_B", "CAM_C"):
        for day in ("2026-11-01", "2026-11-02"):
            for k in range(3):
                name = f"{cam}_{day}_{k}.png"
                _write(root / "images" / name, _img(seed))
                (root / "labels").mkdir(parents=True, exist_ok=True)
                (root / "labels" / f"{Path(name).stem}.txt").write_text("" if k == 2 else "0 0.5 0.5 0.2 0.2\n")
                rows.append({"image": name, "camera_id": cam, "session_id": f"{cam}_{day}", "date": day})
                seed += 1
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "camera_id", "session_id", "date"])
        w.writeheader()
        w.writerows(rows)
    return root


def test_split_never_splits_group_and_is_deterministic(tmp_path):
    pool = make_pool(tmp_path / "pool")
    r1 = build(pool, tmp_path / "out1", test_groups=["camera_id=CAM_C"], val_frac=0.25, seed=3)
    r2 = build(pool, tmp_path / "out2", test_groups=["camera_id=CAM_C"], val_frac=0.25, seed=3)
    m1 = list(csv.DictReader(open(tmp_path / "out1" / "manifest.csv", encoding="utf-8")))
    m2 = list(csv.DictReader(open(tmp_path / "out2" / "manifest.csv", encoding="utf-8")))
    assert [(r["image"], r["split"]) for r in m1] == [(r["image"], r["split"]) for r in m2]
    sess = {}
    for r in m1:
        sess.setdefault(r["session_id"], set()).add(r["split"])
        assert (r["camera_id"] == "CAM_C") == (r["split"] == "test")      # designated test kept, nothing else
    assert all(len(v) == 1 for v in sess.values())
    assert r1["verify"]["ok"] and r1["independent_test"] and r1["splits"]["val"]["images"] > 0
    with pytest.raises(SplitError):
        build(pool, tmp_path / "out1")                                  # immutable output


def test_split_rejects_missing_label_and_unknown_test_group(tmp_path):
    pool = make_pool(tmp_path / "pool")
    with pytest.raises(SplitError):
        build(pool, tmp_path / "x", test_groups=["camera_id=CAM_Z"], run_verify=False)
    next((pool / "labels").glob("*.txt")).unlink()
    with pytest.raises(SplitError, match="label file missing"):
        build(pool, tmp_path / "y", run_verify=False)


def test_assign_splits_seed_changes_but_groups_hold():
    items = [{"name": f"i{k}", "groups": [f"session_id=S{k // 2}"], "row": {}, "dhash": f"{k:064x}"}
             for k in range(20)]
    for seed in range(5):
        sp, _ = assign_splits(items, [], val_frac=0.2, test_frac=0.2, seed=seed, near_bits=-1)
        for k in range(0, 20, 2):
            assert sp[k] == sp[k + 1]
        assert assign_splits(items, [], 0.2, 0.2, seed, -1)[0] == sp


# ---------------------------------------------------------------- evaluation core
class FakePredictor:
    """Fixed pixel boxes on the all-zero 'litter' frame, `bg_boxes` on any other full frame, nothing on tiles."""

    def __init__(self, names, boxes, bg_boxes=()):
        self.names, self.calls = names, []
        self.boxes, self.bg = np.array(boxes, float).reshape(-1, 6), np.array(bg_boxes, float).reshape(-1, 6)

    def __call__(self, image, imgsz, conf):
        self.calls.append((image.shape[:2], imgsz))
        if image.shape[:2] != (H, W):                 # tiles: no detections, keeps the test exact
            return np.zeros((0, 6))
        b = self.boxes if image.max() == 0 else self.bg
        return b[b[:, 4] >= conf]


def _gt(rows):
    return np.array(rows, float).reshape(-1, 5)


def test_greedy_match_and_ap():
    gts = _gt([[0, 10, 10, 30, 30], [1, 50, 10, 70, 30]])
    preds = np.array([[10, 10, 30, 30, 0.9, 0], [50, 10, 70, 30, 0.8, 0], [0, 40, 5, 45, 0.7, 1]])
    tp, which = greedy_match(preds, gts, 0.5, class_aware=True)
    assert tp.tolist() == [True, False, False] and which[0] == 0
    tp_a, _ = greedy_match(preds, gts, 0.5, class_aware=False)
    assert tp_a.tolist() == [True, True, False]
    assert average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)
    assert average_precision([0.9, 0.8], [False, True], 2) == pytest.approx(0.25)
    assert average_precision([], [], 0) is None


def test_load_gt_missing_label_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gt(tmp_path / "none.txt", W, H)


def test_small_object_experiment_fake_predictor(tmp_path):
    gts = _gt([[0, 10, 10, 26, 26], [2, 40, 5, 90, 60]])    # small (16x16) plastic_packet, large paper
    # v3 model: plastic_packaging (1 -> plastic_packet) hits the small box at 0.15;
    # newspaper (7 -> paper) hits the large box at 0.6; cup (6, unmapped) floats elsewhere at 0.5 -> FP
    pred = FakePredictor(schema.V3_CLASSES, [[10, 10, 26, 26, 0.15, 1], [40, 5, 90, 60, 0.6, 7], [0, 40, 8, 48, 0.5, 6]],
                         bg_boxes=[[5, 5, 20, 20, 0.4, 0], [30, 30, 40, 40, 0.2, 0]])   # 1 FP >= 0.3, 2 FP >= 0.1
    images = [("a.png", np.zeros((H, W, 3), np.uint8), gts), ("bg.png", np.ones((H, W, 3), np.uint8), _gt([]))]
    settings = make_settings([640], ["off", "on"], tile_size=48, overlap=0.25)
    res = run_experiment(pred, images, settings, thresholds=[0.1, 0.3], conf_floor=0.01, match_iou=0.3)
    s = res["summary"]
    assert s["model_schema"] == "v3" and s["n_gt"] == 2 and s["n_background_images"] == 1
    off = s["settings"]["imgsz640"]["by_threshold"]
    assert off["0.1"]["recall_class"] == 1.0 and off["0.3"]["recall_class"] == 0.5
    assert off["0.3"]["by_bucket"]["small"]["recall_any"] == 0.0 and off["0.1"]["by_bucket"]["small"]["n_gt"] == 1
    assert off["0.3"]["by_class"]["paper"]["recall_class"] == 1.0
    assert off["0.3"]["fp_per_image"] == pytest.approx(1.0)          # (cup + 1 bg FP) / 2 images
    assert off["0.3"]["fp_per_background_image"] == pytest.approx(1.0)
    assert off["0.1"]["fp_per_image"] == pytest.approx(1.5) and off["0.1"]["fp_per_background_image"] == 2.0
    assert any(c[0] != (H, W) for c in pred.calls)                    # tiling ran on crops
    small = [r for r in res["per_gt"] if r["bucket"] == "small" and r["setting"] == "imgsz640"][0]
    assert small["best_conf_same_class"] == 0.15 and small["matched"]
    write_outputs(res, tmp_path / "exp")
    assert (tmp_path / "exp" / "per_gt.csv").is_file() and (tmp_path / "exp" / "summary.csv").is_file()


def test_tiling_offsets_boxes_back():
    class TileOnly:
        names = schema.CLASSES

        def __call__(self, image, imgsz, conf):
            return np.array([[1, 1, 5, 5, 0.9, 0]]) if image.shape[:2] != (H, W) else np.zeros((0, 6))

    out = predict_setting(TileOnly(), np.zeros((H, W, 3), np.uint8),
                          {"imgsz": 640, "tiling": True, "tile_size": 48, "overlap": 0.0}, 0.01)
    assert {(int(b[0]), int(b[1])) for b in out} >= {(1, 1), (49, 1)}


def test_compare_models_fake_and_insufficient_report(tmp_path):
    gts = _gt([[0, 10, 10, 26, 26], [5, 40, 5, 90, 60]])
    imgs = [("a.png", np.zeros((H, W, 3), np.uint8), gts), ("bg.png", np.ones((H, W, 3), np.uint8), _gt([]))]
    v3 = FakePredictor(schema.V3_CLASSES, [[10, 10, 26, 26, 0.5, 2], [40, 5, 90, 60, 0.7, 6]],   # food_wrapper, cup
                       bg_boxes=[[5, 5, 20, 20, 0.6, 0]])                                         # plastic_bag on clean
    cand = FakePredictor(schema.CLASSES, [[10, 10, 26, 26, 0.8, 0], [40, 5, 90, 60, 0.9, 5]])
    r3 = evaluate_model(v3, iter(imgs), 640)
    assert r3["per_class"]["plastic_packet"]["recall"] == 1.0
    assert r3["per_class"]["glass"]["recall"] == 0.0 and r3["per_class"]["glass"]["no_source_class"]
    assert r3["unmapped_v3_fp"] == 2 and r3["unmapped_v3_fp_by_class"] == {"cup": 1, "plastic_bag": 1}
    assert r3["agnostic"]["recall"] == 1.0 and r3["fp_per_clean_image"] == 1.0
    rc = evaluate_model(cand, iter(imgs), 1280)
    assert rc["per_class"]["glass"]["ap50"] == pytest.approx(1.0) and rc["fp_per_clean_image"] == 0.0
    doc = compare({"v3_store_cctv": (v3, 640), "candidate": (cand, 1280)}, lambda: iter(imgs), "synthetic",
                  tmp_path / "cmp", independence=None)
    md = (tmp_path / "cmp" / "comparison_report.md").read_text()
    assert not doc["sufficient"] and "INSUFFICIENT EVIDENCE" in md and "NOT verified" in md
    empty = compare({}, lambda: iter(()), "none", tmp_path / "cmp2")
    assert "NO TEST SET" in (tmp_path / "cmp2" / "comparison_report.md").read_text() and not empty["sufficient"]
    assert sufficiency({}, "verified")[0].startswith("NO TEST SET")


# ---------------------------------------------------------------- packaging contract
def test_package_matches_install_contract(tmp_path):
    from training.install_trained_model import verify_package
    ds = make_dataset(tmp_path / "ds")
    rep = verify(ds, min_instances=1)
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"not-a-real-checkpoint")
    (run / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.1\n")
    pkg = build_package(run, tmp_path / "staging", ds, rep, {"epochs": 1}, "yolo11s.pt", 1280, smoke=True,
                        check_weights=False, make_zip=True)
    assert pkg.name == schema.MODEL_VERSION
    meta = verify_package(pkg, allow_smoke=True, load_model=False)
    assert meta["classes"] == schema.CLASSES and meta["dataset_hash"] == rep["dataset_hash"] and meta["sha256"]
    for f in ("best.pt", "class_names.json", "data.yaml", "train_args.json", "results.csv", "evaluation.json",
              "model_meta.json", "SHA256SUMS", "dataset_info.json"):
        assert (pkg / f).is_file(), f
    assert (tmp_path / "staging" / f"ldm_model_{schema.MODEL_VERSION}.zip").is_file()
    with pytest.raises(Exception):
        verify_package(pkg, allow_smoke=False, load_model=False)       # smoke packages refused by default
    with pytest.raises(TrainError):
        build_package(run, tmp_path / "staging", ds, rep, {}, "x", 640, check_weights=False)   # immutable


def test_train_guards():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(TrainError, match="never from v3"):
        check_base(root / "models" / "litter_yolo" / "v3_store_cctv" / "best.pt")
    with pytest.raises(TrainError, match="inside models"):
        check_staging(root / "models" / "litter_yolo", "v4_ldm7_yolo11s")
