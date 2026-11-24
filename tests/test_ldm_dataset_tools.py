"""Dataset preparation tools (training/*ldm*, extract_frames, import_images): no model, no real data needed."""
from __future__ import annotations

import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from training import build_ldm_dataset, extract_frames, import_images, ldm_review, validate_ldm_dataset
from training.ldm_common import (label_path, model_classes, overlay_suspect, read_manifest, write_manifest)


def scene(seed: int, w: int = 320, h: int = 240) -> np.ndarray:
    """A textured, distinct 'scene' per seed (distinct dHash), deterministic."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (12, 16, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), (5, 5), 0)
    grey = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 0.35, grey, 0.65, 0)      # CCTV-like muted colours


def clean_scene(key: str) -> np.ndarray:
    """A scene for a fixture image: the same for the same key in every run (hash() of a str changes per process
    with PYTHONHASHSEED, which made these tests flaky), and never one the overlay heuristic flags by chance (the
    tests set overlay_suspect themselves where they need it)."""
    seed = zlib.crc32(key.encode()) % 10**6
    while overlay_suspect(img := scene(seed)):
        seed += 1
    return img


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    return tmp_path / "ws"


# ---------------------------------------------------------------- overlay heuristic
def test_overlay_detects_drawn_box_but_not_a_coloured_edge():
    clean = scene(1)
    boxed = clean.copy()
    cv2.rectangle(boxed, (60, 40), (200, 200), (0, 255, 0), 1)
    edge = clean.copy()
    cv2.line(edge, (150, 0), (150, 239), (255, 120, 0), 1)       # one straight coloured line, no corner
    assert overlay_suspect(boxed)
    assert not overlay_suspect(clean)
    assert not overlay_suspect(edge)


# ---------------------------------------------------------------- frame extraction
def _video(path: Path, fps: int = 10, seconds: int = 9) -> Path:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    assert vw.isOpened()
    for i in range(fps * seconds):
        vw.write(scene(100 + i // (fps * 3)))      # the scene changes every 3 s, otherwise static
    vw.release()
    return path


def test_extract_frames_samples_dedups_and_records_metadata(tmp_path: Path, ws: Path):
    v = _video(tmp_path / "cam.avi")
    assert extract_frames.main([str(v), "--workspace", str(ws), "--every", "1", "--session", "cam_day1",
                                "--recorded-at", "2026-10-01T10:00:00+05:30", "--conditions", "day"]) == 0
    rows = read_manifest(ws)
    assert len(rows) == 3                                   # 9 samples, 3 distinct scenes
    assert [float(r["timestamp_sec"]) for r in rows] == [0.0, 3.0, 6.0]
    assert rows[1]["recorded_at"] == "2026-10-01T10:00:03+05:30"
    assert all(r["session"] == "cam_day1" and r["review_status"] == "unreviewed" for r in rows)
    assert all((ws / "images" / "train" / r["image"]).is_file() for r in rows)
    assert not any(label_path(ws, r["image"]).exists() for r in rows)   # extraction never writes labels
    assert v.is_file()                                                  # source untouched
    # re-running adds nothing
    assert extract_frames.main([str(v), "--workspace", str(ws), "--every", "1", "--session", "cam_day1"]) == 0
    assert len(read_manifest(ws)) == 3
    # the workspace cannot be trained on by accident (no val split) and uses the model's class list
    data = yaml.safe_load((ws / "data.yaml").read_text(encoding="utf-8"))
    assert "val" not in data and list(data["names"].values()) == model_classes()


def test_extract_refuses_app_annotated_video(tmp_path: Path, ws: Path):
    d = tmp_path / "storage" / "videos" / "job1"
    d.mkdir(parents=True)
    v = _video(d / "annotated.avi")
    assert extract_frames.main([str(v), "--workspace", str(ws)]) == 1
    assert read_manifest(ws) == []


# ---------------------------------------------------------------- image import
def test_import_skips_duplicates_and_corrupt_files(tmp_path: Path, ws: Path):
    src = tmp_path / "shots"
    src.mkdir()
    cv2.imwrite(str(src / "a.jpg"), scene(1))
    cv2.imwrite(str(src / "b.jpg"), scene(2))
    (src / "copy_of_a.jpg").write_bytes((src / "a.jpg").read_bytes())
    (src / "broken.jpg").write_bytes(b"not an image")
    assert import_images.main([str(src), "--workspace", str(ws), "--source-type", "real_cctv"]) == 0
    rows = read_manifest(ws)
    assert sorted(r["image"] for r in rows) == ["shots__a.jpg", "shots__b.jpg"]
    assert all(r["source_type"] == "real_cctv" for r in rows)
    # importing again is a no-op (exact duplicates of workspace images)
    assert import_images.main([str(src), "--workspace", str(ws), "--source-type", "real_cctv"]) == 0
    assert len(read_manifest(ws)) == 2


def test_import_reads_scheduled_capture_layout(tmp_path: Path, ws: Path):
    cap = tmp_path / "captures" / "cam01" / "20261005_142233_ab12cd"
    cap.mkdir(parents=True)
    cv2.imwrite(str(cap / "original.jpg"), scene(5))
    cv2.imwrite(str(cap / "annotated.jpg"), scene(6))
    assert import_images.main([str(tmp_path / "captures"), "--workspace", str(ws), "--source-type", "real_cctv"]) == 0
    (r,) = read_manifest(ws)
    assert (r["camera"], r["session"], r["recorded_at"]) == ("cam01", "cam01_20261005", "2026-10-05T14:22:33")


# ---------------------------------------------------------------- review bookkeeping
def _populate(ws: Path, tmp_path: Path, sessions: dict[str, int], source_type: str = "real_cctv") -> None:
    src = tmp_path / f"src_{source_type}"
    for s, n in sessions.items():
        d = src / s
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cv2.imwrite(str(d / f"{i:03d}.jpg"), clean_scene(f"{s}:{i}"))
    assert import_images.main([str(src), "--workspace", str(ws), "--source-type", source_type]) == 0


def _annotate(ws: Path, image: str, boxes: list[str]) -> None:
    """What the Ground Truth page does when a person saves (an empty list = saved with no boxes)."""
    label_path(ws, image).write_text("".join(b + "\n" for b in boxes), encoding="utf-8")


def test_sync_and_mark_empty_never_label_unreviewed_images(tmp_path: Path, ws: Path):
    _populate(ws, tmp_path, {"s1": 4})
    names = sorted(r["image"] for r in read_manifest(ws))
    _annotate(ws, names[0], ["2 0.5 0.5 0.1 0.1"])
    _annotate(ws, names[1], [])
    assert ldm_review.main(["sync", "--workspace", str(ws)]) == 0
    st = {r["image"]: (r["review_status"], r["category"]) for r in read_manifest(ws)}
    assert st[names[0]] == ("annotated", "litter")
    assert st[names[1]] == ("reviewed_empty", "unknown")
    assert st[names[2]][0] == st[names[3]][0] == "unreviewed"
    assert ldm_review.main(["mark-empty", "--workspace", str(ws)]) == 0
    assert not label_path(ws, names[2]).exists()                       # unreviewed: still no label
    assert ldm_review.main(["mark-empty", "--workspace", str(ws), "--images", names[2], "--category", "clean"]) == 0
    assert label_path(ws, names[2]).read_text() == ""
    rep = ldm_review.report(ws, model_classes())
    assert rep["usable_images"] == 3 and rep["needs_annotation"] == 1
    assert (ws / "reports" / "needs_annotation.txt").read_text().split() == [names[3]]


# ---------------------------------------------------------------- build + validate
def _review_all(ws: Path) -> None:
    rows = read_manifest(ws)
    for i, r in enumerate(rows):
        if i % 2:
            _annotate(ws, r["image"], [f"{i % 10} 0.5 0.5 0.2 0.2"])
            r["review_status"], r["category"] = "annotated", "litter"
        else:
            _annotate(ws, r["image"], [])
            r["review_status"], r["category"] = "reviewed_empty", "hard_negative"
    write_manifest(ws, rows)


def test_build_keeps_sessions_together_and_synthetic_in_train(tmp_path: Path, ws: Path, monkeypatch):
    monkeypatch.setattr(build_ldm_dataset, "PROJECT_ROOT", tmp_path)
    _populate(ws, tmp_path, {f"day{i}": 6 for i in range(8)})
    _populate(ws, tmp_path, {"aug": 6}, source_type="synthetic")
    _review_all(ws)
    assert build_ldm_dataset.main(["--name", "ds_v1", "--workspace", str(ws)]) == 0
    out = tmp_path / "dataset" / "ds_v1"
    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert "path" not in data and list(data["names"].values()) == model_classes()
    import csv
    rows = list(csv.DictReader(open(out / "split_manifest.csv", encoding="utf-8-sig")))
    by_session: dict[str, set] = {}
    for r in rows:
        by_session.setdefault(r["session"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_session.values())
    assert by_session["aug"] == {"train"}
    assert {s for v in by_session.values() for s in v} == {"train", "val", "test"}
    # versioned: never overwritten
    assert build_ldm_dataset.main(["--name", "ds_v1", "--workspace", str(ws)]) == 2
    # the validator accepts it ...
    assert validate_ldm_dataset.main([str(out)]) == 0
    # ... and catches leakage: a train image copied into test
    tr = next((out / "images" / "train").glob("*.jpg"))
    (out / "images" / "test" / ("leak_" + tr.name)).write_bytes(tr.read_bytes())
    (out / "labels" / "test" / ("leak_" + tr.stem + ".txt")).write_text("")
    assert validate_ldm_dataset.main([str(out)]) == 1
    rep = (out / "validation_report.md").read_text(encoding="utf-8")
    assert "cross_split_near_duplicate" in rep


def test_build_excludes_unreviewed_and_overlay_frames(tmp_path: Path, ws: Path, monkeypatch):
    monkeypatch.setattr(build_ldm_dataset, "PROJECT_ROOT", tmp_path)
    _populate(ws, tmp_path, {"a": 3, "b": 3, "c": 3})
    rows = read_manifest(ws)
    _annotate(ws, rows[0]["image"], ["0 0.5 0.5 0.2 0.2"])
    rows[0]["review_status"] = "annotated"
    _annotate(ws, rows[1]["image"], ["0 0.5 0.5 0.2 0.2"])
    rows[1]["review_status"], rows[1]["overlay_suspect"] = "annotated", "yes"
    write_manifest(ws, rows)
    assert build_ldm_dataset.main(["--name", "ds_x", "--workspace", str(ws), "--dry-run"]) == 0
    items, excluded = build_ldm_dataset.collect_workspace(ws, model_classes(), False)
    assert [i["name"] for i in items] == [rows[0]["image"]]
    reasons = {e["image"]: e["reason"] for e in excluded}
    assert "overlay_suspect" in reasons[rows[1]["image"]]
    assert reasons[rows[2]["image"]] == "review_status unreviewed"


# ---------------------------------------------------------------- annotation tool round trip
def _cvat_export(tmp_path: Path, names: list[str], labels: dict[str, str]) -> Path:
    exp = tmp_path / "cvat_export"
    (exp / "obj_train_data").mkdir(parents=True)
    (exp / "obj.names").write_text("\n".join(names) + "\n")
    (exp / "train.txt").write_text("data/obj_train_data/x.jpg\n")       # CVAT bookkeeping, not a label
    for stem, text in labels.items():
        (exp / "obj_train_data" / f"{stem}.txt").write_text(text)
    return exp


def test_import_labels_skips_empty_cvat_files_unless_told(tmp_path: Path, ws: Path):
    from training import annotation_io
    _populate(ws, tmp_path, {"s1": 3})
    a, b, c = sorted(Path(r["image"]).stem for r in read_manifest(ws))
    exp = _cvat_export(tmp_path, model_classes(), {a: "2 0.5 0.5 0.1 0.1\n", b: "", c: "4 0.98 0.5 0.2 0.2\n"})
    assert annotation_io.main(["import", str(exp), "--workspace", str(ws)]) == 1      # c: box outside the image
    assert label_path(ws, a + ".jpg").read_text().startswith("2 0.500000")
    assert not label_path(ws, b + ".jpg").exists()                                    # empty != reviewed
    assert not label_path(ws, c + ".jpg").exists()
    assert annotation_io.main(["import", str(exp), "--workspace", str(ws), "--empty-means-reviewed"]) == 1
    assert label_path(ws, b + ".jpg").read_text() == ""
    # a class list in a different order is refused outright
    bad = _cvat_export(tmp_path / "x", list(reversed(model_classes())), {a: "0 0.5 0.5 0.1 0.1\n"})
    assert annotation_io.main(["import", str(bad), "--workspace", str(ws)]) == 2


def test_export_batch_puts_real_first_and_skips_overlay(tmp_path: Path, ws: Path):
    from training import annotation_io
    _populate(ws, tmp_path, {"real": 2})
    _populate(ws, tmp_path, {"aug": 2}, source_type="synthetic")
    rows = read_manifest(ws)
    rows[0]["overlay_suspect"] = "yes"
    write_manifest(ws, rows)
    assert annotation_io.main(["export", "--batch", "b1", "--workspace", str(ws), "--include-synthetic"]) == 0
    out = ws / "annotation_batches" / "b1"
    import csv
    batch = [r["source_type"] for r in csv.DictReader(open(out / "batch.csv", encoding="utf-8-sig"))]
    assert batch == ["real_cctv", "synthetic", "synthetic"]
    assert (out / "obj.names").read_text().split() == model_classes()


# ---------------------------------------------------------------- workspace validation
def test_workspace_validation_reports_bad_labels_and_status(tmp_path: Path, ws: Path):
    _populate(ws, tmp_path, {"s1": 5})
    rows = read_manifest(ws)
    n = [r["image"] for r in rows]
    _annotate(ws, n[0], ["2 0.5 0.5 0.1 0.1"]); rows[0]["review_status"], rows[0]["category"] = "annotated", "litter"
    _annotate(ws, n[1], ["12 0.5 0.5 0.1 0.1"]); rows[1]["review_status"] = "annotated"        # bad class id
    _annotate(ws, n[2], ["1 0.95 0.5 0.3 0.2"]); rows[2]["review_status"] = "annotated"        # outside image
    rows[3]["review_status"] = "reviewed_empty"                                                # no label file
    _annotate(ws, n[4], ["0 0.5 0.5 0.1 0.1"])                                                 # not synced
    write_manifest(ws, rows)
    assert validate_ldm_dataset.main([str(ws)]) == 1
    import csv
    fix = {(r["type"], r["image"]) for r in csv.DictReader(open(ws / "reports" / "files_to_fix.csv", encoding="utf-8-sig"))}
    assert ("class_id", n[1]) in fix
    assert ("box_outside_image", n[2]) in fix
    assert ("reviewed_without_label", n[3]) in fix
    assert ("needs_sync", n[4]) in fix
    assert not any(img == n[0] for _, img in fix)
    rep = (ws / "reports" / "validation_report.md").read_text(encoding="utf-8")
    assert "Verified (reviewed by a person" in rep
