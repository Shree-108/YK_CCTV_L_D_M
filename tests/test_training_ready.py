"""training/prepare_training_ready.py and training/install_trained_model.py on small synthetic fixtures."""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from training import install_trained_model as inst
from training import prepare_training_ready as prep

SCENES = {"A_shopfront": 6, "B_roadside_bin": 4}


def _fixture(tmp: Path, *, unverified: str = "", tamper: str = "") -> tuple[Path, Path]:
    src, lab = tmp / "src", tmp / "labels"
    (lab / "labels" / "train").mkdir(parents=True)
    (lab / "annotations" / "per_image").mkdir(parents=True)
    src.mkdir()
    rows, sums = [], []
    n = 0
    for scene, k in SCENES.items():
        for i in range(k):
            n += 1
            name = f"cctv_litter_{n:03d}.jpg"
            rng = np.random.default_rng(n)                       # deterministic, distinct images
            img = rng.integers(0, 255, (72, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(src / name), img)
            digest = hashlib.sha256((src / name).read_bytes()).hexdigest()
            sums.append(f"{digest}  {name}")
            (lab / "labels" / "train" / f"cctv_litter_{n:03d}.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n0 0.2 0.3 0.05 0.05\n")
            (lab / "annotations" / "per_image" / f"cctv_litter_{n:03d}.json").write_text(json.dumps(
                {"ignore": [{"xyxy": [1, 2, 3, 4], "why": "ambiguous"}]}))
            rows.append({"file": name, "scene": scene, "role": "reference" if i == 0 else "variant",
                         "reference_id": "", "sha256": digest,
                         "label_status": "draft" if name == unverified else "verified",
                         "verified_by": "test", "source_type": "synthetic"})
    with (lab / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (lab / "source_sha256.txt").write_text("# test\n" + "\n".join(sums) + "\n")
    if tamper:
        cv2.imwrite(str(src / tamper), np.zeros((72, 128, 3), np.uint8))
    return src, lab


def test_build_scene_split_and_reports(tmp_path):
    src, lab = _fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    res = prep.build(src, lab, tmp_path / "out", test_scene="B_roadside_bin", val_count=2, make_zip=True)
    assert res["splits"] == {"train": 4, "val": 2, "test": 4}
    out = tmp_path / "out"
    man = list(csv.DictReader((out / "dataset_manifest.csv").open()))
    assert {r["scene"] for r in man if r["split"] == "test"} == {"B_roadside_bin"}
    assert {r["scene"] for r in man if r["split"] in ("train", "val")} == {"A_shopfront"}
    assert all(r["role"] != "reference" for r in man if r["split"] == "val")       # reference stays in train
    for f in ("data.yaml", "dataset_manifest.csv", "annotation_audit.csv", "dataset_summary.csv",
              "class_distribution.csv", "split_summary.csv", "ignore_regions.json", "README.md"):
        assert (out / f).is_file(), f
    assert "0: litter" in (out / "data.yaml").read_text()
    assert len(list((out / "labels" / "test").glob("*.txt"))) == 4
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before                # source untouched
    assert zipfile.is_zipfile(res["zip"])


def test_refuses_existing_output(tmp_path):
    src, lab = _fixture(tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep.txt").write_text("x")
    with pytest.raises(prep.PrepError, match="refusing to overwrite"):
        prep.build(src, lab, tmp_path / "out", make_zip=False)
    assert (tmp_path / "out" / "keep.txt").read_text() == "x"


def test_refuses_unverified_labels_and_changed_images(tmp_path):
    src, lab = _fixture(tmp_path / "a", unverified="cctv_litter_002.jpg")
    with pytest.raises(prep.PrepError, match="unverified"):
        prep.build(src, lab, tmp_path / "a" / "out", make_zip=False)
    src, lab = _fixture(tmp_path / "b", tamper="cctv_litter_003.jpg")
    with pytest.raises(prep.PrepError, match="source_hash_differs"):
        prep.build(src, lab, tmp_path / "b" / "out", make_zip=False)


def test_invalid_box_is_reported_not_fixed(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("0 0.5 0.5 0.1 0.1\n3 0.5 0.5 0.1 0.1\n0 0.99 0.5 0.2 0.1\n0 0.5 0.5 0 0.1\n")
    boxes, issues = prep.parse_yolo(p, 100, 100)
    assert len(boxes) == 4
    assert any("class_id_3" in i for i in issues) and any("outside_image" in i for i in issues)
    assert any("zero_or_negative" in i for i in issues)


# --------------------------------------------------------------------------------------------- installer
def _package(tmp: Path, version: str = "v9_test", smoke: bool = False) -> Path:
    d = tmp / "pkg" / version
    d.mkdir(parents=True)
    (d / "best.pt").write_bytes(b"weights")
    (d / "class_names.json").write_text(json.dumps(["litter"]))
    (d / "model_meta.json").write_text(json.dumps({"version": version, "classes": ["litter"], "smoke_test": smoke}))
    sums = [f"{hashlib.sha256((d / f).read_bytes()).hexdigest()}  {f}" for f in ("best.pt", "class_names.json",
                                                                                 "model_meta.json")]
    (d / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return d


def test_install_new_version_only_and_never_v3(tmp_path):
    models = tmp_path / "models"
    (models / "v3_store_cctv").mkdir(parents=True)
    (models / "v3_store_cctv" / "best.pt").write_bytes(b"prod")
    pkg = _package(tmp_path)
    res = inst.install(pkg, models, load_model=False)
    assert res["active"] is False and (models / "v9_test" / "best.pt").read_bytes() == b"weights"
    with pytest.raises(inst.InstallError, match="already exists"):
        inst.install(pkg, models, load_model=False)
    v3 = _package(tmp_path / "x", version="v3_store_cctv")
    with pytest.raises(inst.InstallError, match="production model"):
        inst.install(v3, models, load_model=False)
    assert (models / "v3_store_cctv" / "best.pt").read_bytes() == b"prod"


def test_install_rejects_tampered_and_smoke_packages(tmp_path):
    pkg = _package(tmp_path / "t")
    (pkg / "best.pt").write_bytes(b"changed")
    with pytest.raises(inst.InstallError, match="checksum"):
        inst.install(pkg, tmp_path / "m", load_model=False)
    smoke = _package(tmp_path / "s", smoke=True)
    with pytest.raises(inst.InstallError, match="SMOKE"):
        inst.install(smoke, tmp_path / "m", load_model=False)
    assert inst.install(smoke, tmp_path / "m", allow_smoke=True, load_model=False)["version"] == "v9_test"
