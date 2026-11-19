"""Lab endpoints: datasets, test lab, evaluation, hard examples, experiments, health, search, exports.

Datasets and training runs are SYNTHETIC and created in temporary folders (cfg.dataset_dir / cfg.storage_dir), so the
tests have exact expected values and never read or write the project's dataset/ or storage/. The synthetic images are
drawn shapes: they exercise the evaluation code, and the resulting precision/recall mean nothing about real litter.
The original assertions against the v1 dataset (210 images, 13 classes) are kept in test_v1_legacy.py.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from .conftest import V3_CLASSES, V3_VERSION, needs_v3

pytestmark = needs_v3
AUTH = {"Authorization": "Bearer test-admin-token"}
DS = "synthetic_v3"
# (split, boxes as (class_id, x, y, w, h) in pixels on a 640x480 frame)
LABELLED = [
    ("test", [(3, 100, 100, 20, 14), (5, 300, 200, 60, 50)]),       # small + medium
    ("test", [(0, 200, 150, 150, 110)]),                             # large
    ("test", [(2, 400, 300, 40, 30), (8, 80, 380, 24, 18)]),
    ("train", [(6, 320, 240, 50, 60)]),
]
NEGATIVES = [("test", 2)]


def _draw(boxes, seed) -> np.ndarray:
    img = np.full((480, 640, 3), 110, np.uint8) + np.random.default_rng(seed).integers(0, 15, (480, 640, 3), dtype=np.uint8)
    for k, (_, x, y, w, h) in enumerate(boxes):
        cv2.rectangle(img, (x, y), (x + w, y + h), (230, 230 - 40 * k, 220), -1)
    return img


def _make_dataset(root) -> dict:
    d = root / DS
    for i, (split, boxes) in enumerate(LABELLED):
        (d / "images" / split).mkdir(parents=True, exist_ok=True); (d / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "images" / split / f"img{i}.jpg"), _draw(boxes, i))
        (d / "labels" / split / f"img{i}.txt").write_text("".join(
            f"{c} {(x + w / 2) / 640:.6f} {(y + h / 2) / 480:.6f} {w / 640:.6f} {h / 480:.6f}\n" for c, x, y, w, h in boxes))
    for split, n in NEGATIVES:
        (d / "images" / "negative" / split).mkdir(parents=True, exist_ok=True)
        for j in range(n):
            cv2.imwrite(str(d / "images" / "negative" / split / f"neg{j}.jpg"), _draw([], 100 + j))
    (d / "data.yaml").write_text(yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/test",
                                                  "test": "images/test", "names": {i: n for i, n in enumerate(V3_CLASSES)}}))
    n_test = sum(1 for s, _ in LABELLED if s == "test")
    return {"images": len(LABELLED) + sum(n for _, n in NEGATIVES), "negatives": sum(n for _, n in NEGATIVES),
            "objects": sum(len(b) for _, b in LABELLED), "test_images": n_test + sum(n for s, n in NEGATIVES if s == "test")}


def _make_run(storage) -> None:
    run = storage / "runs" / "synthetic_run"
    run.mkdir(parents=True)
    (run / "results.csv").write_text(
        "epoch,train/box_loss,train/cls_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,1.2,2.0,0.5,0.4,0.3,0.1\n2,1.1,1.5,0.6,0.5,0.4,0.2\n")
    (run / "args.yaml").write_text(yaml.safe_dump({"model": "yolo11s.pt", "data": "synthetic.yaml", "epochs": 2, "imgsz": 640}))


@pytest.fixture
def expected(cfg, tmp_path):
    cfg.dataset_dir = tmp_path / "datasets"
    cfg.dataset_dir.mkdir()
    exp = _make_dataset(cfg.dataset_dir)
    _make_run(cfg.storage_dir)
    return exp


@pytest.fixture
def client(cfg, expected):
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        assert c.post("/api/models/activate", json={"version": V3_VERSION}, headers=AUTH).status_code == 200
        yield c


def _wait(client, job_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.5)
    raise AssertionError("job timeout")


def test_dataset_stats_and_validation(client, expected):
    names = [d["name"] for d in client.get("/api/datasets").json()["datasets"]]
    assert names == [DS]                                      # only the temporary dataset is visible
    s = client.get(f"/api/datasets/{DS}").json()
    assert s["totals"]["images"] == expected["images"] and s["totals"]["negatives"] == expected["negatives"]
    assert sum(s["objects_per_class"].values()) == s["totals"]["objects"] == expected["objects"]
    assert set(s["size_distribution"]) == {"small", "medium", "large"} and all(s["size_distribution"].values())
    assert "issue_counts" in s
    imgs = client.get(f"/api/datasets/{DS}/images", params={"split": "test"}).json()
    assert imgs["total"] == expected["test_images"]
    one = next(i["id"] for i in imgs["items"] if not i.get("negative") and i.get("objects"))
    ann = client.get(f"/api/datasets/{DS}/annotation", params={"id": one}).json()
    assert ann["ground_truth"] and ann["editable"]
    assert client.get(f"/api/datasets/{DS}/image", params={"id": one}).status_code == 200
    assert client.get(f"/api/datasets/{DS}/image", params={"id": "../../.env"}).status_code == 404
    assert client.get("/api/datasets/nope").status_code == 404


def test_testlab_counts_are_consistent(client):
    one = next(i["id"] for i in client.get(f"/api/datasets/{DS}/images", params={"split": "test"}).json()["items"]
               if i.get("objects"))
    active = client.get("/api/models").json()["active"]
    assert active == V3_VERSION
    r = client.post("/api/testlab/run", json={"dataset": DS, "image": one, "versions": [active], "confidence": 0.5})
    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    c = res["counts"]
    assert c["true_positive"] + c["missed"] == c["ground_truth"] >= 1
    assert c["true_positive"] + c["false_positive"] == c["predicted"]
    assert res["latency_ms"] > 0 and all(p["match"] in ("tp", "fp") for p in res["detections"])
    bad = client.post("/api/testlab/run", json={"dataset": DS, "image": one, "versions": ["nope"]})
    assert bad.json()["results"][0]["error"]


def test_evaluation_job_and_hard_examples(client):
    from backend.core.config import AppConfig
    from backend.alerts.manager import ClassMetadata
    r = client.post("/api/evaluations", json={"dataset": DS, "split": "test", "compare_tiling": True})
    assert r.status_code == 202, r.text
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done", job.get("error")
    ev = client.get(f"/api/evaluations/{r.json()['job_id']}").json()
    o = ev["overall"]
    assert 0 <= o["precision"] <= 1 and 0 <= o["recall"] <= 1
    assert len(ev["threshold_sweep"]) == 19 and ev["tiling_comparison"] is not None
    assert {"small", "medium", "large"} == set(ev["by_size"])
    assert set(ev["report_groups"]) == set(ClassMetadata(AppConfig().class_metadata_path).report_groups)
    assert [p["class"] for p in ev["per_class"]] == V3_CLASSES     # v3: 10 classes
    rec = [p["recall"] for p in ev["threshold_sweep"]]
    assert all(a >= b for a, b in zip(rec, rec[1:]))               # recall never rises with the threshold
    hx = client.get("/api/hard-examples").json()
    assert hx["total"] == ev["hard_examples_found"]
    if hx["items"]:
        hid = hx["items"][0]["id"]
        assert client.patch(f"/api/hard-examples/{hid}", json={"review_status": "approved"}).json()[
            "review_status"] == "approved"
    assert any(e["id"] == r.json()["job_id"] for e in client.get("/api/evaluations").json()["evaluations"])


def test_evaluation_rejects_dataset_with_other_classes(client, cfg):
    d = cfg.dataset_dir / "other_classes"
    (d / "images" / "test").mkdir(parents=True); (d / "labels" / "test").mkdir(parents=True)
    cv2.imwrite(str(d / "images" / "test" / "a.jpg"), _draw([], 7))
    (d / "labels" / "test" / "a.txt").write_text("")
    (d / "data.yaml").write_text(yaml.safe_dump({"test": "images/test", "names": {0: "chips_packet"}}))
    r = client.post("/api/evaluations", json={"dataset": "other_classes", "split": "test"})
    if r.status_code == 202:
        job = _wait(client, r.json()["job_id"])
        assert job["status"] == "failed" and "do not match" in (job.get("error") or "")
    else:
        assert r.status_code in (400, 422)


def test_experiments_health_search_export(client):
    exps = client.get("/api/experiments").json()["experiments"]
    assert [e["id"] for e in exps] and all("id" in e for e in exps)
    d = client.get(f"/api/experiments/{exps[0]['id']}").json()
    assert d["epochs"] and "config" in d
    h = client.get("/api/system/health").json()
    comps = {c["component"]: c["status"] for c in h["components"]}
    assert comps["inference"] == "operational" and comps["database"] == "operational"
    assert comps["camera_services"] == "not_configured"
    assert client.get("/api/search", params={"q": "v3"}).json()["models"]
    csv = client.get("/api/export/alerts.csv")
    assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
    assert client.get("/api/auth/check").status_code == 401
    assert client.get("/api/auth/check", headers=AUTH).json() == {"admin": True}


def test_alert_notes_and_timeline(client, litter_image):
    # no ROI / camera registration needed: confirmed full-frame litter creates alerts (ROI removed 2026-11-19)
    body = client.post("/api/detect/image", files={"file": ("a.jpg", cv2.imencode(".jpg", litter_image)[1].tobytes(),
                                                            "image/jpeg")}, data={"camera_id": "CAM01"}).json()
    aid = body["alerts"][0]["id"]
    r = client.patch(f"/api/alerts/{aid}", json={"notes": "picked up by staff", "status": "resolved"}).json()
    assert r["notes"] == "picked up by staff" and r["status"] == "resolved"
    assert client.get(f"/api/alerts/{aid}/timeline").status_code == 200
    # detection overrides: a very high threshold confirms nothing
    strict = client.post("/api/detect/image", files={"file": ("a.jpg", cv2.imencode(".jpg", litter_image)[1].tobytes(),
                                                              "image/jpeg")},
                         data={"confidence": "0.99", "create_alerts": "false", "tiling": "off"}).json()
    assert strict["counts"]["confirmed"] <= body["count"] and strict["alerts"] == []
