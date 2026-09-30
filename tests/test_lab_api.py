"""Lab endpoints: datasets, test lab, evaluation, hard examples, experiments, health, search, exports."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from .conftest import needs_model

pytestmark = needs_model
AUTH = {"Authorization": "Bearer test-admin-token"}
DS = "cctv_litter_dataset_v1"


@pytest.fixture
def client(cfg):
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        # tests were written against the fine-tuned v1 model (not the zero-shot v2)
        c.post("/api/models/activate", json={"version": "v1_litter_s1_synthetic"}, headers=AUTH)
        yield c


def _wait(client, job_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.5)
    raise AssertionError("job timeout")


def test_dataset_stats_and_validation(client):
    names = [d["name"] for d in client.get("/api/datasets").json()["datasets"]]
    assert DS in names
    s = client.get(f"/api/datasets/{DS}").json()
    assert s["totals"]["images"] == 210 and s["totals"]["negatives"] == 60
    assert sum(s["objects_per_class"].values()) == s["totals"]["objects"] > 0
    assert set(s["size_distribution"]) == {"small", "medium", "large"}
    assert "issue_counts" in s
    imgs = client.get(f"/api/datasets/{DS}/images", params={"split": "test"}).json()
    assert imgs["total"] == 10
    one = imgs["items"][0]["id"]
    ann = client.get(f"/api/datasets/{DS}/annotation", params={"id": one}).json()
    assert ann["ground_truth"] and ann["editable"]
    assert client.get(f"/api/datasets/{DS}/image", params={"id": one}).status_code == 200
    assert client.get(f"/api/datasets/{DS}/image", params={"id": "../../.env"}).status_code == 404
    assert client.get("/api/datasets/nope").status_code == 404


def test_testlab_matches_ground_truth(client):
    one = client.get(f"/api/datasets/{DS}/images", params={"split": "test"}).json()["items"][0]["id"]
    active = client.get("/api/models").json()["active"]
    r = client.post("/api/testlab/run", json={"dataset": DS, "image": one, "versions": [active], "confidence": 0.5})
    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    c = res["counts"]
    assert c["true_positive"] + c["missed"] == c["ground_truth"]
    assert c["true_positive"] + c["false_positive"] == c["predicted"]
    assert res["latency_ms"] > 0 and all(p["match"] in ("tp", "fp") for p in res["detections"])
    bad = client.post("/api/testlab/run", json={"dataset": DS, "image": one, "versions": ["nope"]})
    assert bad.json()["results"][0]["error"]


def test_evaluation_job_and_hard_examples(client):
    r = client.post("/api/evaluations", json={"dataset": DS, "split": "test", "compare_tiling": True})
    assert r.status_code == 202, r.text
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done", job.get("error")
    ev = client.get(f"/api/evaluations/{r.json()['job_id']}").json()
    o = ev["overall"]
    assert 0 <= o["precision"] <= 1 and 0 <= o["recall"] <= 1 and o["mAP50"] is not None
    assert len(ev["threshold_sweep"]) == 19 and ev["tiling_comparison"] is not None
    assert {"small", "medium", "large"} == set(ev["by_size"])
    assert {"chips_packet", "wrapper", "plastic_packaging"} <= set(ev["report_groups"])
    assert len(ev["per_class"]) == 13
    # threshold sweep is monotone in recall
    rec = [p["recall"] for p in ev["threshold_sweep"]]
    assert all(a >= b for a, b in zip(rec, rec[1:]))
    hx = client.get("/api/hard-examples").json()
    assert hx["total"] == ev["hard_examples_found"]
    if hx["items"]:
        hid = hx["items"][0]["id"]
        assert client.patch(f"/api/hard-examples/{hid}", json={"review_status": "approved"}).json()[
            "review_status"] == "approved"
    assert any(e["id"] == r.json()["job_id"] for e in client.get("/api/evaluations").json()["evaluations"])


def test_experiments_health_search_export(client):
    exps = client.get("/api/experiments").json()["experiments"]
    assert exps and all("id" in e for e in exps)
    d = client.get(f"/api/experiments/{exps[0]['id']}").json()
    assert d["epochs"] and "config" in d
    h = client.get("/api/system/health").json()
    comps = {c["component"]: c["status"] for c in h["components"]}
    assert comps["inference"] == "operational" and comps["database"] == "operational"
    assert comps["camera_services"] == "not_configured"
    assert client.get("/api/search", params={"q": "v1"}).json()["models"]
    csv = client.get("/api/export/alerts.csv")
    assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
    assert client.get("/api/auth/check").status_code == 401
    assert client.get("/api/auth/check", headers=AUTH).json() == {"admin": True}


def test_alert_notes_and_timeline(client, litter_image):
    import cv2
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
