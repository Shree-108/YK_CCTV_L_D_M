"""Random-interval screenshot capture, tested with a sample video as a mock stream (no real camera)."""
from __future__ import annotations

import datetime as dt
import time

import pytest
from fastapi.testclient import TestClient

from .conftest import needs_model

pytestmark = needs_model
AUTH = {"Authorization": "Bearer test-admin-token"}


@pytest.fixture
def client(cfg):
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        c.post("/api/models/activate", json={"version": "v1_litter_s1_synthetic"}, headers=AUTH)
        yield c


@pytest.fixture
def cam(client, litter_image, make_video):
    clip = make_video([litter_image], repeat=20, name="mock_stream.mp4")   # every frame identical
    r = client.post("/api/cameras", headers=AUTH, json={"id": "mock1", "name": "Mock stream", "source": str(clip),
                                                        "enabled": False})
    assert r.status_code == 201, r.text
    return "mock1"


def sched(client, cam_id):
    return client.get(f"/api/schedule/{cam_id}").json()


def test_capture_now_saves_original_annotated_and_results(client, cam):
    assert client.post(f"/api/schedule/{cam}/capture-now").status_code == 401          # controls are admin-only
    r = client.post(f"/api/schedule/{cam}/capture-now", headers=AUTH)
    assert r.status_code == 200, r.text
    c = r.json()["capture"]
    assert c["status"] == "done" and c["trigger"] == "manual" and c["task"] == "litter"
    assert c["counts"]["raw"] >= 1 and isinstance(c["detections"], list)
    for kind in ("original", "annotated"):
        img = client.get(c[f"{kind}_url"])
        assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg" and len(img.content) > 1000
    assert client.get("/api/captures", params={"camera_id": cam}).json()["captures"][0]["id"] == c["id"]
    assert sched(client, cam)["last_status"] == "done"


def test_controls_and_random_next_capture(client, cam):
    assert client.put(f"/api/schedule/{cam}", json={"min_minutes": 5, "max_minutes": 2}, headers=AUTH).status_code == 422
    client.put(f"/api/schedule/{cam}", json={"min_minutes": 120, "max_minutes": 180}, headers=AUTH)
    s = client.post(f"/api/schedule/{cam}/start", headers=AUTH).json()["schedule"]
    assert s["state"] == "running" and 120 * 60 - 5 <= s["seconds_to_next"] <= 180 * 60
    nexts = set()
    for _ in range(5):   # random, not a fixed interval
        nexts.add(client.post(f"/api/schedule/{cam}/start", headers=AUTH).json()["schedule"]["next_run_at"])
    assert len(nexts) > 1
    assert client.post(f"/api/schedule/{cam}/pause", headers=AUTH).json()["schedule"]["state"] == "paused"
    assert client.post(f"/api/schedule/{cam}/resume", headers=AUTH).json()["schedule"]["state"] == "running"
    s = client.post(f"/api/schedule/{cam}/stop", headers=AUTH).json()["schedule"]
    assert s["state"] == "stopped" and s["next_run_at"] is None


def _make_due(client, cam):
    st = client.app.state.app_state
    s = st.scheduler.get(cam)
    s["next_run_at"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)).isoformat()
    st.scheduler._save(cam, s)


def _wait(pred, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.5)
    return False


def test_scheduled_capture_runs_and_reschedules(client, cam):
    client.post(f"/api/schedule/{cam}/start", headers=AUTH)
    _make_due(client, cam)
    assert _wait(lambda: any(c["trigger"] == "scheduled" and c["status"] == "done"
                             for c in client.get("/api/captures", params={"camera_id": cam}).json()["captures"]))
    s = sched(client, cam)
    assert s["state"] == "running" and s["seconds_to_next"] >= 120 * 60 - 5


def test_no_overlapping_captures(client, cam):
    lock = client.app.state.app_state.scheduler._lock(cam)
    with lock:                                   # a capture is "in progress"
        assert client.post(f"/api/schedule/{cam}/capture-now", headers=AUTH).status_code == 409
        assert sched(client, cam)["busy"] is True


def test_connection_failure_is_recorded_and_retried(client):
    client.post("/api/cameras", headers=AUTH, json={"id": "dead1", "name": "Dead", "source": "rtsp://admin:Hunter22@127.0.0.1:1/x",
                                                    "enabled": False})
    client.put("/api/schedule/dead1", json={"attempts": 1, "retry_minutes": 7}, headers=AUTH)
    c = client.post("/api/schedule/dead1/capture-now", headers=AUTH).json()["capture"]
    assert c["status"] == "failed" and "Hunter22" not in str(c)
    client.post("/api/schedule/dead1/start", headers=AUTH)
    _make_due(client, "dead1")
    assert _wait(lambda: sched(client, "dead1")["consecutive_failures"] >= 1, 90)
    s = sched(client, "dead1")
    assert s["state"] == "running" and 7 * 60 - 10 <= s["seconds_to_next"] <= 7 * 60   # retry, not 2-3 h
    assert "Hunter22" not in client.get("/api/events?limit=100").text


def test_restart_recovery(client, cam):
    st = client.app.state.app_state
    client.post(f"/api/schedule/{cam}/start", headers=AUTH)
    s = st.scheduler.get(cam)
    s["next_run_at"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=5)).isoformat()   # missed while down
    st.scheduler._save(cam, s)
    st.db.insert_capture({"id": "crashed", "camera_id": cam, "ts": "2026-01-01T00:00:00", "trigger": "scheduled",
                          "task": "litter", "status": "running"})
    st.scheduler.shutdown()
    from backend.cameras.scheduler import CaptureScheduler
    fresh = CaptureScheduler(st)                 # what a restarted server does
    fresh._recover()
    s = fresh.status(cam)
    assert 50 <= s["seconds_to_next"] <= 185                                   # ONE catch-up in 1-3 min, not a burst
    assert st.db.get_capture("crashed")["status"] == "interrupted"


def test_uniform_without_model_is_skipped_honestly(client, cam):
    client.put(f"/api/schedule/{cam}", json={"task": "uniform"}, headers=AUTH)
    s = sched(client, cam)
    assert s["model_available"] is False and "Uniform model not installed" in s["model_note"]
    c = client.post(f"/api/schedule/{cam}/capture-now", headers=AUTH).json()["capture"]
    assert c["status"] == "skipped" and c["original_url"] and not c["annotated_url"] and not c["detections"]


def test_repeated_captures_do_not_duplicate_alerts(client, cam):
    a = client.post(f"/api/schedule/{cam}/capture-now", headers=AUTH).json()["capture"]
    b = client.post(f"/api/schedule/{cam}/capture-now", headers=AUTH).json()["capture"]
    if not a["alert_ids"]:
        pytest.skip("model produced no confirmed litter on the sample frame")
    assert b["alert_ids"] == [] or set(b["alert_ids"]) <= set(a["alert_ids"])
    alerts = client.get("/api/alerts", params={"camera_id": cam}).json()["alerts"]
    assert len(alerts) == len(a["alert_ids"]) and all(x["occurrences"] >= 2 for x in alerts)
