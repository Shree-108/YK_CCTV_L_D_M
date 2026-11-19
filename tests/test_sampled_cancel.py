"""A single cancel must stop a four-screenshot analysis, whether it is queued or already analysing.

(The former "cancel during ROI identification" scenario no longer exists: ROI was removed from the application,
2026-11-19. The same guarantee - one request is enough - is now tested on the analysis stage and on the queue.)
The worker is held deterministically inside the first screenshot's detection, so the test does not depend on timing."""
from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.sampling import runner as runner_mod

from .test_sampled import _upload, _wait, write_video


@pytest.fixture
def client(cfg):
    with TestClient(create_app(cfg)) as c:
        yield c


def test_one_cancel_stops_a_running_and_a_queued_analysis(client, tmp_path, monkeypatch):
    if client.app.state.app_state.engine is None:
        pytest.skip("no detection model loaded")
    entered, release = threading.Event(), threading.Event()
    real = runner_mod.detect_image
    calls: list[str] = []

    def held_detect(*args, **kwargs):
        calls.append(args[8] if len(args) > 8 else kwargs.get("job_id"))
        entered.set()
        assert release.wait(120), "test never released the analysis"
        return real(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "detect_image", held_detect)
    runner = client.app.state.app_state.sampled
    m1 = _upload(client, write_video(tmp_path / "a.mp4", 150, fps=10)).json()
    m2 = _upload(client, write_video(tmp_path / "b.mp4", 160, fps=10)).json()
    j1 = client.post("/api/sampled/jobs", json={"media_id": m1["id"]}).json()["job_id"]
    assert entered.wait(120), "analysis never started"
    j2 = client.post("/api/sampled/jobs", json={"media_id": m2["id"]}).json()["job_id"]   # waits behind j1

    s1 = client.get(f"/api/sampled/jobs/{j1}").json()
    assert s1["state"] == "running" and s1["stage"] == "analysing"
    assert client.get(f"/api/sampled/jobs/{j2}").json()["state"] == "queued"

    r2 = client.post(f"/api/sampled/jobs/{j2}/cancel")                # queued: cancelled at once
    assert r2.status_code == 200 and r2.json()["state"] == "cancelled"
    r1 = client.post(f"/api/sampled/jobs/{j1}/cancel")                # running: exactly one request
    assert r1.status_code == 200, r1.text
    release.set()

    j = _wait(client, j1, ("cancelled", "done", "failed"))
    assert j["state"] == "cancelled", (j["state"], j["stage"])
    runner.wait_idle(60)
    for jid in (j1, j2):
        j = client.get(f"/api/sampled/jobs/{jid}").json()
        assert j["state"] == "cancelled" and j["error"] == "Cancelled by the operator."
        assert j["standard_outputs"] is None                           # no CSV / litter outputs for a cancelled job
        assert client.post(f"/api/sampled/jobs/{jid}/cancel").status_code == 409   # already finished
    assert calls == [j1]                                               # j1 stopped after its 1st screenshot, j2 never ran
    assert not client.get("/api/detections", params={"job_id": j2}).json()["detections"]
