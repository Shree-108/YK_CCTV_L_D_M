"""A cancel sent while the outdoor ROI is being identified must stop the job with that single request.

Before the fix the worker ignored a cancel that arrived during ROI identification (the slow, CPU-bound step): the job
moved on to awaiting_roi (or, with auto-accept, to analysis) and the operator had to cancel a second time."""
from __future__ import annotations

import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app

from .test_sampled import FakeSegmenter, _upload, _wait, write_video


class BlockingSegmenter(FakeSegmenter):
    """Signals when ROI identification has started, then holds it until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()

    def masks(self, image: np.ndarray):
        self.entered.set()
        assert self.release.wait(60), "test never released the segmenter"
        return super().masks(image)


@pytest.fixture
def client(cfg):
    cfg.roi_segmenter = "off"
    with TestClient(create_app(cfg)) as c:
        yield c


@pytest.mark.parametrize("auto_accept", [False, True])
def test_one_cancel_during_roi_identification_stops_the_job(client, tmp_path, auto_accept):
    seg = BlockingSegmenter()
    runner = client.app.state.app_state.sampled
    runner._segmenter = lambda: seg
    m = _upload(client, write_video(tmp_path / "a.mp4", 150, fps=10)).json()
    jid = client.post("/api/sampled/jobs", json={"media_id": m["id"], "auto_accept": auto_accept}).json()["job_id"]

    assert seg.entered.wait(120), "ROI identification never started"
    j = client.get(f"/api/sampled/jobs/{jid}").json()
    assert j["state"] == "running" and j["stage"] == "identifying_roi"
    r = client.post(f"/api/sampled/jobs/{jid}/cancel")                      # the operator cancels exactly once
    assert r.status_code == 200, r.text
    seg.release.set()

    j = _wait(client, jid, ("cancelled", "awaiting_roi", "done", "failed"))
    assert j["state"] == "cancelled", (j["state"], j["stage"])
    runner.wait_idle(60)
    j = client.get(f"/api/sampled/jobs/{jid}").json()
    assert j["state"] == "cancelled" and j["error"] == "Cancelled by the operator."
    assert client.put(f"/api/sampled/jobs/{jid}/roi", json={"zones": []}).status_code in (409, 422)   # nothing to confirm
    assert not client.get("/api/detections", params={"job_id": jid}).json()["detections"]           # never analysed


def test_cancel_while_waiting_for_the_roi_still_works(client, tmp_path):
    """The existing path (cancel in awaiting_roi) is unchanged by the fix."""
    client.app.state.app_state.sampled._segmenter = lambda: FakeSegmenter(outdoor=False)
    m = _upload(client, write_video(tmp_path / "b.mp4", 150, fps=10)).json()
    jid = client.post("/api/sampled/jobs", json={"media_id": m["id"]}).json()["job_id"]
    assert _wait(client, jid, ("awaiting_roi", "failed"))["state"] == "awaiting_roi"
    assert client.post(f"/api/sampled/jobs/{jid}/cancel").json()["state"] == "cancelled"
