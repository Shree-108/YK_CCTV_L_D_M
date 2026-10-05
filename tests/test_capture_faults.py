"""Fault tolerance of CCTV capture: invalid frames, hanging/unreachable/invalid sources, isolation from
the API process, and honest notification delivery records. No real camera is used."""
from __future__ import annotations

import socket
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.alerts.notifier import Notifier
from backend.cameras.grabber import frame_problem, grab_frame

from .conftest import needs_model

AUTH = {"Authorization": "Bearer test-admin-token"}


# ------------------------------------------------------------------ frame validation (no model needed)
def test_frame_problem_detects_unusable_frames(litter_image):
    assert frame_problem(None) == "empty frame"
    assert "too small" in frame_problem(np.zeros((40, 40, 3), np.uint8))
    assert "blank" in frame_problem(np.full((360, 640, 3), 30, np.uint8))
    broken = litter_image.copy()
    broken[broken.shape[0] // 3:] = (128, 128, 128)        # lower 2/3 grey: typical lost-reference-frame decode
    assert "corrupted" in frame_problem(broken)
    assert frame_problem(litter_image) is None


class HangingServer:
    """Accepts TCP connections and never answers - like a frozen camera/NVR."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        self.conns: list[socket.socket] = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                c, _ = self.sock.accept()
                self.conns.append(c)
            except OSError:
                return

    def close(self):
        for c in self.conns:
            c.close()
        self.sock.close()


def test_grabber_hard_timeout_on_hanging_stream(tmp_path):
    srv = HangingServer()
    try:
        t0 = time.perf_counter()
        r = grab_frame(f"rtsp://user:Secr3tPw@127.0.0.1:{srv.port}/cam", tmp_path / "x.jpg", timeout=6)
        assert not r["ok"] and r["code"] in ("timeout", "open_failed")
        assert time.perf_counter() - t0 < 20                  # bounded, never hangs the caller
        assert "Secr3tPw" not in str(r)
    finally:
        srv.close()


def test_grabber_invalid_sources(tmp_path):
    for src in ("rtsp://", "D:/does/not/exist.mp4", "not a url at all"):
        r = grab_frame(src, tmp_path / "x.jpg", timeout=20)
        assert not r["ok"] and r["code"] in ("open_failed", "no_frame", "error", "timeout"), r


def test_notifier_records_delivery_only_on_success():
    delivered, errors = [], []

    class Ok:
        name = "ok"
        def available(self): return True
        def send(self, a): pass

    class Fails:
        name = "fails"
        def available(self): return True
        def send(self, a): raise ConnectionError("SMTP down")

    n = Notifier(lambda cam: {"min_severity": "LOW"}, providers=[Ok(), Fails()],
                 on_error=lambda p, e: errors.append(p), on_delivered=lambda p, a: delivered.append(p))
    n.dispatch({"id": 1, "camera_id": "c", "severity": "HIGH", "class": "paper", "confidence": 0.9, "ts": "now"})
    t0 = time.time()
    while (not delivered or not errors) and time.time() - t0 < 5:
        time.sleep(0.05)
    assert delivered == ["ok"] and errors == ["fails"]


# ------------------------------------------------------------------ through the running app (model needed)
@pytest.fixture
def client(cfg):
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        c.post("/api/models/activate", json={"version": "v1_litter_s1_synthetic"}, headers=AUTH)
        yield c


@needs_model
def test_invalid_frames_skip_inference(client, make_video):
    blank = make_video([np.full((368, 640, 3), 40, np.uint8)], repeat=20, name="blank.mp4")
    client.post("/api/cameras", headers=AUTH, json={"id": "blank1", "name": "Blank", "source": str(blank), "enabled": False})
    client.put("/api/schedule/blank1", json={"attempts": 2, "backoff_sec": 0}, headers=AUTH)
    c = client.post("/api/schedule/blank1/capture-now", headers=AUTH).json()["capture"]
    assert c["status"] == "invalid_frame" and "blank" in c["error"]
    assert c["detections"] == [] and c["annotated_url"] is None and c["model_version"] is None   # model never ran
    s = client.get("/api/schedule/blank1").json()
    assert s["last_status"] == "invalid_frame" and s["last_success_at"] is None


@needs_model
def test_hanging_camera_does_not_block_api_or_uploads(client, litter_image):
    srv = HangingServer()
    try:
        client.post("/api/cameras", headers=AUTH, json={"id": "hang1", "name": "Hang", "enabled": False,
                                                        "source": f"rtsp://u:HidePw99@127.0.0.1:{srv.port}/cam"})
        client.put("/api/schedule/hang1", json={"attempts": 2, "timeout_sec": 6, "backoff_sec": 1}, headers=AUTH)
        result: dict = {}
        th = threading.Thread(target=lambda: result.update(client.post("/api/schedule/hang1/capture-now", headers=AUTH).json()))
        th.start()
        time.sleep(1.5)                                   # capture is now stuck on the hanging camera
        t0 = time.perf_counter()
        assert client.get("/api/health").status_code == 200
        jpg = cv2.imencode(".jpg", litter_image)[1].tobytes()
        up = client.post("/api/detect/image", files={"file": ("x.jpg", jpg, "image/jpeg")}, data={"create_alerts": "false"})
        assert up.status_code == 200 and "detections" in up.json()          # uploads keep working
        assert time.perf_counter() - t0 < 15
        assert client.post("/api/schedule/hang1/capture-now", headers=AUTH).status_code == 409   # no overlap
        th.join(60)
        c = result["capture"]
        assert c["status"] == "failed" and c["error"].split(":")[0] in ("timeout", "open_failed")
        assert "HidePw99" not in str(result) and "HidePw99" not in client.get("/api/events?limit=100").text
    finally:
        srv.close()


@needs_model
def test_repeated_failures_back_off_and_never_crash(client):
    client.post("/api/cameras", headers=AUTH, json={"id": "gone1", "name": "Gone", "enabled": False,
                                                    "source": "D:/no/such/camera.mp4"})
    client.put("/api/schedule/gone1", json={"attempts": 3, "backoff_sec": 0.5, "timeout_sec": 20}, headers=AUTH)
    t0 = time.perf_counter()
    for _ in range(2):
        c = client.post("/api/schedule/gone1/capture-now", headers=AUTH).json()["capture"]
        assert c["status"] == "failed"
    assert time.perf_counter() - t0 >= 2 * (0.5 + 1.0)                 # 0.5 s + 1 s exponential waits per capture
    assert client.get("/api/health").json()["status"] == "ok"
