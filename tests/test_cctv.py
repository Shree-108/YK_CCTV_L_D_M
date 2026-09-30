"""Live CCTV: RTSP config validation, credential handling/masking, connection-test error classes.
Uses a local fake RTSP server (Digest auth) - no real NVR is needed, and none is pretended."""
from __future__ import annotations

import hashlib
import re
import socket
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.cameras import cctv

SECRET = "S3cr3t!pw@x"
AUTH = {"Authorization": "Bearer test-admin-token"}


class FakeRtsp:
    """Answers DESCRIBE: 401 Digest challenge, then 200 for the right credentials+path, 401 otherwise,
    404 for an unknown channel. Sends no media, so the video stage must fail."""

    def __init__(self, user="admin", pwd=SECRET, channels=(1,)):
        self.user, self.pwd, self.channels = user, pwd, channels
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._client, args=(c,), daemon=True).start()

    def _client(self, c):
        realm, nonce = "NVR", "abc123"
        with c:
            while True:
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = c.recv(4096)
                    if not chunk:
                        return
                    data += chunk
                req = data.decode()
                cseq = re.search(r"CSeq: (\d+)", req).group(1)
                uri = req.split(" ")[1]
                m = re.search(r'Authorization: Digest (.*)', req)
                ch = int(re.search(r"channel=(\d+)", uri).group(1)) if "channel=" in uri else 1
                if not m:
                    code, extra = 401, f'WWW-Authenticate: Digest realm="{realm}", nonce="{nonce}"\r\n'
                else:
                    f = dict(re.findall(r'(\w+)="?([^",]+)"?', m.group(1)))
                    ha1 = hashlib.md5(f"{self.user}:{realm}:{self.pwd}".encode()).hexdigest()
                    ha2 = hashlib.md5(f"DESCRIBE:{uri}".encode()).hexdigest()
                    good = f.get("response") == hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
                    code, extra = (401, "") if not good else ((200, "") if ch in self.channels else (404, ""))
                c.sendall(f"RTSP/1.0 {code} X\r\nCSeq: {cseq}\r\n{extra}\r\n".encode())

    def close(self):
        self.sock.close()


@pytest.fixture
def server():
    s = FakeRtsp()
    yield s
    s.close()


def cfg(port, **kw):
    return cctv.RtspConfig(host="127.0.0.1", port=port, username="admin", **kw)


def test_config_validation_rejects_urls_and_credentials_in_host():
    for bad in ("rtsp://1.2.3.4", "admin:pw@1.2.3.4", "1.2.3.4/cam", "01J05HAR58G8S9T7 x"):
        with pytest.raises(ValidationError):
            cctv.RtspConfig(host=bad)
    with pytest.raises(ValidationError):
        cctv.RtspConfig(host="10.0.0.5", path_template="/x?user={username}&pw={password}")
    c = cctv.RtspConfig(host="10.0.0.5", channel=3, stream_type="main")
    assert c.path() == "/cam/realmonitor?channel=3&subtype=0"


def test_urls_are_masked_everywhere(monkeypatch):
    monkeypatch.setenv("CCTV_CAM1_PASSWORD", SECRET)
    c = cctv.RtspConfig(host="10.0.0.5", username="admin")
    full = cctv.build_url("cam1", c)
    assert SECRET not in cctv.mask_url(full) and "********" in cctv.mask_url(full)
    assert SECRET not in cctv.display_url(c)
    assert SECRET not in cctv.sanitize(f"failed to open {full} ({SECRET})", SECRET)


def test_connection_error_classes(server, monkeypatch):
    monkeypatch.delenv("CCTV_PASSWORD", raising=False)
    closed = socket.socket(); closed.bind(("127.0.0.1", 0)); port = closed.getsockname()[1]; closed.close()
    assert cctv.test_connection("t", cfg(port), SECRET, timeout=4).code == "unreachable"   # Windows retries SYN ~2 s
    assert cctv.test_connection("t", cfg(server.port), "wrong-password", timeout=2).code == "auth_failed"
    assert cctv.test_connection("t", cfg(server.port, channel=7), SECRET, timeout=2).code == "channel_unavailable"
    r = cctv.test_connection("t", cfg(server.port), SECRET, timeout=1)
    assert r.code == "stream_unavailable" and not r.ok          # authenticated, but a fake server sends no video
    assert [s["stage"] for s in r.stages] == ["network", "rtsp", "video"]
    assert cctv.test_connection("t", cfg(server.port), None, timeout=1).code == "invalid_config"  # no password
    for res in (r,):
        assert SECRET not in res.model_dump_json()


@pytest.fixture
def client(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(cctv, "ENV_PATH", tmp_path / ".env")
    monkeypatch.delenv("CCTV_LIVE1_PASSWORD", raising=False)
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        yield c


def test_save_connect_stop_without_leaking_password(client, cfg, tmp_path, server):
    body = {"name": "IFFCO CHOWK - CAM 1", "password": SECRET, "processing_fps": 5, "frame_skip": 1,
            "rtsp": {"host": "127.0.0.1", "port": server.port, "channel": 1, "stream_type": "sub", "username": "admin"}}
    assert client.put("/api/cctv/live1", json=body).status_code == 401            # admin only
    r = client.put("/api/cctv/live1", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    out = r.json()
    assert SECRET not in r.text and out["has_password"] and out["rtsp_url"].startswith("rtsp://admin:********@")
    assert out["enabled"] is False and out["source"] == "cctv:live1"
    assert out["settings"]["video"] == {"live_process_fps": 5, "live_frame_skip": 1}
    # the password lives only in the (gitignored) .env, never in the database
    assert f'CCTV_LIVE1_PASSWORD' in (tmp_path / ".env").read_text()
    with sqlite3.connect(cfg.db_path) as db:
        dump = "\n".join(db.iterdump())
    assert SECRET not in dump
    # API listings and events never contain it either
    for path in ("/api/cameras", "/api/cameras/live1", "/api/events?limit=50"):
        assert SECRET not in client.get(path).text
    # a 422 must not echo the body
    bad = client.put("/api/cctv/live1", json={**body, "rtsp": {"host": "rtsp://x"}}, headers=AUTH)
    assert bad.status_code == 422 and SECRET not in bad.text
    # test endpoint uses the stored password; the fake server sends no video -> honest failure
    t = client.post("/api/cctv/test", json={"camera_id": "live1", "rtsp": body["rtsp"]}, headers=AUTH).json()
    assert t["code"] == "stream_unavailable" and not t["ok"]
    # connect starts the worker (it will keep reconnecting: no real video), stop stops it
    assert client.post("/api/cctv/live1/connect", headers=AUTH).json()["enabled"] is True
    assert client.post("/api/cctv/live1/stop", headers=AUTH).json()["enabled"] is False


def test_connect_requires_configuration(client):
    client.put("/api/cctv/live2", json={"rtsp": {"host": "10.9.9.9", "username": "admin"}}, headers=AUTH)
    r = client.post("/api/cctv/live2/connect", headers=AUTH)
    assert r.status_code == 409 and "credentials" in r.json()["detail"]
