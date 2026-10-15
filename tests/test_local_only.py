"""API security Step 1: local-only safeguard (no login yet) and API docs disabled by default."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.local_only import is_loopback
from backend.core.config import AppConfig
from backend.main import create_app


@pytest.fixture
def strict_cfg(tmp_path: Path) -> AppConfig:
    """Production-like config: no extra trusted hosts, docs at their default (off)."""
    c = AppConfig()
    c.storage_dir = tmp_path / "storage"
    c.db_path = tmp_path / "storage" / "t.db"
    c.admin_token = "test-admin-token"
    c.start_cameras = False
    c.enable_api_docs = False
    c.local_extra_hosts = ()
    c.ensure_dirs()
    return c


def _client(cfg: AppConfig, host: str) -> TestClient:
    return TestClient(create_app(cfg), client=(host, 50000))


@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", True), ("127.5.5.5", True), ("::1", True), ("::ffff:127.0.0.1", True),
    ("192.168.1.20", False), ("10.0.0.5", False), ("0.0.0.0", False), ("::ffff:192.168.1.9", False),
    ("fe80::1", False), ("testclient", False), ("localhost", False), ("", False), (None, False),
])
def test_is_loopback(host, expected):
    assert is_loopback(host) is expected


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_local_requests_still_work(strict_cfg, host):
    with _client(strict_cfg, host) as c:
        r = c.get("/api/health")
        assert r.status_code == 200 and r.json()["status"] in ("ok", "degraded")
        assert c.get("/api/classes").status_code == 200


@pytest.mark.parametrize("path,method", [
    ("/api/health", "GET"), ("/api/alerts", "GET"), ("/api/alerts/1/snapshot", "GET"),
    ("/api/cameras/cam1/stream", "GET"), ("/api/cameras/cam1/frame.jpg", "GET"), ("/api/captures/x/image", "GET"),
    ("/api/datasets/ds/image?id=a.jpg", "GET"), ("/api/export/alerts.csv", "GET"), ("/api/stream", "GET"),
    ("/api/invoices/engine", "GET"), ("/api/alerts/1", "PATCH"), ("/api/detect/image", "POST"),
    ("/api/evaluations", "POST"), ("/docs", "GET"), ("/openapi.json", "GET"),
])
def test_remote_requests_are_rejected_before_any_route(strict_cfg, path, method):
    with _client(strict_cfg, "192.168.0.50") as c:
        r = c.request(method, path)
        assert r.status_code == 403
        assert "Remote access" in r.json()["detail"]


def test_remote_cors_preflight_is_rejected(strict_cfg):
    with _client(strict_cfg, "10.1.2.3") as c:
        r = c.options("/api/detect/image", headers={"Origin": "http://10.1.2.3:3010",
                                                     "Access-Control-Request-Method": "POST"})
        assert r.status_code == 403


def test_remote_rejection_does_not_touch_the_database(strict_cfg):
    with _client(strict_cfg, "192.168.0.50") as c:
        c.patch("/api/alerts/1", json={"status": "false_positive"})
        c.post("/api/detect/image", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    import sqlite3
    db = sqlite3.connect(strict_cfg.db_path)
    assert db.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM system_events WHERE source='alerts'").fetchone()[0] == 0


def test_api_docs_are_off_by_default(strict_cfg):
    with _client(strict_cfg, "127.0.0.1") as c:
        for p in ("/docs", "/redoc", "/openapi.json"):
            assert c.get(p).status_code == 404


def test_api_docs_can_be_enabled_locally_only(strict_cfg):
    strict_cfg.enable_api_docs = True
    with _client(strict_cfg, "127.0.0.1") as c:
        assert c.get("/docs").status_code == 200
        assert c.get("/openapi.json").json()["info"]["title"] == "CCTV Litter Monitor API"
    with _client(strict_cfg, "192.168.0.50") as c:
        assert c.get("/docs").status_code == 403


def test_docs_flag_reads_environment(monkeypatch):
    monkeypatch.delenv("LM_ENABLE_API_DOCS", raising=False)
    assert AppConfig().enable_api_docs is False
    monkeypatch.setenv("LM_ENABLE_API_DOCS", "true")
    assert AppConfig().enable_api_docs is True
    assert AppConfig().local_extra_hosts == ()          # never configurable from the environment
