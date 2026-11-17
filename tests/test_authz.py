"""API security Step 4: route permissions (backend/api/authz.py).

Synthetic accounts and passwords only; every test uses its own temporary storage and auth database, and a fake
clock where expiry matters. The permission matrix below is the reviewed one: changing a route's permission (or adding
a route) must change this table too.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api import authz, routes, routes_auth, routes_lab, routes_sampled
from backend.core.config import AppConfig
from backend.main import create_app

PASS = "correct horse battery"          # synthetic test password
TOKEN = "test-admin-token"
ROLES = ("viewer", "operator", "admin")
RANK = {"public": 0, "viewer": 1, "operator": 2, "admin": 3, "admin_if_login": 3}
EVIL = "http://localhost:5000"           # another local page: same-site, but not an allowed origin

# (method, path) -> permission. "admin" = ADMIN_API_TOKEN needed even with login off (as before Step 4);
# "admin_if_login" = admin when login is on, open (as before) when it is off.
MATRIX = {
    ("GET", "/api/health"): "public",
    ("POST", "/api/auth/login"): "public", ("POST", "/api/auth/logout"): "public", ("GET", "/api/auth/me"): "public",
    ("POST", "/api/auth/activity"): "public",
    ("GET", "/api/classes"): "viewer", ("GET", "/api/summary"): "viewer", ("GET", "/api/events"): "viewer",
    ("GET", "/api/cameras"): "viewer", ("GET", "/api/cameras/{cam_id}"): "viewer",
    ("POST", "/api/cameras"): "admin", ("PUT", "/api/cameras/{cam_id}"): "admin",
    ("DELETE", "/api/cameras/{cam_id}"): "admin",
    ("GET", "/api/cameras/{cam_id}/frame.jpg"): "viewer", ("GET", "/api/cameras/{cam_id}/stream"): "viewer",
    ("GET", "/api/cameras/{cam_id}/live"): "viewer", ("POST", "/api/cameras/{cam_id}/restart"): "admin",
    ("GET", "/api/cctv/defaults"): "admin_if_login", ("POST", "/api/cctv/test"): "admin",
    ("PUT", "/api/cctv/{cam_id}"): "admin", ("POST", "/api/cctv/{cam_id}/connect"): "admin",
    ("POST", "/api/cctv/{cam_id}/stop"): "admin",
    ("GET", "/api/schedule/{cam_id}"): "viewer", ("PUT", "/api/schedule/{cam_id}"): "admin",
    ("POST", "/api/schedule/{cam_id}/{action}"): "admin",
    ("GET", "/api/captures"): "viewer", ("GET", "/api/captures/{cap_id}/image"): "viewer",
    ("GET", "/api/roi/{camera_id}"): "viewer", ("PUT", "/api/roi/{camera_id}"): "operator",
    ("GET", "/api/detections"): "viewer",
    ("POST", "/api/detect/image"): "operator", ("POST", "/api/detect/video"): "operator",
    ("GET", "/api/jobs"): "viewer", ("GET", "/api/jobs/{job_id}"): "viewer",
    ("GET", "/api/jobs/{job_id}/files/{kind}"): "viewer",
    ("GET", "/api/alerts"): "viewer", ("GET", "/api/alerts/{alert_id}"): "viewer",
    ("PATCH", "/api/alerts/{alert_id}"): "operator", ("GET", "/api/alerts/{alert_id}/snapshot"): "viewer",
    ("GET", "/api/alerts/{alert_id}/timeline"): "viewer",
    ("GET", "/api/analytics"): "viewer", ("GET", "/api/models"): "viewer", ("POST", "/api/models/activate"): "admin",
    ("POST", "/api/models/register"): "admin",
    ("GET", "/api/settings"): "viewer", ("PUT", "/api/settings"): "admin",
    ("GET", "/api/auth/check"): "admin", ("GET", "/api/system/health"): "admin_if_login",
    ("GET", "/api/stream"): "viewer", ("GET", "/api/search"): "viewer",
    ("GET", "/api/export/{table}.csv"): "admin_if_login",
    ("GET", "/api/datasets"): "viewer", ("GET", "/api/datasets/{name}"): "viewer",
    ("GET", "/api/datasets/{name}/images"): "viewer", ("GET", "/api/datasets/{name}/image"): "viewer",
    ("GET", "/api/datasets/{name}/annotation"): "viewer", ("PUT", "/api/datasets/{name}/annotation"): "admin",
    ("POST", "/api/datasets/{name}/versions"): "admin", ("GET", "/api/datasets/{name}/export"): "admin",
    ("GET", "/api/experiments"): "viewer", ("GET", "/api/experiments/{exp_id}"): "viewer",
    ("GET", "/api/experiments/{exp_id}/artifacts/{name}"): "viewer",
    ("POST", "/api/evaluations"): "admin_if_login", ("GET", "/api/evaluations"): "viewer",
    ("GET", "/api/evaluations/{job_id}"): "viewer",
    ("POST", "/api/testlab/run"): "operator", ("POST", "/api/testlab/upload"): "operator",
    ("POST", "/api/testlab/reviews"): "operator",
    ("GET", "/api/hard-examples"): "viewer", ("PATCH", "/api/hard-examples/{hid}"): "operator",
    # four-screenshot video analysis (backend/api/routes_sampled.py)
    ("POST", "/api/sampled/media"): "operator", ("GET", "/api/sampled/media/{media_id}"): "viewer",
    ("GET", "/api/sampled/media/{media_id}/preview.jpg"): "viewer",
    ("POST", "/api/sampled/jobs"): "operator", ("GET", "/api/sampled/jobs"): "viewer",
    ("GET", "/api/sampled/jobs/{job_id}"): "viewer", ("PUT", "/api/sampled/jobs/{job_id}/roi"): "operator",
    ("POST", "/api/sampled/jobs/{job_id}/roi/approve"): "operator",
    ("POST", "/api/sampled/jobs/{job_id}/cancel"): "operator",
    ("GET", "/api/sampled/jobs/{job_id}/frames/{idx}/{kind}.jpg"): "viewer",
    ("GET", "/api/sampled/jobs/{job_id}/report.csv"): "admin_if_login",
    ("GET", "/api/sampled/jobs/{job_id}/reports/{name}"): "admin_if_login",
    ("GET", "/api/sampled/reports/overview"): "viewer",
    ("GET", "/api/sampled/reports/months/{month}.csv"): "admin_if_login",
    ("GET", "/api/sampled/reports/months/{month}"): "viewer",
    ("GET", "/api/sampled/reports/history"): "viewer", ("GET", "/api/sampled/reports/detections"): "viewer",
    ("GET", "/api/sampled/reports/hourly"): "viewer", ("GET", "/api/sampled/reports/all.csv"): "admin_if_login",
}
STREAMS = {("GET", "/api/stream")}      # infinite responses: checked separately
PARAMS = {"cam_id": "nocam", "camera_id": "nocam", "cap_id": "nocap", "job_id": "nojob", "kind": "video",
          "alert_id": "999999", "action": "start", "table": "alerts", "name": "nodataset", "exp_id": "noexp",
          "hid": "999999", "media_id": "0123456789abcdef", "idx": "1", "month": "2026-11"}


class Clock:
    def __init__(self) -> None:
        self.t = 1_800_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, minutes: float) -> None:
        self.t += minutes * 60


def _label(p: authz.Permission) -> str:
    if p.role is None:
        return "public"
    return "admin_if_login" if (p.role == "admin" and p.off_mode == "open") else p.role


def _url(path: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: PARAMS[m.group(1)], path)


def _cfg(tmp_path: Path, mode: str, token: Optional[str] = TOKEN) -> AppConfig:
    c = AppConfig()
    c.storage_dir = tmp_path / "storage"
    c.db_path = tmp_path / "storage" / "t.db"
    c.auth_db_path = tmp_path / "storage" / "auth.db"           # never the real storage/auth.db
    c.admin_token = token
    c.start_cameras = False
    c.auth_mode = mode
    c.ensure_dirs()
    return c


def _client(cfg: AppConfig) -> TestClient:
    return TestClient(create_app(cfg), client=("127.0.0.1", 50000))


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def on(tmp_path, clock):
    """Login on, one synthetic account per role, fake clock."""
    with _client(_cfg(tmp_path, "on")) as c:
        svc = c.app.state.auth
        svc.clock = clock
        for role in ROLES:
            svc.add_user(f"qa.{role}", role, PASS)
        yield c


def _login(c: TestClient, role: str) -> None:
    c.cookies.clear()
    r = c.post("/api/auth/login", json={"username": f"qa.{role}", "password": PASS})
    assert r.status_code == 200, r.text


def _hook_stream_rechecks(monkeypatch, on_recheck) -> list:
    """Run ``on_recheck(n, raw)`` before every periodic stream re-check (n = 1, 2, ...); returns the call log."""
    real, calls = authz._stream_session_valid, []

    def hooked(svc, raw):
        calls.append(raw)
        on_recheck(len(calls), raw)
        return real(svc, raw)

    monkeypatch.setattr(authz, "STREAM_RECHECK_SEC", 0.0)
    monkeypatch.setattr(authz, "_stream_session_valid", hooked)
    return calls


def _activity(c: TestClient, origin: Optional[str] = "default"):
    """What the browser sends on user input (allowed Origin by default)."""
    o = c.app.state.auth_cfg.cors_origins[0] if origin == "default" else origin
    return c.post("/api/auth/activity", headers={"Origin": o} if o else {})


def _call(c: TestClient, method: str, path: str, **kw):
    return c.request(method, _url(path), **kw)


# ---------------------------------------------------------------- the matrix itself
def test_every_route_declares_exactly_one_permission_and_matches_the_reviewed_matrix():
    found = {}
    for prefix, router in (("/api", routes.router), ("/api", routes_lab.router), ("", routes_auth.router),
                           ("", routes_sampled.router)):
        for r in router.routes:
            perms = [d.dependency for d in r.dependencies if isinstance(d.dependency, authz.Permission)]
            assert len(perms) == 1, f"{r.path} must declare exactly one permission, has {perms}"
            for m in r.methods:
                found[(m, r.path if r.path.startswith("/api") else prefix + r.path)] = _label(perms[0])
    assert found == MATRIX


def test_the_app_registers_no_route_outside_the_matrix(tmp_path):
    with _client(_cfg(tmp_path, "off")) as c:
        ops = {(m.upper(), p) for p, v in c.app.openapi()["paths"].items() for m in v}
    assert ops == set(MATRIX)
    assert not [p for _, p in ops if "invoice" in p]


# ---------------------------------------------------------------- login off (the default): unchanged behaviour
def test_login_is_off_by_default(monkeypatch):
    monkeypatch.delenv("LM_AUTH_MODE", raising=False)
    assert AppConfig().auth_mode == "off"


def test_login_off_keeps_pre_step4_behaviour(tmp_path):
    cfg = _cfg(tmp_path, AppConfig().auth_mode)
    with _client(cfg) as c:
        for (method, path), perm in MATRIX.items():
            if (method, path) in STREAMS:
                continue
            r = _call(c, method, path)
            if perm == "admin":
                assert r.status_code == 401, (method, path)                       # token still required
                r = _call(c, method, path, headers={"Authorization": f"Bearer {TOKEN}"})
                assert r.status_code not in (401, 403), (method, path, r.status_code)
            else:
                assert r.status_code not in (401, 403), (method, path, r.status_code)
        assert c.get("/api/export/alerts.csv").status_code == 200                  # open, as before
        h = c.get("/api/health").json()
        assert {"model", "cameras", "admin_enabled", "system"} <= set(h)          # full details, as before
    assert not cfg.auth_db_path.exists()


def test_login_off_without_a_token_still_disables_admin_routes(tmp_path):
    with _client(_cfg(tmp_path, "off", token=None)) as c:
        assert c.get("/api/auth/check").status_code == 503
        assert c.put("/api/settings", json={}).status_code == 503
        assert c.get("/api/alerts").status_code == 200


# ---------------------------------------------------------------- login on: 401 / 403 everywhere
def test_unauthenticated_requests_get_401_on_every_protected_route(on):
    on.cookies.clear()
    for (method, path), perm in MATRIX.items():
        r = _call(on, method, path)
        if perm == "public":
            assert r.status_code != 401 or path == "/api/auth/login", (method, path)
        else:
            assert r.status_code == 401, (method, path, r.status_code)
            assert r.json()["detail"].startswith("Authentication required")


@pytest.mark.parametrize("role", ROLES)
def test_role_boundaries_on_every_route(on, role):
    _login(on, role)
    for (method, path), perm in MATRIX.items():
        if perm == "public" or (method, path) in STREAMS:
            continue
        r = _call(on, method, path)
        if RANK[role] >= RANK[perm]:
            assert r.status_code not in (401, 403), (role, method, path, r.status_code, r.text[:200])
        else:
            assert r.status_code == 403, (role, method, path, r.status_code)
            assert f"needs the {'admin' if perm.startswith('admin') else perm} role" in r.json()["detail"]


def test_admin_token_is_an_admin_with_login_on_and_nothing_more(on):
    on.cookies.clear()
    for (method, path), perm in MATRIX.items():
        if perm == "public" or (method, path) in STREAMS:
            continue
        r = _call(on, method, path, headers={"X-Admin-Token": TOKEN})
        assert r.status_code not in (401, 403), (method, path, r.status_code)
    assert on.get("/api/alerts", headers={"Authorization": "Bearer wrong-token"}).status_code == 401
    assert on.get("/api/auth/me", headers={"Authorization": f"Bearer {TOKEN}"}).json()["role"] == "admin"


def test_login_on_without_a_configured_token_needs_a_session(tmp_path):
    with _client(_cfg(tmp_path, "on", token=None)) as c:
        c.app.state.auth.add_user("qa.admin", "admin", PASS)
        assert c.get("/api/auth/check").status_code == 401                         # not 503: login is available
        assert c.get("/api/auth/check", headers={"Authorization": "Bearer anything"}).status_code == 401
        _login(c, "admin")
        assert c.get("/api/auth/check").status_code == 200


# ---------------------------------------------------------------- specific decisions
def test_csv_exports_are_admin_only(on):
    on.cookies.clear()
    assert on.get("/api/export/alerts.csv").status_code == 401
    for role in ("viewer", "operator"):
        _login(on, role)
        assert on.get("/api/export/alerts.csv").status_code == 403
        assert on.get("/api/export/detections.csv").status_code == 403
    _login(on, "admin")
    r = on.get("/api/export/detections.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    on.cookies.clear()
    assert on.get("/api/export/alerts.csv", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_roi_operator_for_uploads_admin_for_registered_cameras(on):
    cam = {"id": "cam1", "name": "Cam 1", "source": "0", "enabled": False}
    assert on.post("/api/cameras", json=cam, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 201
    _login(on, "operator")
    assert on.put("/api/roi/upload", json={"zones": []}).status_code == 200
    r = on.put("/api/roi/cam1", json={"zones": []})
    assert r.status_code == 403 and "admin" in r.json()["detail"]
    _login(on, "admin")
    assert on.put("/api/roi/cam1", json={"zones": []}).status_code == 200


def test_health_says_little_to_anonymous_callers(on):
    on.cookies.clear()
    r = on.get("/api/health")
    assert r.status_code == 200 and set(r.json()) == {"status", "time"}
    assert set(on.get("/api/health", headers={"Authorization": "Bearer wrong"}).json()) == {"status", "time"}
    _login(on, "viewer")
    h = on.get("/api/health").json()
    assert {"model", "cameras", "notifications"} <= set(h)                          # what the dashboard needs
    assert not {"system", "hardware", "admin_enabled", "uptime_sec"} & set(h)      # host details: admin only
    assert h["model"] is None or "path" not in h["model"]
    on.cookies.clear()
    full = on.get("/api/health", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert {"model", "system", "hardware", "admin_enabled"} <= set(full)
    _login(on, "operator")
    assert on.get("/api/system/health").status_code == 403                          # detailed diagnostics: admin
    _login(on, "admin")
    assert on.get("/api/system/health").status_code == 200


NVR_HOST, NVR_USER, CAM_HOST, CAM_SECRET = "nvr.example.test", "qa-nvr-user", "cam.example.test", "qa-cam-secret"


def _add_cameras(c: TestClient) -> None:
    """One Live CCTV camera (NVR host + username, no password) and one plain RTSP camera (credentials in its URL)."""
    admin = {"Authorization": f"Bearer {TOKEN}"}
    r = c.put("/api/cctv/nvr1", json={"name": "NVR 1", "rtsp": {"host": NVR_HOST, "username": NVR_USER}}, headers=admin)
    assert r.status_code == 200, r.text
    cam = {"id": "cam1", "name": "Cam 1", "source": f"rtsp://qa:{CAM_SECRET}@{CAM_HOST}:554/stream1", "enabled": False}
    assert c.post("/api/cameras", json=cam, headers=admin).status_code == 201


@pytest.mark.parametrize("role", ["viewer", "operator"])
def test_non_admins_get_no_camera_infrastructure_details(on, role):
    _add_cameras(on)
    _login(on, role)
    for r in (on.get("/api/cameras"), on.get("/api/cameras/nvr1"), on.get("/api/cameras/cam1")):
        assert r.status_code == 200
        for secret in (NVR_HOST, NVR_USER, CAM_HOST, CAM_SECRET, "stream1"):
            assert secret not in r.text, (role, secret)
    cams = {c["id"]: c for c in on.get("/api/cameras").json()["cameras"]}
    assert cams["nvr1"]["source"] == "cctv:nvr1" and cams["nvr1"]["rtsp_configured"] is True
    assert "rtsp" not in cams["nvr1"] and "rtsp_url" not in cams["nvr1"] and "has_password" not in cams["nvr1"]
    assert cams["cam1"]["source"] == "rtsp://(hidden)" and cams["cam1"]["rtsp_configured"] is False
    assert cams["cam1"]["name"] == "Cam 1" and "status" in cams["cam1"]                 # what the pages need


def test_admins_and_login_off_still_get_full_camera_details(on, tmp_path):
    _add_cameras(on)
    _login(on, "admin")
    cams = {c["id"]: c for c in on.get("/api/cameras").json()["cameras"]}
    assert cams["nvr1"]["rtsp"]["host"] == NVR_HOST and NVR_USER in cams["nvr1"]["rtsp_url"]
    assert CAM_HOST in cams["cam1"]["source"]
    for c in cams.values():
        assert CAM_SECRET not in str(c)                                                 # passwords: never
    with _client(_cfg(tmp_path / "off", "off")) as off:
        _add_cameras(off)
        cams = {c["id"]: c for c in off.get("/api/cameras").json()["cameras"]}          # unchanged behaviour
        assert cams["nvr1"]["rtsp"]["host"] == NVR_HOST and CAM_HOST in cams["cam1"]["source"]
        assert CAM_SECRET not in off.get("/api/cameras").text


def test_model_file_paths_are_relative_for_non_admins(on):
    _login(on, "viewer")
    models = on.get("/api/models").json()["models"]
    for m in models:
        assert not Path(m["path"]).is_absolute() and ":" not in m["path"], m["path"]
    _login(on, "admin")
    admin_models = on.get("/api/models").json()["models"]
    assert all(Path(m["path"]).is_absolute() for m in admin_models)
    assert [m["version"] for m in models] == [m["version"] for m in admin_models]


def test_evaluations_are_admin_only_with_login_on_and_detection_stays_operator(on, tmp_path):
    body = {"dataset": "nodataset", "split": "test"}
    for role in ("viewer", "operator"):
        _login(on, role)
        r = on.post("/api/evaluations", json=body)
        assert r.status_code == 403 and "needs the admin role" in r.json()["detail"]
    _login(on, "operator")
    assert on.post("/api/detect/image").status_code == 422                          # authorised; no file sent
    _login(on, "admin")
    assert on.post("/api/evaluations", json=body).status_code not in (401, 403)
    with _client(_cfg(tmp_path / "off", "off")) as off:                             # login off: as before
        assert off.post("/api/evaluations", json=body).status_code not in (401, 403)


# ---------------------------------------------------------------- free text a viewer can read
# Realistic messages in the shapes the application writes (camera worker, scheduler, alerts, evaluation, notifier),
# carrying infrastructure details in their exception text, plus one message the allowlist does not know.
LEAKY = [  # (source, level, message, what a non-admin must see instead)
    ("camera", "warning", f"camera disconnected: rtsp://qa:***@{NVR_HOST}:554/cam/realmonitor timed out; retrying",
     "camera disconnected; retrying (details visible to administrators)"),
    ("camera", "error", r"processing error: weights not found: C:\SHREE\YATRIKART\models\litter_yolo\v9\best.pt",
     "processing error (details visible to administrators)"),
    ("schedule", "warning", f"capture failed: login rejected for {NVR_USER} at 192.168.1.20:554",
     "capture failed (details visible to administrators)"),
    ("schedule", "warning", f"capture skipped: auth_failed - {NVR_USER} rejected by {NVR_HOST}",
     "capture skipped: auth_failed (details visible to administrators)"),
    ("alerts", "error", "snapshot save failed: [Errno 13] /srv/ldm/storage/alerts/a.jpg",
     "snapshot save failed (details visible to administrators)"),
    ("evaluation", "error", r"evaluation job_9 failed: cannot read \\fileserver\cctv\eval",
     "evaluation job_9 failed (details visible to administrators)"),
    ("notifier", "error", "smtp: connection to mail.corp.example refused",
     "notification delivery failed (details visible to administrators)"),
    ("camera", "warning", f"an unexpected new message mentioning {NVR_HOST} and fe80::1%eth0",
     "camera warning (details visible to administrators)"),
]
LEAKED_BITS = [NVR_HOST, NVR_USER, "192.168.1.20", "SHREE", "fileserver", "/srv/ldm", "realmonitor", "corp.example",
               "fe80::1", "Errno"]
JOB_ERROR = r"model load failed: C:\SHREE\YATRIKART\models\v9\best.pt"


def _leaky_data(c: TestClient) -> None:
    _add_cameras(c)
    db = c.app.state.app_state.db
    for source, level, msg, _ in LEAKY:
        db.log_event(level, source, msg, "nvr1", data={"detail": msg})
    db.create_job("job_leak", "evaluation", None, "eval", "unused")
    db.update_job("job_leak", status="failed", error=JOB_ERROR)


@pytest.mark.parametrize("role", ["viewer", "operator"])
def test_free_text_hides_hosts_paths_and_users_from_non_admins(on, role):
    _leaky_data(on)
    _login(on, role)
    for path in ("/api/events", "/api/jobs", "/api/jobs/job_leak", "/api/evaluations"):
        r = on.get(path)
        assert r.status_code == 200, path
        for bit in LEAKED_BITS:
            assert bit not in r.text, (role, path, bit)
    events = on.get("/api/events").json()["events"]
    messages = [e["message"] for e in events]
    for _, _, _, expected in LEAKY:
        assert expected in messages, expected                                      # safe prefix kept
    assert "camera created" in messages and "CCTV configuration saved" in messages  # known messages unchanged
    assert all(e["data_json"] is None for e in events)                             # event details: admins only
    assert on.get("/api/jobs/job_leak").json()["error"] == "error (details visible to administrators)"


def test_admins_and_login_off_see_the_original_text(on, tmp_path):
    originals = {msg for _, _, msg, _ in LEAKY}
    _leaky_data(on)
    _login(on, "admin")
    events = on.get("/api/events").json()["events"]
    assert {e["message"] for e in events} >= originals
    assert any(e["data_json"] for e in events)
    assert on.get("/api/jobs/job_leak").json()["error"] == JOB_ERROR
    with _client(_cfg(tmp_path / "off", "off")) as off:
        _leaky_data(off)
        assert {e["message"] for e in off.get("/api/events").json()["events"]} >= originals


def test_every_event_message_the_application_writes_is_classified():
    """Each message shape written by a log_event call (backend/) maps to a deliberate non-admin text: shown as written
    (fixed text / identifiers), safe prefix only, or the generic fallback. Guards against a template that leaks."""
    from backend.api.authz import DETAILS_HIDDEN, safe_event_message as safe
    shown = [("camera", "info", "camera created"), ("camera", "info", "camera updated (name, settings)"),
             ("camera", "info", "camera deleted"), ("camera", "info", "camera reconnected"),
             ("cctv", "warning", "connection test: auth_failed"), ("cctv", "info", "CCTV configuration saved"),
             ("cctv", "info", "live stream started"), ("cctv", "info", "live stream stopped"),
             ("roi", "info", "ROI saved (2 zone(s))"), ("roi", "info", "ROI reset (0 zone(s))"),
             ("alerts", "info", "alert 12 -> false_positive"), ("camera", "info", "stream restart requested"),
             ("dataset", "info", "labels updated: ds_v1/images/train/a_01.jpg (3 objects)"),
             ("dataset", "info", "dataset version ds_v2 created from ds_v1"),
             ("models", "info", "registered model v4_test from exp_7"),
             ("schedule", "info", "capture schedule started (next 2026-10-17T09:30:00+00:00)"),
             ("schedule", "info", "capture schedule paused"), ("schedule", "info", "capture schedule resumed"),
             ("schedule", "info", "capture schedule stopped"),
             ("schedule", "info", "missed capture while offline: running once shortly"),
             ("system", "info", "backend started"), ("system", "info", "backend stopped"),
             ("notifier", "info", "alert 7 delivered via telegram"),
             ("settings", "info", "global detection settings updated"), ("models", "info", "activated model v3"),
             ("models", "info", "applied recommended settings of v3"),
             ("evaluation", "info", "evaluation job_1 finished for v3_store_cctv")]
    for source, level, msg in shown:
        assert safe(source, level, msg) == msg, msg
    # identifiers are restricted: a path or URL smuggled into a "shown" template is not shown
    for source, level, msg in [("dataset", "info", r"labels updated: ds/C:\x\a.jpg (1 objects)"),
                               ("dataset", "info", "labels updated: ds//srv/a.jpg (1 objects)"),
                               ("models", "info", "activated model http://evil.example/x"),
                               ("cctv", "info", "connection test: failed for 10.0.0.9")]:
        assert safe(source, level, msg).endswith(DETAILS_HIDDEN), msg


def test_live_event_stream_hides_infrastructure_from_viewers(on, monkeypatch):
    _add_cameras(on)
    _login(on, "viewer")
    svc, db = on.app.state.auth, on.app.state.app_state.db
    source, level, msg, expected = LEAKY[0]

    def emit_then_end(n, raw):
        if n == 1:
            db.log_event(level, source, msg, "nvr1", data={"detail": msg})          # arrives while the stream is open
        else:
            svc.logout(raw)

    _hook_stream_rechecks(monkeypatch, emit_then_end)
    r = on.get("/api/stream")
    assert r.status_code == 200 and expected in r.text
    for bit in LEAKED_BITS:
        assert bit not in r.text


def test_api_responses_carry_a_content_security_policy(on, tmp_path):
    """The API serves no web pages: every response says 'run nothing, frame nothing' and 'do not guess the type'."""
    from backend.api.security_headers import API_CSP
    _login(on, "viewer")
    for path in ("/api/health", "/api/alerts", "/api/auth/me", "/api/does-not-exist"):
        r = on.get(path)
        assert r.headers.get("content-security-policy") == API_CSP, path
        assert r.headers.get("x-content-type-options") == "nosniff", path
    assert "default-src 'none'" in API_CSP and "frame-ancestors 'none'" in API_CSP and "script" not in API_CSP
    on.cookies.clear()
    assert on.get("/api/alerts").headers.get("content-security-policy") == API_CSP   # also on 401
    cfg = _cfg(tmp_path / "docs", "off")
    cfg.enable_api_docs = True
    with _client(cfg) as c:                                                      # optional docs pages: exempt
        r = c.get("/docs")
        assert r.status_code == 200 and "content-security-policy" not in r.headers
        assert r.headers.get("x-content-type-options") == "nosniff"
        assert c.get("/api/health").headers.get("content-security-policy") == API_CSP


def test_login_and_logout_refuse_other_origins(on):
    """Login / logout CSRF: another page (e.g. another port on localhost) must not sign the browser in or out."""
    allowed = on.app.state.auth_cfg.cors_origins[0]
    on.cookies.clear()
    body = {"username": "qa.viewer", "password": PASS}
    for origin in (EVIL, "null"):
        r = on.post("/api/auth/login", json=body, headers={"Origin": origin})
        assert r.status_code == 403 and "set-cookie" not in r.headers, origin
    assert on.post("/api/auth/login", json=body, headers={"Origin": allowed}).status_code == 200
    sid = on.cookies.get("ldm_session")
    for origin in (EVIL, "null"):
        assert on.post("/api/auth/logout", headers={"Origin": origin}).status_code == 403, origin
        assert on.get("/api/alerts").status_code == 200                           # still signed in
    assert on.post("/api/auth/logout", headers={"Origin": allowed}).status_code == 204
    on.cookies.clear()
    on.cookies.set("ldm_session", sid, path="/api")
    assert on.get("/api/alerts").status_code == 401                               # the session really ended
    _login(on, "viewer")                                                          # no Origin (non-browser): allowed
    assert on.post("/api/auth/logout").status_code == 204


def test_redaction_patterns_hide_addresses_and_paths_but_keep_ordinary_text():
    from backend.api.authz import _IPV4_RE, _IPV6_RE, _POSIX_PATH_RE, _URL_RE, _WIN_PATH_RE, HIDDEN

    def red(t: str) -> str:
        for rx in (_URL_RE, _WIN_PATH_RE, _POSIX_PATH_RE, _IPV6_RE, _IPV4_RE):
            t = rx.sub(HIDDEN, t)
        return t

    hidden = ["rtsp://u:***@10.1.2.3:554/x", r"C:\SHREE\models\best.pt", "C:/data/a.jpg", r"\\fileserver\cctv\a.mp4",
              "/srv/ldm/storage/a.mp4", "192.168.1.20:554", "fe80::1%eth0", "[2001:db8::7]:554", "::1",
              "2001:0db8:85a3:0000:0000:8a2e:0370:7334"]
    for h in hidden:
        assert h not in red(f"failed: {h} (timeout)") and HIDDEN in red(f"failed: {h} (timeout)"), h
    for keep in ["checked at 12:30:45", "ratio 16:9", "2026-10-17T05:26:50+00:00", "0/1 enabled cameras online",
                 "alert 5 -> resolved", "mAP 0.512 for v3_store_cctv", "camera created"]:
        assert red(keep) == keep, keep


def test_dataset_and_evaluation_documents_hide_server_paths_from_non_admins(tmp_path):
    """Dataset README / data.yaml and evaluation result files are read from disk and can contain absolute paths."""
    secret_dir = r"C:\SHREE\private\recordings"
    ds_root = tmp_path / "datasets" / "qa_ds"
    (ds_root / "images" / "train").mkdir(parents=True)
    (ds_root / "data.yaml").write_text(f"path: {ds_root.as_posix()}\ntrain: images/train\nval: images/train\n"
                                       "names: ['plastic_bag']\n", encoding="utf-8")
    (ds_root / "README.md").write_text(f"Synthetic test set built from {secret_dir}\\clip1.mp4 on 10.0.0.9\n",
                                       encoding="utf-8")
    cfg = _cfg(tmp_path, "on")
    cfg.dataset_dir = tmp_path / "datasets"
    (cfg.storage_dir / "evaluations").mkdir(parents=True, exist_ok=True)
    (cfg.storage_dir / "evaluations" / "evalqa1.json").write_text(
        '{"model_path": "C:\\\\SHREE\\\\models\\\\v9\\\\best.pt", "dataset": "qa_ds", "overall": {"f1": 0.5}}',
        encoding="utf-8")
    with _client(cfg) as c:
        for role in ROLES:
            c.app.state.auth.add_user(f"qa.{role}", role, PASS)
        for role in ("viewer", "operator"):
            _login(c, role)
            for path in ("/api/datasets", "/api/datasets/qa_ds", "/api/evaluations/evalqa1"):
                r = c.get(path)
                assert r.status_code == 200, (path, r.text[:200])
                for bit in ("SHREE", "private", "10.0.0.9", str(tmp_path), tmp_path.as_posix()):
                    assert bit not in r.text, (role, path, bit)
            ds = c.get("/api/datasets").json()["datasets"][0]
            assert ds["name"] == "qa_ds" and "Synthetic" in (ds.get("readme") or "")    # UI still detects it
            assert c.get("/api/evaluations/evalqa1").json()["overall"] == {"f1": 0.5}
        _login(c, "admin")
        assert "SHREE" in c.get("/api/evaluations/evalqa1").text                         # admins: unchanged


# ---------------------------------------------------------------- camera frames
def test_only_admins_make_the_server_connect_to_a_camera(on, tmp_path):
    cam = {"id": "camx", "name": "Cam X", "source": "env:LDM_TEST_UNSET_CAMERA_URL", "enabled": False}
    assert on.post("/api/cameras", json=cam, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 201
    for role in ("viewer", "operator"):
        _login(on, role)
        r = on.get("/api/cameras/camx/frame.jpg")
        assert r.status_code == 403 and "admin" in r.json()["detail"]
        assert on.get("/api/cameras/camx/frame.jpg?raw=1").status_code == 403
    _login(on, "admin")
    assert on.get("/api/cameras/camx/frame.jpg").status_code == 503               # tried: source not configured
    with _client(_cfg(tmp_path / "off", "off")) as off:                             # login off: as before
        assert off.post("/api/cameras", json=cam, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 201
        assert off.get("/api/cameras/camx/frame.jpg").status_code == 503


# ---------------------------------------------------------------- sessions
def test_invalid_session_cookies_are_401(on):
    for raw in ("not-a-session", "x" * 500, ""):
        on.cookies.clear()
        on.cookies.set("ldm_session", raw, path="/api")
        assert on.get("/api/alerts").status_code == 401


POLLING = ("/api/alerts", "/api/health", "/api/auth/me", "/api/summary", "/api/cameras", "/api/events")


def test_polling_and_other_requests_never_extend_the_idle_timeout(on, clock):
    """Approved policy (all roles): background polling, health/status, /me and any other request only check."""
    _login(on, "viewer")
    for _ in range(5):                                         # a dashboard polling every few minutes...
        clock.advance(5.9)
        for path in POLLING:
            assert on.get(path).status_code == 200, path
    clock.advance(0.6)                                         # ...30.1 min after login, without user input
    assert on.get("/api/alerts").status_code == 401
    assert on.get("/api/auth/me").json()["authenticated"] is False


def test_writes_do_not_extend_it_either(on, clock):
    _login(on, "operator")
    clock.advance(20)
    assert on.patch("/api/alerts/999999", json={"status": "resolved"}).status_code == 404   # allowed, but no refresh
    assert on.put("/api/roi/upload", json={"zones": []}).status_code == 200
    clock.advance(10.1)
    assert on.get("/api/alerts").status_code == 401


def test_user_activity_extends_the_idle_timeout(on, clock):
    _login(on, "viewer")
    clock.advance(25)
    r = _activity(on)
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    assert set(r.json()) == {"idle_expires_at"}                                     # no session ID, nothing else
    clock.advance(25)                                                               # 50 min after login
    assert on.get("/api/alerts").status_code == 200
    clock.advance(5.1)                                                              # 30.1 min after the activity
    assert on.get("/api/alerts").status_code == 401
    assert _activity(on).status_code == 401                                         # expired: activity cannot revive it


@pytest.mark.parametrize("role", ROLES)
def test_the_idle_policy_is_the_same_for_every_role(on, clock, role):
    _login(on, role)
    clock.advance(31)
    assert on.get("/api/summary").status_code == 401


def test_activity_cannot_be_reported_by_arbitrary_requests(on, clock):
    """Only a POST with an allowed browser Origin and a valid session counts; nothing else can stand in for it."""
    on.cookies.clear()
    assert _activity(on).status_code == 401                                         # no session
    _login(on, "admin")
    assert _activity(on, origin=None).status_code == 403                            # no Origin (non-browser)
    assert _activity(on, origin=EVIL).status_code == 403                            # another local page
    assert _activity(on, origin="null").status_code == 403
    assert on.get("/api/auth/activity").status_code == 405                          # GET is not activity
    clock.advance(30.1)                                                             # refused reports extended nothing
    assert on.get("/api/alerts").status_code == 401
    on.cookies.clear()                                                              # the admin token has no session
    assert on.post("/api/auth/activity", headers={"Authorization": f"Bearer {TOKEN}",
                                                  "Origin": on.app.state.auth_cfg.cors_origins[0]}).status_code == 401


def test_activity_is_recorded_at_most_twice_a_minute(on, clock):
    _login(on, "viewer")
    clock.advance(0.25)                                        # 15 s after sign-in: nothing new to record
    first = _activity(on).json()["idle_expires_at"]
    clock.advance(0.2)                                         # 27 s: still within the 30 s recording interval
    assert _activity(on).json()["idle_expires_at"] == first
    clock.advance(0.1)                                         # 33 s: recorded, deadline moves on
    assert _activity(on).json()["idle_expires_at"] > first


def test_activity_never_extends_the_12_hour_limit(on, clock):
    _login(on, "viewer")
    login_t = clock.t
    for _ in range(int(12 * 60 / 25)):                                             # active all day...
        clock.advance(25)
        r = _activity(on)
        assert r.status_code == 200
        idle = r.json()["idle_expires_at"]
        assert idle <= _iso_utc(login_t + 12 * 3600)                               # never past the absolute limit
        assert on.get("/api/alerts").status_code == 200
    clock.advance(25)                                                               # 12 h 25 min after login
    assert _activity(on).status_code == 401
    assert on.get("/api/alerts").status_code == 401                                 # ...still ends after 12 h


def _iso_utc(t: float) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="seconds")


def test_login_off_has_no_activity_endpoint(tmp_path):
    with _client(_cfg(tmp_path, "off")) as c:
        assert c.post("/api/auth/activity", headers={"Origin": "http://localhost:3000"}).status_code == 404


@pytest.mark.parametrize("revoke", ["logout", "disable", "password", "role"])
def test_revoked_sessions_are_401(on, revoke):
    _login(on, "operator")
    sid = on.cookies.get("ldm_session")
    svc = on.app.state.auth
    if revoke == "logout":
        assert on.post("/api/auth/logout").status_code == 204
    elif revoke == "disable":
        svc.set_disabled("qa.operator", True)
    elif revoke == "password":
        svc.set_password("qa.operator", "a brand new passphrase")
    else:
        svc.set_role("qa.operator", "viewer")
    on.cookies.clear()
    on.cookies.set("ldm_session", sid, path="/api")                               # replay the old cookie
    assert on.get("/api/alerts").status_code == 401
    assert on.patch("/api/alerts/1", json={"status": "resolved"}).status_code == 401


def test_role_change_takes_effect_at_next_login(on):
    on.app.state.auth.set_role("qa.viewer", "operator")
    _login(on, "viewer")                                                         # username qa.viewer, now operator
    assert on.patch("/api/alerts/999999", json={"status": "resolved"}).status_code == 404


def test_lockout_blocks_login_and_protected_routes_stay_closed(on, clock):
    on.cookies.clear()
    for _ in range(5):
        assert on.post("/api/auth/login", json={"username": "qa.admin", "password": "wrong password!"}).status_code == 401
    r = on.post("/api/auth/login", json={"username": "qa.admin", "password": PASS})
    assert r.status_code == 429 and "set-cookie" not in r.headers
    assert on.get("/api/auth/check").status_code == 401
    clock.advance(16)
    _login(on, "admin")
    assert on.get("/api/auth/check").status_code == 200


# ---------------------------------------------------------------- cross-site writes
def test_cookie_writes_from_other_origins_are_refused(on):
    _login(on, "operator")
    body = {"status": "resolved"}
    r = on.patch("/api/alerts/999999", json=body, headers={"Origin": EVIL})
    assert r.status_code == 403 and r.json()["detail"] == "Request origin not allowed"
    assert on.patch("/api/alerts/999999", json=body, headers={"Origin": "null"}).status_code == 403
    allowed = on.app.state.auth_cfg.cors_origins[0]
    assert on.patch("/api/alerts/999999", json=body, headers={"Origin": allowed}).status_code == 404
    assert on.patch("/api/alerts/999999", json=body).status_code == 404                # no Origin: non-browser
    assert on.get("/api/alerts", headers={"Origin": EVIL}).status_code == 200          # reads are not writes
    on.cookies.clear()                                                         # header auth cannot be forged
    r = on.post("/api/models/activate", json={"version": "none"}, headers={"Authorization": f"Bearer {TOKEN}",
                                                                           "Origin": EVIL})
    assert r.status_code == 404


# ---------------------------------------------------------------- streams
def test_streams_need_a_session(on):
    on.cookies.clear()
    assert on.get("/api/stream").status_code == 401
    assert on.get("/api/cameras/nocam/stream").status_code == 401


def test_event_stream_ends_when_the_session_ends(on, monkeypatch):
    """The stream re-checks the session; once that check fails the server ends the otherwise endless response."""
    _login(on, "viewer")
    svc = on.app.state.auth
    calls = _hook_stream_rechecks(monkeypatch, lambda n, raw: n > 1 and svc.logout(raw))
    r = on.get("/api/stream")                     # returns only because the server closed the stream
    assert r.status_code == 200 and "event: summary" in r.text
    assert len(calls) == 2 and all(x == calls[0] for x in calls)
    assert on.get("/api/alerts").status_code == 401


def test_event_stream_ends_when_the_session_times_out(on, clock, monkeypatch):
    """No logout: the session simply passes its idle timeout while the stream is open."""
    _login(on, "viewer")
    calls = _hook_stream_rechecks(monkeypatch, lambda n, raw: n == 2 and clock.advance(31))
    r = on.get("/api/stream")
    assert r.status_code == 200 and len(calls) == 2
    assert on.get("/api/alerts").status_code == 401


class _FakeCameraWorker:
    """Stands in for a running camera worker (no real camera, no network): a constant JPEG in memory."""
    latest_jpeg = b"\xff\xd8\xff\xe0qa-frame\xff\xd9"

    def stop(self) -> None:
        pass


@pytest.mark.parametrize("ending", ["logout", "idle"])
def test_mjpeg_camera_stream_closes_when_the_session_ends(on, clock, monkeypatch, ending):
    workers = on.app.state.app_state.cameras.workers
    workers["qafake"] = _FakeCameraWorker()
    try:
        _login(on, "viewer")
        svc = on.app.state.auth

        def end_session(n, raw):
            if n == 2:
                svc.logout(raw) if ending == "logout" else clock.advance(31)

        calls = _hook_stream_rechecks(monkeypatch, end_session)
        r = on.get("/api/cameras/qafake/stream")    # returns only because the server closed the stream
        assert r.status_code == 200 and r.headers["content-type"].startswith("multipart/x-mixed-replace")
        assert b"qa-frame" in r.content and len(calls) == 2
        assert on.get("/api/cameras/qafake/stream").status_code == 401
    finally:
        workers.pop("qafake", None)


def test_open_streams_do_not_keep_a_session_alive(on, clock, monkeypatch):
    """A watched-but-untouched dashboard: the stream keeps re-checking, and the session still times out."""
    _login(on, "viewer")
    calls = _hook_stream_rechecks(monkeypatch, lambda n, raw: clock.advance(10))  # 10 min between re-checks
    r = on.get("/api/stream")
    assert r.status_code == 200
    assert len(calls) == 3                         # 10, 20 min: still valid; 30 min: timed out -> stream closed
    assert on.get("/api/alerts").status_code == 401


# ---------------------------------------------------------------- litter-detection workflows still work
def test_viewer_and_operator_workflows_with_login_on(on):
    _login(on, "viewer")
    for path in ("/api/alerts", "/api/cameras", "/api/analytics", "/api/detections", "/api/summary",
                 "/api/classes", "/api/settings", "/api/models", "/api/jobs"):
        assert on.get(path).status_code == 200, path
    _login(on, "operator")
    img = np.full((240, 320, 3), 120, np.uint8)
    cv2.rectangle(img, (100, 150), (140, 180), (230, 230, 230), -1)
    jpg = cv2.imencode(".jpg", img)[1].tobytes()
    r = on.post("/api/detect/image", files={"file": ("synthetic.jpg", io.BytesIO(jpg), "image/jpeg")},
                data={"create_alerts": "false"})
    assert r.status_code in (200, 503), r.text                         # 503 only when no model is installed
    if r.status_code == 200:
        assert "detections" in r.json()
