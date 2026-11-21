"""A failing ai_garbage_detection export must never change a finished job (backend/services/export_guard.py)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.services import export_guard, garbage_export

from .conftest import V3_VERSION, needs_v3
from .test_sampled import write_video

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
AUTH = {"Authorization": "Bearer test-admin-token"}


def _boom(*_a, **_k):
    raise RuntimeError("forced exporter failure")


# ------------------------------------------------------------------------------------------ unit
def test_guard_swallows_an_exporter_exception_and_records_it(monkeypatch, tmp_path):
    monkeypatch.setattr(garbage_export, "auto_export", _boom)
    res = export_guard.safe_auto_export(SimpleNamespace(), kind="sampled", source_id="sv_x", out_dir=tmp_path / "r")
    assert res["ok"] is False and "forced exporter failure" in res["error"]
    assert json.loads((tmp_path / "r" / export_guard.STATUS_FILE).read_text())["ok"] is False


def test_guard_swallows_an_import_error(monkeypatch, tmp_path):
    monkeypatch.delattr("backend.services.garbage_export", raising=False)   # attribute on the package
    monkeypatch.setitem(sys.modules, "backend.services.garbage_export", None)   # makes the import raise
    res = export_guard.safe_auto_export(SimpleNamespace(), kind="video", source_id="vid_x", out_dir=tmp_path)
    assert res["ok"] is False and res["error"].startswith(("ImportError", "ModuleNotFoundError"))
    assert (tmp_path / export_guard.STATUS_FILE).is_file()


# ------------------------------------------------------------------------------------------ real jobs
@pytest.fixture
def client(cfg):
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        assert c.post("/api/models/activate", json={"version": V3_VERSION}, headers=AUTH).status_code == 200
        yield c


def _wait(get, done, timeout=400):
    end = time.time() + timeout
    while time.time() < end:
        j = get()
        if done(j):
            return j
        time.sleep(0.3)
    raise AssertionError("job did not finish")


@needs_v3
def test_sampled_job_stays_done_when_the_exporter_raises(client, cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(garbage_export, "auto_export", _boom)
    v = write_video(tmp_path / "store.mp4", 150, fps=10)
    m = client.post("/api/sampled/media", files={"file": (v.name, v.read_bytes(), "video/mp4")}).json()
    r = client.post("/api/sampled/jobs", json={"media_id": m["id"], "seed": 5})
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    j = _wait(lambda: client.get(f"/api/sampled/jobs/{jid}").json(), lambda j: j["state"] in ("done", "failed"))
    assert j["state"] == "done" and j["stage"] == "completed", (j["state"], j.get("error"))
    status = json.loads((cfg.storage_dir / "sampled" / jid / "reports" / export_guard.STATUS_FILE).read_text())
    assert status["ok"] is False and "forced exporter failure" in status["error"]
    time.sleep(0.5)                                   # nothing may flip the job afterwards
    assert client.get(f"/api/sampled/jobs/{jid}").json()["state"] == "done"


@needs_v3
def test_video_job_stays_done_when_the_exporter_raises(client, cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(garbage_export, "auto_export", _boom)
    v = write_video(tmp_path / "clip.mp4", 60, fps=10)
    r = client.post("/api/detect/video", files={"file": ("clip.mp4", v.read_bytes(), "video/mp4")},
                    data={"sample_interval_sec": "1", "tiling": "off"})
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    j = _wait(lambda: client.get(f"/api/jobs/{jid}").json(), lambda j: j["status"] in ("done", "failed"))
    assert j["status"] == "done", j.get("error")
    files = list((cfg.storage_dir / "analysis_outputs").glob(f"vid_*/{export_guard.STATUS_FILE}"))
    assert len(files) == 1 and json.loads(files[0].read_text())["ok"] is False
    time.sleep(0.5)
    assert client.get(f"/api/jobs/{jid}").json()["status"] == "done"
