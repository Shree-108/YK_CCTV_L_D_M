"""Upload size limits (MAX_IMAGE_UPLOAD_MB / MAX_VIDEO_UPLOAD_MB): oversized files are rejected with 413, nothing is
left on disk, the server stays healthy, and normal-sized uploads still work. Temporary storage + DB only."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from .conftest import V3_VERSION, needs_v3

pytestmark = needs_v3
AUTH = {"Authorization": "Bearer test-admin-token"}
MB = 1024 * 1024


@pytest.fixture
def client(cfg):
    cfg.max_image_mb = 1
    cfg.max_video_mb = 1
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        assert c.post("/api/models/activate", json={"version": V3_VERSION}, headers=AUTH).status_code == 200
        yield c


def _leftovers(cfg) -> list:
    return [p for p in cfg.uploads_dir.iterdir()] if cfg.uploads_dir.exists() else []


def _big_valid_jpeg() -> bytes:
    """A real, decodable JPEG larger than 1 MB (random noise compresses badly)."""
    img = np.random.default_rng(1).integers(0, 256, (1400, 1400, 3), dtype=np.uint8)
    data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()
    assert len(data) > MB
    return data


def test_image_over_limit_rejected_and_nothing_kept(client, cfg):
    r = client.post("/api/detect/image", files={"file": ("big.jpg", _big_valid_jpeg(), "image/jpeg")})
    assert r.status_code == 413 and "1 MB" in r.json()["detail"]
    assert _leftovers(cfg) == []                                  # partial upload deleted
    assert client.get("/api/health").json()["status"] == "ok"    # server unaffected
    assert client.get("/api/alerts").json()["total"] == 0        # nothing processed


def test_image_size_boundary(client, cfg):
    # exactly the limit is allowed by the size check (it then fails decoding: 400, not 413) ...
    at = client.post("/api/detect/image", files={"file": ("a.jpg", b"x" * MB, "image/jpeg")})
    assert at.status_code == 400
    # ... one byte more is too large
    over = client.post("/api/detect/image", files={"file": ("a.jpg", b"x" * (MB + 1), "image/jpeg")})
    assert over.status_code == 413
    assert _leftovers(cfg) == []


def test_normal_image_still_works_after_rejection(client, cfg, litter_image):
    client.post("/api/detect/image", files={"file": ("big.jpg", _big_valid_jpeg(), "image/jpeg")})
    small = cv2.imencode(".jpg", litter_image)[1].tobytes()
    assert len(small) < MB
    r = client.post("/api/detect/image", files={"file": ("ok.jpg", small, "image/jpeg")},
                    data={"create_alerts": "false", "return_image": "false"})
    assert r.status_code == 200 and r.json()["count"] >= 1
    assert _leftovers(cfg) == []                                  # the processed upload is removed too


def test_video_over_limit_rejected_and_nothing_kept(client, cfg):
    r = client.post("/api/detect/video", files={"file": ("big.mp4", b"\0" * (MB + 1), "video/mp4")})
    assert r.status_code == 413
    assert _leftovers(cfg) == []
    assert client.get("/api/jobs").json()["jobs"] == []           # no job was queued


def test_normal_video_still_works(client, cfg, litter_image, make_video):
    small = cv2.resize(litter_image, (320, 180))
    video = make_video([small], repeat=6, fps=5.0)
    assert video.stat().st_size < MB
    r = client.post("/api/detect/video", files={"file": ("ok.mp4", video.read_bytes(), "video/mp4")})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    deadline = time.time() + 120
    while time.time() < deadline and client.get(f"/api/jobs/{job_id}").json()["status"] not in ("done", "failed"):
        time.sleep(0.5)
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "done"
    assert _leftovers(cfg) == []                                  # upload removed after processing


def test_unsupported_type_rejected_before_size(client, cfg):
    r = client.post("/api/detect/image", files={"file": ("big.exe", b"x" * (MB + 1), "application/octet-stream")})
    assert r.status_code == 415
    assert _leftovers(cfg) == []
