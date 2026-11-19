"""Four-screenshot video analysis: sampling, decoding, job workflow (full frame, no ROI), CSV and reports."""
from __future__ import annotations

import csv
import io
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.sampling import SAMPLE_COUNT, frame_quality
from backend.sampling.extract import extract
from backend.sampling.plan import (TooShort, min_gap, plan_samples, recording_start_from_name, segments)
from backend.services import litter_reports as lr

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------- helpers
def scene(idx: int = 0, w: int = 320, h: int = 192, moved: bool = False) -> np.ndarray:
    """A synthetic, textured CCTV-like frame: 'shop interior' on the right, 'pavement' bottom-left.
    The frame number is written into the top-left so decoded screenshots can be matched to their frame."""
    rng = np.random.default_rng(7)
    img = rng.integers(60, 200, (h, w, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    img[h // 2:, : w // 2] = (90, 110, 120)                       # pavement
    img[h // 2:, : w // 2] += rng.integers(0, 25, (h - h // 2, w // 2, 3), dtype=np.uint8)
    for x in range(w // 2 + 10, w, 22):                            # shelving stripes
        cv2.rectangle(img, (x, 10), (x + 12, h - 10), (30, 30, 200), -1)
    if moved:
        img = np.ascontiguousarray(np.rot90(img, 2))
    cv2.putText(img, str(idx), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def write_video(path: Path, n: int, fps: float = 10.0, moved_after: int | None = None) -> Path:
    h, w = scene().shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        vw.write(scene(i, moved=moved_after is not None and i >= moved_after))
    vw.release()
    return path


# ---------------------------------------------------------------------- planning
def test_exactly_four_one_per_segment_well_separated_and_reproducible():
    frames, fps = 6000, 25.0
    a = plan_samples(frames, fps, SAMPLE_COUNT, seed=42)
    assert len(a) == 4 and [s.idx for s in a] == [1, 2, 3, 4]
    segs = segments(1, frames - 1, 4)
    for s, (lo, hi) in zip(a, segs):
        assert lo <= s.frame <= hi
        assert s.seconds == pytest.approx(s.frame / fps)
    gaps = [b.frame - x.frame for x, b in zip(a, a[1:])]
    assert min(gaps) >= min_gap(frames - 1)
    assert plan_samples(frames, fps, 4, seed=42) == a                    # same seed -> same frames
    assert plan_samples(frames, fps, 4, seed=43) != a                    # another seed -> other frames
    assert all(s.frame >= 1 for s in a)                                  # frame 0 (often incomplete) never used


def test_short_or_unknown_videos_are_refused_not_shortened():
    with pytest.raises(TooShort):
        plan_samples(8, 10.0, 4, seed=1)
    with pytest.raises(TooShort):
        plan_samples(0, 10.0, 4, seed=1)
    with pytest.raises(TooShort):
        plan_samples(1000, 0, 4, seed=1)
    assert len(plan_samples(20, 10.0, 4, seed=1)) == 4


def test_many_seeds_never_cluster():
    for seed in range(1, 300):
        s = plan_samples(500, 25.0, 4, seed)
        assert len({x.frame for x in s}) == 4
        assert min(b.frame - a.frame for a, b in zip(s, s[1:])) >= min_gap(499)


def test_recording_start_is_only_parsed_from_unambiguous_names():
    assert recording_start_from_name("20261029_20261029183009_20261029183120_183009.mp4") == "2026-10-29T18:30:09"
    assert recording_start_from_name("cam1_20261104215126.mp4") == "2026-11-04T21:51:26"
    assert recording_start_from_name("video1.mp4") is None
    assert recording_start_from_name("clip_20261399999999.mp4") is None      # invalid date -> nothing guessed


# ---------------------------------------------------------------------- decoding
def test_frame_quality_rejects_grey_fill_and_blank_frames():
    assert frame_quality.problem(scene()) is None
    grey = np.full((192, 320, 3), 128, np.uint8)
    assert "grey fill" in frame_quality.problem(grey)
    black = np.zeros((192, 320, 3), np.uint8)
    assert "blank" in frame_quality.problem(black)
    assert frame_quality.problem(None)


def test_extract_saves_the_planned_frames_with_their_real_time(tmp_path):
    video = write_video(tmp_path / "v.mp4", 200, fps=10)
    samples = plan_samples(200, 10.0, 4, seed=5)
    out = extract(video, 10.0, samples, tmp_path / "frames", "v")
    assert [e.status for e in out] == ["saved"] * 4
    for e in out:
        assert e.path.is_file() and f"_f{e.sample.frame:07d}_" in e.path.name
        assert e.seconds == pytest.approx(e.sample.frame / 10.0, abs=0.11)
        ref = scene(e.sample.frame)
        assert np.abs(cv2.imread(str(e.path)).astype(int) - ref.astype(int)).mean() < 12   # really that frame


def test_extract_reports_frames_past_the_end_instead_of_substituting(tmp_path):
    video = write_video(tmp_path / "v.mp4", 40, fps=10)
    samples = plan_samples(400, 10.0, 4, seed=3)                         # planned from a wrong (too long) length
    out = extract(video, 10.0, samples, tmp_path / "frames", "v")
    assert len(out) == 4
    assert any(e.status == "failed" and "ends after" in e.error for e in out)
    assert all(e.path is None for e in out if e.status != "saved")


# ---------------------------------------------------------------------- API workflow
@pytest.fixture
def client(cfg):
    with TestClient(create_app(cfg)) as c:
        yield c


def _needs_engine(client):
    if client.app.state.app_state.engine is None:
        pytest.skip("no detection model loaded")


def _wait(client, job_id, states, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/sampled/jobs/{job_id}").json()
        if j["state"] in states:
            return j
        time.sleep(0.2)
    raise AssertionError(f"job stuck in {j['state']} / {j['stage']}")


def _upload(client, path: Path, name: str | None = None):
    return client.post("/api/sampled/media", files={"file": (name or path.name, path.read_bytes(), "video/mp4")})


def test_upload_validates_and_describes_the_video(client, tmp_path):
    v = write_video(tmp_path / "20261029_20261029183009_20261029183120_183009.mp4", 120, fps=10)
    r = _upload(client, v)
    assert r.status_code == 201, r.text
    m = r.json()
    assert (m["width"], m["height"], m["frames"]) == (320, 192, 120)
    assert m["fps"] == pytest.approx(10) and m["duration_sec"] == pytest.approx(12, abs=0.2)
    assert m["suggested_recording_start"] == "2026-10-29T18:30:09"
    assert client.get(m["preview_url"]).status_code == 200
    again = _upload(client, v).json()
    assert again["previously_received"][0]["media_id"] == m["id"]         # same file recognised
    bad = client.post("/api/sampled/media", files={"file": ("x.mp4", b"not a video at all", "video/mp4")})
    assert bad.status_code == 422
    assert client.post("/api/sampled/media", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 415


def test_too_short_video_is_refused_at_job_creation(client, tmp_path):
    _needs_engine(client)
    m = _upload(client, write_video(tmp_path / "short.mp4", 6, fps=10)).json()
    r = client.post("/api/sampled/jobs", json={"media_id": m["id"]})
    assert r.status_code == 422 and "too short" in r.json()["detail"]


ANALYSED_OUTCOMES = ("LITTER_DETECTED", "NO_LITTER", "NO_LITTER_DETECTED")


def _no_roi_keys(obj, path="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert "roi" not in str(k).lower() or k == "awaiting_roi", f"ROI key {path}.{k} in response"
            _no_roi_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _no_roi_keys(v, f"{path}[{i}]")


def test_full_workflow_runs_straight_to_done_csv_and_reports(client, tmp_path, cfg):
    """Manual (seeded) mode: no ROI stage any more - the job goes received -> ... -> completed with one request."""
    _needs_engine(client)
    m = _upload(client, write_video(tmp_path / "store.mp4", 150, fps=10)).json()
    r = client.post("/api/sampled/jobs", json={"media_id": m["id"], "store_id": "Store 12", "camera_id": "CAM-07",
                                               "recording_start": "2026-11-05T10:15:00", "seed": 99})
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    dup = client.post("/api/sampled/jobs", json={"media_id": m["id"]})
    assert dup.status_code == 409 and dup.json()["detail"]["job_id"] == jid

    j = _wait(client, jid, ("done", "failed", "awaiting_roi"))
    assert j["state"] == "done", (j["state"], j["error"])
    assert j["stage"] == "completed"
    _no_roi_keys(j)
    assert len(j["frames"]) == 4 and all(f["status"] == "saved" for f in j["frames"])
    assert [f["frame_idx"] for f in j["frames"]] == sorted(f["frame_idx"] for f in j["frames"])
    assert j["frames"][0]["recording_timestamp"].startswith("2026-11-05T10:15")
    assert client.get(j["frames"][0]["original_url"]).status_code == 200
    assert all(f["outcome"] in ANALYSED_OUTCOMES for f in j["frames"])      # every screenshot analysed, full frame
    assert j["counts"]["screenshots_analysed"] == 4 and j["counts"]["screenshots_extracted"] == 4
    assert all(f["annotated_url"] for f in j["frames"])

    rows = list(csv.DictReader(io.StringIO(client.get(j["csv_url"]).text)))
    assert len({r["screenshot"] for r in rows}) == 4                       # every screenshot is traceable
    assert all(r["store_id"] == "Store 12" and r["camera_id"] == "CAM-07" for r in rows)
    assert all(r["recording_date"] == "2026-11-05" for r in rows)
    assert all(r["outcome"] in ANALYSED_OUTCOMES for r in rows)
    assert (cfg.storage_dir / rows[0]["evidence_image"]).is_file()

    # identity: an unregistered store is accepted as free text, its name is UNKNOWN with a warning
    ident = j["identity"]
    assert ident["store_id"] == "Store 12" and ident["store_name"] == lr.UNKNOWN and ident["camera_id"] == "CAM-07"
    assert ident["warnings"]
    # standard litter outputs (detection + summary CSV in the exact column format)
    std = j["standard_outputs"]
    assert std["processing_status"] in ("COMPLETED_LITTER_DETECTED", "COMPLETED_NO_LITTER_DETECTED")
    det = client.get(std["urls"]["detections_csv"])
    assert det.status_code == 200 and next(csv.reader(io.StringIO(det.text.lstrip("\ufeff")))) == lr.DETECTION_COLUMNS
    summ = client.get(std["urls"]["summary_csv"])
    assert summ.status_code == 200 and next(csv.reader(io.StringIO(summ.text.lstrip("\ufeff")))) == lr.SUMMARY_COLUMNS

    # detections are linked to the job, so history and reports can find them
    dets = client.get("/api/detections", params={"job_id": jid}).json()
    assert all(d["source_type"] == "sample" for d in dets["detections"])
    ov = client.get("/api/sampled/reports/overview").json()
    assert ov["summary"]["videos_received"] == 1 and ov["summary"]["analyses_completed"] == 1
    assert ov["latest"]["id"] == jid and len(ov["latest"]["frames"]) == 4
    month = ov["months"]["completion"][0]
    assert client.get(f"/api/sampled/reports/months/{month}").json()["analyses_completed"] == 1
    mcsv = client.get(f"/api/sampled/reports/months/{month}.csv").text
    assert mcsv.splitlines()[0].startswith("video_name,store_id,camera_id")
    hourly = client.get("/api/sampled/reports/hourly").json()
    assert hourly["available"] and hourly["rows"][0]["hour_start"].startswith("2026-11-05T10:00")
    hist = client.get("/api/sampled/reports/history", params={"store": "Store 12"}).json()
    assert hist["total"] == 1 and hist["analyses"][0]["timestamps"]

    # the same seed and settings reuse the finished analysis instead of running it again
    again = client.post("/api/sampled/jobs", json={"media_id": m["id"], "store_id": "Store 12", "camera_id": "CAM-07",
                                                   "recording_start": "2026-11-05T10:15:00", "seed": 99})
    assert again.json() == {"job_id": jid, "reused": True, "state": "done"}


def test_removed_roi_job_options_are_refused(client, tmp_path):
    """roi_mode / auto_accept are gone from POST /api/sampled/jobs (extra fields forbidden); the ROI routes are gone."""
    m = _upload(client, write_video(tmp_path / "r.mp4", 150, fps=10)).json()
    for extra in ({"roi_mode": "manual"}, {"auto_accept": True}, {"roi": []}):
        r = client.post("/api/sampled/jobs", json={"media_id": m["id"], **extra})
        assert r.status_code == 422, (extra, r.text)
    assert client.put("/api/sampled/jobs/sv_x/roi", json={"zones": []}).status_code in (404, 405)
    assert client.post("/api/sampled/jobs/sv_x/roi/approve").status_code in (404, 405)
