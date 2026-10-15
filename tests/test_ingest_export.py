"""Export-folder ingester (ingest/export_folder.py) with a fake LDM API: no backend, no model, no NVR."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from ingest import export_folder as ef


def _clip(path: Path, seed: int, frames: int = 20) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    rng = np.random.default_rng(seed)
    for _ in range(frames):
        vw.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    vw.release()
    return path


class FakeApi:
    def __init__(self, down: bool = False) -> None:
        self.down, self.uploads = down, []

    def submit(self, p, camera, sample_fps):
        if self.down:
            raise ef.ApiDown("ConnectError")
        self.uploads.append((p.name, camera))
        return f"job{len(self.uploads)}"

    def wait(self, job_id, poll, limit):
        return {"id": job_id, "status": "done", "result": {"stats": {"frames_analysed": 4}, "alert_ids": []}}

    def file(self, job_id, kind):
        if kind == "detections":
            return [{"frame_idx": 5, "video_time_sec": 0.5,
                     "detections": [{"class": "food_wrapper", "confidence": 0.8, "bbox": [1, 2, 3, 4], "status": "confirmed", "track_id": 1}]}]
        return [{"event": "confirmed", "class": "food_wrapper", "confidence": 0.8, "video_time_sec": 0.5, "frame_idx": 5,
                 "track_id": 1, "alert_id": None}]

    def snapshot(self, alert_id):
        return None


def _args(tmp: Path, **kw) -> argparse.Namespace:
    a = dict(export_dir=tmp / "export", ingest_dir=tmp / "ingest", camera=None, start=None, sample_fps=None,
             settle=0.0, max_attempts=3, reprocess=False, dry_run=False, poll=0.01, job_timeout=5,
             time_format="%Y%m%d%H%M%S")
    a.update(kw)
    (tmp / "ingest").mkdir(parents=True, exist_ok=True)
    return argparse.Namespace(**a)


LIMITS = {"max_mb": 500, "max_seconds": 3600.0, "sample_fps": 2.0, "model": "v3_store_cctv"}
PAT = re.compile(r"ch\d+_(?P<start>\d{14})")


def test_camera_and_start_time_come_from_folder_and_name(tmp_path: Path):
    a = _args(tmp_path)
    p = _clip(a.export_dir / "ch2" / "ch2_20260925154200.mp4", 1)
    assert ef.camera_for(p, a.export_dir, None) == "ch2"
    assert ef.camera_for(a.export_dir / "x.mp4", a.export_dir, None) == "nvr_export"
    assert ef.start_for(p, PAT, "%Y%m%d%H%M%S", None) == ("2026-09-25T15:42:00", "file name")
    assert ef.start_for(a.export_dir / "x.mp4", PAT, "%Y%m%d%H%M%S", None) == (None, "unknown")


def test_process_once_skip_duplicates_and_write_timestamped_results(tmp_path: Path):
    a = _args(tmp_path)
    p = _clip(a.export_dir / "ch1" / "ch1_20260925154100.mp4", 2)
    copy = a.export_dir / "ch1" / "renamed_copy.mp4"
    copy.write_bytes(p.read_bytes())
    api, ledger = FakeApi(), ef.Ledger(a.ingest_dir / "ingest.db")
    assert ef.process(p, a, api, ledger, LIMITS, PAT) == "done"
    assert ef.process(copy, a, api, ledger, LIMITS, PAT) == "duplicate"
    assert ef.process(p, a, api, ledger, LIMITS, PAT) == "duplicate"
    assert api.uploads == [("ch1_20260925154100.mp4", "ch1")]
    (res,) = list((a.ingest_dir / "results").iterdir())
    assert "2026-09-25T15:41:00" in (res / "detections.csv").read_text()      # start + 0.5 s, rounded to seconds
    assert p.is_file() and copy.is_file()                                      # export folder never modified


def test_api_down_leaves_recording_pending_and_retries(tmp_path: Path):
    a = _args(tmp_path)
    p = _clip(a.export_dir / "cam" / "a.mp4", 3)
    ledger = ef.Ledger(a.ingest_dir / "ingest.db")
    with pytest.raises(ef.ApiDown):
        ef.process(p, a, FakeApi(down=True), ledger, LIMITS, None)
    row = ledger.get(ef.sha256_file(p))
    assert (row["status"], row["attempts"]) == ("pending", 0)
    assert ef.process(p, a, FakeApi(), ledger, LIMITS, None) == "done"


def test_dry_run_writes_nothing_and_bad_inputs_are_reported(tmp_path: Path):
    a = _args(tmp_path, dry_run=True)
    good = _clip(a.export_dir / "cam" / "good.mp4", 4)
    bad = a.export_dir / "cam" / "broken.mp4"
    bad.write_bytes(b"\0" * 2048)
    dav = a.export_dir / "cam" / "native.dav"
    dav.write_bytes(b"DHAV" + b"\0" * 64)
    api, ledger = FakeApi(), ef.Ledger(a.ingest_dir / "ingest.db")
    assert ef.process(good, a, api, ledger, LIMITS, None) == "would_process"
    assert ef.process(bad, a, api, ledger, LIMITS, None) == "failed"
    assert ef.process(dav, a, api, ledger, LIMITS, None) == "unsupported"
    assert api.uploads == [] and ledger.db.execute("SELECT COUNT(*) FROM recordings").fetchone()[0] == 0
    a.dry_run = False
    assert ef.process(dav, a, api, ledger, LIMITS, None) == "unsupported"
    assert ledger.get(ef.sha256_file(dav))["status"] == "unsupported"


def test_too_large_is_refused_before_upload(tmp_path: Path):
    a = _args(tmp_path)
    p = _clip(a.export_dir / "cam" / "big.mp4", 5)
    api = FakeApi()
    assert ef.process(p, a, api, ef.Ledger(a.ingest_dir / "ingest.db"), {**LIMITS, "max_mb": 0}, None) == "too_large"
    assert api.uploads == []
