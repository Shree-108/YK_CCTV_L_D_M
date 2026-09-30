from __future__ import annotations

import glob
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.core.config import AppConfig, PROJECT_ROOT
from backend.core.settings import DetectionSettings

MODEL = PROJECT_ROOT / "models" / "litter_yolo" / "v1_litter_s1_synthetic" / "best.pt"
TEST_IMAGES = sorted(glob.glob(str(PROJECT_ROOT / "dataset" / "cctv_litter_dataset_v1" / "images" / "test" / "*.jpg")))
NEG_IMAGES = sorted(glob.glob(str(PROJECT_ROOT / "dataset" / "cctv_litter_dataset_v1" / "images" / "negative" / "val" / "*.jpg")))

needs_model = pytest.mark.skipif(not MODEL.is_file() or not TEST_IMAGES, reason="trained model/test images not present")


@pytest.fixture(scope="session")
def engine():
    if not MODEL.is_file():
        pytest.skip("model not present")
    from backend.detection.engine import YoloEngine
    return YoloEngine(MODEL, DetectionSettings(), "test-v1")


@pytest.fixture(scope="session")
def litter_image() -> np.ndarray:
    if not TEST_IMAGES:
        pytest.skip("no test images")
    return cv2.imread(TEST_IMAGES[0])


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    c = AppConfig()
    c.storage_dir = tmp_path / "storage"
    c.db_path = tmp_path / "storage" / "test.db"
    c.admin_token = "test-admin-token"
    c.start_cameras = False
    c.ensure_dirs()
    return c


@pytest.fixture
def make_video(tmp_path: Path):
    """Write a real video from real dataset frames: frames[i] is shown for `repeat` frames each."""
    def _make(frames: list[np.ndarray], repeat: int = 1, fps: float = 5.0, name: str = "clip.mp4") -> Path:
        p = tmp_path / name
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            for _ in range(repeat):
                vw.write(f)
        vw.release()
        return p
    return _make
