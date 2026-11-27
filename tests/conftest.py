from __future__ import annotations

import glob
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.core.config import AppConfig, PROJECT_ROOT
from backend.core.settings import DetectionSettings

# ---------------------------------------------------------------- current production model (v3) + its test fixture
V3_VERSION = "v3_store_cctv"
V3_DIR = PROJECT_ROOT / "models" / "litter_yolo" / V3_VERSION
V3_MODEL = V3_DIR / "best.pt"                         # production weights: only ever READ by the tests
V3_CLASSES = json.loads((V3_DIR / "class_names.json").read_text()) if (V3_DIR / "class_names.json").is_file() else []
FIXTURES = Path(__file__).parent / "fixtures" / "v3"
FIXTURE_IMAGE = FIXTURES / "aug100_A_083.jpg"         # synthetic street frame, see fixtures/v3/README.md
FIXTURE_LABELS = FIXTURES / "aug100_A_083.txt"        # 19 visually verified litter boxes (single class)
# No human-verified clean (litter-free) CCTV frames exist yet; tests that need them skip until they are added here.
CLEAN_FRAMES = sorted(glob.glob(str(PROJECT_ROOT / "tests" / "fixtures" / "clean" / "*.jpg")))

needs_v3 = pytest.mark.skipif(not V3_MODEL.is_file() or not FIXTURE_IMAGE.is_file(),
                              reason="v3_store_cctv weights or tests/fixtures/v3 image not present")

# ---------------------------------------------------------------- historical v1 artifacts (not on this machine)
V1_MODEL = PROJECT_ROOT / "models" / "litter_yolo" / "v1_litter_s1_synthetic" / "best.pt"
V1_TEST_IMAGES = sorted(glob.glob(str(PROJECT_ROOT / "dataset" / "cctv_litter_dataset_v1" / "images" / "test" / "*.jpg")))
V1_NEG_IMAGES = sorted(glob.glob(str(PROJECT_ROOT / "dataset" / "cctv_litter_dataset_v1" / "images" / "negative" / "val" / "*.jpg")))
needs_v1 = pytest.mark.skipif(
    not V1_MODEL.is_file() or not V1_TEST_IMAGES,
    reason="historical v1 artifacts unavailable: models/litter_yolo/v1_litter_s1_synthetic/best.pt and "
           "dataset/cctv_litter_dataset_v1 images were never committed and are not on this machine "
           "(v1 was trained on another PC). Restore both to run these tests; do not substitute other files.")


@pytest.fixture(scope="session")
def engine():
    """The production v3 model, loaded read-only."""
    if not V3_MODEL.is_file():
        pytest.skip("v3_store_cctv weights not present")
    from backend.detection.engine import YoloEngine
    return YoloEngine(V3_MODEL, DetectionSettings(), V3_VERSION)


@pytest.fixture(scope="session")
def litter_image() -> np.ndarray:
    if not FIXTURE_IMAGE.is_file():
        pytest.skip("tests/fixtures/v3 image not present")
    return cv2.imread(str(FIXTURE_IMAGE))


def fixture_boxes(img: np.ndarray) -> list[list[float]]:
    """Verified litter boxes of the fixture image in pixel xyxy."""
    h, w = img.shape[:2]
    out = []
    for line in FIXTURE_LABELS.read_text().splitlines():
        if line.strip():
            _, x, y, bw, bh = map(float, line.split())
            out.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


@pytest.fixture(autouse=True)
def _isolated_export_folder(tmp_path: Path, monkeypatch) -> None:
    """Every AppConfig built in a test writes import CSVs / result bundles (and the export ledger) under tmp_path:
    finished four-screenshot jobs now always produce a bundle, which must never land in the real storage/exports."""
    monkeypatch.setenv("LDM_GARBAGE_EXPORT_DIR", str(tmp_path / "exports" / "ai_garbage_detection"))
    monkeypatch.setenv("LDM_GARBAGE_EXPORT_LEDGER", str(tmp_path / "exports" / "ledger.db"))
    # monitored region (roi_service): off for existing tests (full-frame behaviour); the ROI store never touches the
    # real config/camera_roi.json. tests/test_roi_service.py switches it on.
    monkeypatch.setenv("LDM_ROI", "off")
    monkeypatch.setenv("LDM_ROI_STORE", str(tmp_path / "config" / "camera_roi.json"))
    # S3 upload: never from tests (.env may switch it on and hold real keys). tests/test_s3.py uses a fake client.
    monkeypatch.setenv("LDM_S3_UPLOAD", "off")
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "LDM_S3_BUCKET"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    c = AppConfig()
    c.storage_dir = tmp_path / "storage"
    c.db_path = tmp_path / "storage" / "test.db"
    c.admin_token = "test-admin-token"
    c.start_cameras = False
    c.auth_mode = "off"   # detection/analytics tests; login is covered by tests/test_auth.py + test_authz.py
    c.auth_db_path = tmp_path / "storage" / "auth.db"   # never the real storage/auth.db
    c.ensure_dirs()
    return c


@pytest.fixture
def make_video(tmp_path: Path):
    """Write a real video from real frames: frames[i] is shown for `repeat` frames each."""
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
