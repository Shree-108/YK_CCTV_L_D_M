"""Process-level configuration from environment variables (.env supported).

Secrets (admin token, SMTP, Telegram, webhook, RTSP credentials) are ONLY read
from the environment - never from source code or the database.
Runtime-tunable detection settings live in the database instead (see
``backend.core.settings``) so they can be edited from the admin panel.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_path(name: str, default: Path) -> Path:
    v = os.getenv(name)
    p = Path(v) if v else default
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc


@dataclass
class AppConfig:
    storage_dir: Path = field(default_factory=lambda: _env_path("LM_STORAGE_DIR", PROJECT_ROOT / "storage"))
    models_dir: Path = field(default_factory=lambda: _env_path("LM_MODELS_DIR", PROJECT_ROOT / "models"))
    dataset_dir: Path = field(default_factory=lambda: _env_path("LM_DATASET_DIR", PROJECT_ROOT / "dataset"))
    db_path: Path = field(default_factory=lambda: _env_path("LM_DB_PATH", PROJECT_ROOT / "storage" / "litter_monitor.db"))
    class_metadata_path: Path = field(
        default_factory=lambda: _env_path("LM_CLASS_METADATA", PROJECT_ROOT / "backend" / "core" / "class_metadata.json"))
    admin_token: str | None = field(default_factory=lambda: os.getenv("ADMIN_API_TOKEN") or None)
    max_image_mb: int = field(default_factory=lambda: _env_int("MAX_IMAGE_UPLOAD_MB", 25))
    max_video_mb: int = field(default_factory=lambda: _env_int("MAX_VIDEO_UPLOAD_MB", 500))
    cors_origins: list[str] = field(default_factory=lambda: [
        o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if o.strip()])
    start_cameras: bool = field(default_factory=lambda: os.getenv("LM_START_CAMERAS", "true").lower() == "true")

    @property
    def alerts_dir(self) -> Path:
        return self.storage_dir / "alerts"

    @property
    def videos_dir(self) -> Path:
        return self.storage_dir / "videos"

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def hard_examples_dir(self) -> Path:
        return self.storage_dir / "hard_examples"

    @property
    def yolo_models_dir(self) -> Path:
        return self.models_dir / "litter_yolo"

    def ensure_dirs(self) -> None:
        for d in (self.storage_dir, self.alerts_dir, self.videos_dir, self.uploads_dir,
                  self.hard_examples_dir, self.yolo_models_dir, self.db_path.parent):
            d.mkdir(parents=True, exist_ok=True)
