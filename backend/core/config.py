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
    # /docs, /redoc and /openapi.json are off unless explicitly enabled (and even then only reachable locally)
    enable_api_docs: bool = field(default_factory=lambda: os.getenv("LM_ENABLE_API_DOCS", "false").lower() == "true")
    # Client names treated as local by the local-only safeguard in addition to loopback IPs. Tests only
    # (Starlette's TestClient reports "testclient"); deliberately not configurable from the environment.
    local_extra_hosts: tuple[str, ...] = ()
    # four-screenshot analysis: scene-segmentation model that proposes the outdoor ROI ("off" disables it;
    # the operator then always draws/confirms the ROI). Weights: models/roi_segmenter/upernet-convnext-tiny (MIT).
    roi_segmenter: str = field(default_factory=lambda: os.getenv("LM_ROI_SEGMENTER", "auto").strip().lower())
    roi_segmenter_dir: Path = field(default_factory=lambda: _env_path(
        "LM_ROI_SEGMENTER_DIR", PROJECT_ROOT / "models" / "roi_segmenter" / "upernet-convnext-tiny"))
    # Login / sessions (API security plan Step 2). OFF by default; when on, routes need a role (Step 4, api/authz.py).
    auth_mode: str = field(default_factory=lambda: os.getenv("LM_AUTH_MODE", "off").strip().lower())
    auth_db_path: Path = field(default_factory=lambda: _env_path("LM_AUTH_DB_PATH", PROJECT_ROOT / "storage" / "auth.db"))
    auth_idle_minutes: int = field(default_factory=lambda: _env_int("LM_AUTH_IDLE_MINUTES", 30))
    auth_max_session_hours: int = field(default_factory=lambda: _env_int("LM_AUTH_MAX_SESSION_HOURS", 12))
    auth_max_failures: int = field(default_factory=lambda: _env_int("LM_AUTH_MAX_FAILURES", 5))
    auth_lockout_minutes: int = field(default_factory=lambda: _env_int("LM_AUTH_LOCKOUT_MINUTES", 15))
    auth_min_password_length: int = field(default_factory=lambda: _env_int("LM_AUTH_MIN_PASSWORD_LENGTH", 12))
    # Secure cookie flag: auto = only when the request arrived over HTTPS; true/false force it
    auth_cookie_secure: str = field(default_factory=lambda: os.getenv("LM_AUTH_COOKIE_SECURE", "auto").strip().lower())

    def __post_init__(self) -> None:
        if self.auth_mode not in ("off", "on"):
            raise ValueError(f"LM_AUTH_MODE must be 'off' or 'on' (got {self.auth_mode!r})")
        if self.auth_cookie_secure not in ("auto", "true", "false"):
            raise ValueError(f"LM_AUTH_COOKIE_SECURE must be auto, true or false (got {self.auth_cookie_secure!r})")
        for name in ("auth_idle_minutes", "auth_max_session_hours", "auth_max_failures", "auth_lockout_minutes"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.auth_min_password_length < 8:
            raise ValueError("LM_AUTH_MIN_PASSWORD_LENGTH must be at least 8")

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
    def media_dir(self) -> Path:
        """Uploaded videos kept for the four-screenshot analysis (unlike uploads/, which is temporary)."""
        return self.storage_dir / "media"

    @property
    def sampled_dir(self) -> Path:
        """Per-job outputs of the four-screenshot analysis: frames/, annotated/, roi/, report CSV."""
        return self.storage_dir / "sampled"

    @property
    def yolo_models_dir(self) -> Path:
        return self.models_dir / "litter_yolo"

    def ensure_dirs(self) -> None:
        for d in (self.storage_dir, self.alerts_dir, self.videos_dir, self.uploads_dir, self.media_dir, self.sampled_dir,
                  self.hard_examples_dir, self.yolo_models_dir, self.db_path.parent):
            d.mkdir(parents=True, exist_ok=True)
