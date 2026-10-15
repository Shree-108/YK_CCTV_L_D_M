"""Login accounts and sessions (API security plan Step 2). Off unless LM_AUTH_MODE=on."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.auth.service import AuthService
    from backend.core.config import AppConfig


def build_service(cfg: "AppConfig") -> "AuthService":
    """Auth service on the configured auth database (creates the file if needed)."""
    from backend.auth.service import AuthService, AuthSettings
    from backend.auth.store import AuthStore
    return AuthService(AuthStore(cfg.auth_db_path), AuthSettings(
        idle_minutes=cfg.auth_idle_minutes, max_session_hours=cfg.auth_max_session_hours,
        max_failures=cfg.auth_max_failures, window_minutes=cfg.auth_lockout_minutes,
        lockout_minutes=cfg.auth_lockout_minutes, min_password_length=cfg.auth_min_password_length))
