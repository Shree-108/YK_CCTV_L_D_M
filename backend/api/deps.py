"""Shared API dependencies: app state, admin auth, safe uploads."""
from __future__ import annotations

import hmac
import re
import uuid
from pathlib import Path

from fastapi import Depends, Header, HTTPException, Request, UploadFile

from backend.services.state import AppState

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")


def get_state(request: Request) -> AppState:
    return request.app.state.app_state


def require_admin(authorization: str | None = Header(default=None),
                  x_admin_token: str | None = Header(default=None),
                  state: AppState = Depends(get_state)) -> None:
    expected = state.cfg.admin_token
    if not expected:
        raise HTTPException(503, "Admin functions are disabled: set ADMIN_API_TOKEN in the server's .env")
    supplied = x_admin_token or (authorization[7:] if authorization and authorization.lower().startswith("bearer ")
                                 else None)
    if not supplied or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(401, "Invalid or missing admin token", headers={"WWW-Authenticate": "Bearer"})


def safe_display_name(name: str | None) -> str:
    """Sanitised original filename, for display/metadata only (never used as a path)."""
    base = Path(name or "upload").name
    return re.sub(r"[^A-Za-z0-9._\- ]", "_", base)[:120] or "upload"


def check_camera_id(camera_id: str) -> str:
    if not CAMERA_ID_RE.match(camera_id):
        raise HTTPException(422, "camera_id may only contain letters, digits, '-' and '_' (max 40)")
    return camera_id


async def save_upload(file: UploadFile, allowed: set[str], max_mb: int, dest_dir: Path) -> Path:
    """Stream an upload to disk under a random name, enforcing extension and size limits."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(415, f"Unsupported file type '{ext or '?'}'. Allowed: {', '.join(sorted(allowed))}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{uuid.uuid4().hex}{ext}"
    limit, size = max_mb * 1024 * 1024, 0
    try:
        with path.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File too large (limit {max_mb} MB)")
                fh.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "Empty upload")
    return path
