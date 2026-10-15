"""Login endpoints (API security plan Step 2): POST /api/auth/login, POST /api/auth/logout, GET /api/auth/me.

Nothing else is protected by these yet - route-level protection is Step 4. With LM_AUTH_MODE=off, login and logout
answer 404 "login is disabled" and /me reports auth_mode "off" (the existing ADMIN_API_TOKEN still identifies an
admin, exactly as require_admin accepts it).

* The session ID travels only in the Set-Cookie header (HttpOnly, SameSite=Strict, Path=/api; Secure over HTTPS or
  when LM_AUTH_COOKIE_SECURE=true). It is never in a response body or a log.
* The login body is parsed by hand: FastAPI's default 422 errors echo the submitted values, which would put a
  password into the response.
* Every response here carries Cache-Control: no-store.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
from typing import Any, Optional

from fastapi import APIRouter, Header, Request, Response
from fastapi.responses import JSONResponse

from backend.auth.service import AuthService, LoginFailed, LoginLocked, Principal, admin_token_principal

router = APIRouter(prefix="/api/auth", tags=["auth"])
COOKIE = "ldm_session"
MAX_BODY = 4096
NO_STORE = {"Cache-Control": "no-store", "Pragma": "no-cache"}


def _auth(request: Request) -> Optional[AuthService]:
    return getattr(request.app.state, "auth", None)


def _cfg(request: Request):
    return request.app.state.auth_cfg


def _iso(t: Optional[float]) -> Optional[str]:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="seconds") if t else None


def _secure(request: Request) -> bool:
    mode = _cfg(request).auth_cookie_secure
    return mode == "true" or (mode == "auto" and request.url.scheme == "https")


def _error(status: int, detail: str, headers: Optional[dict[str, str]] = None) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status, headers={**NO_STORE, **(headers or {})})


def _disabled() -> JSONResponse:
    return _error(404, "Login is disabled (LM_AUTH_MODE=off)")


def current_principal(request: Request, authorization: Optional[str], x_admin_token: Optional[str]) -> Optional[Principal]:
    """Session cookie (when login is on) or the existing admin token (always), else None."""
    supplied = x_admin_token or (authorization[7:] if authorization and authorization.lower().startswith("bearer ")
                                 else None)
    p = admin_token_principal(supplied, _cfg(request).admin_token)
    if p:
        return p
    svc = _auth(request)
    return svc.session_principal(request.cookies.get(COOKIE)) if svc else None


@router.post("/login")
async def login(request: Request) -> Response:
    svc = _auth(request)
    if svc is None:
        return _disabled()
    raw = await request.body()
    if len(raw) > MAX_BODY or "application/json" not in request.headers.get("content-type", ""):
        return _error(400, "Invalid login request: send JSON {\"username\": ..., \"password\": ...}")
    try:
        body = json.loads(raw)
        username, password = body["username"], body["password"]
        if not isinstance(username, str) or not isinstance(password, str):
            raise TypeError
    except (ValueError, KeyError, TypeError):
        return _error(400, "Invalid login request: send JSON {\"username\": ..., \"password\": ...}")
    client = request.client.host if request.client else "?"
    try:
        # scrypt takes ~0.15 s: keep it off the event loop (live streams keep flowing during a login)
        session_id, p = await asyncio.to_thread(svc.login, username, password, client)
    except LoginLocked as exc:
        return _error(429, "Too many failed login attempts. Try again later.", {"Retry-After": str(exc.retry_after)})
    except LoginFailed:
        return _error(401, "Invalid username or password")
    max_age = int(svc.s.max_session_hours * 3600)
    resp = JSONResponse({"username": p.username, "role": p.role, "idle_expires_at": _iso(p.session_expires_at)},
                        headers=NO_STORE)
    resp.set_cookie(COOKIE, session_id, max_age=max_age, path="/api", httponly=True, samesite="strict",
                    secure=_secure(request))
    return resp


@router.post("/logout")
def logout(request: Request) -> Response:
    svc = _auth(request)
    if svc is None:
        return _disabled()
    svc.logout(request.cookies.get(COOKIE))
    resp = Response(status_code=204, headers=NO_STORE)
    resp.delete_cookie(COOKIE, path="/api", httponly=True, samesite="strict", secure=_secure(request))
    return resp


@router.get("/me")
def me(request: Request, authorization: Optional[str] = Header(default=None),
       x_admin_token: Optional[str] = Header(default=None)) -> JSONResponse:
    p = current_principal(request, authorization, x_admin_token)
    body: dict[str, Any] = {"auth_mode": _cfg(request).auth_mode, "authenticated": p is not None}
    if p:
        body.update(username=p.username, role=p.role, via=p.via, idle_expires_at=_iso(p.session_expires_at))
    return JSONResponse(body, headers=NO_STORE)
