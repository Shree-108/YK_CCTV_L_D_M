"""Login endpoints (API security plan Step 2): POST /api/auth/login, POST /api/auth/logout, GET /api/auth/me.

These four are public (POST /api/auth/activity added in Step 4); every other route's permission is declared in
backend/api/authz.py. With LM_AUTH_MODE=off, login, logout and activity answer 404 "login is disabled" and /me reports auth_mode "off" (the existing ADMIN_API_TOKEN still identifies an
admin, exactly as require_admin accepts it).

* The session ID travels only in the Set-Cookie header (HttpOnly, SameSite=Strict, Path=/; Secure over HTTPS or
  when LM_AUTH_COOKIE_SECURE=true). It is never in a response body or a log. Path=/ because the browser reaches the
  API under a proxy prefix (/ldm-api/api/...), which a Path=/api cookie would never be sent to.
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

from backend.api.authz import COOKIE, PUBLIC, allowed_origin, current_principal
from backend.auth.service import AuthService, LoginFailed, LoginLocked

router = APIRouter(prefix="/api/auth", tags=["auth"])
MAX_BODY = 4096
COOKIE_PATH = "/"
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


def _foreign_origin(request: Request) -> bool:
    """A browser request from a page that is not this application (another site, or another port on localhost -
    SameSite=Strict does not separate ports). Requests without an Origin header (non-browser clients) pass, as for
    every other cookie-authenticated write (backend/api/authz.py)."""
    origin = request.headers.get("origin")
    return origin is not None and not allowed_origin(request, origin)


@router.post("/login", dependencies=PUBLIC)
async def login(request: Request) -> Response:
    svc = _auth(request)
    if svc is None:
        return _disabled()
    if _foreign_origin(request):                   # login CSRF: another page signing this browser in
        return _error(403, "Request origin not allowed")
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
    resp.set_cookie(COOKIE, session_id, max_age=max_age, path=COOKIE_PATH, httponly=True, samesite="strict",
                    secure=_secure(request))
    return resp


@router.post("/logout", dependencies=PUBLIC)
def logout(request: Request) -> Response:
    svc = _auth(request)
    if svc is None:
        return _disabled()
    if _foreign_origin(request):                   # logout CSRF: another local page signing the user out
        return _error(403, "Request origin not allowed")
    svc.logout(request.cookies.get(COOKIE))
    resp = Response(status_code=204, headers=NO_STORE)
    resp.delete_cookie(COOKIE, path=COOKIE_PATH, httponly=True, samesite="strict", secure=_secure(request))
    return resp


@router.post("/activity", dependencies=PUBLIC)
def activity(request: Request) -> JSONResponse:
    """The person at the browser interacted (click, key, scroll, touch): extend the idle deadline (never beyond the
    12-hour absolute limit). This is the ONLY request that extends a session - polling, streams and every other API
    call only check it. Needs the session cookie and an allowed browser Origin (CORS_ORIGINS or this API's own)."""
    svc = _auth(request)
    if svc is None:
        return _disabled()
    origin = request.headers.get("origin")
    if origin is None or not allowed_origin(request, origin):
        return _error(403, "Request origin not allowed")
    p = svc.record_activity(request.cookies.get(COOKIE))
    if p is None:
        return _error(401, "Authentication required: sign in")
    return JSONResponse({"idle_expires_at": _iso(p.session_expires_at)}, headers=NO_STORE)


@router.get("/me", dependencies=PUBLIC)
def me(request: Request, authorization: Optional[str] = Header(default=None),
       x_admin_token: Optional[str] = Header(default=None)) -> JSONResponse:
    p = current_principal(request, authorization, x_admin_token)
    body: dict[str, Any] = {"auth_mode": _cfg(request).auth_mode, "authenticated": p is not None}
    if p:
        body.update(username=p.username, role=p.role, via=p.via, idle_expires_at=_iso(p.session_expires_at))
    return JSONResponse(body, headers=NO_STORE)
