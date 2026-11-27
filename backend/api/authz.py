"""Route permissions (API security Step 4): one place that decides who may call which API route.

Every route declares exactly one permission with ``dependencies=PUBLIC | VIEWER | OPERATOR | ADMIN | ADMIN_IF_LOGIN``
(tests/test_authz.py fails when a route has none). Roles are ordered viewer < operator < admin.

LM_AUTH_MODE=on (the default)
  * The caller is identified by the session cookie, or by the existing ADMIN_API_TOKEN (which identifies an admin,
    exactly as before - it is not given anything it could not already do).
  * No identity -> 401. Identity with a lower role than the route needs -> 403.
  * Cookie-authenticated writes (POST/PUT/PATCH/DELETE) must come from an allowed origin (CORS_ORIGINS or the API's
    own origin, as seen through Nginx's Host / X-Forwarded-Proto) when the browser sends an Origin header.
    SameSite=Strict does not separate ports or apps on the same host, so this check is what blocks cross-app posts.
  * Long-lived streams re-check the session every STREAM_RECHECK_SEC and end when it has expired or been revoked.
  * Idle timeout (approved policy, all roles): no request extends a session - not polling, health, /me, streams or
    any other API call. Only POST /api/auth/activity does, which the browser sends on real user input.

LM_AUTH_MODE=off - automated tests only, never on a reachable server
  * VIEWER, OPERATOR and ADMIN_IF_LOGIN routes are open.
  * ADMIN routes need ADMIN_API_TOKEN exactly as require_admin always did (503 when it is not set).

Nothing here logs or returns a session ID, password or token.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Callable, Optional
from urllib.parse import urlsplit

from fastapi import Depends, Header, HTTPException, Request

from backend.api.deps import get_state, require_admin
from backend.auth.service import Principal, admin_token_principal
from backend.services.state import AppState

log = logging.getLogger(__name__)

COOKIE = "ldm_session"
ROLE_RANK = {"viewer": 1, "operator": 2, "admin": 3}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
STREAM_RECHECK_SEC = 30.0


def login_enabled(request: Request) -> bool:
    return getattr(request.app.state, "auth", None) is not None


def current_principal(request: Request, authorization: Optional[str], x_admin_token: Optional[str]) -> Optional[Principal]:
    """Session cookie (when login is on) or the existing admin token (always), else None."""
    supplied = x_admin_token or (authorization[7:] if authorization and authorization.lower().startswith("bearer ")
                                 else None)
    p = admin_token_principal(supplied, request.app.state.auth_cfg.admin_token)
    if p:
        return p
    svc = getattr(request.app.state, "auth", None)
    return svc.session_principal(request.cookies.get(COOKIE)) if svc else None


def _unauthenticated() -> HTTPException:
    return HTTPException(401, "Authentication required: sign in", headers={"WWW-Authenticate": "Bearer"})


def allowed_origin(request: Request, origin: str) -> bool:
    if origin == "null":
        return False
    allowed = set(request.app.state.auth_cfg.cors_origins or [])
    own = f"{request.url.scheme}://{request.url.netloc}"
    o = urlsplit(origin)
    return f"{o.scheme}://{o.netloc}" in allowed | {own}


def authorize(request: Request, role: str, off_mode: str, authorization: Optional[str],
              x_admin_token: Optional[str], state: AppState) -> Optional[Principal]:
    """Raise 401/403 unless the caller may use a route needing ``role``; returns the principal (None when open)."""
    if not login_enabled(request):
        if off_mode == "admin_token":
            require_admin(authorization, x_admin_token, state)       # pre-Step-4 behaviour, unchanged
            return Principal("admin-token", "admin", "admin_token")
        return None
    p = current_principal(request, authorization, x_admin_token)
    if p is None:
        raise _unauthenticated()
    if ROLE_RANK.get(p.role, 0) < ROLE_RANK[role]:
        log.warning("authz: %s (%s) denied %s %s - needs %s", p.username, p.role, request.method,
                    request.url.path, role)
        raise HTTPException(403, f"Your role ({p.role}) does not allow this action; it needs the {role} role")
    if p.via == "session" and request.method not in SAFE_METHODS:
        origin = request.headers.get("origin")
        if origin is not None and not allowed_origin(request, origin):
            log.warning("authz: %s %s from origin %r refused for %s", request.method, request.url.path, origin,
                        p.username)
            raise HTTPException(403, "Request origin not allowed")
    request.state.principal = p
    return p


class Permission:
    """FastAPI dependency for one permission level. ``off_mode`` says what happens while login is switched off."""

    def __init__(self, role: Optional[str], off_mode: str = "open") -> None:
        self.role = role
        self.off_mode = off_mode

    def __call__(self, request: Request, authorization: Optional[str] = Header(default=None),
                 x_admin_token: Optional[str] = Header(default=None),
                 state: AppState = Depends(get_state)) -> Optional[Principal]:
        if self.role is None:
            return None
        return authorize(request, self.role, self.off_mode, authorization, x_admin_token, state)

    def __repr__(self) -> str:
        return f"Permission({self.role!r}, off_mode={self.off_mode!r})"


PUBLIC_PERMISSION = Permission(None)
VIEWER_PERMISSION = Permission("viewer")
OPERATOR_PERMISSION = Permission("operator")
ADMIN_PERMISSION = Permission("admin", off_mode="admin_token")
ADMIN_IF_LOGIN_PERMISSION = Permission("admin")          # admin with login on; open (as before) with login off

PUBLIC = [Depends(PUBLIC_PERMISSION)]
VIEWER = [Depends(VIEWER_PERMISSION)]
OPERATOR = [Depends(OPERATOR_PERMISSION)]
ADMIN = [Depends(ADMIN_PERMISSION)]
ADMIN_IF_LOGIN = [Depends(ADMIN_IF_LOGIN_PERMISSION)]


def require_admin_here(request: Request, authorization: Optional[str], x_admin_token: Optional[str],
                       state: AppState, off_mode: str = "admin_token") -> None:
    """For a route whose permission depends on its input (e.g. a camera frame that needs a new
    connection). ``off_mode="open"`` keeps the pre-Step-4 behaviour while login is off."""
    authorize(request, "admin", off_mode, authorization, x_admin_token, state)


def actor_of(request: Request) -> Optional[str]:
    """Who made this change, for audit fields such as an event review (None when login is off)."""
    p = getattr(request.state, "principal", None)
    return getattr(p, "username", None) if p is not None else None


def optional_principal(request: Request) -> Optional[Principal]:
    """Who is calling, without raising (used to decide how much a public route may say)."""
    return current_principal(request, request.headers.get("authorization"), request.headers.get("x-admin-token"))


def full_details(request: Request) -> bool:
    """May this caller see infrastructure details (NVR host/username, camera source URLs, model file paths, host
    diagnostics)? Always with login off (unchanged behaviour); with login on only an admin (session or token)."""
    if not login_enabled(request):
        return True
    p = getattr(request.state, "principal", None) or optional_principal(request)
    return p is not None and p.role == "admin"


_URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s'\"<>]+", re.I)
_WIN_PATH_RE = re.compile(r"(?:\b[A-Za-z]:[\\/]|\\\\)[^\s'\"<>]*")         # C:\... and \\server\share
_POSIX_PATH_RE = re.compile(r"(?<![\w.])/(?:[\w.-]+/)+[\w.-]*")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b")
# IPv6: full form (5+ groups) or compressed with "::" - not times like 12:30:45
_IPV6_RE = re.compile(r"(?<![\w:.])\[?(?:(?:[0-9a-f]{1,4}:){4,7}[0-9a-f]{1,4}"
                      r"|(?:[0-9a-f]{1,4}:){1,6}:(?:[0-9a-f]{1,4}(?::[0-9a-f]{1,4}){0,5})?"
                      r"|::(?:[0-9a-f]{1,4}(?::[0-9a-f]{1,4}){0,6})?)(?:%\w+)?\]?(?::\d+)?(?![\w:])", re.I)
HIDDEN = "[hidden]"


def _infrastructure_terms(state: AppState) -> list[str]:
    """Host names and user names of the configured cameras / NVRs (longest first)."""
    terms: set[str] = set()
    for cam in state.db.list_cameras():
        rtsp = cam.get("rtsp") or {}
        terms.update(t for t in (rtsp.get("host"), rtsp.get("username")) if t)
        src = cam.get("source") or ""
        if "://" in src:
            u = urlsplit(src)
            terms.update(t for t in (u.hostname, u.username) if t)
    return sorted((t for t in terms if len(t) >= 3), key=len, reverse=True)


def text_redactor(request: Request, state: AppState) -> Callable[[Optional[str]], Optional[str]]:
    """Pattern-based redaction for documents read from disk that a non-admin can read (dataset README / data.yaml,
    experiment and evaluation files): hides URLs, IP addresses, file paths and the configured camera host and user
    names. Event messages and error fields use the stricter allowlist (MessageFilter) instead. Identity for admins
    and while login is off. Stored data is not changed - only what this caller receives."""
    if full_details(request):
        return lambda text: text
    terms = [re.compile(r"(?<![\w.-])" + re.escape(t) + r"(?![\w-])", re.I) for t in _infrastructure_terms(state)]

    def redact(text: Optional[str]) -> Optional[str]:
        if not text:
            return text
        for rx in (_URL_RE, _WIN_PATH_RE, _POSIX_PATH_RE, _IPV6_RE, _IPV4_RE, *terms):
            text = rx.sub(HIDDEN, text)
        return text

    return redact


def redact_json(value, redact: Callable[[Optional[str]], Optional[str]]):
    """``redact`` applied to every string inside a JSON-like value (dataset / experiment / evaluation documents read
    from disk, which can contain absolute paths in fields and free text)."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_json(v, redact) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_json(v, redact) for v in value]
    return value


# ---------------------------------------------------------------------------------------------------------------
# Event and error text for non-admins: an allowlist, not a pattern filter. The application writes a known, fixed set
# of event messages (backend log_event calls). Non-admins see a message only when it matches one of these templates,
# whose variable parts are restricted to safe identifiers / numbers / enums. Messages that carry raw exception text
# keep their safe prefix and lose the details; anything else (unknown or new messages) becomes a generic line.
# Error fields (camera / capture / schedule / job errors) are replaced by fixed text. Admins (and login off) see the
# original text. Stored data is never changed.
DETAILS_HIDDEN = "(details visible to administrators)"
_ID = r"[A-Za-z0-9_.\-]{1,64}"
_SAFE_EVENTS: list[tuple[re.Pattern, Optional[str]]] = [(re.compile(p, re.S), r) for p, r in [
    # fixed messages, or messages whose variable parts are identifiers, numbers or enums: shown as written
    (r"camera (created|deleted|reconnected)", None),
    (r"camera updated \([a-z_]+(, [a-z_]+)*\)", None),
    (r"connection test: [a-z_]{1,40}", None),
    (r"CCTV configuration saved|live stream (started|stopped)|stream restart requested", None),
    (r"ROI (saved|reset) \(\d{1,3} zone\(s\)\)", None),          # history (feature removed 2026-11-19)
    (r"alert \d{1,12} -> (new|acknowledged|resolved|false_positive)", None),
    (rf"labels updated: {_ID}/[A-Za-z0-9_.\-]{{1,80}}(/[A-Za-z0-9_.\-]{{1,80}}){{0,6}} \(\d{{1,6}} objects\)", None),
    (rf"dataset version {_ID} created from {_ID}", None),
    (rf"registered model {_ID} from {_ID}", None),
    (r"capture schedule started \(next ([0-9T:.+\-Z]{1,40}|None)\)", None),
    (r"capture schedule (paused|resumed|stopped)", None),
    (r"missed capture while offline: running once shortly", None),
    (r"backend (started|stopped)", None),
    (rf"alert \d{{1,12}} delivered via {_ID}", None),
    (r"global detection settings updated", None),
    (rf"activated model {_ID}|applied recommended settings of {_ID}", None),
    (rf"evaluation {_ID} finished for {_ID}", None),
    (rf"(automatic )?four-screenshot analysis {_ID} queued", None),
    (rf"analysis {_ID}: ROI (approved|not_configured|disabled_in_settings) "
     r"\((registered_camera|saved_camera_roi|full_frame_fallback)\)", None),
    (rf"analysis {_ID} is waiting for ROI confirmation", None),
    (rf"ROI confirmed for analysis {_ID} \((auto|camera|manual|whole_frame)\)", None),
    (rf"four-screenshot analysis {_ID} finished: \d{{1,6}} litter detection\(s\)", None),
    (rf"video analysis {_ID} finished: \d{{1,7}} litter event\(s\), \d{{1,9}} observation\(s\)", None),
    (rf"store {_ID} (created|renamed)", None),
    (rf"litter event {_ID} -> (DETECTED|VERIFIED|REJECTED|RESOLVED)", None),
    # messages that end in raw exception text: keep the safe prefix, drop the details
    (rf"four-screenshot analysis ({_ID}) failed: .*", rf"four-screenshot analysis \1 failed {DETAILS_HIDDEN}"),
    (r"camera disconnected: .*; retrying", f"camera disconnected; retrying {DETAILS_HIDDEN}"),
    (r"processing error: .*", f"processing error {DETAILS_HIDDEN}"),
    (r"capture ([a-z_]{1,20}): ([a-z_]{1,40}) - .*", rf"capture \1: \2 {DETAILS_HIDDEN}"),
    (r"capture failed: .*", f"capture failed {DETAILS_HIDDEN}"),
    (rf"evaluation ({_ID}) failed: .*", rf"evaluation \1 failed {DETAILS_HIDDEN}"),
    (r"snapshot save failed: .*", f"snapshot save failed {DETAILS_HIDDEN}"),
]]
_ERROR_CODE_RE = re.compile(r"^([a-z][a-z_]{1,39}): ")


def safe_event_message(source: str, level: str, message: Optional[str]) -> str:
    """The text a non-admin may see for one event message (see the allowlist above)."""
    text = message or ""
    for rx, replacement in _SAFE_EVENTS:
        if rx.fullmatch(text):
            return text if replacement is None else rx.sub(replacement, text)
    if source == "notifier":
        return f"notification delivery failed {DETAILS_HIDDEN}"
    word = "error" if level == "error" else "warning" if level == "warning" else "event"
    return f"{source or 'system'} {word} {DETAILS_HIDDEN}"


def safe_error_text(text: Optional[str]) -> Optional[str]:
    """Fixed text for an error field shown to a non-admin; a leading machine code (e.g. 'auth_failed: ...') is kept."""
    if not text:
        return text
    m = _ERROR_CODE_RE.match(text)
    return f"{m.group(1)} {DETAILS_HIDDEN}" if m else f"error {DETAILS_HIDDEN}"


class MessageFilter:
    """What one caller may see of event messages and error fields (identity for admins and while login is off)."""

    def __init__(self, full: bool) -> None:
        self.full = full

    def event(self, e: dict) -> dict:
        if self.full:
            return e
        return {**e, "message": safe_event_message(e.get("source", ""), e.get("level", ""), e.get("message")),
                "data_json": None}

    def error(self, text: Optional[str]) -> Optional[str]:
        return text if self.full else safe_error_text(text)


def message_filter(request: Request) -> MessageFilter:
    return MessageFilter(full_details(request))


def _stream_session_valid(svc, raw: Optional[str]) -> bool:
    """A stream's periodic re-check: only checks, never extends the session (separate so tests can observe it)."""
    return svc.session_principal(raw) is not None


def stream_guard(request: Request) -> Callable[[], bool]:
    """For long-lived responses: a cheap callable that is False once the caller's session has ended.

    Re-checks at most every STREAM_RECHECK_SEC and does not extend the session's idle timer.
    """
    p: Optional[Principal] = getattr(request.state, "principal", None)
    svc = getattr(request.app.state, "auth", None)
    if p is None or p.via != "session" or svc is None:
        return lambda: True
    raw = request.cookies.get(COOKIE)
    checked = [time.monotonic(), True]

    def still_valid() -> bool:
        now = time.monotonic()
        if checked[1] and now - checked[0] >= STREAM_RECHECK_SEC:
            checked[0] = now
            checked[1] = _stream_session_valid(svc, raw)
            if not checked[1]:
                log.info("authz: stream %s closed - session of %s ended", request.url.path, p.username)
        return checked[1]

    return still_valid
