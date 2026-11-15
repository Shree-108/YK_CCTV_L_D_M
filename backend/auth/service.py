"""Login, sessions and lock-out (API security plan Step 2). Route permissions use it via backend/api/authz.py.

* A session ID is 256 random bits (``secrets.token_urlsafe(32)``), created fresh at every login, returned only to
  be placed in an HttpOnly cookie, and stored only as its SHA-256.
* Sessions end after ``idle_minutes`` without *user activity* or ``max_session_hours`` after login, at logout, when
  the user is disabled, and when the user's password changes. Checking a session (every API request, polling, streams)
  never extends it; only ``record_activity`` does - called by POST /api/auth/activity when the person at the browser
  actually interacts (Step 4, approved idle policy). Activity can never move the 12-hour absolute limit.
* Failed logins are counted per attempted username (whether or not it exists) and per client address. After
  ``max_failures`` within ``window_minutes`` that key is locked for ``lockout_minutes``. Unknown usernames are
  checked against a dummy hash, so timing and messages are the same as for a wrong password.
* Logs contain usernames only when the account exists, a short fingerprint otherwise (a password typed into the
  username field must not end up in a log). Passwords, session IDs and tokens are never logged.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
import time
from dataclasses import dataclass
from typing import Callable, Optional

from backend.auth.passwords import DUMMY_HASH, MAX_LENGTH, hash_password, needs_rehash, verify_password
from backend.auth.store import ROLES, AuthStore

log = logging.getLogger(__name__)
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._@-]{0,63}$")
MAX_SESSION_ID_LENGTH = 128
ACTIVITY_MIN_INTERVAL_SEC = 30   # record user activity at most twice a minute per session (database writes)


class LoginFailed(Exception):
    """Wrong username or password (deliberately not more specific)."""


class LoginLocked(Exception):
    def __init__(self, retry_after: int) -> None:
        super().__init__("too many failed login attempts")
        self.retry_after = retry_after


@dataclass(frozen=True)
class AuthSettings:
    idle_minutes: float = 30
    max_session_hours: float = 12
    max_failures: int = 5
    window_minutes: float = 15
    lockout_minutes: float = 15
    min_password_length: int = 12


@dataclass(frozen=True)
class Principal:
    username: str
    role: str
    via: str                                   # "session" | "admin_token"
    session_expires_at: Optional[float] = None


def session_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalise_username(username: str) -> str:
    u = (username or "").strip().lower()
    if not USERNAME_RE.match(u):
        raise ValueError("username: 1-64 characters, letters, digits and . _ @ - (starting with a letter or digit)")
    return u


def _fingerprint(text: str) -> str:
    return "#" + hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:8]


def admin_token_principal(supplied: Optional[str], expected: Optional[str]) -> Optional[Principal]:
    """The existing ADMIN_API_TOKEN keeps working: it identifies an admin (constant-time comparison)."""
    if not supplied or not expected:
        return None
    if hmac.compare_digest(supplied.encode(), expected.encode()):
        return Principal("admin-token", "admin", "admin_token")
    return None


class AuthService:
    def __init__(self, store: AuthStore, settings: AuthSettings = AuthSettings(),
                 clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.s = settings
        self.clock = clock

    # ------------------------------------------------------------------ accounts (used by the CLI / tests)
    def add_user(self, username: str, role: str, password: str) -> None:
        u = normalise_username(username)
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        self.store.add_user(u, role, hash_password(password, self.s.min_password_length), self.clock())
        log.info("auth: user %s created with role %s", u, role)

    def set_password(self, username: str, password: str) -> int:
        u = normalise_username(username)
        if not self.store.get_user(u):
            raise ValueError(f"user '{u}' does not exist")
        self.store.update_user(u, password_hash=hash_password(password, self.s.min_password_length),
                               password_changed_at=self.clock())
        ended = self.store.delete_user_sessions(u)
        log.info("auth: password changed for %s; %d session(s) ended", u, ended)
        return ended

    def set_disabled(self, username: str, disabled: bool) -> None:
        u = normalise_username(username)
        if not self.store.update_user(u, disabled=int(disabled)):
            raise ValueError(f"user '{u}' does not exist")
        if disabled:
            self.store.delete_user_sessions(u)
        log.info("auth: user %s %s", u, "disabled" if disabled else "enabled")

    def set_role(self, username: str, role: str) -> None:
        u = normalise_username(username)
        if not self.store.update_user(u, role=role):
            raise ValueError(f"user '{u}' does not exist")
        self.store.delete_user_sessions(u)
        log.info("auth: role of %s changed to %s", u, role)

    def unlock(self, username: str) -> None:
        self.store.clear_failures("user:" + normalise_username(username))

    # ------------------------------------------------------------------ lock-out bookkeeping
    def _locked_for(self, key: str, now: float) -> int:
        f = self.store.get_failures(key)
        if f and f["locked_until"] and f["locked_until"] > now:
            return int(f["locked_until"] - now) + 1
        return 0

    def _register_failure(self, key: str, now: float) -> None:
        f = self.store.get_failures(key)
        window = self.s.window_minutes * 60
        if not f or f["first_failed_at"] is None or now - f["first_failed_at"] > window or \
                (f["locked_until"] and f["locked_until"] <= now):
            count, first = 1, now
        else:
            count, first = f["failed_count"] + 1, f["first_failed_at"]
        locked = now + self.s.lockout_minutes * 60 if count >= self.s.max_failures else None
        self.store.set_failures(key, count, first, locked)

    # ------------------------------------------------------------------ login / sessions
    def login(self, username: str, password: str, client: str) -> tuple[str, Principal]:
        """Returns (raw session ID for the cookie, principal). Raises LoginFailed or LoginLocked."""
        now = self.clock()
        try:
            u = normalise_username(username)
        except ValueError:
            u = None
        client_key = "client:" + (client or "?")
        user_key = "user:" + (u or _fingerprint(username or ""))
        wait = max(self._locked_for(client_key, now), self._locked_for(user_key, now))
        if wait:
            log.warning("auth: login refused (locked) for %s from %s", u or _fingerprint(username or ""), client)
            raise LoginLocked(wait)
        user = self.store.get_user(u) if u else None
        ok_input = isinstance(password, str) and 0 < len(password) <= MAX_LENGTH
        valid = verify_password(password if ok_input else "", user["password_hash"] if user else DUMMY_HASH)
        if not (user and ok_input and valid and not user["disabled"]):
            self._register_failure(client_key, now)
            self._register_failure(user_key, now)
            log.warning("auth: failed login for %s from %s", u if user else _fingerprint(username or ""), client)
            raise LoginFailed()
        self.store.clear_failures(user_key)
        self.store.clear_failures(client_key)
        if needs_rehash(user["password_hash"]):
            self.store.update_user(u, password_hash=hash_password(password, 1))
        self.store.update_user(u, last_login_at=now)
        raw = secrets.token_urlsafe(32)
        absolute = now + self.s.max_session_hours * 3600
        idle = min(now + self.s.idle_minutes * 60, absolute)
        self.store.add_session(session_hash(raw), u, now, idle, absolute)
        self.store.purge_expired_sessions(now)
        log.info("auth: %s logged in from %s", u, client)
        return raw, Principal(u, user["role"], "session", idle)

    def _valid_row(self, raw: Optional[str]) -> Optional[tuple[str, dict, float]]:
        if not raw or len(raw) > MAX_SESSION_ID_LENGTH:
            return None
        h = session_hash(raw)
        row = self.store.get_session(h)
        if not row:
            return None
        now = self.clock()
        if row["disabled"] or now >= row["idle_expires_at"] or now >= row["absolute_expires_at"]:
            self.store.delete_session(h)
            return None
        return h, row, now

    def session_principal(self, raw: Optional[str]) -> Optional[Principal]:
        """Valid session -> Principal; anything else -> None. Only checks: never extends the idle deadline (API
        requests, background polling and stream re-checks must not keep an unattended session alive)."""
        v = self._valid_row(raw)
        if not v:
            return None
        _, row, _ = v
        return Principal(row["username"], row["role"], "session", row["idle_expires_at"])

    def record_activity(self, raw: Optional[str]) -> Optional[Principal]:
        """The person using this session interacted with the application: move the idle deadline to now + idle
        minutes, never beyond the absolute limit. Returns None (and changes nothing) for an invalid/expired session."""
        v = self._valid_row(raw)
        if not v:
            return None
        h, row, now = v
        idle = min(now + self.s.idle_minutes * 60, row["absolute_expires_at"])
        if now - row["last_seen_at"] >= ACTIVITY_MIN_INTERVAL_SEC:
            self.store.touch_session(h, now, idle)
        else:
            idle = row["idle_expires_at"]
        return Principal(row["username"], row["role"], "session", idle)

    def logout(self, raw: Optional[str]) -> bool:
        if not raw or len(raw) > MAX_SESSION_ID_LENGTH:
            return False
        return self.store.delete_session(session_hash(raw))
