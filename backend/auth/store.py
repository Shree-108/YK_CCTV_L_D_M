"""SQLite storage for users, sessions and failed-login counters - a SEPARATE file from the detection database.

Only password hashes and SHA-256 hashes of session IDs are stored; a raw session ID never touches the disk.
Usernames are normalised to lower case. Times are UNIX seconds (UTC).
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

ROLES = ("viewer", "operator", "admin")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('viewer', 'operator', 'admin')),
    password_hash TEXT NOT NULL,
    disabled INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    password_changed_at REAL NOT NULL,
    last_login_at REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    id_hash TEXT PRIMARY KEY,
    username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    idle_expires_at REAL NOT NULL,
    absolute_expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(username);
CREATE TABLE IF NOT EXISTS login_failures (       -- key: 'user:<name>' (existing or not) or 'client:<address>'
    key TEXT PRIMARY KEY,
    failed_count INTEGER NOT NULL DEFAULT 0,
    first_failed_at REAL,
    locked_until REAL
);
"""


class AuthStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        try:
            with self._lock:
                yield c
                c.commit()
        finally:
            c.close()

    # ------------------------------------------------------------------ users
    def get_user(self, username: str) -> Optional[dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM users WHERE username=?", (username.lower(),)).fetchone()
        return dict(r) if r else None

    def list_users(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT username, role, disabled, created_at, password_changed_at, last_login_at "
                "FROM users ORDER BY username")]

    def add_user(self, username: str, role: str, password_hash: str, now: float) -> None:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        with self._conn() as c:
            try:
                c.execute("INSERT INTO users (username, role, password_hash, created_at, password_changed_at) "
                          "VALUES (?,?,?,?,?)", (username.lower(), role, password_hash, now, now))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"user '{username.lower()}' already exists") from exc

    _UPDATABLE = {"role", "password_hash", "disabled", "password_changed_at", "last_login_at"}

    def update_user(self, username: str, **fields: Any) -> bool:
        if not fields:
            return False
        bad = set(fields) - self._UPDATABLE
        if bad:
            raise ValueError(f"cannot update {sorted(bad)}")
        if "role" in fields and fields["role"] not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as c:
            return c.execute(f"UPDATE users SET {cols} WHERE username=?",
                             (*fields.values(), username.lower())).rowcount > 0

    # ------------------------------------------------------------------ sessions
    def add_session(self, id_hash: str, username: str, now: float, idle_expires: float, absolute_expires: float) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)",
                      (id_hash, username.lower(), now, now, idle_expires, absolute_expires))

    def get_session(self, id_hash: str) -> Optional[dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT s.*, u.role, u.disabled FROM sessions s JOIN users u ON u.username = s.username "
                          "WHERE s.id_hash=?", (id_hash,)).fetchone()
        return dict(r) if r else None

    def touch_session(self, id_hash: str, now: float, idle_expires: float) -> None:
        with self._conn() as c:
            c.execute("UPDATE sessions SET last_seen_at=?, idle_expires_at=? WHERE id_hash=?", (now, idle_expires, id_hash))

    def delete_session(self, id_hash: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM sessions WHERE id_hash=?", (id_hash,)).rowcount > 0

    def delete_user_sessions(self, username: str) -> int:
        with self._conn() as c:
            return c.execute("DELETE FROM sessions WHERE username=?", (username.lower(),)).rowcount

    def purge_expired_sessions(self, now: float) -> int:
        with self._conn() as c:
            return c.execute("DELETE FROM sessions WHERE idle_expires_at<=? OR absolute_expires_at<=?",
                             (now, now)).rowcount

    def count_sessions(self, username: Optional[str] = None) -> int:
        with self._conn() as c:
            if username:
                return c.execute("SELECT COUNT(*) FROM sessions WHERE username=?", (username.lower(),)).fetchone()[0]
            return c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    # ------------------------------------------------------------------ failed-login counters
    # Same table for usernames (existing or not) and client addresses, so a lock-out behaves identically whether
    # or not an account exists (no username enumeration).
    def get_failures(self, key: str) -> Optional[dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM login_failures WHERE key=?", (key,)).fetchone()
        return dict(r) if r else None

    def set_failures(self, key: str, failed_count: int, first_failed_at: Optional[float],
                     locked_until: Optional[float]) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO login_failures VALUES (?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
                      "failed_count=excluded.failed_count, first_failed_at=excluded.first_failed_at, "
                      "locked_until=excluded.locked_until", (key, failed_count, first_failed_at, locked_until))

    def clear_failures(self, key: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM login_failures WHERE key=?", (key,))
