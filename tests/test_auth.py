"""API security Step 2: password hashing, login, sessions, lock-out, login endpoints. Synthetic credentials only,
isolated temporary storage, fake clock for expiry/lock-out."""
from __future__ import annotations

import io
import logging
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.auth import passwords as pw
from backend.auth.service import (AuthService, AuthSettings, LoginFailed, LoginLocked, admin_token_principal,
                                  session_hash)
from backend.auth.store import AuthStore
from backend.core.config import AppConfig
from backend.main import create_app

PASS = "correct horse battery"          # synthetic test passwords
OTHER = "another long passphrase"


class Clock:
    def __init__(self) -> None:
        self.t = 1_800_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, minutes: float) -> None:
        self.t += minutes * 60


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def svc(tmp_path: Path, clock: Clock) -> AuthService:
    s = AuthService(AuthStore(tmp_path / "auth.db"), AuthSettings(), clock=clock)
    s.add_user("Alice", "operator", PASS)
    return s


# ---------------------------------------------------------------- passwords
def test_hash_roundtrip_unique_salt_and_no_plaintext():
    h1, h2 = pw.hash_password(PASS), pw.hash_password(PASS)
    assert h1 != h2 and h1.startswith("scrypt$32768$8$1$")
    assert PASS not in h1
    assert pw.verify_password(PASS, h1) and not pw.verify_password(OTHER, h1)
    assert not pw.needs_rehash(h1)


@pytest.mark.parametrize("bad", ["short", "x" * 257, " leading space pw", "trailing space pw ", "            "])
def test_password_policy(bad):
    with pytest.raises(pw.PasswordPolicyError):
        pw.hash_password(bad)


def test_verify_rejects_malformed_and_oversize():
    h = pw.hash_password(PASS)
    assert not pw.verify_password(PASS, "md5$abc")
    assert not pw.verify_password(PASS, "garbage")
    assert not pw.verify_password("y" * 10_000, h)
    assert not pw.verify_password(PASS, pw.DUMMY_HASH)


# ---------------------------------------------------------------- service
def test_login_creates_hashed_session_only(svc, tmp_path):
    raw, p = svc.login("ALICE", PASS, "127.0.0.1")
    assert (p.username, p.role, p.via) == ("alice", "operator", "session")
    assert len(raw) >= 40
    assert svc.session_principal(raw).username == "alice"
    db = (tmp_path / "auth.db").read_bytes() + (tmp_path / "auth.db-wal").read_bytes() if (tmp_path / "auth.db-wal").exists() \
        else (tmp_path / "auth.db").read_bytes()
    assert raw.encode() not in db and PASS.encode() not in db
    assert sqlite3.connect(tmp_path / "auth.db").execute("SELECT id_hash FROM sessions").fetchone()[0] == session_hash(raw)


def test_new_session_id_every_login(svc):
    a, _ = svc.login("alice", PASS, "127.0.0.1")
    b, _ = svc.login("alice", PASS, "127.0.0.1")
    assert a != b and svc.session_principal(a) and svc.session_principal(b)


@pytest.mark.parametrize("user,password", [("alice", OTHER), ("nobody", PASS), ("bad name!", PASS), ("alice", ""),
                                           ("alice", "z" * 300)])
def test_failed_logins_are_indistinguishable(svc, user, password):
    with pytest.raises(LoginFailed):
        svc.login(user, password, "127.0.0.1")


def test_disabled_user_cannot_log_in_and_loses_sessions(svc):
    raw, _ = svc.login("alice", PASS, "127.0.0.1")
    svc.set_disabled("alice", True)
    assert svc.session_principal(raw) is None
    with pytest.raises(LoginFailed):
        svc.login("alice", PASS, "127.0.0.1")
    svc.set_disabled("alice", False)
    assert svc.login("alice", PASS, "127.0.0.1")


def test_username_lockout_then_recovery(svc, clock):
    for i in range(5):
        with pytest.raises(LoginFailed):
            svc.login("alice", OTHER, f"127.0.0.{i + 1}")          # different clients: the USERNAME locks
    with pytest.raises(LoginLocked) as e:
        svc.login("alice", PASS, "127.0.0.9")                   # even the right password is refused
    assert 800 < e.value.retry_after <= 901
    clock.advance(16)
    raw, _ = svc.login("alice", PASS, "127.0.0.9")
    assert svc.session_principal(raw)


def test_unknown_username_locks_the_same_way(svc):
    for _ in range(5):
        with pytest.raises(LoginFailed):
            svc.login("ghost", OTHER, "127.0.0.5")
    with pytest.raises(LoginLocked):
        svc.login("ghost", OTHER, "127.0.0.6")


def test_client_lockout_across_usernames(svc):
    for name in ("u1", "u2", "u3", "u4", "u5"):
        with pytest.raises(LoginFailed):
            svc.login(name, OTHER, "127.0.0.7")
    with pytest.raises(LoginLocked):
        svc.login("alice", PASS, "127.0.0.7")                   # this client is locked
    assert svc.login("alice", PASS, "127.0.0.8")                # another client is not


def test_failures_outside_the_window_do_not_lock(svc, clock):
    for _ in range(4):
        with pytest.raises(LoginFailed):
            svc.login("alice", OTHER, "127.0.0.1")
    clock.advance(16)
    with pytest.raises(LoginFailed):
        svc.login("alice", OTHER, "127.0.0.1")                  # new window: count restarts at 1
    assert svc.login("alice", PASS, "127.0.0.1")


def test_success_clears_failure_count(svc):
    for _ in range(4):
        with pytest.raises(LoginFailed):
            svc.login("alice", OTHER, "127.0.0.1")
    svc.login("alice", PASS, "127.0.0.1")
    for _ in range(4):
        with pytest.raises(LoginFailed):
            svc.login("alice", OTHER, "127.0.0.1")
    assert svc.login("alice", PASS, "127.0.0.1")                # 4 + 4 failures never reached 5 in a row


def test_idle_and_absolute_expiry(svc, clock):
    raw, _ = svc.login("alice", PASS, "127.0.0.1")
    clock.advance(31)
    assert svc.session_principal(raw) is None                   # idle 30 min
    raw, _ = svc.login("alice", PASS, "127.0.0.1")
    start, ended_after = clock.t, None
    while clock.t - start < 13 * 3600:                          # user activity every 20 min ...
        clock.advance(20)
        if svc.record_activity(raw) is None:
            ended_after = clock.t - start
            break
    assert ended_after is not None                              # ... the session still ends,
    assert 12 * 3600 <= ended_after < 12 * 3600 + 20 * 60       # at the 12 h absolute limit, not before


def test_checking_a_session_never_extends_it(svc, clock):
    raw, _ = svc.login("alice", PASS, "127.0.0.1")
    for _ in range(29):                                         # checked every minute (polling, streams, requests)
        clock.advance(1)
        assert svc.session_principal(raw) is not None
    clock.advance(1.1)
    assert svc.session_principal(raw) is None                   # 30 min after login: ended anyway
    raw, _ = svc.login("alice", PASS, "127.0.0.1")
    clock.advance(20)
    p = svc.record_activity(raw)                                # user activity moves the deadline
    assert p and p.session_expires_at == clock.t + 30 * 60
    clock.advance(29)
    assert svc.session_principal(raw) is not None


def test_logout_password_change_and_role_change_end_sessions(svc):
    a, _ = svc.login("alice", PASS, "127.0.0.1")
    assert svc.logout(a) and svc.session_principal(a) is None
    b, _ = svc.login("alice", PASS, "127.0.0.1")
    c, _ = svc.login("alice", PASS, "127.0.0.1")
    assert svc.set_password("alice", OTHER) == 2
    assert svc.session_principal(b) is None and svc.session_principal(c) is None
    with pytest.raises(LoginFailed):
        svc.login("alice", PASS, "127.0.0.1")
    d, _ = svc.login("alice", OTHER, "127.0.0.1")
    svc.set_role("alice", "viewer")
    assert svc.session_principal(d) is None
    assert svc.login("alice", OTHER, "127.0.0.1")[1].role == "viewer"


@pytest.mark.parametrize("raw", [None, "", "x" * 500, "not-a-real-session-id"])
def test_invalid_session_ids(svc, raw):
    assert svc.session_principal(raw) is None and not svc.logout(raw)


def test_admin_token_principal():
    assert admin_token_principal("tok-123", "tok-123").role == "admin"
    assert admin_token_principal("tok-124", "tok-123") is None
    assert admin_token_principal(None, "tok-123") is None
    assert admin_token_principal("tok-123", None) is None


def test_add_user_validation(svc):
    with pytest.raises(ValueError):
        svc.add_user("alice", "viewer", OTHER)                  # duplicate (case-insensitive)
    with pytest.raises(ValueError):
        svc.add_user("bob", "superuser", OTHER)
    with pytest.raises(pw.PasswordPolicyError):
        svc.add_user("bob", "viewer", "short")


def test_logs_never_contain_secrets(svc, caplog):
    caplog.set_level(logging.DEBUG)
    raw, _ = svc.login("alice", PASS, "127.0.0.1")
    with pytest.raises(LoginFailed):
        svc.login("alice", OTHER, "127.0.0.1")
    with pytest.raises(LoginFailed):
        svc.login(PASS, OTHER, "127.0.0.1")                     # password typed into the username field
    text = caplog.text
    assert PASS not in text and OTHER not in text and raw not in text
    assert "alice" in text                                      # existing usernames are logged, nothing else


# ---------------------------------------------------------------- endpoints
def _cfg(tmp_path: Path, mode: str, secure: str = "auto") -> AppConfig:
    c = AppConfig()
    c.storage_dir = tmp_path / "storage"
    c.db_path = tmp_path / "storage" / "t.db"
    c.auth_db_path = tmp_path / "storage" / "auth.db"
    c.admin_token = "test-admin-token"
    c.start_cameras = False
    c.auth_mode = mode
    c.auth_cookie_secure = secure
    c.ensure_dirs()
    return c


def _client(cfg: AppConfig) -> TestClient:
    return TestClient(create_app(cfg), client=("127.0.0.1", 50000))


def test_auth_off_is_default_and_inert(tmp_path):
    cfg = _cfg(tmp_path, AppConfig().auth_mode)
    assert cfg.auth_mode == "off"
    with _client(cfg) as c:
        r = c.post("/api/auth/login", json={"username": "alice", "password": PASS})
        assert r.status_code == 404 and "disabled" in r.json()["detail"]
        assert c.post("/api/auth/logout").status_code == 404
        assert c.get("/api/auth/me").json() == {"auth_mode": "off", "authenticated": False}
        me = c.get("/api/auth/me", headers={"Authorization": "Bearer test-admin-token"}).json()
        assert (me["authenticated"], me["role"], me["via"]) == (True, "admin", "admin_token")
        # existing admin token behaviour is unchanged
        assert c.get("/api/auth/check", headers={"Authorization": "Bearer test-admin-token"}).status_code == 200
        assert c.get("/api/auth/check", headers={"X-Admin-Token": "test-admin-token"}).status_code == 200
        assert c.get("/api/auth/check").status_code == 401
    assert not cfg.auth_db_path.exists()                        # no auth database while off


def test_login_cookie_me_logout_flow(tmp_path):
    cfg = _cfg(tmp_path, "on")
    with _client(cfg) as c:
        c.app.state.auth.add_user("alice", "operator", PASS)
        r = c.post("/api/auth/login", json={"username": "alice", "password": PASS})
        assert r.status_code == 200
        body = r.json()
        assert body["username"] == "alice" and body["role"] == "operator"
        cookie = r.headers["set-cookie"]
        sid = c.cookies.get("ldm_session")
        assert sid and sid not in r.text                         # session ID only in Set-Cookie
        low = cookie.lower()
        assert "httponly" in low and "samesite=strict" in low and "path=/api" in low and "secure" not in low
        assert r.headers["cache-control"] == "no-store"
        me = c.get("/api/auth/me").json()
        assert (me["authenticated"], me["username"], me["role"], me["via"]) == (True, "alice", "operator", "session")
        assert "ldm_session" not in str(me) and sid not in str(me)
        assert c.post("/api/auth/logout").status_code == 204
        assert c.get("/api/auth/me").json()["authenticated"] is False
        c.cookies.set("ldm_session", sid, path="/api")           # replaying the old cookie does not work
        assert c.get("/api/auth/me").json()["authenticated"] is False


def test_secure_cookie_can_be_forced(tmp_path):
    with _client(_cfg(tmp_path, "on", secure="true")) as c:
        c.app.state.auth.add_user("alice", "viewer", PASS)
        r = c.post("/api/auth/login", json={"username": "alice", "password": PASS})
        assert "secure" in r.headers["set-cookie"].lower()


def test_login_errors_are_generic_and_never_echo_the_password(tmp_path):
    with _client(_cfg(tmp_path, "on")) as c:
        c.app.state.auth.add_user("alice", "viewer", PASS)
        wrong = c.post("/api/auth/login", json={"username": "alice", "password": OTHER})
        unknown = c.post("/api/auth/login", json={"username": "nobody", "password": OTHER})
        assert wrong.status_code == unknown.status_code == 401
        assert wrong.json() == unknown.json() == {"detail": "Invalid username or password"}
        for bad in ({"username": "alice", "password": 12345678901234}, {"username": "alice"},
                    {"username": ["alice"], "password": OTHER}):
            r = c.post("/api/auth/login", json=bad)
            assert r.status_code == 400 and OTHER not in r.text and "12345678901234" not in r.text
        r = c.post("/api/auth/login", content=f"username=alice&password={OTHER}",
                   headers={"content-type": "application/x-www-form-urlencoded"})
        assert r.status_code == 400 and OTHER not in r.text
        r = c.post("/api/auth/login", content=b'{"username":"alice","password":"' + b"p" * 5000 + b'"}',
                   headers={"content-type": "application/json"})
        assert r.status_code == 400 and "pppp" not in r.text
        assert "set-cookie" not in wrong.headers


def test_api_lockout_returns_429_with_retry_after(tmp_path):
    with _client(_cfg(tmp_path, "on")) as c:
        c.app.state.auth.add_user("alice", "viewer", PASS)
        for _ in range(5):
            assert c.post("/api/auth/login", json={"username": "alice", "password": OTHER}).status_code == 401
        r = c.post("/api/auth/login", json={"username": "alice", "password": PASS})
        assert r.status_code == 429 and int(r.headers["retry-after"]) > 0


def test_auth_on_keeps_admin_token_and_protects_routes(tmp_path):
    with _client(_cfg(tmp_path, "on")) as c:
        assert c.get("/api/auth/check", headers={"Authorization": "Bearer test-admin-token"}).status_code == 200
        assert c.get("/api/auth/check").status_code == 401
        assert c.get("/api/classes").status_code == 401           # Step 4: routes need a login (tests/test_authz.py)
        assert c.get("/api/classes", headers={"Authorization": "Bearer test-admin-token"}).status_code == 200


def test_auth_endpoints_respect_local_only(tmp_path):
    with TestClient(create_app(_cfg(tmp_path, "on")), client=("192.168.1.20", 50000)) as c:
        assert c.post("/api/auth/login", json={"username": "a", "password": PASS}).status_code == 403


# ---------------------------------------------------------------- config + CLI
@pytest.mark.parametrize("var,val", [("LM_AUTH_MODE", "maybe"), ("LM_AUTH_COOKIE_SECURE", "yes"),
                                     ("LM_AUTH_IDLE_MINUTES", "0"), ("LM_AUTH_MIN_PASSWORD_LENGTH", "4")])
def test_invalid_auth_config_refuses_to_start(monkeypatch, var, val):
    monkeypatch.setenv(var, val)
    with pytest.raises(ValueError):
        AppConfig()


def test_cli_manages_users_without_echoing_passwords(tmp_path, monkeypatch, capsys):
    from backend.auth.__main__ import main
    monkeypatch.setenv("LM_AUTH_DB_PATH", str(tmp_path / "cli_auth.db"))
    monkeypatch.setattr("sys.stdin", io.StringIO(PASS + "\n"))
    assert main(["add-user", "Carol", "--role", "admin", "--password-stdin"]) == 0
    monkeypatch.setattr("sys.stdin", io.StringIO("short\n"))
    assert main(["set-password", "carol", "--password-stdin"]) == 2          # policy enforced
    assert main(["disable", "carol"]) == 0 and main(["list"]) == 0
    out = capsys.readouterr()
    assert "carol" in out.out and "disabled" in out.out
    assert PASS not in out.out + out.err
    with pytest.raises(SystemExit):
        main(["add-user", "dave", "--role", "viewer", PASS])                  # no password arguments


# ---------------------------------------------------------------- Step 3: cookies across the UI/API origins
def test_cors_allows_credentials_only_for_listed_origins(tmp_path):
    cfg = _cfg(tmp_path, "on")
    cfg.cors_origins = ["http://localhost:3010"]
    with _client(cfg) as c:
        ok = c.get("/api/auth/me", headers={"Origin": "http://localhost:3010"})
        assert ok.headers["access-control-allow-origin"] == "http://localhost:3010"     # exact origin, never "*"
        assert ok.headers["access-control-allow-credentials"] == "true"
        pre = c.options("/api/auth/login", headers={"Origin": "http://localhost:3010",
                                                     "Access-Control-Request-Method": "POST",
                                                     "Access-Control-Request-Headers": "content-type"})
        assert pre.status_code == 200 and pre.headers["access-control-allow-credentials"] == "true"
        evil = c.get("/api/auth/me", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in evil.headers
        assert "access-control-allow-credentials" not in evil.headers or \
            evil.headers.get("access-control-allow-origin") is None
        c.app.state.auth.add_user("alice", "viewer", PASS)
        for _ in range(5):
            c.post("/api/auth/login", json={"username": "alice", "password": OTHER}, headers={"Origin": "http://localhost:3010"})
        locked = c.post("/api/auth/login", json={"username": "alice", "password": OTHER}, headers={"Origin": "http://localhost:3010"})
        assert locked.status_code == 429
        assert "retry-after" in locked.headers.get("access-control-expose-headers", "").lower()


def test_cli_strips_powershell_byte_order_mark(tmp_path, monkeypatch):
    """Windows PowerShell 5.1 pipes text with a UTF-8 BOM; the stored password must be the typed one."""
    from backend.auth.__main__ import main
    from backend.auth import build_service
    monkeypatch.setenv("LM_AUTH_DB_PATH", str(tmp_path / "bom_auth.db"))
    monkeypatch.setattr("sys.stdin", io.StringIO("\ufeff" + PASS + "\r\n"))
    assert main(["add-user", "erin", "--role", "viewer", "--password-stdin"]) == 0
    svc = build_service(AppConfig())
    assert svc.login("erin", PASS, "127.0.0.1")[1].username == "erin"
