# API security remediation: Step 2, login and session backend (2026-10-15)

Approved scope: the login and session backend only, **off by default** (`LM_AUTH_MODE=off`). Not included: the
frontend login (Step 3), protecting routes (Step 4) and rate limits (Step 5). Branch `security/step1-local-only`,
based on Step 1 commit `42ae11c`.

## Design

| Topic | Implementation |
|---|---|
| Mode | `LM_AUTH_MODE=off` (default) or `on`; any other value refuses to start. **While off, no auth database is opened or created**; login and logout answer 404 "Login is disabled" |
| Storage | separate SQLite file `LM_AUTH_DB_PATH` (default `storage/auth.db`, git-ignored). The detection database gets no schema change |
| Passwords | stdlib `hashlib.scrypt` (N=2^15, r=8, p=1, 16-byte random salt, parameters stored with each hash, ~0.15 s on this laptop); constant-time comparison; 12–256 characters, no leading or trailing spaces; unknown users are checked against a dummy hash (same timing and message) |
| Sessions | 256-bit random ID (`secrets.token_urlsafe(32)`), new at every login, **stored only as SHA-256**; idle timeout 30 min (sliding), absolute 12 h; ended by logout, password change, role change or disabling the user |
| Cookie | `ldm_session`, **HttpOnly, SameSite=Strict, Path=/api**, Max-Age 12 h; **Secure** when the request is HTTPS (`LM_AUTH_COOKIE_SECURE=auto`) or forced with `true` |
| Lock-out | after 5 failures within 15 min, the **attempted username** (existing or not, so no username enumeration) and the **client address** are each locked for 15 min → `429` with `Retry-After`. A success clears the counters |
| Endpoints | `POST /api/auth/login` (JSON `{username, password}`, body at most 4 KB, parsed by hand so errors never echo input) · `POST /api/auth/logout` (204, clears cookie) · `GET /api/auth/me` (`auth_mode`, `authenticated`, `username`, `role`, `via`, idle expiry: never the session ID). All with `Cache-Control: no-store` |
| ADMIN_API_TOKEN | **unchanged**: `require_admin` and every admin route behave exactly as before in both modes; `/me` also recognises the token as `admin` via `admin_token` |
| Logging | Python logger only: usernames of existing accounts, a short fingerprint for anything else (a password typed into the username field never reaches a log); never passwords, session IDs or tokens |
| Accounts | `py -m backend.auth add-user <name> --role viewer\|operator\|admin`, `set-password`, `set-role`, `disable`, `enable`, `unlock`, `list`. Passwords come from a hidden prompt (typed twice) or `--password-stdin`, **never as arguments** |
| Still unchanged | local-only safeguard (active in both modes); no route requires login yet |

## Files

| File | Change |
|---|---|
| `backend/auth/__init__.py`, `passwords.py`, `store.py`, `service.py`, `__main__.py` | **new**: hashing, storage, login/session/lock-out service, account CLI |
| `backend/api/routes_auth.py` | **new**: `/api/auth/login`, `/logout`, `/me` |
| `backend/core/config.py` | `LM_AUTH_*` settings with validation |
| `backend/main.py` | includes the auth router; creates the auth service only when `LM_AUTH_MODE=on` |
| `.env.example` | documents `LM_AUTH_*` (no values) |
| `tests/conftest.py` | shared test config uses a temporary `auth.db` (never the real one) |
| `tests/test_auth.py` | **new**: 41 tests |
| `docs/security/STEP2_REPORT.md` | this report |

Not changed (verified empty `git diff`): `backend/api/routes.py`, `routes_lab.py`, `deps.py` (`require_admin`),
`routes_invoice.py`, `invoice_extractor/`, `frontend/`, `app/`, `backend/services/`, `backend/detection/`,
`backend/cameras/`, `backend/alerts/`, `backend/database/`, `backend/core/settings.py`, `models/`, `requirements.txt`,
`package.json`.

## Test results (actual, 2026-10-15)

- `tests/test_auth.py`: **41 passed**. These cover:
  - hashing (unique salts, no plaintext, policy, malformed or oversize input);
  - only a hash of the session ID is stored;
  - a new session ID at every login;
  - 5 kinds of failed login all look the same;
  - disabled users;
  - username lock-out with recovery after 15 min;
  - unknown usernames lock the same way;
  - client lock-out across usernames;
  - the failure window resets;
  - a success clears the counters;
  - idle expiry, and the sliding session ending at exactly the 12 h limit;
  - logout, password change and role change end sessions;
  - invalid session IDs;
  - admin-token identification;
  - logs contain no passwords or session IDs;
  - auth off: endpoints inert, admin token unchanged, **no auth.db created**;
  - login → cookie (HttpOnly, SameSite=Strict, Path=/api, not Secure on HTTP) → `/me` → logout, and a replayed cookie rejected;
  - Secure can be forced;
  - generic 401 for wrong or unknown users;
  - bad bodies get 400 without echoing the password;
  - 429 with Retry-After;
  - admin token still works with auth on, while routes stay unprotected until Step 4;
  - auth endpoints are still local-only;
  - invalid config refuses to start;
  - CLI without password arguments, policy enforced, nothing echoed.
- **Full suite: 164 passed, 38 skipped** (22:46 IST) = 123 (Step 1) + 41. Same 38 skips: `test_integration` 19, `test_scheduler` 8, `test_lab_api` 5, `test_capture_faults` 4, `test_roi_upload` 2 (missing v1 model and images).
- **Live database** fingerprint unchanged (`f7c83c2d…` before and after); no real `storage/auth.db` created; model sha256 and effective detection settings identical to the Step 0 baseline; scan of changed files: no real secrets, no real LAN address.

## Remaining limitations (by design for Step 2)

- **No route requires login yet.** Turning `LM_AUTH_MODE=on` only enables login, logout and `/me`; footage and data endpoints are still protected solely by the local-only safeguard (Step 4).
- **The frontend has no login screen yet** (Step 3), and cookies are not yet sent with API calls or EventSource (`credentials: 'include'`, Step 3/4).
- **No Origin/CSRF check yet** beyond SameSite=Strict (Step 4); **no rate limits** other than the login lock-out (Step 5).
- **Lock-out counters can be used to lock someone out:** an attacker who can reach the login (only local users today) can lock a known username for 15 min. `py -m backend.auth unlock <name>` clears it.
- **scrypt cost** (N=2^15) is a compromise for this CPU; raise it on stronger hardware (existing hashes keep working and are upgraded at the next login).
- **No audit trail in the app's event log yet** (logins go to the Python log only); no password expiry or multi-factor login.
