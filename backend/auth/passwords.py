"""Password hashing with the standard library's scrypt (no extra dependency).

Stored format: ``scrypt$<n>$<r>$<p>$<salt b64>$<hash b64>``. The cost parameters travel with each hash, so they can
be raised later without invalidating existing accounts (``needs_rehash``). N=2**15, r=8 (32 MiB, ~0.15 s on the
project laptop) is a compromise for a CPU-only PC; raise ``N`` on stronger hardware.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

N, R, P, DKLEN, SALT_BYTES = 2 ** 15, 8, 1, 32, 16
MAXMEM = 64 * 1024 * 1024
MIN_LENGTH, MAX_LENGTH = 12, 256          # MAX bounds the work an attacker can force per request


class PasswordPolicyError(ValueError):
    pass


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def check_policy(password: str, min_length: int = MIN_LENGTH) -> None:
    if not isinstance(password, str):
        raise PasswordPolicyError("password must be text")
    if len(password) < min_length:
        raise PasswordPolicyError(f"password must be at least {min_length} characters")
    if len(password) > MAX_LENGTH:
        raise PasswordPolicyError(f"password must be at most {MAX_LENGTH} characters")
    if password.strip() != password or not password.strip():
        raise PasswordPolicyError("password must not start or end with spaces")


def hash_password(password: str, min_length: int = MIN_LENGTH) -> str:
    check_policy(password, min_length)
    salt = secrets.token_bytes(SALT_BYTES)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=N, r=R, p=P, maxmem=MAXMEM, dklen=DKLEN)
    return f"scrypt${N}${R}${P}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check. Any malformed hash or oversize input simply fails."""
    try:
        if not isinstance(password, str) or len(password) > MAX_LENGTH:
            return False
        algo, n, r, p, salt_b64, dk_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        expected = base64.b64decode(dk_b64)
        dk = hashlib.scrypt(password.encode("utf-8"), salt=base64.b64decode(salt_b64), n=int(n), r=int(r), p=int(p),
                            maxmem=MAXMEM, dklen=len(expected))
        return hmac.compare_digest(dk, expected)
    except (ValueError, TypeError):
        return False


def needs_rehash(stored: str) -> bool:
    try:
        algo, n, r, p, *_ = stored.split("$")
        return algo != "scrypt" or (int(n), int(r), int(p)) != (N, R, P)
    except ValueError:
        return True


# Verified against when the username does not exist, so a login takes the same time either way.
DUMMY_HASH = f"scrypt${N}${R}${P}${_b64(b'0' * SALT_BYTES)}${_b64(b'0' * DKLEN)}"
