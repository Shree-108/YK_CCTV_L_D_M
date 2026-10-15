"""Local-only safeguard: while login protection is not in place, only this computer may use the API.

The API has no user login yet (API security plan, Step 2-4), so every route - including live CCTV video,
snapshots and data exports - would be readable by anyone who can reach the port. This ASGI middleware rejects
any request whose client address is not a loopback address (127.0.0.0/8, ::1, IPv4-mapped loopback), no matter
which --host uvicorn was started with. It is pure ASGI so streaming responses (MJPEG, server-sent events) are
untouched for allowed clients.

Notes
* uvicorn replaces the client address with the X-Forwarded-For address only for trusted proxies
  (--forwarded-allow-ips, default 127.0.0.1). Do not put a reverse proxy in front of the API before login
  protection exists: a proxy on this PC that does not forward the real client address would look local.
* ``extra_hosts`` exists for automated tests (Starlette's TestClient reports the client as "testclient").
  Production code never sets it.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import time
from typing import Any, Awaitable, Callable, Iterable

log = logging.getLogger(__name__)

MESSAGE = ("Remote access to the Litter Detection API is disabled until login protection is enabled. "
           "Open the application on this computer (http://localhost).")


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


class LocalOnlyMiddleware:
    def __init__(self, app: Callable[..., Awaitable[None]], extra_hosts: Iterable[str] = ()) -> None:
        self.app = app
        self.extra_hosts = frozenset(extra_hosts)
        self._last_log: dict[str, float] = {}

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        host = client[0] if client else None
        if is_loopback(host) or (host is not None and host in self.extra_hosts):
            await self.app(scope, receive, send)
            return
        now = time.monotonic()
        if now - self._last_log.get(host or "?", 0.0) > 60:      # at most one log line per client per minute
            self._last_log[host or "?"] = now
            log.warning("Rejected non-local request from %s to %s (API is local-only until login protection "
                        "is enabled)", host, scope.get("path"))
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"detail": MESSAGE}).encode()
        await send({"type": "http.response.start", "status": 403,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
