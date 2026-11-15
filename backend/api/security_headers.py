"""Security headers on every API response (API security Step 4).

The API never serves web pages: it returns JSON, JPEG frames/snapshots, MJPEG and server-sent event streams, MP4
videos and CSV/ZIP downloads. Its Content-Security-Policy is therefore "nothing", except the media types it does
return, so a response opened directly in a browser tab can never run a script or be framed. The web UI has its own
policy (proxy.ts / frontend/lib/csp.ts). ``X-Content-Type-Options: nosniff`` stops browsers from guessing a different
content type than the one declared.

The optional API docs (/docs, /redoc, off by default) are HTML pages that load Swagger/ReDoc from a CDN, so they are
left without this policy when enabled. Pure ASGI, so streaming responses are not buffered.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable

API_CSP = ("default-src 'none'; img-src 'self'; media-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
           "form-action 'none'")
DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


class SecurityHeadersMiddleware:
    def __init__(self, app: Callable[..., Awaitable[None]], exempt_prefixes: Iterable[str] = ()) -> None:
        self.app = app
        self.exempt = tuple(exempt_prefixes)

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        with_csp = not (self.exempt and scope.get("path", "").startswith(self.exempt))

        async def send_with_headers(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", [])
                           if k.lower() not in (b"content-security-policy", b"x-content-type-options")]
                headers.append((b"x-content-type-options", b"nosniff"))
                if with_csp:
                    headers.append((b"content-security-policy", API_CSP.encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)
