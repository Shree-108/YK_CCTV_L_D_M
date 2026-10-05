"""Authenticated RTSP (NVR / IP camera) configuration, credentials and connection testing.

* The camera row stores only NON-secret connection details (host, port, channel, stream type,
  username, URL template) in ``rtsp_json``. The password is never stored in the database.
* Credentials come from the environment (``.env`` for local development):
      CCTV_<CAMERA_ID>_USERNAME / CCTV_<CAMERA_ID>_PASSWORD   (per camera, preferred)
      CCTV_USERNAME / CCTV_PASSWORD                           (fallback shared by all cameras)
  A password typed in the UI is written to the gitignored ``.env`` under the per-camera key.
* The full URL (with password) is only built in memory at connect time and never logged or returned;
  anything shown to a user goes through :func:`mask_url`.
* The URL template differs between NVR brands/firmware. The default below is the common
  CP Plus / Dahua pattern and MUST be confirmed by the CCTV administrator for each installation.
"""
from __future__ import annotations

import base64
from pathlib import Path
import hashlib
import os
import re
import socket
import time
import uuid
from typing import Any, Literal, Optional
from urllib.parse import quote

import cv2
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.core.config import PROJECT_ROOT

# FFmpeg prints failing stream URLs (with credentials) to stderr: keep it quiet process-wide
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

DEFAULT_TEMPLATE = "/cam/realmonitor?channel={channel}&subtype={subtype}"   # CP Plus / Dahua style: CONFIRM per site
HOST_RE = re.compile(r"^(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)*[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?$"
                     r"|^\d{1,3}(?:\.\d{1,3}){3}$|^\[[0-9A-Fa-f:]+\]$")
CRED_URL_RE = re.compile(r"(\w+://)([^:/@\s]+):([^@/\s]+)@")
ENV_PATH = PROJECT_ROOT / ".env"


class RtspConfig(BaseModel):
    """Non-secret connection details of one NVR channel."""
    model_config = ConfigDict(extra="forbid")
    host: str = Field(min_length=1, max_length=253, description="NVR/camera IP or hostname on the LAN/VPN")
    port: int = Field(554, ge=1, le=65535)
    channel: int = Field(1, ge=1, le=256)
    stream_type: Literal["main", "sub"] = "sub"
    username: str = Field("", max_length=64)
    path_template: str = Field(DEFAULT_TEMPLATE, max_length=300,
                               description="RTSP path; placeholders {channel} {subtype} (0=main, 1=sub)")

    @field_validator("host")
    @classmethod
    def _host(cls, v: str) -> str:
        v = v.strip()
        if "://" in v or "@" in v or "/" in v or not HOST_RE.match(v):
            raise ValueError("host must be a plain IP address or hostname (no scheme, path or credentials)")
        return v

    @field_validator("path_template")
    @classmethod
    def _tpl(cls, v: str) -> str:
        v = v.strip() or DEFAULT_TEMPLATE
        if not v.startswith("/"):
            v = "/" + v
        if "@" in v or "{password}" in v or "{username}" in v:
            raise ValueError("the path template must not contain credentials")
        try:
            v.format(channel=1, subtype=0)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError("unknown placeholder in path template (use {channel} and {subtype})") from exc
        return v

    @property
    def subtype(self) -> int:
        return 0 if self.stream_type == "main" else 1

    def path(self) -> str:
        return self.path_template.format(channel=self.channel, subtype=self.subtype)


# ------------------------------------------------------------------ credentials
def _env_key(camera_id: str, what: str) -> str:
    return f"CCTV_{re.sub(r'[^A-Za-z0-9]', '_', camera_id).upper()}_{what}"


def credentials(camera_id: str, cfg: RtspConfig) -> tuple[str, str]:
    user = cfg.username or os.getenv(_env_key(camera_id, "USERNAME")) or os.getenv("CCTV_USERNAME", "")
    pwd = os.getenv(_env_key(camera_id, "PASSWORD")) or os.getenv("CCTV_PASSWORD", "")
    return user, pwd


def has_password(camera_id: str) -> bool:
    return bool(os.getenv(_env_key(camera_id, "PASSWORD")) or os.getenv("CCTV_PASSWORD"))


def store_password(camera_id: str, password: str) -> None:
    """Persist a password to the gitignored .env (never to the database) and load it into the process."""
    from dotenv import set_key

    key = _env_key(camera_id, "PASSWORD")
    if not ENV_PATH.exists():
        ENV_PATH.touch()
    set_key(str(ENV_PATH), key, password, quote_mode="always")
    os.environ[key] = password


def env_defaults() -> dict[str, Any]:
    """Non-secret defaults from CCTV_* variables, to prefill the form."""
    out: dict[str, Any] = {}
    for key, field in (("CCTV_HOST", "host"), ("CCTV_RTSP_PORT", "port"), ("CCTV_CHANNEL", "channel"),
                       ("CCTV_STREAM_TYPE", "stream_type"), ("CCTV_USERNAME", "username"),
                       ("CCTV_RTSP_PATH", "path_template")):
        if os.getenv(key):
            out[field] = os.getenv(key)
    out["has_password"] = bool(os.getenv("CCTV_PASSWORD"))
    return out


# ------------------------------------------------------------------ URLs
def build_url(camera_id: str, cfg: RtspConfig, password: Optional[str] = None) -> str:
    """Full RTSP URL WITH credentials. Keep it in memory only; never log or return it."""
    user, pwd = credentials(camera_id, cfg)
    if password is not None:
        pwd = password
    auth = f"{quote(user, safe='')}:{quote(pwd, safe='')}@" if user else ""
    return f"rtsp://{auth}{cfg.host}:{cfg.port}{cfg.path()}"


def mask_url(url: str) -> str:
    return CRED_URL_RE.sub(r"\1\2:********@", url or "")


def display_url(cfg: RtspConfig) -> str:
    user = f"{cfg.username or '<user>'}:********@"
    return f"rtsp://{user}{cfg.host}:{cfg.port}{cfg.path()}"


def sanitize(text: str, *secrets: str) -> str:
    """Remove credentials from any message before it reaches a log, the UI or the database."""
    text = mask_url(str(text))
    for s in secrets:
        if s:
            text = text.replace(s, "********").replace(quote(s, safe=""), "********")
    return text


# ------------------------------------------------------------------ connection test
class TestResult(BaseModel):
    ok: bool
    code: str            # connected | invalid_config | unreachable | timeout | auth_failed | channel_unavailable | stream_unavailable
    message: str
    stages: list[dict[str, Any]] = []
    resolution: Optional[list[int]] = None
    fps: Optional[float] = None
    url: Optional[str] = None     # masked


MESSAGES = {
    "connected": "CONNECTED - camera stream received successfully.",
    "invalid_config": "Invalid configuration.",
    "unreachable": "Unable to reach NVR (connection refused or host unreachable).",
    "timeout": "Network timeout - the NVR did not answer. Check the IP, port and that this computer is on the NVR's LAN/VPN.",
    "auth_failed": "Authentication failed - check the username and password.",
    "channel_unavailable": "Channel unavailable - the NVR rejected this channel/stream path.",
    "stream_unavailable": "RTSP stream unavailable - the NVR accepted the request but sent no video.",
}


def _digest(user: str, pwd: str, method: str, uri: str, hdr: str) -> str:
    f = dict(re.findall(r'(\w+)="?([^",]+)"?', hdr))
    realm, nonce, qop = f.get("realm", ""), f.get("nonce", ""), f.get("qop")
    ha1 = hashlib.md5(f"{user}:{realm}:{pwd}".encode()).hexdigest()
    ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
    if qop:
        nc, cnonce = "00000001", uuid.uuid4().hex[:16]
        resp = hashlib.md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}".encode()).hexdigest()
        return (f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{uri}", '
                f'response="{resp}", qop=auth, nc={nc}, cnonce="{cnonce}"')
    resp = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
    return f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{uri}", response="{resp}"'


def rtsp_describe(host: str, port: int, path: str, user: str, pwd: str, timeout: float = 4.0) -> int:
    """RTSP DESCRIBE with Basic/Digest auth. Returns the RTSP status code (200, 401, 404, ...)."""
    uri = f"rtsp://{host}:{port}{path}"

    def send(sock: socket.socket, cseq: int, auth: Optional[str]) -> tuple[int, dict[str, str]]:
        lines = [f"DESCRIBE {uri} RTSP/1.0", f"CSeq: {cseq}", "Accept: application/sdp", "User-Agent: LitterMonitor"]
        if auth:
            lines.append(f"Authorization: {auth}")
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        data = b""
        while b"\r\n\r\n" not in data and len(data) < 65536:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        head = data.split(b"\r\n\r\n", 1)[0].decode("latin-1", "replace").split("\r\n")
        m = re.match(r"RTSP/1\.\d (\d{3})", head[0] if head else "")
        if not m:
            raise ConnectionError("not an RTSP server")
        headers: dict[str, str] = {}
        for line in head[1:]:
            k, _, v = line.partition(":")
            headers.setdefault(k.strip().lower(), v.strip())
            if k.strip().lower() == "www-authenticate" and "digest" in v.lower():
                headers["www-authenticate"] = v.strip()   # prefer Digest over Basic
        return int(m.group(1)), headers

    with socket.create_connection((host.strip("[]"), port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        code, headers = send(sock, 1, None)
        if code == 401 and user:
            challenge = headers.get("www-authenticate", "")
            if challenge.lower().startswith("digest"):
                auth = _digest(user, pwd, "DESCRIBE", uri, challenge)
            else:
                auth = "Basic " + base64.b64encode(f"{user}:{pwd}".encode()).decode()
            code, _ = send(sock, 2, auth)
        return code


def test_connection(camera_id: str, cfg: RtspConfig, password: Optional[str] = None,
                    frames: int = 5, timeout: float = 4.0) -> TestResult:
    """1) TCP reachability 2) RTSP DESCRIBE (auth + channel) 3) decode real frames with OpenCV/FFmpeg."""
    user, pwd = credentials(camera_id, cfg)
    if password is not None:
        pwd = password
    stages: list[dict[str, Any]] = []

    def fail(code: str, detail: str = "", stage: str = "network") -> TestResult:
        stages.append({"stage": stage, "ok": False, "detail": sanitize(detail, pwd)})
        return TestResult(ok=False, code=code, message=MESSAGES[code], stages=stages, url=display_url(cfg))

    if not user:
        return TestResult(ok=False, code="invalid_config", message="Invalid configuration: username is required.",
                          url=display_url(cfg))
    if not pwd:
        return TestResult(ok=False, code="invalid_config", message="Invalid configuration: password is required.",
                          url=display_url(cfg))
    # 1) network
    t0 = time.perf_counter()
    try:
        socket.create_connection((cfg.host.strip("[]"), cfg.port), timeout=timeout).close()
    except socket.timeout:
        return fail("timeout", "TCP connect timed out")
    except OSError as exc:
        return fail("unreachable", exc.strerror or type(exc).__name__)
    stages.append({"stage": "network", "ok": True, "detail": f"TCP {cfg.port} open ({(time.perf_counter() - t0) * 1000:.0f} ms)"})
    # 2) RTSP handshake: tells authentication failures apart from wrong channel/path
    try:
        code = rtsp_describe(cfg.host, cfg.port, cfg.path(), user, pwd, timeout)
    except socket.timeout:
        return fail("timeout", "RTSP DESCRIBE timed out", "rtsp")
    except (OSError, ConnectionError) as exc:
        return fail("stream_unavailable", f"RTSP handshake failed: {type(exc).__name__}", "rtsp")
    if code in (401, 403):
        return fail("auth_failed", f"RTSP {code}", "rtsp")
    if code in (400, 404, 451, 454, 457):
        return fail("channel_unavailable", f"RTSP {code}", "rtsp")
    if code != 200:
        return fail("stream_unavailable", f"RTSP {code}", "rtsp")
    stages.append({"stage": "rtsp", "ok": True, "detail": "RTSP DESCRIBE 200 (authenticated)"})
    # 3) a real, valid frame - decoded in a separate killable process (FFmpeg can block in native code)
    import tempfile

    from backend.cameras.grabber import grab_frame

    with tempfile.TemporaryDirectory() as tmp:
        t1 = time.perf_counter()
        res = grab_frame(build_url(camera_id, cfg, pwd), Path(tmp) / "test.jpg", timeout=max(10.0, timeout * 4))
    if not res.get("ok"):
        code = "timeout" if res.get("code") == "timeout" else "stream_unavailable"
        return fail(code, f"{res.get('code')}: {res.get('message')}", "video")
    stages.append({"stage": "video", "ok": True,
                   "detail": f"valid frame decoded ({res['width']}x{res['height']}, {time.perf_counter() - t1:.1f} s)"})
    return TestResult(ok=True, code="connected", message=MESSAGES["connected"], stages=stages,
                      resolution=[res["width"], res["height"]], fps=None, url=display_url(cfg))


def camera_url(cam: dict[str, Any]) -> str:
    """Resolve a camera row whose source is 'cctv:<id>' into its (secret) RTSP URL."""
    raw = cam.get("rtsp") or {}
    if not raw.get("host"):
        raise ValueError("CCTV configuration required")
    cfg = RtspConfig.model_validate(raw)
    user, pwd = credentials(cam["id"], cfg)
    if not user or not pwd:
        raise ValueError("CCTV credentials are not configured")
    return build_url(cam["id"], cfg)


def is_cctv(cam: Optional[dict[str, Any]]) -> bool:
    return bool(cam) and str(cam.get("source", "")).startswith("cctv:")

