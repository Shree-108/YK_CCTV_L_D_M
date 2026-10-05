"""Grab ONE valid frame from a camera in a separate, killable process.

FFmpeg/OpenCV reads can block in native code where Python timeouts cannot interrupt them, so the
scheduler never calls them inside the API process: it runs ``python -m backend.cameras.grabber``,
waits with a hard deadline and kills the child if it overruns.

Protocol (no secrets on the command line - process lists are readable by other users):
    stdin : JSON {"source": "<url|path|index>", "out": "<jpg path>", "random_seek": bool, "warmup": int}
    stdout: JSON {"ok": bool, "code": "...", "message": "...", "width": int, "height": int}
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")   # FFmpeg would print the URL (with credentials) on errors


def frame_problem(frame: Optional[np.ndarray]) -> Optional[str]:
    """Why a decoded frame must not be analysed (None = frame is usable)."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return "empty frame"
    if frame.ndim != 3 or frame.shape[2] != 3:
        return "unexpected frame format"
    h, w = frame.shape[:2]
    if w < 64 or h < 64:
        return f"frame too small ({w}x{h})"
    small = frame[:: max(1, h // 90), :: max(1, w // 160)].astype(np.float32)
    if float(small.std()) < 3.0:
        return "blank/uniform frame (no video signal)"
    # broken H.264 decodes (missing reference frames) fill large areas with exactly mid-grey or pure green
    for bgr in ((128, 128, 128), (0, 135, 0)):
        if (np.abs(small - np.array(bgr, np.float32)).max(axis=2) <= 3).mean() > 0.35:
            return "corrupted frame (decoder artefacts)"
    return None


def _child() -> int:
    import cv2

    from backend.cameras.manager import open_capture

    req = json.loads(sys.stdin.read())
    out: dict[str, Any] = {"ok": False, "code": "no_frame", "message": "stream returned no frames"}
    cap = None
    try:
        src = req["source"]
        cap = open_capture(int(src) if isinstance(src, str) and src.isdigit() else src)
        if not cap.isOpened():
            out.update(code="open_failed", message="could not open camera stream")
        else:
            if req.get("random_seek"):
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if n > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, random.SystemRandom().randrange(n))
            frame, problem = None, "stream returned no frames"
            for _ in range(max(1, int(req.get("warmup", 8)))):   # first RTSP frames are often stale/grey
                ok, f = cap.read()
                if ok and f is not None:
                    p = frame_problem(f)
                    if p is None:
                        frame, problem = f, None
                    elif frame is None:
                        problem = p
            if frame is None:
                out.update(code="invalid_frame" if problem != "stream returned no frames" else "no_frame", message=problem)
            else:
                cv2.imwrite(req["out"], frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                out = {"ok": True, "code": "ok", "message": "frame captured", "width": int(frame.shape[1]),
                       "height": int(frame.shape[0])}
    except Exception as exc:  # noqa: BLE001
        out.update(code="error", message=type(exc).__name__)   # never the exception text: it may contain the URL
    finally:
        if cap is not None:
            cap.release()
    sys.stdout.write(json.dumps(out))
    return 0


def grab_frame(source: Any, out_path: Path, timeout: float = 30.0, random_seek: bool = False) -> dict[str, Any]:
    """Run the child; returns its JSON result, or code 'timeout' / 'crashed'. Never raises for camera problems."""
    from backend.core.config import PROJECT_ROOT

    req = json.dumps({"source": source if not isinstance(source, int) else str(source), "out": str(out_path),
                      "random_seek": random_seek})
    try:
        proc = subprocess.run([sys.executable, "-m", "backend.cameras.grabber"], input=req, capture_output=True,
                              text=True, timeout=timeout, cwd=str(PROJECT_ROOT),
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:   # subprocess.run kills the child before raising
        return {"ok": False, "code": "timeout", "message": f"camera did not deliver a frame within {timeout:.0f} s"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "code": "crashed", "message": f"frame grabber exited with code {proc.returncode}"}


if __name__ == "__main__":
    raise SystemExit(_child())
