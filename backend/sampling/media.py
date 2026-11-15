"""Uploaded videos for the four-screenshot analysis: store, fingerprint, validate and describe them.

The frame count of NVR exports is often wrong in the container (e.g. hours for a 3-minute clip), so the
video packets are counted without decoding; samples planned from a wrong length would fall past the end.
(Approach adapted from the Yatrikart UDM ``probe``.)
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.sampling import frame_quality
from backend.sampling.plan import recording_start_from_name

PREVIEW_SCAN_FRAMES = 400


class Unreadable(ValueError):
    pass


def new_media_id() -> str:
    return uuid.uuid4().hex[:16]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _packets(path: Path) -> Optional[tuple[int, float]]:
    """(video packets, seconds from first to last packet) read WITHOUT decoding, or None if unsupported."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened() or not cap.set(cv2.CAP_PROP_FORMAT, -1):
            return None
        n, first, last = 0, None, 0.0
        while cap.grab():
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if ms > 0:
                first = ms if first is None else min(first, ms)
                last = max(last, ms)
            n += 1
        return n, ((last - first) / 1000 if first is not None else 0.0)
    finally:
        cap.release()


def _real_length(cap: cv2.VideoCapture, path: Path, reported: int, fps: float) -> tuple[int, Optional[float]]:
    counted = _packets(path)
    if counted is None:                          # no raw mode: count the frames that actually decode
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        n = 0
        while cap.grab():
            n += 1
        return n, (n / fps if fps and n else None)
    packets, span = counted
    if packets == reported:
        return reported, (reported / fps if fps and reported else None)
    n = max(packets - 1, 0)                      # cut exports end with an incomplete packet
    return n, (span + 1 / fps if fps and n else None)


def probe(path: Path, filename: str) -> tuple[dict, np.ndarray]:
    """Metadata + a readable preview frame. Raises Unreadable with a clear reason."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise Unreadable("This video could not be opened (unsupported codec or damaged file).")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if not (0 < fps <= 240):
            raise Unreadable(f"This video reports an invalid frame rate ({fps:g} fps).")
        frames, duration = _real_length(cap, path, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), fps)
        if frames <= 0:
            raise Unreadable("No video frames were found in this file.")
    finally:
        cap.release()
    # preview: the first readable frame, decoded in order (seeking returns grey frames on some NVR exports)
    cap, preview, first = cv2.VideoCapture(str(path)), None, None
    try:
        for _ in range(PREVIEW_SCAN_FRAMES):
            ok, f = cap.read()
            if not ok:
                break
            first = f if first is None else first
            if frame_quality.problem(f) is None:
                preview = f
                break
    finally:
        cap.release()
    if preview is None:
        if first is None:
            raise Unreadable("No frames could be decoded from this video.")
        preview = first
    h, w = preview.shape[:2]
    if w < 16 or h < 16:
        raise Unreadable(f"The video resolution ({w}x{h}) is too small to analyse.")
    meta = {"filename": filename, "width": int(w), "height": int(h), "fps": round(fps, 3), "frames": int(frames),
            "duration_sec": round(duration, 3) if duration else round(frames / fps, 3),
            "suggested_recording_start": recording_start_from_name(filename)}
    return meta, preview
