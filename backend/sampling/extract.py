"""Decode the planned frames in order (never by seeking) and save them as screenshots.

Seeking returns grey or wrong frames on HEVC/MPEG-PS NVR exports, so the video is decoded once from the start
up to the last planned frame. Each frame's time is its own presentation timestamp relative to frame 0.
An unreadable frame is recorded as a failure and is never replaced by a neighbouring frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from backend.sampling import frame_quality
from backend.sampling.plan import Sample, hhmmss


class Stopped(Exception):
    pass


@dataclass
class Extracted:
    sample: Sample
    status: str                      # saved | unreadable | failed
    seconds: Optional[float]         # decoded presentation time (or the planned time when not decoded)
    path: Optional[Path] = None
    image: Optional[np.ndarray] = None
    error: Optional[str] = None


class SequentialReader:
    """Frames of a cv2.VideoCapture in decode order, from frame 0. Ask for frames in increasing order."""

    def __init__(self, cap: cv2.VideoCapture, fps: float) -> None:
        self.cap, self.fps = cap, fps
        self.next = 0
        self.origin_ms = 0.0
        self.known = (0, 0.0)

    def _grab(self) -> Optional[float]:
        if not self.cap.grab():
            return None
        n, ms = self.next, self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.next += 1
        if n == 0:
            self.origin_ms = ms
            return 0.0
        if ms > 0:
            t = (ms - self.origin_ms) / 1000
            if t > self.known[1]:
                self.known = (n, t)
                return t
        return self.known[1] + (n - self.known[0]) / self.fps

    def read(self, number: int, should_stop: Callable[[], bool]) -> tuple[Optional[np.ndarray], Optional[float], str]:
        if number < self.next:
            return None, None, f"frame {number} was already passed"
        while self.next < number:
            if self.next % 250 == 0 and should_stop():
                raise Stopped()
            if self._grab() is None:
                return None, None, f"the video ends after {self.next} decodable frames, before frame {number}"
        t = self._grab()
        if t is None:
            return None, None, f"the video ends after {self.next} decodable frames, before frame {number}"
        ok, frame = self.cap.retrieve()
        if not ok or frame is None:
            return None, t, "the decoder returned no image for this frame"
        return frame, t, ""


def frame_name(stem: str, s: Sample, seconds: Optional[float]) -> str:
    t = hhmmss(seconds if seconds is not None else s.seconds, millis=True).replace(":", "").replace(".", "")
    return f"{stem}_s{s.idx}_f{s.frame:07d}_t{t}.jpg"


def extract(video: Path, fps: float, samples: list[Sample], out_dir: Path, stem: str,
            should_stop: Callable[[], bool] = lambda: False,
            on_frame: Callable[[Extracted], None] = lambda e: None) -> list[Extracted]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("The video could not be opened (missing codec or unreadable file).")
    reader = SequentialReader(cap, fps)
    results: list[Extracted] = []
    try:
        for s in sorted(samples, key=lambda x: x.frame):
            if should_stop():
                raise Stopped()
            try:
                img, t, err = reader.read(s.frame, should_stop)
            except cv2.error as exc:
                img, t, err = None, None, f"decoder error: {exc}"
            if img is None:
                e = Extracted(s, "failed", t if t is not None else s.seconds, error=err or "frame could not be decoded")
            elif (why := frame_quality.problem(img)) is not None:
                e = Extracted(s, "unreadable", t, error=why)
            else:
                path = out_dir / frame_name(stem, s, t)
                if cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    e = Extracted(s, "saved", t, path=path, image=img)
                else:
                    e = Extracted(s, "failed", t, error=f"could not write {path.name}")
            results.append(e)
            on_frame(e)
    finally:
        cap.release()
    return results
