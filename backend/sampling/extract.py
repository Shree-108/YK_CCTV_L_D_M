"""Decode the planned frames in order (never by seeking) and save them as screenshots.

Seeking returns grey or wrong frames on HEVC/MPEG-PS NVR exports, so the video is decoded once from the start
up to the last planned frame. Each frame's time is its own presentation timestamp relative to frame 0.
Random (operator) sampling: an unreadable frame is recorded as a failure and is never replaced by a neighbouring
frame. Fixed (automatic) sampling, ``adjust=True``: when the planned frame cannot be decoded, the nearest readable
frame inside the sample's window (``Sample.segment``, at most about one second away) is used instead, and the
substitution is recorded (actual frame number, actual time, reason). If no frame in the window is readable, the
screenshot is a recorded failure - nothing is ever fabricated.
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
    frame: Optional[int] = None      # frame actually used (differs from sample.frame only after an adjustment)
    note: Optional[str] = None       # why and how the planned frame was replaced

    def __post_init__(self) -> None:
        if self.frame is None:
            self.frame = self.sample.frame


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

    def skip_to(self, number: int, should_stop: Callable[[], bool]) -> Optional[str]:
        """Decode (without converting) until the next frame is ``number``. An error text when that is impossible."""
        if number < self.next:
            return f"frame {number} was already passed"
        while self.next < number:
            if self.next % 250 == 0 and should_stop():
                raise Stopped()
            if self._grab() is None:
                return f"the video ends after {self.next} decodable frames, before frame {number}"
        return None

    def take(self) -> tuple[Optional[np.ndarray], Optional[float], str]:
        """The next frame as an image with its time; (None, None, why) at the end of the video."""
        t = self._grab()
        if t is None:
            return None, None, f"the video ends after {self.next} decodable frames"
        ok, frame = self.cap.retrieve()
        if not ok or frame is None:
            return None, t, "the decoder returned no image for this frame"
        return frame, t, ""

    def read(self, number: int, should_stop: Callable[[], bool]) -> tuple[Optional[np.ndarray], Optional[float], str]:
        err = self.skip_to(number, should_stop)
        if err:
            return None, None, err
        img, t, err = self.take()
        if img is None and t is None:
            return None, None, f"the video ends after {self.next} decodable frames, before frame {number}"
        return img, t, err


def _why(img: Optional[np.ndarray], err: str) -> Optional[str]:
    """None when ``img`` is a usable picture, else the reason it is not."""
    if img is None:
        return err or "frame could not be decoded"
    return frame_quality.problem(img)


def _read_nearby(reader: SequentialReader, s: Sample, should_stop: Callable[[], bool]
                 ) -> tuple[Optional[np.ndarray], Optional[float], int, str, Optional[str]]:
    """(image, time, frame used, error, adjustment note) for a sample allowed to move inside ``s.segment``.
    Holds at most two decoded pictures (the last good one before the target, the first good one after it)."""
    lo, hi = s.segment
    err = reader.skip_to(max(lo, reader.next), should_stop)
    if err:
        return None, None, s.frame, err, None
    below: Optional[tuple[int, float, np.ndarray]] = None
    while reader.next < s.frame:                       # frames just before the target: keep the last good one
        n = reader.next
        img, t, e = reader.take()
        if img is None and t is None:
            break
        if _why(img, e) is None:
            below = (n, t, img)                        # type: ignore[assignment]
    if reader.next == s.frame:
        target_img, target_t, target_err = reader.take()
    else:
        target_img, target_t, target_err = None, None, f"the video ends before frame {s.frame}"
    reason = _why(target_img, target_err)
    if reason is None:
        return target_img, target_t, s.frame, "", None
    above: Optional[tuple[int, float, np.ndarray]] = None
    # look forward only while a frame after the target could still be closer than the one found before it
    while reader.next <= hi and (below is None or reader.next - s.frame < s.frame - below[0]):
        n = reader.next
        img, t, e = reader.take()
        if img is None and t is None:
            break
        if _why(img, e) is None:
            above = (n, t, img)                        # type: ignore[assignment]
            break
    found = [c for c in (below, above) if c is not None]
    if not found:
        return (target_img, target_t, s.frame,
                f"{reason}; no readable frame within frames {lo}-{hi} either", None)
    n, t, img = min(found, key=lambda c: (abs(c[0] - s.frame), c[0]))
    planned_t = target_t if target_t is not None else s.seconds
    note = (f"planned frame {s.frame} was not usable ({reason}); used the nearest readable frame {n} "
            f"({n - s.frame:+d} frames, {t - planned_t:+.3f} s)")
    return img, t, n, "", note


def frame_name(stem: str, s: Sample, seconds: Optional[float], frame: Optional[int] = None) -> str:
    t = hhmmss(seconds if seconds is not None else s.seconds, millis=True).replace(":", "").replace(".", "")
    return f"{stem}_s{s.idx}_f{(s.frame if frame is None else frame):07d}_t{t}.jpg"


def extract(video: Path, fps: float, samples: list[Sample], out_dir: Path, stem: str,
            should_stop: Callable[[], bool] = lambda: False,
            on_frame: Callable[[Extracted], None] = lambda e: None, adjust: bool = False) -> list[Extracted]:
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
            n, note = s.frame, None
            try:
                if adjust and s.segment[0] <= s.frame <= s.segment[1] and s.segment[0] < s.segment[1]:
                    img, t, n, err, note = _read_nearby(reader, s, should_stop)
                else:
                    img, t, err = reader.read(s.frame, should_stop)
            except cv2.error as exc:
                img, t, err = None, None, f"decoder error: {exc}"
            if img is None:
                e = Extracted(s, "failed", t if t is not None else s.seconds, error=err or "frame could not be decoded")
            elif err:
                e = Extracted(s, "unreadable", t, error=err)
            elif (why := frame_quality.problem(img)) is not None:
                e = Extracted(s, "unreadable", t, error=why)
            else:
                path = out_dir / frame_name(stem, s, t, n)
                if cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    e = Extracted(s, "saved", t, path=path, image=img, frame=n, note=note)
                else:
                    e = Extracted(s, "failed", t, error=f"could not write {path.name}")
            results.append(e)
            on_frame(e)
    finally:
        cap.release()
    return results
