"""Is a decoded video frame a real picture? Checked before a screenshot is saved and before inference.

Two kinds of frame carry no evidence and must never be reported as "no litter":
  decoder grey fill  where the data a frame depends on is missing (a gap in an NVR recording, or a decoder
                     started inside a group of pictures) H.264/HEVC decoders paint flat neutral mid-grey.
  blank picture      (nearly) one flat colour: covered lens, "no video" screen, complete darkness.

Both are recognised from 16x16 blocks of a 640-px-wide copy. Thresholds adapted from the Yatrikart UDM
(calibrated there on 14,000+ sequentially decoded NVR frames: readable frames had 0 % grey-fill blocks,
broken ones 50 % or more).
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

WIDTH, BLOCK = 640, 16
FLAT_STD = 3.0             # brightness std inside a block (0-255 levels)
GREY_MID, GREY_TOL, NEUTRAL = 128, 20, 6
GREY_FILL_LIMIT = 0.25     # share of the picture painted with decoder grey fill
BLANK_LIMIT = 0.98         # share of the picture that is flat (any colour)


def problem(frame: Optional[np.ndarray]) -> Optional[str]:
    """Why ``frame`` (BGR) is not a usable picture, or None when it is readable."""
    if frame is None or getattr(frame, "size", 0) == 0 or frame.ndim != 3 or min(frame.shape[:2]) < BLOCK:
        return "the decoder returned no usable image data"
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (WIDTH, max(BLOCK, round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    bh, bw = ycc.shape[0] // BLOCK, WIDTH // BLOCK
    blocks = ycc[:bh * BLOCK, :bw * BLOCK].reshape(bh, BLOCK, bw, BLOCK, 3).swapaxes(1, 2).reshape(bh, bw, -1, 3)
    y, chroma = blocks[..., 0], np.abs(blocks[..., 1:] - 128).max(axis=-1)
    flat = y.std(axis=-1) < FLAT_STD
    grey = flat & (np.abs(y.mean(axis=-1) - GREY_MID) < GREY_TOL) & (chroma.mean(axis=-1) < NEUTRAL)
    if grey.mean() >= GREY_FILL_LIMIT:
        return (f"corrupted frame: {grey.mean():.0%} of the picture is decoder grey fill (the video data this "
                "frame depends on is missing, e.g. a gap in the recording)")
    if flat.mean() >= BLANK_LIMIT:
        return "blank frame: the picture is one flat colour (no video signal, covered lens or complete darkness)"
    return None


# ---------------------------------------------------------------------- view quality (reporting only)
def view_metrics(frame: np.ndarray) -> dict[str, float]:
    """Sharpness (Laplacian variance), mean brightness and grey std of a 640-px-wide grey copy of ``frame``."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (WIDTH, max(BLOCK, round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return {"laplacian_var": float(cv2.Laplacian(grey, cv2.CV_64F).var()), "mean": float(grey.mean()),
            "std": float(grey.std())}


def view_blocked_reason(frame: Optional[np.ndarray], q: Any) -> Optional[str]:
    """Why a screenshot shows no usable view of the scene, or None. ``q`` is ViewQualitySettings (thresholds).
    Used by the dashboard import rows only (``view_blocked``); it does not change what is analysed or detected.
    The existing problem() check (decoder grey fill, blank frame) comes first and is unchanged."""
    p = problem(frame)
    if p:
        return p
    m = view_metrics(frame)
    if m["mean"] < q.dark_max_mean:
        return f"too dark (mean brightness {m['mean']:.0f} < {q.dark_max_mean:g})"
    if m["mean"] > q.bright_min_mean:
        return f"too bright (mean brightness {m['mean']:.0f} > {q.bright_min_mean:g})"
    if m["std"] < q.uniform_max_std:
        return f"mostly uniform picture (grey std {m['std']:.1f} < {q.uniform_max_std:g})"
    if m["laplacian_var"] < q.blur_min_laplacian_var:
        return f"blurred (sharpness {m['laplacian_var']:.0f} < {q.blur_min_laplacian_var:g})"
    return None
