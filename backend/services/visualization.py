"""Readable litter annotation with label-collision avoidance. Full-frame only: nothing about regions is drawn.

Every detection the model returned (above the weak confidence floor) is drawn, styled by its status:
  DETECTED (pipeline ``confirmed``)        solid orange-red box, ``Plastic Bag 87%``
  UNDER REVIEW (``candidate``)             dashed amber box, ``REVIEW Plastic Bag 41%``
  LOW (``weak``, below the candidate rule)  dashed blue-grey box, ``LOW Plastic Bag 30%`` - drawn, never counted
  REJECTED (a filter: size/aspect/class)    dotted grey box (+ the reason on still images)
so an operator always sees what the model saw - nothing is silently hidden.

"Under review" counts ``candidate`` detections only - the same rule as the import CSV remark and the dashboard
Results view (owner decision 2026-11-21); the ``weak`` count is kept in the result bundle's manifest.json only.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from backend.detection.types import CANDIDATE, CONFIRMED, REJECTED, WEAK, Detection

COLORS = {CONFIRMED: (40, 40, 235), CANDIDATE: (0, 190, 255), WEAK: (200, 160, 120), REJECTED: (130, 130, 130)}
FONT = cv2.FONT_HERSHEY_SIMPLEX
LITTER_BOX = (0, 90, 255)          # DETECTED: orange-red, solid
REVIEW_BOX = (0, 215, 255)         # UNDER REVIEW (candidate): amber, dashed
LOW_BOX = (200, 160, 120)          # LOW (weak): blue-grey, dashed, not counted
REJECTED_BOX = (150, 150, 150)     # REJECTED by a filter: grey, dotted


def _overlaps(r: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    x1, y1, x2, y2 = r
    return any(not (x2 <= a1 or x1 >= a2 or y2 <= b1 or y1 >= b2) for a1, b1, a2, b2 in placed)


def draw_detections(img: np.ndarray, dets: Sequence[Detection], show_weak: bool = True,
                    show_rejected: bool = False) -> None:
    """Compact engineering view (live camera wall): colour by status, ``CLASS 91%  ID:17``."""
    h, w = img.shape[:2]
    scale = float(np.clip(w / 1600, 0.4, 0.9))
    thick = max(1, int(round(scale * 2)))
    placed: list[tuple[int, int, int, int]] = []
    order = {CONFIRMED: 0, CANDIDATE: 1, WEAK: 2, REJECTED: 3}
    for d in sorted(dets, key=lambda d: (order.get(d.status, 9), -d.confidence)):
        if (d.status == WEAK and not show_weak) or (d.status == REJECTED and not show_rejected):
            continue
        color = COLORS.get(d.status, (255, 255, 255))
        x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick if d.status != WEAK else 1, cv2.LINE_AA)
        if d.status in (WEAK, REJECTED):
            continue  # no label: keeps busy scenes readable
        label = f"{d.cls_name.upper()} {d.confidence * 100:.0f}%"
        if d.track_id is not None:
            label += f"  ID:{d.track_id}"
        (tw, th), base = cv2.getTextSize(label, FONT, scale, 1)
        lh = th + base + 4
        rect = None
        for cx, cy in [(x1, y1 - lh), (x1, y2), (x1, y1), (x2 - tw - 6, y1 - lh)]:
            cx, cy = int(np.clip(cx, 0, max(0, w - tw - 6))), int(np.clip(cy, 0, max(0, h - lh)))
            r = (cx, cy, cx + tw + 6, cy + lh)
            if not _overlaps(r, placed):
                rect = r
                break
        if rect is None:
            continue  # box still drawn; label skipped rather than stacked illegibly
        placed.append(rect)
        cv2.rectangle(img, rect[:2], rect[2:], color, -1)
        cv2.putText(img, label, (rect[0] + 3, rect[3] - base - 2), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay(img: np.ndarray, lines: Sequence[str]) -> None:
    """Top-left HUD (camera, LIVE, FPS, model, inference time, confirmed count)."""
    scale = float(np.clip(img.shape[1] / 1600, 0.45, 0.8))
    y = 8
    for text in lines:
        (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
        cv2.rectangle(img, (8, y), (8 + tw + 10, y + th + base + 8), (20, 20, 20), -1)
        cv2.putText(img, text, (13, y + th + 4), FONT, scale, (235, 235, 235), 1, cv2.LINE_AA)
        y += th + base + 12


def annotate(frame: np.ndarray, dets: Sequence[Detection], hud: Sequence[str] | None = None,
             show_weak: bool = True) -> np.ndarray:
    img = frame.copy()
    draw_detections(img, dets, show_weak=show_weak)
    if hud:
        draw_overlay(img, hud)
    return img


# ------------------------------------------------------------------ operator view (images, video, evidence)
def display_name(cls_name: str) -> str:
    """``plastic_bag`` -> ``Plastic Bag``."""
    return cls_name.replace("_", " ").strip().title()


def _panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x2 <= x or y2 <= y:
        return
    part = img[y:y2, x:x2]
    cv2.addWeighted(part, 1 - alpha, np.full_like(part, 12), alpha, 0, dst=part)


def _dashed_rect(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int],
                 thick: int, dash: int) -> None:
    (x1, y1), (x2, y2) = p1, p2
    for a, b, fixed, horiz in ((x1, x2, y1, True), (x1, x2, y2, True), (y1, y2, x1, False), (y1, y2, x2, False)):
        for s in range(a, b, dash * 2):
            e = min(s + dash, b)
            cv2.line(img, (s, fixed) if horiz else (fixed, s), (e, fixed) if horiz else (fixed, e), color, thick,
                     cv2.LINE_AA)


def _tag(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], fs: float,
         placed: list[tuple[int, int, int, int]], avoid: Sequence[tuple[int, int, int, int]] = ()) -> None:
    """Label at the first free spot above / on / below (x, y); skipped rather than stacked illegibly."""
    h, w = img.shape[:2]
    (tw, th), base = cv2.getTextSize(text, FONT, fs, 1)
    lh = th + base + 6
    for cx, cy in ((x, y - lh), (x, y), (x, y + lh), (x, y - 2 * lh)):
        cx, cy = int(np.clip(cx, 0, max(0, w - tw - 8))), int(np.clip(cy, 0, max(0, h - lh)))
        r = (cx, cy, cx + tw + 8, cy + lh)
        if not _overlaps(r, placed) and not _overlaps(r, list(avoid)):
            placed.append(r)
            cv2.rectangle(img, r[:2], r[2:], color, -1)
            dark = sum(color) > 450
            cv2.putText(img, text, (r[0] + 4, r[3] - base - 3), FONT, fs, (20, 20, 20) if dark else (255, 255, 255),
                        1, cv2.LINE_AA)
            return


def hud_counts(dets: Sequence[Detection]) -> dict[str, int]:
    """The counts written in the annotated image's header: litter = confirmed, under_review = candidate only."""
    return {"litter": sum(1 for d in dets if d.status == CONFIRMED),
            "under_review": sum(1 for d in dets if d.status == CANDIDATE)}


def annotate_litter(frame: np.ndarray, dets: Sequence[Detection], header: Sequence[str] = (),
                    show_rejected: bool = True, label_rejected: bool = True, max_listed: int = 5) -> np.ndarray:
    """Operator view of one frame: every detection by status (see module doc) and a HUD with the counts.
    ``header``: extra first HUD lines (camera / store / time). Labels never cover a DETECTED box."""
    img = frame.copy()
    h, w = img.shape[:2]
    s = float(np.clip(w / 900, 0.6, 1.4))
    thick = max(2, int(round(2 * s)))
    fs = 0.5 * s
    litter = sorted((d for d in dets if d.status == CONFIRMED), key=lambda d: -d.confidence)
    review = sorted((d for d in dets if d.status == CANDIDATE), key=lambda d: -d.confidence)
    low = sorted((d for d in dets if d.status == WEAK), key=lambda d: -d.confidence)
    rejected = [d for d in dets if d.status == REJECTED] if show_rejected else []
    box = lambda d: tuple(int(round(v)) for v in d.bbox)     # noqa: E731
    litter_boxes = [box(d) for d in litter]
    placed: list[tuple[int, int, int, int]] = []
    for d in litter:   # most important labels are placed first
        x1, y1, _, _ = box(d)
        _tag(img, f"{display_name(d.cls_name)} {d.confidence * 100:.0f}%", x1, y1, LITTER_BOX, fs * 1.15, placed,
             [b for b in litter_boxes if b != box(d)])
    for d in review:
        x1, y1, x2, y2 = box(d)
        _dashed_rect(img, (x1, y1), (x2, y2), REVIEW_BOX, max(1, thick - 1), max(4, int(6 * s)))
        _tag(img, f"REVIEW {display_name(d.cls_name)} {d.confidence * 100:.0f}%", x1, y1, REVIEW_BOX, fs, placed,
             litter_boxes)
    for d in low:
        x1, y1, x2, y2 = box(d)
        _dashed_rect(img, (x1, y1), (x2, y2), LOW_BOX, max(1, thick - 1), max(4, int(6 * s)))
        _tag(img, f"LOW {display_name(d.cls_name)} {d.confidence * 100:.0f}%", x1, y1, LOW_BOX, fs * 0.9, placed,
             litter_boxes)
    for d in rejected:
        x1, y1, x2, y2 = box(d)
        _dashed_rect(img, (x1, y1), (x2, y2), REJECTED_BOX, 1, 3)
        if label_rejected:
            _tag(img, f"{d.cls_name} {d.confidence * 100:.0f}% {d.reason or 'rejected'}", x1, y1, REJECTED_BOX,
                 fs * 0.85, placed, litter_boxes)
    for x1, y1, x2, y2 in litter_boxes:   # outlines last: a DETECTED box is never hidden under a label
        cv2.rectangle(img, (x1, y1), (x2, y2), LITTER_BOX, thick + 1, cv2.LINE_AA)

    lines = [(t, (235, 235, 235)) for t in header]
    counts = hud_counts(dets)
    lines += [(f"Litter detected: {counts['litter']}", (255, 255, 255)),
              (f"Under review: {counts['under_review']}", REVIEW_BOX)]
    if rejected:
        lines.append((f"Rejected by filters: {len(rejected)}", REJECTED_BOX))
    lines += [(f"{display_name(d.cls_name)} - {d.confidence * 100:.0f}%", (200, 200, 200)) for d in litter[:max_listed]]
    if len(litter) > max_listed:
        lines.append((f"+{len(litter) - max_listed} more", (170, 170, 170)))
    hs = 0.52 * s
    sizes = [cv2.getTextSize(t, FONT, hs, 1) for t, _ in lines]
    lh = max(th + base for (_, th), base in sizes) + 8
    pw = max(tw for (tw, _), _ in sizes) + 20
    _panel(img, 10, 10, pw, lh * len(lines) + 8, alpha=0.6)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (20, 10 + 4 + lh * (i + 1) - 8), FONT, hs, color, 1, cv2.LINE_AA)
    return img
