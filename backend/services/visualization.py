"""Clean, readable annotation with label-collision avoidance.

Box colour by status: confirmed = red, candidate = amber, weak = thin grey-blue,
rejected = hidden by default (optionally dim grey). Labels look like
``CHIPS_PACKET 91%  ID:17`` and are placed above the box, below it, or inside it,
whichever position does not overlap an already-placed label.
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from backend.detection.filters import ROISet
from backend.detection.types import CANDIDATE, CONFIRMED, REJECTED, WEAK, Detection

COLORS = {CONFIRMED: (40, 40, 235), CANDIDATE: (0, 190, 255), WEAK: (200, 160, 120), REJECTED: (130, 130, 130)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _overlaps(r: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    x1, y1, x2, y2 = r
    return any(not (x2 <= a1 or x1 >= a2 or y2 <= b1 or y1 >= b2) for a1, b1, a2, b2 in placed)


def draw_roi(img: np.ndarray, roi: Optional[ROISet], alpha: float = 0.18) -> None:
    if roi is None or not roi.zones:
        return
    h, w = img.shape[:2]
    overlay = img.copy()
    for z in roi.zones:
        pts = (np.asarray(z.polygon) * [w, h]).astype(np.int32)
        color = (255, 200, 0) if z.mode == "include" else (60, 60, 200)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def draw_detections(img: np.ndarray, dets: Sequence[Detection], show_weak: bool = True,
                    show_rejected: bool = False) -> None:
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
        if d.status == WEAK:
            continue  # weak boxes get no label: keeps busy scenes readable
        label = f"{d.cls_name.upper()} {d.confidence * 100:.0f}%"
        if d.track_id is not None:
            label += f"  ID:{d.track_id}"
        (tw, th), base = cv2.getTextSize(label, FONT, scale, 1)
        lh = th + base + 4
        candidates = [(x1, y1 - lh), (x1, y2), (x1, y1), (x2 - tw - 6, y1 - lh)]
        rect = None
        for cx, cy in candidates:
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


def annotate(frame: np.ndarray, dets: Sequence[Detection], roi: Optional[ROISet] = None,
             hud: Optional[Sequence[str]] = None, show_weak: bool = True) -> np.ndarray:
    img = frame.copy()
    draw_roi(img, roi)
    draw_detections(img, dets, show_weak=show_weak)
    if hud:
        draw_overlay(img, hud)
    return img


# ------------------------------------------------------------------ operator view (uploads)
LITTER_BOX = (0, 90, 255)       # orange-red: litter
ROI_LINE = (255, 220, 0)        # cyan: monitored ground area
EXCLUDE_LINE = (80, 80, 210)


def display_name(cls_name: str) -> str:
    """``plastic_bag`` -> ``Plastic Bag``."""
    return cls_name.replace("_", " ").strip().title()


def _panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x2 <= x or y2 <= y:
        return
    roi = img[y:y2, x:x2]
    cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, 12), alpha, 0, dst=roi)


def annotate_litter(frame: np.ndarray, dets: Sequence[Detection], roi: Optional[ROISet] = None,
                    roi_active: bool = False, max_listed: int = 5, extra: Sequence[str] = ()) -> np.ndarray:
    """Clean CCTV-analytics overlay: ROI outline, a tight box + ``Plastic Bag 78%`` label on every
    confirmed litter object, and a small HUD with the ROI state and the current litter count.
    Candidates, weak and rejected detections (incl. everything outside the ROI) are not drawn."""
    img = frame.copy()
    h, w = img.shape[:2]
    s = float(np.clip(w / 900, 0.6, 1.4))
    thick = max(2, int(round(2 * s)))

    if roi is not None and roi.zones:
        overlay = img.copy()
        for z in roi.zones:
            pts = (np.asarray(z.polygon) * [w, h]).astype(np.int32)
            color = ROI_LINE if z.mode == "include" else EXCLUDE_LINE
            cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.08, img, 0.92, 0, dst=img)
        for z in roi.zones:
            pts = (np.asarray(z.polygon) * [w, h]).astype(np.int32)
            color = ROI_LINE if z.mode == "include" else EXCLUDE_LINE
            cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)
            if z.mode == "include":
                tx, ty = pts[np.argmin(pts[:, 1])]
                label = "LITTER DETECTION ROI"
                (tw, th), base = cv2.getTextSize(label, FONT, 0.5 * s, 1)
                tx = int(np.clip(tx, 0, w - tw - 8))
                ty = int(np.clip(ty - 6, th + 6, h - 2))
                cv2.rectangle(img, (tx, ty - th - 6), (tx + tw + 8, ty + base - 2), color, -1)
                cv2.putText(img, label, (tx + 4, ty - 3), FONT, 0.5 * s, (20, 20, 20), 1, cv2.LINE_AA)

    litter = sorted((d for d in dets if d.status == CONFIRMED), key=lambda d: -d.confidence)
    boxes = [tuple(int(round(v)) for v in d.bbox) for d in litter]
    placed: list[tuple[int, int, int, int]] = []
    fs = 0.6 * s
    for i, (d, (x1, y1, x2, y2)) in enumerate(zip(litter, boxes)):
        label = f"{display_name(d.cls_name)} {d.confidence * 100:.0f}%"
        (tw, th), base = cv2.getTextSize(label, FONT, fs, 1)
        lh = th + base + 6
        # a label must not cover another label or another litter box; try above, below, left, right
        others = [b for j, b in enumerate(boxes) if j != i]
        spots = [(x1, y1 - lh), (x1, y2 + 1), (x2 - tw - 8, y1 - lh), (x2 - tw - 8, y2 + 1),
                 (x2 + 2, y1), (x1 - tw - 10, y1), (x1, y1 - 2 * lh), (x1, y2 + lh + 1)]
        for cx, cy in spots:
            cx, cy = int(np.clip(cx, 0, max(0, w - tw - 8))), int(np.clip(cy, 0, max(0, h - lh)))
            r = (cx, cy, cx + tw + 8, cy + lh)
            if not _overlaps(r, placed) and not _overlaps(r, others):
                placed.append(r)
                cv2.rectangle(img, r[:2], r[2:], LITTER_BOX, -1)
                cv2.putText(img, label, (r[0] + 4, r[3] - base - 3), FONT, fs, (255, 255, 255), 1, cv2.LINE_AA)
                break
    for x1, y1, x2, y2 in boxes:   # outlines last: a box is never hidden under a label
        cv2.rectangle(img, (x1, y1), (x2, y2), LITTER_BOX, thick, cv2.LINE_AA)

    lines = [(t, (0, 230, 120)) for t in extra] + [("LITTER DETECTION ROI - ACTIVE" if roi_active else "ROI NOT SET - FULL FRAME",
              ROI_LINE if roi_active else (170, 170, 170)),
             (f"Litter Detected: {len(litter)}", (255, 255, 255))]
    lines += [(f"{display_name(d.cls_name)} - {d.confidence * 100:.0f}%", (200, 200, 200)) for d in litter[:max_listed]]
    if len(litter) > max_listed:
        lines.append((f"+{len(litter) - max_listed} more", (170, 170, 170)))
    hs = 0.52 * s
    sizes = [cv2.getTextSize(t, FONT, hs, 1) for t, _ in lines]
    lh = max(th + base for (_, th), base in sizes) + 8
    pw = max(tw for (tw, _), _ in sizes) + 20
    _panel(img, 10, 10, pw, lh * len(lines) + 8)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (20, 10 + 4 + lh * (i + 1) - 8), FONT, hs, color, 1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ pavement-only view
PAVEMENT_GREEN = (60, 200, 60)      # approved pavement ROI
EXCLUSION_RED = (40, 40, 220)       # exclusion zones (hatched)
CONFIRMED_BOX = (0, 90, 255)        # confirmed litter: solid, labelled
CANDIDATE_BOX = (0, 215, 255)       # small/weak candidate: dashed amber, labelled "CANDIDATE"
REJECTED_BOX = (150, 150, 150)      # diagnostic view only: dotted grey with the reason


def _dashed_rect(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int],
                 thick: int, dash: int) -> None:
    (x1, y1), (x2, y2) = p1, p2
    for a, b, fixed, horiz in ((x1, x2, y1, True), (x1, x2, y2, True), (y1, y2, x1, False), (y1, y2, x2, False)):
        for s in range(a, b, dash * 2):
            e = min(s + dash, b)
            cv2.line(img, (s, fixed) if horiz else (fixed, s), (e, fixed) if horiz else (fixed, e), color, thick, cv2.LINE_AA)


def _hatch(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], step: int) -> None:
    """Diagonal hatching clipped to a polygon."""
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    x, y, w, h = cv2.boundingRect(pts)
    lines = np.zeros_like(mask)
    for k in range(-h, w, step):
        cv2.line(lines, (x + k, y + h), (x + k + h, y), 255, 1)
    img[(mask > 0) & (lines > 0)] = color


def _tag(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], fs: float,
         placed: list[tuple[int, int, int, int]]) -> None:
    h, w = img.shape[:2]
    (tw, th), base = cv2.getTextSize(text, FONT, fs, 1)
    lh = th + base + 6
    for cx, cy in ((x, y - lh), (x, y), (x, y + lh), (x, y - 2 * lh)):
        cx, cy = int(np.clip(cx, 0, max(0, w - tw - 8))), int(np.clip(cy, 0, max(0, h - lh)))
        r = (cx, cy, cx + tw + 8, cy + lh)
        if not _overlaps(r, placed):
            placed.append(r)
            cv2.rectangle(img, r[:2], r[2:], color, -1)
            dark = sum(color) > 450
            cv2.putText(img, text, (r[0] + 4, r[3] - base - 3), FONT, fs, (20, 20, 20) if dark else (255, 255, 255),
                        1, cv2.LINE_AA)
            return


def draw_pavement_roi(img: np.ndarray, roi: Optional[ROISet], s: float) -> None:
    """Green outline = approved pavement ROI; red outline with hatching = exclusion zone. Only the camera's saved
    polygon is drawn; nothing is inferred."""
    if roi is None or not roi.zones:
        return
    h, w = img.shape[:2]
    thick = max(2, int(round(2 * s)))
    for z in roi.excludes:
        pts = (np.asarray(z.polygon) * [w, h]).astype(np.int32)
        _hatch(img, pts, EXCLUSION_RED, max(10, int(18 * s)))
        cv2.polylines(img, [pts], True, EXCLUSION_RED, thick, cv2.LINE_AA)
    for z in roi.includes:
        pts = (np.asarray(z.polygon) * [w, h]).astype(np.int32)
        cv2.polylines(img, [pts], True, PAVEMENT_GREEN, thick + 1, cv2.LINE_AA)


def annotate_pavement(frame: np.ndarray, dets: Sequence[Detection], roi: Optional[ROISet], status: str,
                      camera_id: Optional[str] = None, diagnostic: bool = False) -> np.ndarray:
    """Pavement-only operator view. Confirmed litter: solid box + ``Food Wrapper 78%``. Small/weak candidates:
    dashed amber box + ``CANDIDATE``. ``diagnostic=True`` also draws rejected detections (dotted grey + reason).
    The HUD names the camera and the ROI status; without an approved ROI it says pavement-only detection is
    unavailable and draws no polygon."""
    from backend.detection import pavement as pv

    img = frame.copy()
    h, w = img.shape[:2]
    s = float(np.clip(w / 900, 0.6, 1.4))
    thick = max(2, int(round(2 * s)))
    fs = 0.5 * s
    active = status == pv.ROI_ACTIVE
    if active:
        draw_pavement_roi(img, roi, s)
    placed: list[tuple[int, int, int, int]] = []
    by = {o: [d for d in dets if d.outcome == o] for o in pv.OUTCOMES}
    by[pv.CONFIRMED_LITTER] = by[pv.CONFIRMED_LITTER] + by[pv.GENERAL_CONFIRMED]
    by[pv.SMALL_OR_WEAK_CANDIDATE] = by[pv.SMALL_OR_WEAK_CANDIDATE] + by[pv.GENERAL_CANDIDATE]
    if diagnostic:
        for d in by[pv.REJECTED_OUTSIDE_ROI] + by[pv.REJECTED_FILTER] + by[pv.UNVERIFIED]:
            x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
            _dashed_rect(img, (x1, y1), (x2, y2), REJECTED_BOX, 1, 3)
            _tag(img, f"{d.cls_name} {d.confidence * 100:.0f}% {(d.outcome_reason or '').replace('exclusion_zone:', 'excl:')}",
                 x1, y1, REJECTED_BOX, fs * 0.85, placed)
    for d in sorted(by[pv.SMALL_OR_WEAK_CANDIDATE], key=lambda d: -d.confidence):
        x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
        _dashed_rect(img, (x1, y1), (x2, y2), CANDIDATE_BOX, thick, max(4, int(6 * s)))
        _tag(img, f"CANDIDATE {display_name(d.cls_name)} {d.confidence * 100:.0f}%", x1, y1, CANDIDATE_BOX, fs, placed)
    for d in sorted(by[pv.CONFIRMED_LITTER], key=lambda d: -d.confidence):
        x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), CONFIRMED_BOX, thick + 1, cv2.LINE_AA)
        _tag(img, f"{display_name(d.cls_name)} {d.confidence * 100:.0f}%", x1, y1, CONFIRMED_BOX, fs * 1.1, placed)

    lines: list[tuple[str, tuple[int, int, int]]] = [(f"Camera: {camera_id or 'not given'}", (235, 235, 235))]
    if active:
        lines.append(("CAMERA_ROI_MODE - pavement ROI approved (green), exclusions hatched red", PAVEMENT_GREEN))
    elif status == pv.GENERAL_IMAGE_MODE:
        lines.append(("GENERAL_IMAGE_MODE - not pavement-validated", (200, 200, 200)))
    elif status == pv.ROI_FRAME_MISMATCH:
        lines.append(("ROI_FRAME_MISMATCH - pavement-only detection unavailable", (60, 160, 255)))
    else:
        lines.append(("ROI_NOT_CONFIGURED - pavement-only detection unavailable", (60, 160, 255)))
    lines.append((f"Confirmed{' (general)' if status == pv.GENERAL_IMAGE_MODE else ' litter'}: "
                  f"{len(by[pv.CONFIRMED_LITTER])}", (255, 255, 255)))
    lines.append((f"Candidates (review): {len(by[pv.SMALL_OR_WEAK_CANDIDATE])}", CANDIDATE_BOX))
    if diagnostic:
        lines.append((f"Rejected outside ROI/exclusion: {len(by[pv.REJECTED_OUTSIDE_ROI])}  other filters: "
                      f"{len(by[pv.REJECTED_FILTER])}  unverified: {len(by[pv.UNVERIFIED])}", REJECTED_BOX))
        lines.append(("DIAGNOSTIC VIEW", (200, 200, 200)))
    hs = 0.52 * s
    sizes = [cv2.getTextSize(t, FONT, hs, 1) for t, _ in lines]
    lh = max(th + base for (_, th), base in sizes) + 8
    pw = max(tw for (tw, _), _ in sizes) + 20
    _panel(img, 10, 10, pw, lh * len(lines) + 8, alpha=0.65)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (20, 10 + 4 + lh * (i + 1) - 8), FONT, hs, color, 1, cv2.LINE_AA)
    return img
