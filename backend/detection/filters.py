"""Per-detection filtering: confidence tiers, class, size, aspect ratio. Detection runs on the FULL frame; there is
no spatial-region filter (ROI removed from the application, owner decision 2026-11-19).

Filters TAG detections (status/reason) instead of deleting them so every
rejection is visible in logs and the "rejected" counter. Confirmation is never
decided here for video - that is the tracker's temporal job.
"""
from __future__ import annotations

from backend.core.settings import DetectionSettings
from backend.detection.types import CANDIDATE, CONFIRMED, REJECTED, WEAK, Detection



def confidence_tier(conf: float, s: DetectionSettings) -> str:
    c = s.confidence
    if conf < c.weak_threshold:
        return REJECTED
    if conf < c.candidate_threshold:
        return WEAK
    return CANDIDATE


def apply_filters(dets: list[Detection], shape: tuple[int, ...], s: DetectionSettings) -> list[Detection]:
    """Tag each detection: rejected(reason) | weak | candidate. First rejection wins."""
    h, w = shape[:2]
    f = s.filters
    frame_area = float(w * h)
    for d in dets:
        tier = confidence_tier(d.confidence, s)
        if tier == REJECTED:
            d.reject("low_confidence")
            continue
        if f.class_allowlist is not None and d.cls_name not in f.class_allowlist:
            d.reject("class_not_allowed")
            continue
        if d.cls_name in f.class_blocklist:
            d.reject("class_blocked")
            continue
        bw, bh = d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]
        if bw <= 0 or bh <= 0 or d.area < f.min_box_area_px:
            d.reject("too_small")
            continue
        if d.area / frame_area > f.max_box_area_frac:
            d.reject("too_large")
            continue
        if max(bw / bh, bh / bw) > f.max_aspect_ratio:
            d.reject("bad_aspect_ratio")
            continue
        d.status = tier
    return dets


def confirm_single_image(dets: list[Detection], s: DetectionSettings) -> list[Detection]:
    """Still images have no temporal evidence: a candidate is confirmed only if its own
    confidence reaches the confirmed threshold."""
    for d in dets:
        if d.status == CANDIDATE and d.confidence >= s.confidence.confirmed_threshold:
            d.status = CONFIRMED
    return dets
