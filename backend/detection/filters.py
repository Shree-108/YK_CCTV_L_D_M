"""Per-detection filtering: confidence tiers, class, size, ROI zones.

Filters TAG detections (status/reason) instead of deleting them so every
rejection is visible in logs and the "rejected" counter. Confirmation is never
decided here for video - that is the tracker's temporal job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.core.settings import DetectionSettings
from backend.detection.types import CANDIDATE, CONFIRMED, REJECTED, WEAK, Detection

ZoneType = Literal["sidewalk", "road_edge", "ground", "parking", "entrance", "custom"]


class ROIZone(BaseModel):
    """A polygon in NORMALIZED (0-1) coordinates, so it survives resolution changes."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field("zone", max_length=60)
    type: ZoneType = "ground"
    mode: Literal["include", "exclude"] = "include"
    polygon: list[tuple[float, float]]

    @field_validator("polygon")
    @classmethod
    def _poly(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(v) < 3:
            raise ValueError("an ROI polygon needs at least 3 points")
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in v):
            raise ValueError("ROI polygon points must be normalized to [0, 1]")
        return v


@dataclass
class ROISet:
    zones: list[ROIZone]

    @classmethod
    def from_json(cls, raw: Sequence[dict[str, Any]] | None) -> "ROISet":
        return cls([ROIZone.model_validate(z) for z in (raw or [])])

    @property
    def includes(self) -> list[ROIZone]:
        return [z for z in self.zones if z.mode == "include"]

    @property
    def excludes(self) -> list[ROIZone]:
        return [z for z in self.zones if z.mode == "exclude"]

    @staticmethod
    def _dist(zone: ROIZone, x: float, y: float) -> float:
        """Signed distance in normalized units (>0 inside)."""
        cnt = np.asarray(zone.polygon, dtype=np.float32).reshape(-1, 1, 2)
        return float(cv2.pointPolygonTest(cnt, (float(x), float(y)), True))

    def crop_box(self, w: int, h: int, pad: float) -> Optional[tuple[int, int, int, int]]:
        """Pixel rectangle covering every include zone (+ ``pad`` of the frame, doubled upwards so
        objects whose foot point is inside still fit). None when there is no include zone."""
        if not self.includes:
            return None
        pts = np.concatenate([np.asarray(z.polygon, dtype=np.float32) for z in self.includes])
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        x1, x2 = max(0.0, x1 - pad), min(1.0, x2 + pad)
        y1, y2 = max(0.0, y1 - 2 * pad), min(1.0, y2 + pad)
        return int(x1 * w), int(y1 * h), int(np.ceil(x2 * w)), int(np.ceil(y2 * h))

    def zone_for(self, x: float, y: float, margin: float = 0.0) -> tuple[bool, Optional[str]]:
        """(allowed, zone_name) for a NORMALIZED point.

        Exclude zones win. With no include zones the whole frame is allowed, so an
        unconfigured camera never silently drops litter. ``margin`` lets points
        just outside an include polygon count (the ROI must not be too strict).
        """
        for z in self.excludes:
            if self._dist(z, x, y) >= 0:
                return False, z.name
        if not self.includes:
            return True, None
        for z in self.includes:
            if self._dist(z, x, y) >= -margin:
                return True, z.name
        return False, None


def anchor_point(bbox: Sequence[float], anchor: str) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2) if anchor == "footpoint" else ((x1 + x2) / 2, (y1 + y2) / 2)


def confidence_tier(conf: float, s: DetectionSettings) -> str:
    c = s.confidence
    if conf < c.weak_threshold:
        return REJECTED
    if conf < c.candidate_threshold:
        return WEAK
    return CANDIDATE


def apply_filters(dets: list[Detection], shape: tuple[int, ...], s: DetectionSettings,
                  roi: ROISet | None) -> list[Detection]:
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
        if f.roi_enabled and roi is not None and roi.zones:
            ax, ay = anchor_point(d.bbox, f.roi_anchor)
            ok, _ = roi.zone_for(ax / w, ay / h, f.roi_margin_frac)
            if not ok:
                d.reject("outside_roi")
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
