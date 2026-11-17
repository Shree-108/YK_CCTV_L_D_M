"""Operator ROI calibration for automatic four-screenshot analyses.

An automatic job without an approved camera ROI is analysed on the whole frame and reported ROI_NOT_CONFIGURED. The
operator can then draw the monitored ground area on one of the job's four screenshots and approve it. That polygon
(normalised 0-1 coordinates) is stored on THIS job only - never reused for another video unless the operator also
saves it for the job's camera ID - and the same four screenshots are analysed again with it.

The polygon is a post-detection filter, as for every other ROI: full-frame inference is unchanged, and a detection
counts as inside when its ground-contact point (bottom-centre of the box, filters.roi_anchor) lies inside an include
zone (with the existing filters.roi_margin_frac tolerance) and not inside an exclude zone.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from backend.detection.filters import ROISet, anchor_point

MAX_ZONES = 10
MAX_POINTS = 64
MIN_AREA_FRAC = 0.002        # an include area smaller than 0.2 % of the frame is a mis-click, not a ground area
MAX_AREA_FRAC = 0.98         # the whole picture is not a monitoring region: draw the ground only
MIN_POINT_GAP_PX = 1.0

OPERATOR_SOURCE = "OPERATOR_APPROVED_FOR_JOB"
REVALIDATION = "REVALIDATION_REQUIRED"


class InvalidPolygon(ValueError):
    pass


def _area(pts: Sequence[tuple[float, float]]) -> float:
    """Unsigned shoelace area (same units as the points)."""
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, [*pts[1:], pts[0]]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """Proper or touching intersection of segments p1p2 and p3p4."""
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 and d2 and d3 and d4:
        return True

    def on(p, q, r) -> bool:   # r on segment pq (collinear case)
        return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))
    return ((d1 == 0 and on(p3, p4, p1)) or (d2 == 0 and on(p3, p4, p2)) or
            (d3 == 0 and on(p1, p2, p3)) or (d4 == 0 and on(p1, p2, p4)))


def self_intersects(pts: Sequence[tuple[float, float]]) -> bool:
    n = len(pts)
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):      # neighbouring edges share a vertex
                continue
            if _segments_intersect(*edges[i], *edges[j]):
                return True
    return False


def validate_zones(zones: list[dict[str, Any]], width: int, height: int) -> None:
    """Raise InvalidPolygon with an operator-readable reason. ``zones`` are ROIZone dicts (already range-checked
    to [0, 1] and >= 3 points by the ROIZone model); measured in the reference screenshot's pixels."""
    if not zones:
        raise InvalidPolygon("Draw the monitored ground area before approving.")
    if len(zones) > MAX_ZONES:
        raise InvalidPolygon(f"At most {MAX_ZONES} zones are allowed.")
    if not any(z["mode"] == "include" for z in zones):
        raise InvalidPolygon("At least one include zone (the monitored ground area) is needed.")
    if width <= 0 or height <= 0:
        raise InvalidPolygon("The screenshot size is unknown.")
    total = 0.0
    for z in zones:
        name = z.get("name") or "zone"
        pts = [(float(x) * width, float(y) * height) for x, y in z["polygon"]]
        if len(pts) > MAX_POINTS:
            raise InvalidPolygon(f"Zone '{name}' has {len(pts)} points; at most {MAX_POINTS} are allowed.")
        for a, b in zip(pts, [*pts[1:], pts[0]]):
            if abs(a[0] - b[0]) < MIN_POINT_GAP_PX and abs(a[1] - b[1]) < MIN_POINT_GAP_PX:
                raise InvalidPolygon(f"Zone '{name}' has two points on top of each other.")
        if self_intersects(pts):
            raise InvalidPolygon(f"Zone '{name}' crosses itself. Click the boundary points in order around the area.")
        frac = _area(pts) / (width * height)
        if frac < MIN_AREA_FRAC:
            raise InvalidPolygon(f"Zone '{name}' is too small ({frac * 100:.2f} % of the screenshot).")
        if z["mode"] == "include":
            total += frac
    if total >= MAX_AREA_FRAC:
        raise InvalidPolygon("The ground area covers the whole screenshot. Draw only the ground in front of the shop, "
                             "leaving out walls, shutters, interiors, road traffic and vehicles.")


def aspect_mismatch(frame_size: Optional[Sequence[int]], reference_size: Optional[Sequence[int]],
                    tolerance: float) -> bool:
    """True when both sizes are known and their aspect ratios differ by more than ``tolerance``."""
    if not frame_size or not reference_size:
        return False
    fw, fh = frame_size[:2]
    rw, rh = reference_size[:2]
    if min(fw, fh, rw, rh) <= 0:
        return False
    return abs((fw / fh) / (rw / rh) - 1) > tolerance


def tag_membership(detections: list[dict[str, Any]], roi: ROISet, width: int, height: int,
                   anchor: str, margin: float) -> dict[str, int]:
    """Write ``roi_member`` (bool) into every raw detection dict and return the per-screenshot counts. Uses the same
    point and tolerance as the filter, so the counts agree with what was confirmed. Nothing is removed."""
    inside = 0
    review = 0
    for d in detections:
        ax, ay = anchor_point(d["bbox"], anchor)
        ok, _ = roi.zone_for(ax / width, ay / height, margin)
        d["roi_member"] = bool(ok)
        inside += ok
        if ok and d.get("status") in ("candidate", "weak"):
            review += 1
    return {"raw": len(detections), "inside_roi": inside, "outside_roi": len(detections) - inside,
            "review_required": review, "confirmed": sum(1 for d in detections if d.get("status") == "confirmed")}
