"""Pavement-only mode: camera-specific ROI enforcement and the significant-litter size filter.

The ROI is camera CONFIGURATION (an approved polygon saved for one camera), never a model prediction. This module
only reads the polygon it is given; it never creates, guesses, copies or approves one.

Analysis mode (decide_mode), per frame:

  GENERAL_IMAGE_MODE  no camera association (manual upload, camera_id "upload"/empty). The EXISTING image analysis
                      runs unchanged (filters, confirmation, alerts); outcomes are GENERAL_* and are never presented
                      as pavement-validated. Size is measured and recorded but does not change anything.
  CAMERA_ROI_MODE     a REGISTERED camera with an approved include polygon and a matching frame shape: the pavement
                      gate below decides.
  ROI_NOT_CONFIGURED  a camera is named but is not registered, or has no approved include polygon: nothing can be
                      confirmed (no confirmed pavement alert). Detections are kept as UNVERIFIED_ROI_NOT_CONFIGURED.
  ROI_FRAME_MISMATCH  the frame's aspect ratio differs from roi_reference_size: as ROI_NOT_CONFIGURED.

Membership is decided AFTER inference on the full frame: the gate never changes a box's coordinates. (ROI-crop
inference is a separate, experimental setting - filters.roi_crop_inference - and is OFF by default.)

Order, per detection in CAMERA_ROI_MODE (after the existing confidence/class/size/aspect filters, unchanged):

  1. Frame check   (above).
  2. Membership    ground-contact point = bottom-centre of the box, in ORIGINAL-frame pixels. Inside the pavement
                   polygon, or within ``edge_tolerance_frac`` of the frame's shorter side of it -> member. With
                   ``footpoint_or_overlap`` a box with >= ``min_include_overlap`` of its area inside also counts, so
                   litter whose bottom edge is hidden just behind the boundary is kept. A box that merely crosses
                   the boundary is NOT rejected for that reason alone.
  3. Exclusions    applied after membership: foot point inside an exclusion polygon, or >= ``max_exclusion_overlap``
                   of the box inside it -> REJECTED_OUTSIDE_ROI (reason ``exclusion_zone:<name>``). No tolerance.
  4. Size          normalised to the original frame: shorter box side / shorter frame side >= ``min_side_frac`` AND
                   box area / frame area >= ``min_area_frac`` (per-class overrides possible). Below -> candidate.
  5. Confidence    unchanged thresholds. Still images: confirmed_threshold; video/live: the tracker's temporal
                   confirmation. Only a member, outside every exclusion, of significant size, that the existing rule
                   confirms becomes CONFIRMED_LITTER.

Outcomes (Detection.outcome):
  CONFIRMED_LITTER               valid class, sufficient confidence, inside the approved ROI, significant size.
  SMALL_OR_WEAK_CANDIDATE        inside the ROI but below the size or confidence rule. Kept for review, no alert.
  REJECTED_OUTSIDE_ROI           foot point outside the pavement polygon, or inside an exclusion zone.
  REJECTED_FILTER                rejected by the pre-existing class/size/aspect filters (unchanged; reason kept).
  UNVERIFIED_ROI_NOT_CONFIGURED  the camera has no approved ROI: pavement membership cannot be decided.

Nothing is deleted: every raw detection keeps its box, confidence, outcome, reason and measurements.

How "never confirmed" is enforced: detections that may not be confirmed are demoted to status ``weak`` (outside the
ROI: ``rejected``) BEFORE confirmation. ``confirm_single_image`` and the tracker only confirm ``candidate``
detections, and alerts are only created for confirmed ones, so no alert path can be reached.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from backend.core.settings import PavementSettings, SizeLimit
from backend.detection.filters import ROISet
from backend.detection.types import CANDIDATE, CONFIRMED, REJECTED, WEAK, Detection

# detection outcomes
CONFIRMED_LITTER = "CONFIRMED_LITTER"
SMALL_OR_WEAK_CANDIDATE = "SMALL_OR_WEAK_CANDIDATE"
REJECTED_OUTSIDE_ROI = "REJECTED_OUTSIDE_ROI"
REJECTED_FILTER = "REJECTED_FILTER"
UNVERIFIED = "UNVERIFIED_ROI_NOT_CONFIGURED"
GENERAL_CONFIRMED = "GENERAL_CONFIRMED"           # general image mode: existing rule, NOT pavement-validated
GENERAL_CANDIDATE = "GENERAL_CANDIDATE"
OUTCOMES = (CONFIRMED_LITTER, SMALL_OR_WEAK_CANDIDATE, REJECTED_OUTSIDE_ROI, REJECTED_FILTER, UNVERIFIED,
            GENERAL_CONFIRMED, GENERAL_CANDIDATE)
CONFIRMED_OUTCOMES = (CONFIRMED_LITTER, GENERAL_CONFIRMED)
CANDIDATE_OUTCOMES = (SMALL_OR_WEAK_CANDIDATE, GENERAL_CANDIDATE)

# analysis mode (frame level)
CAMERA_ROI_MODE = "CAMERA_ROI_MODE"
GENERAL_IMAGE_MODE = "GENERAL_IMAGE_MODE"
ROI_NOT_CONFIGURED = "ROI_NOT_CONFIGURED"
ROI_FRAME_MISMATCH = "ROI_FRAME_MISMATCH"
ROI_ACTIVE = CAMERA_ROI_MODE                       # alias used by callers
NO_CAMERA_IDS = ("", "upload")

# membership verdicts (geometry["membership"])
INSIDE, EDGE, OVERLAP, OUTSIDE, EXCLUDED = "INSIDE", "EDGE_TOLERANCE", "AREA_OVERLAP", "OUTSIDE", "EXCLUDED"

GRID = 8   # 8x8 sample points per box for the area-overlap measurement


@dataclass
class PixelZone:
    name: str
    mode: str
    contour: np.ndarray            # (N,1,2) float32, pixels of THIS frame

    def signed(self, x: float, y: float) -> float:
        """Signed distance in pixels (>0 inside)."""
        return float(cv2.pointPolygonTest(self.contour, (float(x), float(y)), True))


def pixel_zones(roi: Optional[ROISet], w: int, h: int) -> tuple[list[PixelZone], list[PixelZone]]:
    """(includes, excludes) scaled from normalised polygon coordinates to this frame's pixels. The frame passed in
    must be the ORIGINAL camera frame: detections are mapped back to it before this runs (engine tiles / ROI crop)."""
    if roi is None:
        return [], []
    zs = [PixelZone(z.name, z.mode, (np.asarray(z.polygon, dtype=np.float32) * np.float32([w, h])).reshape(-1, 1, 2))
          for z in roi.zones]
    return [z for z in zs if z.mode == "include"], [z for z in zs if z.mode == "exclude"]


def decide_mode(camera_id: Optional[str], registered: bool, roi: Optional[ROISet], approved: bool,
                shape: Sequence[int], p: PavementSettings) -> tuple[str, str]:
    """(mode, note). Only the ROI given for THIS camera is considered; an unknown camera never borrows another's."""
    if (camera_id or "") in NO_CAMERA_IDS:
        return GENERAL_IMAGE_MODE, ("General image analysis (no camera association): existing behaviour, not "
                                    "validated against a pavement ROI.")
    if not registered:
        return ROI_NOT_CONFIGURED, (f"Camera {camera_id} is not registered: pavement-only detection is unavailable. "
                                    "Detections are kept for review and none is confirmed.")
    return frame_status(roi, approved, shape, p)


def frame_status(roi: Optional[ROISet], approved: bool, shape: Sequence[int], p: PavementSettings) -> tuple[str, str]:
    h, w = shape[:2]
    if roi is None or not roi.includes or not approved:
        return ROI_NOT_CONFIGURED, ("No approved pavement ROI for this camera: pavement-only detection is unavailable. "
                                    "Detections are kept for review and none is confirmed.")
    if p.roi_reference_size:
        rw, rh = p.roi_reference_size
        if rw > 0 and rh > 0 and abs((w / h) / (rw / rh) - 1) > p.aspect_tolerance:
            return ROI_FRAME_MISMATCH, (f"The ROI was drawn on a {rw}x{rh} frame but this frame is {w}x{h} (different "
                                        "aspect ratio): it may belong to another camera or stream. Nothing is confirmed.")
    return ROI_ACTIVE, ""


def inside_fraction(zones: Sequence[PixelZone], bbox: Sequence[float], n: int = GRID) -> float:
    """Share of the box area inside any of ``zones`` (n x n cell-centre samples)."""
    if not zones:
        return 0.0
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return 0.0
    hit = 0
    for i in range(n):
        y = y1 + (i + 0.5) * bh / n
        for j in range(n):
            x = x1 + (j + 0.5) * bw / n
            if any(z.signed(x, y) >= 0 for z in zones):
                hit += 1
    return hit / (n * n)


def membership(bbox: Sequence[float], w: int, h: int, includes: Sequence[PixelZone], excludes: Sequence[PixelZone],
               p: PavementSettings) -> tuple[str, Optional[str], dict[str, Any]]:
    """(verdict, zone name, measurements). Verdict: INSIDE | EDGE_TOLERANCE | AREA_OVERLAP | OUTSIDE | EXCLUDED."""
    x1, y1, x2, y2 = bbox
    fx, fy = (x1 + x2) / 2, y2
    tol = p.edge_tolerance_frac * min(w, h)
    best, zone = -float("inf"), None
    for z in includes:
        d = z.signed(fx, fy)
        if d > best:
            best, zone = d, z.name
    info: dict[str, Any] = {"foot_point": [round(fx, 1), round(fy, 1)], "foot_distance_px": round(best, 1)}
    if best >= 0:
        verdict = INSIDE
    elif best >= -tol:
        verdict = EDGE
    else:
        verdict = OUTSIDE
        if p.membership == "footpoint_or_overlap":
            frac = inside_fraction(includes, bbox)
            info["include_overlap"] = round(frac, 3)
            if frac >= p.min_include_overlap:
                verdict = OVERLAP
    if verdict == OUTSIDE:
        return verdict, None, info
    for z in excludes:                                   # exclusions AFTER membership, no tolerance
        if z.signed(fx, fy) >= 0:
            info["exclusion_hit"] = "foot_point"
            return EXCLUDED, z.name, info
        frac = inside_fraction([z], bbox)
        if frac >= p.max_exclusion_overlap:
            info["exclusion_hit"], info["exclusion_overlap"] = "area", round(frac, 3)
            return EXCLUDED, z.name, info
    return verdict, zone, info


def size_limit(cls_name: str, p: PavementSettings) -> SizeLimit:
    return p.class_size.get(cls_name) or p.size


def size_metrics(bbox: Sequence[float], w: int, h: int) -> dict[str, float]:
    bw, bh = max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1])
    return {"box_w_px": round(bw, 1), "box_h_px": round(bh, 1),
            "min_side_frac": round(min(bw, bh) / min(w, h), 5), "area_frac": round(bw * bh / (w * h), 7)}


def significant(cls_name: str, m: dict[str, float], p: PavementSettings) -> bool:
    lim = size_limit(cls_name, p)
    return m["min_side_frac"] >= lim.min_side_frac and m["area_frac"] >= lim.min_area_frac


def tag_general(dets: list[Detection], shape: Sequence[int], p: PavementSettings) -> None:
    """GENERAL_IMAGE_MODE: record measurements and a provisional outcome; statuses are NOT changed (the existing
    filters and confirmation decide, exactly as before)."""
    h, w = shape[:2]
    for d in dets:
        geo: dict[str, Any] = size_metrics(d.bbox, w, h)
        geo["size_ok"] = significant(d.cls_name, geo, p)
        geo["membership"] = "NOT_EVALUATED"
        d.geometry = geo
        if d.status == REJECTED:
            d.outcome, d.outcome_reason = REJECTED_FILTER, d.reason
        else:
            d.outcome, d.outcome_reason = GENERAL_CANDIDATE, "general_mode_not_pavement_validated"


def gate(dets: list[Detection], shape: Sequence[int], roi: Optional[ROISet], status: str,
         p: PavementSettings) -> tuple[str, str]:
    """Tag every detection with its pavement outcome and demote what may not be confirmed. Run AFTER
    ``apply_filters`` (called WITHOUT the ROI) and BEFORE confirmation. ``status`` comes from decide_mode (never
    GENERAL_IMAGE_MODE here). Boxes are never modified. Returns (status, note)."""
    h, w = shape[:2]
    note = ""
    includes, excludes = pixel_zones(roi, w, h) if status == ROI_ACTIVE else ([], [])
    for d in dets:
        geo: dict[str, Any] = size_metrics(d.bbox, w, h)
        big = significant(d.cls_name, geo, p)
        geo["size_ok"] = big
        if status != ROI_ACTIVE:
            geo["membership"] = "NOT_EVALUATED"
            d.geometry = geo
            if d.status == REJECTED:
                d.outcome, d.outcome_reason = REJECTED_FILTER, d.reason
            else:
                d.status = WEAK                                    # can never be confirmed
                d.outcome, d.outcome_reason = UNVERIFIED, status.lower()
            continue
        verdict, zone, info = membership(d.bbox, w, h, includes, excludes, p)
        geo.update(info, membership=verdict, zone=zone)
        d.geometry = geo
        if verdict in (OUTSIDE, EXCLUDED):
            reason = f"exclusion_zone:{zone}" if verdict == EXCLUDED else "outside_pavement_roi"
            if d.status != REJECTED:
                d.reject("outside_roi")                            # same reason string the reports already count
            d.outcome, d.outcome_reason = REJECTED_OUTSIDE_ROI, reason
            continue
        if d.status == REJECTED:
            d.outcome, d.outcome_reason = REJECTED_FILTER, d.reason
            continue
        geo["accepted_because"] = f"membership={verdict}" + (f";zone={zone}" if zone else "")
        why = []
        if not big:
            why.append("below_min_size")
        if d.status == WEAK:
            why.append("low_confidence")
        if why:
            d.status = WEAK                                        # retained for review, never confirmed
        d.outcome, d.outcome_reason = SMALL_OR_WEAK_CANDIDATE, "+".join(why) or "awaiting_confirmation"
    return status, note


def finalize(dets: Sequence[Detection]) -> None:
    """After confirmation (still image: confirm_single_image; video: tracker): a member that was confirmed becomes
    CONFIRMED_LITTER. Only detections the gate left as candidates can have been confirmed."""
    for d in dets:
        if d.outcome == SMALL_OR_WEAK_CANDIDATE and d.status == CONFIRMED:
            d.outcome = CONFIRMED_LITTER
            d.outcome_reason = (f"accepted:{(d.geometry or {}).get('membership', '')}+significant_size+"
                                "confirmed_by_existing_rule")
        elif d.outcome == SMALL_OR_WEAK_CANDIDATE and d.status == CANDIDATE and d.outcome_reason == "awaiting_confirmation":
            d.outcome_reason = "below_confirmation_rule"
        elif d.outcome == GENERAL_CANDIDATE and d.status == CONFIRMED:
            d.outcome = GENERAL_CONFIRMED


def counts(dets: Sequence[Detection]) -> dict[str, int]:
    c = Counter(d.outcome for d in dets if d.outcome)
    return {k.lower(): c[k] for k in OUTCOMES} | {"raw": len(dets)}


def report(status: str, note: str, dets: Sequence[Detection], p: PavementSettings,
           camera_id: Optional[str] = None) -> dict[str, Any]:
    """The ``pavement`` block of an image/screenshot result."""
    return {"status": status, "mode": status, "pavement_validated": status == CAMERA_ROI_MODE, "note": note,
            "camera_id": camera_id, "counts": counts(dets),
            "size_rule": {"min_side_frac": p.size.min_side_frac, "min_area_frac": p.size.min_area_frac,
                          "class_overrides": {k: v.model_dump() for k, v in p.class_size.items()},
                          "normalised_to": "original frame (shorter side / area)"},
            "membership_rule": {"method": p.membership, "edge_tolerance_frac": p.edge_tolerance_frac,
                                "min_include_overlap": p.min_include_overlap,
                                "max_exclusion_overlap": p.max_exclusion_overlap,
                                "anchor": "bottom-centre of the box (ground contact)"}}
