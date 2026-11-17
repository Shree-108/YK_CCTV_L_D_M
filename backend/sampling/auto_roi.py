"""ROI policy of the zero-click analysis: which ROI an automatic job uses, without asking the operator.

Only ROIs a person drew and saved are ever applied. Nothing is approved automatically:

  1. REGISTERED_CAMERA  the ROI of a camera registered on the Cameras page (administrator configuration).
  2. SAVED_CAMERA_ROI   an ROI saved for that camera ID without a camera registration (PUT /api/roi/{id}, or
                        "Save this ROI for the camera" when an operator confirmed an ROI in the manual workflow).
  3. FULL_FRAME_FALLBACK no camera given, or no saved ground area (include zone) for it. The whole frame is
                        analysed and roi_status says NOT_CONFIGURED - it is never presented as an approved ground area.
                        Exclude zones saved for the camera are still applied (they only ever remove detections).

The automatic outdoor-ROI segmenter is not used here: its proposals need operator confirmation, and it is the most
expensive step on CPU. There is no store-scoped ROI in LDM (ROIs are stored per camera); store_id is recorded on the
job and in the report but does not select an ROI.

If ROI filtering is switched off in Settings (filters.roi_enabled = false), no ROI is applied and the status says so.

Revalidation: when the size of the picture the ROI was drawn on is known (``reference_size``) and this video's frames
have a different aspect ratio, the camera view has changed (other stream, other camera, re-mounted camera): the ROI is
NOT applied, the status is REVALIDATION_REQUIRED and nothing is confirmed until an operator approves an ROI again.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from backend.detection.filters import ROIZone

APPROVED = "APPROVED"
NOT_CONFIGURED = "NOT_CONFIGURED"
DISABLED = "DISABLED_IN_SETTINGS"
REVALIDATION = "REVALIDATION_REQUIRED"

REGISTERED_CAMERA = "REGISTERED_CAMERA"
SAVED_CAMERA_ROI = "SAVED_CAMERA_ROI"
FULL_FRAME = "FULL_FRAME_FALLBACK"


@dataclass(frozen=True)
class RoiDecision:
    status: str                       # APPROVED | NOT_CONFIGURED | DISABLED_IN_SETTINGS | REVALIDATION_REQUIRED
    source: str                       # REGISTERED_CAMERA | SAVED_CAMERA_ROI | FULL_FRAME_FALLBACK
    zones: list[dict] = field(default_factory=list)   # zones actually applied (normalised ROIZone dicts)
    name: str = ""                    # include-zone names, ";"-separated ("" for the full frame)
    configuration_id: str = ""        # "<kind>:<camera>#<hash of the zones>" so a later ROI change is traceable
    note: str = ""
    reference_size: Optional[list[int]] = None   # [w, h] of the picture the ROI was drawn on, when known

    @property
    def approved(self) -> bool:
        """An operator/administrator-drawn ground area is actually applied to the detections."""
        return self.status == APPROVED

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "approved": self.approved}


def _clean(raw: Any) -> list[dict]:
    """Valid zones only (a damaged stored ROI must not crash an automatic job)."""
    out = []
    for z in raw or []:
        try:
            out.append(ROIZone.model_validate(z).model_dump())
        except Exception:  # noqa: BLE001
            continue
    return out


def _config_id(kind: str, camera_id: str, zones: list[dict]) -> str:
    digest = hashlib.sha256(json.dumps(zones, sort_keys=True).encode()).hexdigest()[:10]
    return f"{kind}:{camera_id}#{digest}"


def decide(camera_id: Optional[str], camera: Optional[dict[str, Any]], saved_zones: Any,
           roi_enabled: bool = True, *, frame_size: Optional[tuple[int, int]] = None,
           reference_size: Optional[Any] = None, aspect_tolerance: float = 0.02) -> RoiDecision:
    """``camera`` = the registered camera row (or None); ``saved_zones`` = settings["roi:<camera_id>"] (or None);
    ``frame_size`` = this video's (w, h); ``reference_size`` = (w, h) the ROI was drawn on, if recorded."""
    found: Optional[tuple[str, str, list[dict]]] = None
    if camera_id and camera is not None:
        found = (REGISTERED_CAMERA, "camera", _clean(camera.get("roi")))
    elif camera_id and saved_zones:
        found = (SAVED_CAMERA_ROI, "saved_roi", _clean(saved_zones))
    includes = [z for z in (found[2] if found else []) if z["mode"] == "include"]

    if not roi_enabled:
        return RoiDecision(DISABLED, found[0] if found else FULL_FRAME, [], "", "",
                           "ROI filtering is switched off in Settings: the whole frame is analysed.")
    ref = [int(v) for v in reference_size[:2]] if reference_size else None
    if found and includes:
        src, kind, zones = found
        from backend.sampling.roi_calibration import aspect_mismatch
        if aspect_mismatch(frame_size, ref, aspect_tolerance):
            return RoiDecision(REVALIDATION, src, [], "", _config_id(kind, camera_id or "", zones),
                               f"The ROI of camera {camera_id} was drawn on a {ref[0]}x{ref[1]} picture but this video is "
                               f"{frame_size[0]}x{frame_size[1]} (different aspect ratio): the camera view may have "
                               "changed, so the ROI was not applied. Approve an ROI again for this view. Nothing is "
                               "confirmed until then.", ref)
        return RoiDecision(APPROVED, src, zones, ";".join(z["name"] for z in includes),
                           _config_id(kind, camera_id or "", zones),
                           f"The ROI saved for camera {camera_id} was applied automatically.", ref)
    excludes = [z for z in (found[2] if found else []) if z["mode"] == "exclude"]
    if not camera_id:
        why = "No camera was given for this upload"
    elif not found:
        why = f"No ROI is saved for camera {camera_id}"
    else:
        why = f"The ROI saved for camera {camera_id} has no ground area (include zone)"
    note = (f"{why}: the whole frame was analysed (ROI not configured). Detections need review: shop interior or "
            "other non-ground areas may be included.")
    if excludes:
        note += f" {len(excludes)} saved exclude zone(s) were still applied."
    return RoiDecision(NOT_CONFIGURED, FULL_FRAME, excludes, "",
                       _config_id(found[1], camera_id or "", excludes) if excludes and found else "", note)
