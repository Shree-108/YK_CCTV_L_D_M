"""Monitored region (automatic pavement ROI) API - backend/services/roi_service.py, store config/camera_roi.json.

Owner brief 2026-11-24: the ROI is computed automatically (image/video upload, live start, app start), so these routes
are for inspecting and correcting it, never a required step.

Permissions (backend/api/authz.py): reads and preview images = VIEWER; computing, approving, editing and clearing a
camera's or the global ROI = ADMIN (camera configuration); an on-the-fly ROI for one uploaded image = OPERATOR.

Records carry ``polygons`` ([{points, holes}], normalised 0..1). For the existing Cameras editor every response also
has ``polygon`` / ``holes`` = the largest polygon. Precedence: request > manual > auto > global.

Phase 2 (pilot): Edit / Re-run / Approve. A computed (Re-run) or edited region is saved with ``approved: false`` and
stays ``needs_review`` (remark tag) until Approve.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from backend.api.authz import ADMIN, OPERATOR, VIEWER, actor_of
from backend.api.deps import IMAGE_EXTS, check_camera_id, get_state
from backend.cameras.manager import camera_source, open_capture
from backend.services import roi_service as rs
from backend.services.state import AppState

router = APIRouter(prefix="/api")
MAX_IMAGE_BYTES = 25 * 1024 * 1024


class ComputeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frames: Optional[int] = Field(None, ge=1, le=120)        # default: settings.roi.vote_frames


class EditIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    polygon: Optional[list[list[float]]] = Field(None, min_length=3, max_length=500)
    holes: list[list[list[float]]] = Field(default_factory=list, max_length=20)
    polygons: Optional[list[Any]] = Field(None, min_length=1, max_length=50)
    source: str = Field("manual", pattern="^(manual|sam_click)$")


def _compat(rec: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not rec:
        return None
    out = {k: v for k, v in rec.items() if k != "frame_signature"}
    polys = rec.get("polygons") or []
    out["polygon"] = polys[0]["points"] if polys else []
    out["holes"] = polys[0].get("holes", []) if polys else []
    return out


def _out(key: str, rec: Optional[dict[str, Any]], resolved_from: str = "") -> dict[str, Any]:
    return {"key": key, "roi": _compat(rec), "resolved_from": resolved_from or ("camera" if rec else "none")}


def _camera(state: AppState, cam_id: str) -> dict[str, Any]:
    check_camera_id(cam_id)
    cam = state.db.get_camera(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    return cam


def _jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def _camera_frames(state: AppState, cam: dict[str, Any], n: int, seconds: float) -> list[np.ndarray]:
    """N frames of the camera: from its running worker (no second connection to the NVR) or its source."""
    w = state.cameras.workers.get(cam["id"])
    if w is not None:
        out, last = [], None
        gap = seconds / max(1, n)
        deadline = time.monotonic() + max(10.0, seconds * 2)
        while len(out) < n and time.monotonic() < deadline:
            raw = w.latest_live()[1]
            if raw and raw is not last:
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    out.append(img)
                last = raw
            time.sleep(gap if len(out) < n else 0)
        if out:
            return out
    r = state.settings(cam).roi.model_copy(update={"vote_frames": n})
    try:
        return rs.frames_from_source(cam, r)
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from exc


def _edit(state: AppState, key: str, body: EditIn, actor: str) -> dict[str, Any]:
    try:
        polys = rs.validate_polygons(body.polygons if body.polygons else
                                     [{"points": body.polygon, "holes": body.holes}] if body.polygon else [])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    old = state.roi.store.get(key) or {}
    t = rs.now_iso()
    rec = {**old, "polygons": polys, "source": body.source, "confidence": 1.0, "needs_review": True,
           "approved": False, "approved_at": None, "approved_by": None, "edited_at": t, "edited_by": actor or None,
           "updated_at": t, "review_reason": "edited: awaiting approval"}
    rec.setdefault("created_at", t)
    state.roi.store.put(key, rec)
    return rec


# ---------------------------------------------------------------- listing / global
@router.get("/auto-roi", dependencies=VIEWER)
def list_rois(state: AppState = Depends(get_state)) -> dict[str, Any]:
    doc = state.roi.store.all()
    return {"global": _compat(doc.get(rs.GLOBAL_KEY)),
            "cameras": {k: _compat(v) for k, v in doc.items() if k != rs.GLOBAL_KEY},
            "precedence": ["request", "manual", "auto", "global", "on_failure"],
            "enabled": state.roi.enabled and state.settings().roi.enabled, "warmup": state.roi_warmup,
            "last_error": state.roi.last_error}


@router.put("/auto-roi/global", dependencies=ADMIN)
def set_global(body: EditIn, request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    rec = _edit(state, rs.GLOBAL_KEY, body, actor_of(request) or "")
    state.db.log_event("info", "auto_roi", "global ROI saved")
    return _out(rs.GLOBAL_KEY, rec, "global")


@router.delete("/auto-roi/global", dependencies=ADMIN)
def clear_global(state: AppState = Depends(get_state)) -> dict[str, Any]:
    existed = state.roi.store.get(rs.GLOBAL_KEY) is not None
    state.roi.store.put(rs.GLOBAL_KEY, None)
    state.db.log_event("info", "auto_roi", "global ROI cleared")
    return {"cleared": existed}


# ---------------------------------------------------------------- one camera
@router.get("/cameras/{cam_id}/auto-roi", dependencies=VIEWER)
def get_camera_roi(cam_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    """The camera's own ROI, or the global one it falls back to (``resolved_from`` says which)."""
    _camera(state, cam_id)
    doc = state.roi.store.all()
    if doc.get(cam_id):
        return _out(cam_id, doc[cam_id], "camera")
    if doc.get(rs.GLOBAL_KEY):
        return _out(cam_id, doc[rs.GLOBAL_KEY], "global")
    return _out(cam_id, None, "none")


@router.post("/cameras/{cam_id}/auto-roi", dependencies=ADMIN)
def compute_camera_roi(cam_id: str, body: Optional[ComputeIn] = None,
                       state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Re-run now: ``vote_frames`` frames (a file source: spread over the whole video; a live source: over
    ``vote_seconds``) -> segmentation -> majority vote -> polygons (source=auto, needs_review until approved).
    Replaces any existing record, manual ones included. Synchronous; about 1-3 s per frame on CPU."""
    cam = _camera(state, cam_id)
    s = state.settings(cam)
    n = (body.frames if body and body.frames else None) or s.roi.vote_frames
    frames = _camera_frames(state, cam, n, s.roi.vote_seconds)
    if not frames:
        raise HTTPException(503, "Camera source is not reachable: no frame could be read")
    if state.roi.store.get(cam_id):
        state.roi.store.put(cam_id, None)                       # Re-run: force a rebuild (manual ones too)
    ctx = state.roi.resolve(cam_id, frames, s.model_copy(update={"roi": s.roi.model_copy(update={"enabled": True})}))
    if ctx is None or not ctx.active or ctx.source != "auto":
        note = (ctx.warning if ctx else None) or "ROI is switched off (LDM_ROI=off)"
        state.db.log_event("warning", "auto_roi", f"auto ROI not determined: {note}", cam_id)
        return {**_out(cam_id, state.roi.store.get(cam_id)), "computed": False, "note": note}
    rec = ctx.record
    state.db.log_event("info", "auto_roi", f"auto ROI computed from {len(frames)} frame(s)", cam_id,
                       data={"area_frac": rec.get("area_frac"), "confidence": rec.get("confidence")})
    return {**_out(cam_id, rec, "camera"), "computed": True, "note": rec.get("note", "")}


@router.post("/cameras/{cam_id}/auto-roi/approve", dependencies=ADMIN)
def approve_camera_roi(cam_id: str, request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    _camera(state, cam_id)
    rec = state.roi.store.update(cam_id, needs_review=False, approved=True, approved_at=rs.now_iso(),
                                 approved_by=actor_of(request) or None, review_reason=None)
    if rec is None:
        raise HTTPException(404, "No ROI saved for this camera")
    state.db.log_event("info", "auto_roi", "ROI approved", cam_id)
    return _out(cam_id, rec, "camera")


@router.put("/cameras/{cam_id}/auto-roi", dependencies=ADMIN)
def edit_camera_roi(cam_id: str, body: EditIn, request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    _camera(state, cam_id)
    rec = _edit(state, cam_id, body, actor_of(request) or "")
    state.db.log_event("info", "auto_roi", "ROI edited (manual): needs approval", cam_id)
    return _out(cam_id, rec, "camera")


@router.delete("/cameras/{cam_id}/auto-roi", dependencies=ADMIN)
def clear_camera_roi(cam_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    _camera(state, cam_id)
    existed = state.roi.store.get(cam_id) is not None
    state.roi.store.put(cam_id, None)
    p = state.roi.reference_path(cam_id)
    if p:
        p.unlink(missing_ok=True)
    state.db.log_event("info", "auto_roi", "ROI cleared (recomputed automatically on next use)", cam_id)
    return {"cleared": existed}


@router.get("/cameras/{cam_id}/auto-roi/frame.jpg", dependencies=VIEWER)
def camera_roi_frame(cam_id: str, state: AppState = Depends(get_state)) -> Response:
    """The (median) frame the ROI was computed on, without drawing - the editor draws the polygon itself."""
    check_camera_id(cam_id)
    p = state.roi.reference_path(cam_id)
    if not p:
        raise HTTPException(404, "No reference frame: compute the ROI first")
    return Response(p.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/cameras/{cam_id}/auto-roi/overlay.jpg", dependencies=VIEWER)
def camera_roi_overlay(cam_id: str, state: AppState = Depends(get_state)) -> Response:
    check_camera_id(cam_id)
    p = state.roi.reference_path(cam_id)
    if not p:
        raise HTTPException(404, "No reference frame: compute the ROI first")
    img = cv2.imread(str(p))
    rec = state.roi.store.get(cam_id)
    block = rs.RoiContext(cam_id, rec, (rec or {}).get("source", "none")).summary()
    rs.draw_roi(img, block)
    return Response(_jpeg(img), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- single image, no camera needed
@router.post("/auto-roi/image", dependencies=OPERATOR)
async def image_roi(file: UploadFile = File(...), polygon: Optional[str] = Form(None),
                    camera_id: Optional[str] = Form(None), state: AppState = Depends(get_state)) -> dict[str, Any]:
    """ROI for one image: request polygon(s) > the camera's ROI (manual > auto) > global > computed now (not saved
    for an unknown camera)."""
    ext = "." + (file.filename or "").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else ""
    if ext not in IMAGE_EXTS:
        raise HTTPException(415, f"Unsupported file type '{ext or '?'}'")
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "File too large (limit 25 MB)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None:
        raise HTTPException(400, "The file is not a readable image")
    if camera_id:
        check_camera_id(camera_id)
    req = None
    if polygon:
        try:
            raw = json.loads(polygon)
            req = rs.validate_polygons(raw if raw and isinstance(raw[0], dict) or
                                       (raw and isinstance(raw[0], list) and raw[0] and isinstance(raw[0][0], list))
                                       else [raw])
        except (ValueError, TypeError, IndexError) as exc:
            raise HTTPException(422, f"invalid polygon: {exc}") from exc
    s = state.settings(state.db.get_camera(camera_id) if camera_id else None)
    s = s.model_copy(update={"roi": s.roi.model_copy(update={"enabled": True})})
    ctx = state.roi.resolve(camera_id or None, [img], s, req)
    block = ctx.summary() if ctx else None
    out = img.copy()
    rs.draw_roi(out, block)
    return {"roi": _compat(ctx.record) if ctx and ctx.active else None,
            "resolved_from": "computed" if ctx and ctx.source == "auto" and not camera_id else (ctx.source if ctx else "none"),
            "note": (ctx.warning or "; ".join(ctx.notes)) if ctx else "ROI is switched off (LDM_ROI=off)",
            "frame_size": [img.shape[1], img.shape[0]],
            "overlay_jpeg_base64": base64.b64encode(_jpeg(out)).decode()}
