"""Automatic ground ROI API (backend/services/auto_roi.py). Owner request 2026-11-24.

Permissions (backend/api/authz.py): reads and preview images = VIEWER; computing, approving, editing and clearing a
camera's or the global ROI = ADMIN (camera configuration); an on-the-fly ROI for one uploaded image = OPERATOR.

Precedence for the image endpoint: polygon in the request > ROI saved for camera_id > global ROI > computed now.
The camera object (/api/cameras) is unchanged: ROIs live in storage/camera_roi.json, not in the camera row.
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
from backend.services import auto_roi as ar
from backend.services.state import AppState

router = APIRouter(prefix="/api")
MAX_IMAGE_BYTES = 25 * 1024 * 1024


class ComputeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frames: int = Field(ar.DEFAULT_FRAMES, ge=1, le=15)


class EditIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    polygon: list[list[float]] = Field(min_length=3, max_length=200)
    holes: list[list[list[float]]] = Field(default_factory=list, max_length=20)


def _store(state: AppState) -> ar.RoiStore:
    return ar.store_for(state.cfg.storage_dir)


def _camera(state: AppState, cam_id: str) -> dict[str, Any]:
    check_camera_id(cam_id)
    cam = state.db.get_camera(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    return cam


def _jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def _camera_frames(state: AppState, cam: dict[str, Any], n: int) -> list[np.ndarray]:
    """N frames of the camera: from its running worker (no second connection to the NVR) or its source."""
    w = state.cameras.workers.get(cam["id"])
    if w is not None:
        out, last = [], None
        deadline = time.monotonic() + max(10.0, n * 3.0)
        while len(out) < n and time.monotonic() < deadline:
            raw = w.latest_live()[1]
            if raw and raw is not last:
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    out.append(img)
                last = raw
            time.sleep(1.0 if len(out) < n else 0)
        if out:
            return out
    try:
        return ar.frames_from_capture(lambda: open_capture(camera_source(cam)), n)
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from exc


def _out(key: str, rec: Optional[dict[str, Any]], resolved_from: str = "") -> dict[str, Any]:
    return {"key": key, "roi": rec, "resolved_from": resolved_from or ("camera" if rec else "none")}


# ---------------------------------------------------------------- listing / global
@router.get("/auto-roi", dependencies=VIEWER)
def list_rois(state: AppState = Depends(get_state)) -> dict[str, Any]:
    doc = _store(state).all()
    return {"global": doc.get("global"), "cameras": doc.get("cameras", {}),
            "precedence": ["request", "camera", "global", "none"]}


@router.put("/auto-roi/global", dependencies=ADMIN)
def set_global(body: EditIn, request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    try:
        rec = _store(state).edit(ar.GLOBAL_KEY, body.polygon, body.holes, actor_of(request) or "")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    state.db.log_event("info", "auto_roi", "global ROI saved")
    return _out(ar.GLOBAL_KEY, rec, "global")


@router.delete("/auto-roi/global", dependencies=ADMIN)
def clear_global(state: AppState = Depends(get_state)) -> dict[str, Any]:
    existed = _store(state).clear(ar.GLOBAL_KEY)
    state.db.log_event("info", "auto_roi", "global ROI cleared")
    return {"cleared": existed}


# ---------------------------------------------------------------- one camera
@router.get("/cameras/{cam_id}/auto-roi", dependencies=VIEWER)
def get_camera_roi(cam_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    """The camera's own ROI, or the global one it falls back to (``resolved_from`` says which)."""
    _camera(state, cam_id)
    rec, src = _store(state).resolve(None, cam_id)
    return _out(cam_id, rec, src)


@router.post("/cameras/{cam_id}/auto-roi", dependencies=ADMIN)
def compute_camera_roi(cam_id: str, body: Optional[ComputeIn] = None,
                       state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Median of N frames -> segmentation -> polygon, saved with source=auto and needs_review=true.
    Synchronous: about 10-20 s on CPU (plus ~15 s the first time, while the segmenter loads)."""
    cam = _camera(state, cam_id)
    n = (body or ComputeIn()).frames
    frames = _camera_frames(state, cam, n)
    if not frames:
        raise HTTPException(503, "Camera source is not reachable: no frame could be read")
    try:
        result = ar.compute_auto_roi(frames)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Automatic ROI is not available on this server: {exc}") from exc
    store = _store(state)
    rec = store.save_auto(cam_id, result)
    if rec is None:
        state.db.log_event("warning", "auto_roi", "auto ROI found no ground region", cam_id)
        return {**_out(cam_id, store.get(cam_id)), "computed": False, "note": result.note}
    state.db.log_event("info", "auto_roi", f"auto ROI computed from {len(frames)} frame(s), needs review", cam_id,
                       data={"ground_share": rec.get("ground_share")})
    return {**_out(cam_id, rec, "camera"), "computed": True, "note": result.note}


@router.post("/cameras/{cam_id}/auto-roi/approve", dependencies=ADMIN)
def approve_camera_roi(cam_id: str, request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    _camera(state, cam_id)
    try:
        rec = _store(state).approve(cam_id, actor_of(request) or "")
    except KeyError as exc:
        raise HTTPException(404, "No ROI saved for this camera") from exc
    state.db.log_event("info", "auto_roi", "ROI approved", cam_id)
    return _out(cam_id, rec, "camera")


@router.put("/cameras/{cam_id}/auto-roi", dependencies=ADMIN)
def edit_camera_roi(cam_id: str, body: EditIn, request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    _camera(state, cam_id)
    try:
        rec = _store(state).edit(cam_id, body.polygon, body.holes, actor_of(request) or "")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    state.db.log_event("info", "auto_roi", "ROI edited", cam_id)
    return _out(cam_id, rec, "camera")


@router.delete("/cameras/{cam_id}/auto-roi", dependencies=ADMIN)
def clear_camera_roi(cam_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    _camera(state, cam_id)
    existed = _store(state).clear(cam_id)
    state.db.log_event("info", "auto_roi", "ROI cleared", cam_id)
    return {"cleared": existed}


@router.get("/cameras/{cam_id}/auto-roi/frame.jpg", dependencies=VIEWER)
def camera_roi_frame(cam_id: str, state: AppState = Depends(get_state)) -> Response:
    """The (median) frame the ROI was computed on, without drawing - the editor draws the polygon itself."""
    check_camera_id(cam_id)
    p = _store(state).reference_path(cam_id)
    if not p:
        raise HTTPException(404, "No reference frame: compute the ROI first")
    return Response(p.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/cameras/{cam_id}/auto-roi/overlay.jpg", dependencies=VIEWER)
def camera_roi_overlay(cam_id: str, state: AppState = Depends(get_state)) -> Response:
    check_camera_id(cam_id)
    store = _store(state)
    p = store.reference_path(cam_id)
    if not p:
        raise HTTPException(404, "No reference frame: compute the ROI first")
    img = cv2.imread(str(p))
    rec, _ = store.resolve(None, cam_id)
    return Response(_jpeg(ar.render_overlay(img, rec)), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- single image, no camera needed
@router.post("/auto-roi/image", dependencies=OPERATOR)
async def image_roi(file: UploadFile = File(...), polygon: Optional[str] = Form(None),
                    camera_id: Optional[str] = Form(None), state: AppState = Depends(get_state)) -> dict[str, Any]:
    """ROI for one image: request polygon > camera ROI > global ROI > computed on the fly (not saved)."""
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
    try:
        req_poly = json.loads(polygon) if polygon else None
        rec, src = _store(state).resolve(req_poly, camera_id or None)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, f"invalid polygon: {exc}") from exc
    note = ""
    if rec is None:
        try:
            result = ar.compute_auto_roi(img)
        except FileNotFoundError as exc:
            raise HTTPException(503, f"Automatic ROI is not available on this server: {exc}") from exc
        rec, src, note = result.record, ("computed" if result.record else "none"), result.note
    overlay = _jpeg(ar.render_overlay(img, rec))
    return {"roi": rec, "resolved_from": src, "note": note, "frame_size": [img.shape[1], img.shape[0]],
            "overlay_jpeg_base64": base64.b64encode(overlay).decode()}
