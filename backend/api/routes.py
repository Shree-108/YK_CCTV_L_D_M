"""REST API (JSON). Every route declares its permission (backend/api/authz.py): PUBLIC, VIEWER, OPERATOR, ADMIN or
ADMIN_IF_LOGIN. With login off (the default) only ADMIN routes need the ADMIN_API_TOKEN bearer token, as before."""
from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.api.authz import (ADMIN, actor_of, ADMIN_IF_LOGIN, OPERATOR, PUBLIC, VIEWER, full_details, login_enabled,
                               message_filter, optional_principal, require_admin_here, stream_guard)
from backend.api.deps import IMAGE_EXTS, VIDEO_EXTS, check_camera_id, get_state, safe_display_name, save_upload
from backend.cameras import cctv
from backend.cameras.manager import camera_source, mask_source, open_capture
from backend.core.settings import DetectionSettings
from backend.database.db import utcnow
from backend.sampling.plan import parse_recording_start
from backend.services import analysis_outputs as ao
from backend.services import daily_report as dr
from backend.services import litter_reports as lr
from backend.services import litter_store as ls
from backend.services import video_evidence as ve
from backend.services.pipeline import detect_image
from backend.services.state import AppState

router = APIRouter(prefix="/api")


# =========================================================================== schemas
class CameraIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,40}$")
    name: str = Field(min_length=1, max_length=80)
    source: str = Field(min_length=1, max_length=500, description="rtsp://..., http(s)://..., file path, webcam index or env:VAR")
    location: str = Field("", max_length=120)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    enabled: bool = True
    store_id: Optional[str] = Field(None, max_length=64)   # must be a registered store (GET /api/stores)
    settings: dict[str, Any] = {}
    notify: dict[str, Any] = {}


class CameraUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Optional[str] = Field(None, min_length=1, max_length=80)
    source: Optional[str] = Field(None, min_length=1, max_length=500)
    location: Optional[str] = Field(None, max_length=120)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    enabled: Optional[bool] = None
    store_id: Optional[str] = Field(None, max_length=64)   # "" removes the store assignment
    settings: Optional[dict[str, Any]] = None
    notify: Optional[dict[str, Any]] = None


class AlertUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Optional[Literal["new", "acknowledged", "resolved", "false_positive"]] = None
    notes: Optional[str] = Field(None, max_length=2000)


class StoreIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z0-9._ \-]{1,64}$")
    name: str = Field(min_length=1, max_length=120)


class StoreUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)


class EventReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_status: Literal["DETECTED", "VERIFIED", "REJECTED", "RESOLVED"]
    note: Optional[str] = Field(None, max_length=1000)


class ActivateModel(BaseModel):
    version: str = Field(min_length=1, max_length=64)


# =========================================================================== helpers
def _source_summary(src: str) -> str:
    """What a non-admin may see of a camera source: its kind, never host names, user names, paths or env names."""
    if src.startswith("cctv:"):
        return src                                        # "cctv:<camera id>" - no infrastructure detail
    if "://" in src:
        return src.split("://", 1)[0].lower() + "://(hidden)"
    if src.startswith("env:"):
        return "env:(hidden)"
    return "webcam" if src.isdigit() else "file:(hidden)"


def _camera_out(state: AppState, cam: dict[str, Any], full: bool = True,
                redact: Callable[[Optional[str]], Optional[str]] = lambda t: t) -> dict[str, Any]:
    """``full=False`` (a non-admin with login on): no NVR host/username, RTSP URL or source location, and
    ``redact`` (MessageFilter.error) applied to the camera's last error."""
    out = {**cam, "source": mask_source(cam["source"])}
    configured = bool(cctv.is_cctv(cam) and cam.get("rtsp", {}).get("host"))
    out["rtsp_configured"] = configured
    if not full:
        out["source"] = _source_summary(cam["source"])
        out.pop("rtsp", None)
    elif configured:
        cfg = cctv.RtspConfig.model_validate(cam["rtsp"])
        out["rtsp_url"] = cctv.display_url(cfg)          # masked: never contains the password
        out["has_password"] = cctv.has_password(cam["id"])
    store = ls.get_store(state.db, cam["store_id"]) if cam.get("store_id") else None
    out["store_name"] = store["name"] if store else None
    st = state.cameras.status(cam["id"])
    out["status"] = ({"state": st.state, "fps": st.fps, "inference_ms": st.inference_ms, "last_error": redact(st.last_error),
                      "reconnects": st.reconnects, "frames_processed": st.frames_processed,
                      "frames_dropped": st.frames_dropped, "summary": st.summary, "resolution": st.resolution,
                     "capture_fps": st.capture_fps, "frames_captured": st.frames_captured,
                      "last_frame_age_sec": round(time.time() - st.last_frame_ts, 1) if st.last_frame_ts else None}
                     if st else {"state": "disabled" if not cam["enabled"] else "stopped"})
    return out


def _validate_cam_settings(state: AppState, cam_settings: dict[str, Any]) -> None:
    try:
        state.settings({"settings": cam_settings})
    except ValidationError as exc:
        raise HTTPException(422, f"invalid camera settings: {exc.errors()[0]['msg']}") from exc


def _alert_out(a: dict[str, Any]) -> dict[str, Any]:
    a = dict(a)
    a["bbox"] = [a.pop("x1"), a.pop("y1"), a.pop("x2"), a.pop("y2")]
    a["snapshot_url"] = f"/api/alerts/{a['id']}/snapshot" if a.get("snapshot_path") else None
    a["crop_url"] = f"/api/alerts/{a['id']}/snapshot?kind=crop" if a.get("crop_path") else None
    a.pop("snapshot_path", None)
    a.pop("crop_path", None)
    return a


def _system_usage(state: AppState) -> dict[str, Any]:
    import psutil

    out: dict[str, Any] = {"cpu_percent": psutil.cpu_percent(interval=None),
                           "memory_percent": psutil.virtual_memory().percent, "gpu": None}
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = {"name": torch.cuda.get_device_name(0),
                          "memory_used_mb": round(torch.cuda.memory_allocated(0) / 2**20),
                          "memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20)}
            try:
                out["gpu"]["utilization_percent"] = torch.cuda.utilization(0)
            except Exception:  # needs pynvml
                pass
    except ImportError:
        pass
    return out


# =========================================================================== health / system
@router.get("/health", dependencies=PUBLIC)
def health(request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Liveness for everyone. With login on: anonymous callers get the status alone; signed-in non-admins what the
    dashboard shows (model version, camera counts) without file paths, host usage or configuration; admins all."""
    eng = state.engine
    workers = state.cameras.workers
    if login_enabled(request) and optional_principal(request) is None:
        return {"status": "ok" if eng else "degraded", "time": utcnow()}
    if not full_details(request):
        return {
            "status": "ok" if eng else "degraded",
            "time": utcnow(),
            "model": ({"version": eng.version, "device": eng.device, "classes": eng.class_names,
                       "train_imgsz": eng.train_imgsz} if eng else None),
            "model_error": "The detection model could not be loaded (details for administrators)"
                           if state.engine_error else None,
            "cameras": {"configured": len(state.db.list_cameras()), "running": len(workers),
                        "online": sum(1 for w in workers.values() if w.status.state == "online")},
            "notifications": state.notifier.status(),
            "limits": _upload_limits(state),
        }
    return {
        "status": "ok" if eng else "degraded",
        "time": utcnow(),
        "uptime_sec": round(time.time() - state.started),
        "model": ({"version": eng.version, "path": eng.weights, "device": eng.device, "classes": eng.class_names,
                   "train_imgsz": eng.train_imgsz} if eng else None),
        "model_error": state.engine_error,
        "cameras": {"configured": len(state.db.list_cameras()), "running": len(workers),
                    "online": sum(1 for w in workers.values() if w.status.state == "online")},
        "admin_enabled": bool(state.cfg.admin_token),
        "notifications": state.notifier.status(),
        "system": _system_usage(state),
        "hardware": state.hardware,
        "limits": _upload_limits(state),
    }


def _upload_limits(state: AppState) -> dict[str, Any]:
    """What the upload page checks before sending a video (the server enforces the same limits)."""
    return {"max_video_mb": state.cfg.max_video_mb, "video_extensions": sorted(VIDEO_EXTS)}


@router.get("/classes", dependencies=VIEWER)
def classes(state: AppState = Depends(get_state)) -> dict[str, Any]:
    model_classes = state.engine.class_names if state.engine else []
    all_names = list(dict.fromkeys(model_classes + list(state.meta.classes)))
    return {"classes": [{"name": n, "priority": state.meta.priority(n), "bin": state.meta.bin(n),
                         "color": state.meta.colors.get(state.meta.priority(n)),
                         "in_active_model": n in model_classes} for n in all_names]}


@router.get("/summary", dependencies=VIEWER)
def summary(state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Live control-room summary aggregated over running cameras (real counters only)."""
    agg: dict[str, int] = {}
    for w in state.cameras.workers.values():
        for k, v in w.status.summary.items():
            agg[k] = agg.get(k, 0) + v
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    _, new_alerts = state.db.list_alerts(limit=1, status="new")
    _, today_alerts = state.db.list_alerts(limit=1, since=today)
    return {"live": agg, "new_alerts": new_alerts, "alerts_today": today_alerts,
            "cameras": [{"id": cid, "state": w.status.state, "fps": w.status.fps, "inference_ms": w.status.inference_ms}
                        for cid, w in state.cameras.workers.items()]}


@router.get("/events", dependencies=VIEWER)
def system_events(request: Request, limit: int = Query(100, ge=1, le=1000), camera_id: Optional[str] = None,
                  state: AppState = Depends(get_state)) -> dict[str, Any]:
    mf = message_filter(request)
    return {"events": [mf.event(e) for e in state.db.list_events(limit, camera_id)]}


# =========================================================================== cameras
@router.get("/cameras", dependencies=VIEWER)
def list_cameras(request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    full, mf = full_details(request), message_filter(request)
    return {"cameras": [_camera_out(state, c, full, mf.error) for c in state.db.list_cameras()]}


@router.get("/cameras/{cam_id}", dependencies=VIEWER)
def get_camera(cam_id: str, request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    cam = state.db.get_camera(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    return _camera_out(state, cam, full_details(request), message_filter(request).error)


def _check_store(state: AppState, store_id: Optional[str]) -> Optional[str]:
    """A camera may only be assigned to a registered store (no invented store IDs)."""
    store_id = (store_id or "").strip() or None
    if store_id and not ls.get_store(state.db, store_id):
        raise HTTPException(422, f"Store {store_id} is not registered. Add it under Stores first.")
    return store_id


@router.post("/cameras", status_code=201, dependencies=ADMIN)
def create_camera(body: CameraIn, request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    if state.db.get_camera(body.id):
        raise HTTPException(409, f"Camera {body.id} already exists")
    _validate_cam_settings(state, body.settings)
    data = body.model_dump(mode="json")
    data["store_id"] = _check_store(state, data.get("store_id"))
    cam = state.db.upsert_camera(data)
    state.db.log_event("info", "camera", "camera created", body.id)
    state.sync_cameras()
    return _camera_out(state, cam)  # type: ignore[arg-type]


@router.put("/cameras/{cam_id}", dependencies=ADMIN)
def update_camera(cam_id: str, body: CameraUpdate, request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    cam = state.db.get_camera(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    patch = body.model_dump(mode="json", exclude_none=True)
    if "settings" in patch:
        _validate_cam_settings(state, patch["settings"])
    if "store_id" in patch:
        patch["store_id"] = _check_store(state, patch["store_id"])
    cam = state.db.upsert_camera({**cam, **patch})
    state.db.log_event("info", "camera", f"camera updated ({', '.join(patch)})", cam_id)
    state.sync_cameras()
    return _camera_out(state, cam)


@router.delete("/cameras/{cam_id}", dependencies=ADMIN)
def delete_camera(cam_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    if not state.db.delete_camera(cam_id):
        raise HTTPException(404, "Camera not found")
    state.db.log_event("info", "camera", "camera deleted", cam_id)
    state.sync_cameras()
    return {"deleted": cam_id}


@router.get("/cameras/{cam_id}/frame.jpg", dependencies=VIEWER)
def camera_frame(cam_id: str, request: Request, raw: bool = False, authorization: Optional[str] = Header(default=None),
                 x_admin_token: Optional[str] = Header(default=None), state: AppState = Depends(get_state)) -> Response:
    """Latest annotated frame of a running camera; otherwise one raw frame grabbed from the source
    (camera setup preview). Opening a new connection to the camera/NVR is admin-only with login on: a viewer
    must not be able to make the server connect to a camera on demand."""
    w = state.cameras.workers.get(cam_id)
    if w and raw and w.latest_live()[1]:   # un-annotated frame (camera setup preview)
        return Response(w.latest_live()[1], media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    if w and w.latest_jpeg:
        return Response(w.latest_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    cam = state.db.get_camera(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    require_admin_here(request, authorization, x_admin_token, state, off_mode="open")
    try:
        cap = open_capture(camera_source(cam))
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from exc
    if not ok:
        raise HTTPException(503, "Camera source is not reachable")
    return Response(cv2.imencode(".jpg", frame)[1].tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@router.get("/cameras/{cam_id}/stream", dependencies=VIEWER)
async def camera_stream(cam_id: str, request: Request, state: AppState = Depends(get_state)) -> StreamingResponse:
    """MJPEG stream of annotated frames (multipart/x-mixed-replace). Ends when the viewer's session ends."""
    w = state.cameras.workers.get(cam_id)
    if not w:
        raise HTTPException(404, "Camera is not running")
    authorized = stream_guard(request)

    async def gen():
        last = None
        while cam_id in state.cameras.workers and authorized():
            jpg = w.latest_jpeg
            if jpg is not None and jpg is not last:
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(0.05)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# =========================================================================== live CCTV (RTSP / NVR)
LIVE_TUNING = {"processing_fps": ("video", "live_process_fps"), "frame_skip": ("video", "live_frame_skip"),
               "confidence": ("confidence", "confirmed_threshold"), "image_size": ("inference", "image_size")}


def _cctv_body(body: dict[str, Any]) -> tuple[cctv.RtspConfig, Optional[str]]:
    """Validate a CCTV form WITHOUT FastAPI's default 422 body (which would echo the password)."""
    if not isinstance(body, dict):
        raise HTTPException(422, "Invalid configuration")
    password = body.get("password")
    if password is not None and (not isinstance(password, str) or len(password) > 128):
        raise HTTPException(422, "Invalid configuration: password")
    try:
        cfg = cctv.RtspConfig.model_validate(body.get("rtsp") or {})
    except ValidationError as exc:
        e = exc.errors()[0]
        raise HTTPException(422, f"Invalid configuration: {'.'.join(map(str, e['loc']))} - {e['msg']}") from None
    return cfg, (password or None)


@router.get("/cctv/defaults", dependencies=ADMIN_IF_LOGIN)
def cctv_defaults() -> dict[str, Any]:
    """Non-secret defaults from CCTV_* environment variables (prefills the form)."""
    return {"defaults": cctv.env_defaults(), "path_template": cctv.DEFAULT_TEMPLATE,
            "template_note": "Common CP Plus/Dahua RTSP path. Confirm the exact path with the CCTV administrator."}


@router.post("/cctv/test", dependencies=ADMIN)
async def cctv_test(body: dict[str, Any], state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Validate the configuration, connect with authentication and decode a few real frames."""
    cam_id = str(body.get("camera_id") or "cctv_cam1")
    check_camera_id(cam_id)
    cfg, password = _cctv_body(body)
    res = await asyncio.to_thread(cctv.test_connection, cam_id, cfg, password)
    state.db.log_event("info" if res.ok else "warning", "cctv", f"connection test: {res.code}", cam_id)
    return res.model_dump()


@router.put("/cctv/{cam_id}", dependencies=ADMIN)
def cctv_save(cam_id: str, body: dict[str, Any], state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Create/update a Live CCTV camera. Only non-secret details go to the database; a password is
    written to the gitignored .env as CCTV_<ID>_PASSWORD."""
    check_camera_id(cam_id)
    cfg, password = _cctv_body(body)
    name = str(body.get("name") or cam_id).strip()[:80] or cam_id
    existing = state.db.get_camera(cam_id)
    if existing and not cctv.is_cctv(existing):
        raise HTTPException(409, "A non-CCTV camera already uses this id")
    settings = dict((existing or {}).get("settings") or {})
    for key, (section, field) in LIVE_TUNING.items():
        if body.get(key) not in (None, ""):
            settings.setdefault(section, {})[field] = body[key]
    _validate_cam_settings(state, settings)
    if password:
        cctv.store_password(cam_id, password)
    cam = state.db.upsert_camera({**(existing or {"notify": {}, "location": ""}), "id": cam_id, "name": name,
                                  "source": f"cctv:{cam_id}", "rtsp": cfg.model_dump(), "settings": settings,
                                  "enabled": bool(existing and existing["enabled"])})
    state.db.log_event("info", "cctv", "CCTV configuration saved", cam_id)
    state.sync_cameras()
    return _camera_out(state, cam)


def _cctv_toggle(state: AppState, cam_id: str, enabled: bool) -> dict[str, Any]:
    cam = state.db.get_camera(cam_id)
    if not cam or not cctv.is_cctv(cam):
        raise HTTPException(404, "CCTV camera not found")
    if enabled:
        try:
            cctv.camera_url(cam)   # configuration + credentials present (the URL itself is discarded)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
    cam = state.db.upsert_camera({**cam, "enabled": enabled})
    state.db.log_event("info", "cctv", "live stream started" if enabled else "live stream stopped", cam_id)
    state.sync_cameras()
    return _camera_out(state, cam)


@router.post("/cctv/{cam_id}/connect", dependencies=ADMIN)
def cctv_connect(cam_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    return _cctv_toggle(state, cam_id, True)


@router.post("/cctv/{cam_id}/stop", dependencies=ADMIN)
def cctv_stop(cam_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    return _cctv_toggle(state, cam_id, False)


# =========================================================================== scheduled captures
def _sched_cam(state: AppState, cam_id: str) -> dict[str, Any]:
    check_camera_id(cam_id)
    cam = state.db.get_camera(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    return cam


def _capture_out(c: dict[str, Any]) -> dict[str, Any]:
    c = dict(c)
    base = f"/api/captures/{c['id']}/image"
    c["original_url"] = f"{base}?kind=original" if c.pop("original_path", None) else None
    c["annotated_url"] = f"{base}?kind=annotated" if c.pop("annotated_path", None) else None
    return c


@router.get("/schedule/{cam_id}", dependencies=VIEWER)
def schedule_status(cam_id: str, request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    _sched_cam(state, cam_id)
    s = state.scheduler.status(cam_id)
    return {**s, "last_error": message_filter(request).error(s.get("last_error"))}


@router.put("/schedule/{cam_id}", dependencies=ADMIN)
def schedule_configure(cam_id: str, body: dict[str, Any], state: AppState = Depends(get_state)) -> dict[str, Any]:
    """min_minutes / max_minutes / task (litter|uniform) / model_version / retry_minutes / attempts."""
    _sched_cam(state, cam_id)
    try:
        state.scheduler.configure(cam_id, body)
    except ValidationError as exc:
        e = exc.errors()[0]
        raise HTTPException(422, f"{'.'.join(map(str, e['loc'])) or 'schedule'}: {e['msg']}") from None
    if body.get("model_version") and body["model_version"] not in {m["version"] for m in state.db.list_models()}:
        raise HTTPException(404, f"model {body['model_version']} not found")
    return state.scheduler.status(cam_id)


@router.post("/schedule/{cam_id}/{action}", dependencies=ADMIN)
async def schedule_action(cam_id: str, action: Literal["start", "pause", "resume", "stop", "capture-now"],
                          state: AppState = Depends(get_state)) -> dict[str, Any]:
    _sched_cam(state, cam_id)
    if action == "capture-now":
        res = await asyncio.to_thread(state.scheduler.capture, cam_id, "manual")
        if res is None:
            raise HTTPException(409, "A capture is already running for this camera")
        return {"capture": _capture_out(res), "schedule": state.scheduler.status(cam_id)}
    getattr(state.scheduler, action)(cam_id)
    return {"schedule": state.scheduler.status(cam_id)}


@router.get("/captures", dependencies=VIEWER)
def list_captures(request: Request, camera_id: Optional[str] = None, limit: int = Query(30, ge=1, le=200),
                  offset: int = Query(0, ge=0), state: AppState = Depends(get_state)) -> dict[str, Any]:
    mf = message_filter(request)
    return {"captures": [{**(o := _capture_out(c)), "error": mf.error(o.get("error"))}
                         for c in state.db.list_captures(camera_id, limit, offset)]}


@router.get("/captures/{cap_id}/image", dependencies=VIEWER)
def capture_image(cap_id: str, kind: Literal["original", "annotated"] = "annotated",
                  state: AppState = Depends(get_state)) -> FileResponse:
    c = state.db.get_capture(cap_id)
    path = c and c.get(f"{kind}_path")
    if not path:
        raise HTTPException(404, "Image not available")
    p = Path(path).resolve()
    if not p.is_file() or (state.cfg.storage_dir / "captures").resolve() not in p.parents:
        raise HTTPException(404, "Image not available")
    return FileResponse(p, media_type="image/jpeg")


# =========================================================================== stores
@router.get("/stores", dependencies=VIEWER)
def list_stores(state: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"stores": ls.list_stores(state.db)}


@router.post("/stores", status_code=201, dependencies=ADMIN)
def create_store(body: StoreIn, state: AppState = Depends(get_state)) -> dict[str, Any]:
    if ls.get_store(state.db, body.id.strip()):
        raise HTTPException(409, f"Store {body.id} already exists")
    st = ls.upsert_store(state.db, body.id.strip(), body.name.strip())
    state.db.log_event("info", "store", f"store {st['id']} created")
    return st


@router.put("/stores/{store_id}", dependencies=ADMIN)
def update_store(store_id: str, body: StoreUpdate, state: AppState = Depends(get_state)) -> dict[str, Any]:
    if not ls.get_store(state.db, store_id):
        raise HTTPException(404, "Store not found")
    st = ls.upsert_store(state.db, store_id, body.name.strip())
    state.db.log_event("info", "store", f"store {store_id} renamed")
    return st


def _identity(state: AppState, store_id: Optional[str], camera_id: Optional[str]) -> lr.Identity:
    try:
        return ls.resolve_identity(state.db, store_id, camera_id)
    except ls.IdentityError as exc:
        raise HTTPException(422, str(exc)) from None


# =========================================================================== litter events / analytics
@router.get("/litter/events", dependencies=VIEWER)
def litter_events(store_id: Optional[str] = None, camera_id: Optional[str] = None,
                  cls: Optional[str] = Query(None, alias="class"),
                  status: Optional[Literal["DETECTED", "VERIFIED", "REJECTED", "RESOLVED"]] = None,
                  start: Optional[str] = None, end: Optional[str] = None, limit: int = Query(100, ge=1, le=1000),
                  offset: int = Query(0, ge=0), state: AppState = Depends(get_state)) -> dict[str, Any]:
    _check_dates(start, end)
    rows, total = ls.list_events(state.db, start=start, end=end, store_id=store_id, camera_id=camera_id, cls=cls,
                                 status=status, limit=limit, offset=offset)
    return {"events": rows, "total": total}


@router.patch("/litter/events/{event_id}", dependencies=OPERATOR)
def review_litter_event(event_id: str, body: EventReview, request: Request,
                        state: AppState = Depends(get_state)) -> dict[str, Any]:
    if not ls.review_event(state.db, event_id, body.review_status, actor_of(request), body.note):
        raise HTTPException(404, "Litter event not found")
    state.db.log_event("info", "litter", f"litter event {event_id} -> {body.review_status}")
    return ls._event_out(ls.get_event(state.db, event_id))  # type: ignore[arg-type]


@router.get("/litter/events/{event_id}/image", dependencies=VIEWER)
def litter_event_image(event_id: str, state: AppState = Depends(get_state)) -> FileResponse:
    e = ls.get_event(state.db, event_id)
    p = Path(e["image_path"]).resolve() if e and e.get("image_path") else None
    if p is None or not p.is_file() or ao.output_root(state.cfg.storage_dir).resolve() not in p.parents:
        raise HTTPException(404, "Evidence image not available")
    return FileResponse(p, media_type="image/jpeg")


@router.get("/litter/analyses", dependencies=VIEWER)
def litter_analyses(request: Request, limit: int = Query(50, ge=1, le=500),
                    state: AppState = Depends(get_state)) -> dict[str, Any]:
    mf = message_filter(request)
    out = []
    for a in ls.list_analyses(state.db, limit):
        a.pop("output_dir", None)
        a["urls"] = ao.urls_for(a["id"], a.pop("files") or {})
        a["error"] = mf.error(a.get("error"))
        a.update(analysis_id=a["id"], processing_status=a["status"],
                 source_type={"quick_check": "sampled"}.get(a["kind"], a["kind"]),
                 identity={k: a.get(k) for k in ("store_id", "store_name", "camera_id", "camera_name",
                                                  "recording_id")} | {"warnings": a.get("warnings") or []})
        out.append(a)
    return {"analyses": out}


@router.get("/litter/analytics", dependencies=VIEWER)
def litter_analytics(start: Optional[str] = None, end: Optional[str] = None, store_id: Optional[str] = None,
                     camera_id: Optional[str] = None, granularity: Literal["hour", "day", "week"] = "day",
                     state: AppState = Depends(get_state)) -> dict[str, Any]:
    _check_dates(start, end)
    return ls.analytics(state.db, start=start, end=end, store_id=store_id, camera_id=camera_id,
                        granularity=granularity)


def _check_dates(*values: Optional[str]) -> None:
    for v in values:
        if v:
            try:
                dt.datetime.fromisoformat(v)
            except ValueError as exc:
                raise HTTPException(422, f"invalid ISO date: {v}") from exc


# =========================================================================== detections
@router.get("/detections", dependencies=VIEWER)
def list_detections(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                    camera_id: Optional[str] = None,
                    status: Optional[Literal["rejected", "weak", "candidate", "confirmed"]] = None,
                    job_id: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None,
                    cls: Optional[str] = Query(None, alias="class"), model_version: Optional[str] = None,
                    min_confidence: Optional[float] = Query(None, ge=0, le=1),
                    state: AppState = Depends(get_state)) -> dict[str, Any]:
    args = (camera_id, status, job_id, start, end, cls, model_version, min_confidence)
    rows = state.db.list_detections(limit, offset, *args)
    return {"detections": rows, "count": len(rows), "total": state.db.count_detections(*args)}


@router.post("/detect/image", dependencies=OPERATOR)
async def detect_image_endpoint(file: UploadFile = File(...), camera_id: str = Form(""),
                                return_image: bool = Form(True), model_version: Optional[str] = Form(None),
                                confidence: Optional[float] = Form(None, ge=0.01, le=0.99),
                                image_size: Optional[int] = Form(None, ge=160, le=4096),
                                tiling: Literal["auto", "on", "off"] = Form("auto"),
                                create_alerts: bool = Form(True),
                                store_id: Optional[str] = Form(None, max_length=64),
                                save_outputs: bool = Form(True),
                                state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Single-image litter detection on the FULL frame. Every detection the model returned (above the weak floor) is
    listed and drawn with its detection_status. With ``save_outputs`` (default) the analysis is stored: original,
    annotated image, detection CSV, summary CSV and metadata under LDM_OUTPUT/<store>/<date>/<camera>/, plus one
    litter event per DETECTED object. Store/camera identity: backend/services/litter_store.py (422 on a camera
    that belongs to another store)."""
    if camera_id and camera_id != "upload":
        check_camera_id(camera_id)
    ident = _identity(state, store_id, camera_id)
    _engine(state)
    try:
        engine = state.get_engine(model_version)
    except KeyError as exc:
        raise HTTPException(404, f"model {model_version} not found") from exc
    path = await save_upload(file, IMAGE_EXTS, state.cfg.max_image_mb, state.cfg.uploads_dir)
    name = safe_display_name(file.filename)
    cam_key = ident.camera_id if ident.camera_id != lr.UNKNOWN else "upload"
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Could not decode image (corrupt or not an image)")
        cam = state.db.get_camera(cam_key)
        s = _settings_override(state, cam, confidence, image_size, tiling)
        when = lr.local_now()
        res = await asyncio.to_thread(detect_image, engine, img, s, state.meta, state.db,
                                      state.alerts if create_alerts else None, cam_key, name,
                                      location=" · ".join(x for x in (ident.store_id, ident.camera_id)
                                                          if x != lr.UNKNOWN),
                                      header=ao.header_lines(ident, when))
        summary = {"confidence": s.confidence.model_dump(), "inference_image_size": s.inference.image_size,
                   "tiling": tiling, "create_alerts": create_alerts}
        res["analysis"] = await asyncio.to_thread(
            ao.image_analysis, state.db, state.cfg.storage_dir, result=res, image_bytes_path=path,
            original_name=name, ident=ident, settings_summary=summary, save=save_outputs)
    finally:
        path.unlink(missing_ok=True)
    res["analysis"].pop("output_dir", None)
    res["identity"] = ident.as_dict()
    annotated = res.pop("annotated")
    res["alerts"] = [_alert_out(a) for a in res["alerts"]]
    res["filename"] = name
    res["camera_id"] = ident.camera_id
    if return_image:
        res["annotated_image"] = base64.b64encode(cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])[1]
                                                  .tobytes()).decode()
    return res


@router.get("/analyses/{analysis_id}", dependencies=VIEWER)
def get_analysis(analysis_id: str, request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    """One stored analysis (image / video / quick check): identity, status, counts and file URLs."""
    a = ls.get_analysis(state.db, analysis_id) if lr.ANALYSIS_ID_RE.match(analysis_id) else None
    if a is None:
        raise HTTPException(404, "Analysis not found")
    a.pop("output_dir", None)
    a["urls"] = ao.urls_for(analysis_id, a.pop("files") or {})
    a["error"] = message_filter(request).error(a.get("error"))
    return a


@router.get("/analyses/{analysis_id}/files/{kind}", dependencies=VIEWER)
def analysis_file(analysis_id: str, kind: str, state: AppState = Depends(get_state)) -> Response:
    """original | annotated | video | detections_csv | summary_csv | metadata | report_csv | screenshot_01..04 |
    screenshots_json of one analysis. ``report_csv`` (six columns) is rebuilt from the DB on every download, so an
    operator's VERIFIED / REJECTED review is reflected (backend/services/video_evidence.py)."""
    p = ao.analysis_file(state.db, analysis_id, kind)
    if p is None:
        raise HTTPException(404, "Not found")
    if kind == "report_csv":
        a = ls.get_analysis(state.db, analysis_id)
        return Response(ve.live_report(state.db, a).encode("utf-8-sig"), media_type=ao.MEDIA_TYPES[kind],
                        headers={"Content-Disposition": f'attachment; filename="{analysis_id}_litter_report.csv"'})
    media = ao.MEDIA_TYPES.get(kind) or ("image/png" if p.suffix == ".png" else "image/jpeg")
    return FileResponse(p, media_type=media, filename=p.name)


@router.get("/reports/daily", dependencies=VIEWER)
def daily_reports(state: AppState = Depends(get_state)) -> dict[str, Any]:
    """The consolidated six-column CSV of each day on which videos were analysed (newest first)."""
    return {"columns": ve.REPORT_COLUMNS, "days": dr.list_days(state.cfg.storage_dir)}


@router.get("/reports/daily/{day}.csv", dependencies=VIEWER)
def daily_report_csv(day: str, state: AppState = Depends(get_state)) -> FileResponse:
    """litter_report_<YYYY-MM-DD>.csv: every video analysed that day, appended in completion order."""
    p = dr.find(state.cfg.storage_dir, day)
    if p is None:
        raise HTTPException(404, "No report for that date")
    return FileResponse(p, media_type=ao.MEDIA_TYPES["daily_report_csv"], filename=p.name)


@router.post("/detect/video", status_code=202, dependencies=OPERATOR)
async def detect_video_endpoint(file: UploadFile = File(...), camera_id: str = Form(""),
                                store_id: Optional[str] = Form(None, max_length=64),
                                recording_start: Optional[str] = Form(None, max_length=40),
                                sample_interval_sec: Optional[float] = Form(None, ge=0.1, le=60),
                                confidence: Optional[float] = Form(None, ge=0.01, le=0.99),
                                image_size: Optional[int] = Form(None, ge=160, le=4096),
                                tiling: Literal["auto", "on", "off"] = Form("auto"),
                                sample_fps: Optional[float] = Form(None, gt=0, le=60),
                                temporal_window: Optional[int] = Form(None, ge=1, le=100),
                                min_confirmations: Optional[int] = Form(None, ge=1, le=100),
                                model_version: Optional[str] = Form(None),
                                screenshot_seed: Optional[int] = Form(None, ge=0, le=2**31 - 1),
                                state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Full-video litter detection (full frame, tracking, unique litter events). ``sample_interval_sec``: one frame
    analysed every N seconds (wins over ``sample_fps``). ``recording_start``: local date-time of the first frame;
    without it, timestamps are processing time (and say so)."""
    if camera_id and camera_id != "upload":
        check_camera_id(camera_id)
    ident = _identity(state, store_id, camera_id)
    try:
        start = parse_recording_start(recording_start)
    except ValueError:
        raise HTTPException(422, "recording_start must be a date and time such as 2026-11-19T14:05:00") from None
    if start is None:
        ident.warnings.append("Recording start not given: timestamps are the time of processing, not of recording.")
    _engine(state)
    path = await save_upload(file, VIDEO_EXTS, state.cfg.max_video_mb, state.cfg.uploads_dir)
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "Could not decode video (corrupt file or unsupported codec)")
    cam = state.db.get_camera(ident.camera_id) if ident.camera_id != lr.UNKNOWN else None
    patch = _override_patch(state, cam, confidence, image_size, tiling)
    if sample_interval_sec:
        patch.setdefault("video", {})["sample_fps"] = round(1.0 / sample_interval_sec, 6)
    elif sample_fps:
        patch.setdefault("video", {})["sample_fps"] = sample_fps
    if tiling in ("on", "off"):
        patch.setdefault("video", {})["tiling"] = tiling == "on"
    if screenshot_seed is not None:
        patch.setdefault("video", {})["screenshot_seed"] = screenshot_seed
    if temporal_window or min_confirmations:
        patch["tracking"] = {k: v for k, v in (("temporal_window", temporal_window),
                                                 ("min_confirmations", min_confirmations)) if v}
    try:
        state.settings_with(cam, patch)
    except ValidationError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, f"invalid settings: {exc.errors()[0]['msg']}") from exc
    if model_version and model_version not in {m["version"] for m in state.db.list_models()}:
        path.unlink(missing_ok=True)
        raise HTTPException(404, f"model {model_version} not found")
    job_id = state.submit_video(path, ident, safe_display_name(file.filename), patch, model_version or None, start)
    return {"job_id": job_id, "status": "queued", "status_url": f"/api/jobs/{job_id}", "identity": ident.as_dict()}


@router.get("/jobs", dependencies=VIEWER)
def list_jobs(request: Request, limit: int = Query(50, ge=1, le=500), state: AppState = Depends(get_state)) -> dict[str, Any]:
    mf = message_filter(request)
    return {"jobs": [{**j, "error": mf.error(j.get("error"))} for j in state.db.list_jobs(limit)]}


@router.get("/jobs/{job_id}", dependencies=VIEWER)
def get_job(job_id: str, request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    job = state.db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job.pop("output_dir", None)
    job["error"] = message_filter(request).error(job.get("error"))
    if job["status"] == "done":
        job["files"] = {k: f"/api/jobs/{job_id}/files/{k}" for k in JOB_FILES}
        a = ls.analysis_for_job(state.db, job_id)
        if a:
            job["files"].update(ao.urls_for(a["id"], {k: v for k, v in a["files"].items() if k != "video"}))
    return job


JOB_FILES = {"video": ("annotated.mp4", "video/mp4"), "detections": ("detections.json", "application/json"),
             "events": ("events.json", "application/json"), "stats": ("stats.json", "application/json")}


@router.get("/jobs/{job_id}/evidence/{event_id}.jpg", dependencies=VIEWER)
def job_evidence(job_id: str, event_id: str, state: AppState = Depends(get_state)) -> FileResponse:
    e = ls.get_event(state.db, event_id)
    if not e or e.get("job_id") != job_id:
        raise HTTPException(404, "Evidence not found")
    return litter_event_image(event_id, state)


@router.get("/jobs/{job_id}/files/{kind}", dependencies=VIEWER)
def job_file(job_id: str, kind: str, state: AppState = Depends(get_state)) -> FileResponse:
    job = state.db.get_job(job_id)
    if not job or kind not in JOB_FILES:
        raise HTTPException(404, "Not found")
    name, media = JOB_FILES[kind]
    p = Path(job["output_dir"]) / name
    if not p.is_file():
        raise HTTPException(404, "File not available")
    return FileResponse(p, media_type=media, filename=f"{job_id}_{name}")


# =========================================================================== alerts
@router.get("/alerts", dependencies=VIEWER)
def list_alerts(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                camera_id: Optional[str] = None, cls: Optional[str] = Query(None, alias="class"),
                severity: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None,
                status: Optional[Literal["new", "acknowledged", "resolved", "false_positive"]] = None,
                min_confidence: Optional[float] = Query(None, ge=0, le=1),
                start: Optional[str] = None, end: Optional[str] = None,
                state: AppState = Depends(get_state)) -> dict[str, Any]:
    rows, total = state.db.list_alerts(limit, offset, start, end, min_confidence, camera_id=camera_id,
                                       severity=severity, status=status, **{"class": cls})
    return {"alerts": [_alert_out(a) for a in rows], "total": total}


@router.get("/alerts/{alert_id}", dependencies=VIEWER)
def get_alert(alert_id: int, state: AppState = Depends(get_state)) -> dict[str, Any]:
    a = state.db.get_alert(alert_id)
    if not a:
        raise HTTPException(404, "Alert not found")
    return _alert_out(a)


@router.patch("/alerts/{alert_id}", dependencies=OPERATOR)
def update_alert(alert_id: int, body: AlertUpdate, state: AppState = Depends(get_state)) -> dict[str, Any]:
    fields = body.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(422, "nothing to update")
    if not state.db.update_alert(alert_id, **fields, updated_at=utcnow()):
        raise HTTPException(404, "Alert not found")
    if "status" in fields:
        state.db.log_event("info", "alerts", f"alert {alert_id} -> {fields['status']}")
    return _alert_out(state.db.get_alert(alert_id))  # type: ignore[arg-type]


@router.get("/alerts/{alert_id}/snapshot", dependencies=VIEWER)
def alert_snapshot(alert_id: int, kind: Literal["full", "crop"] = "full",
                   state: AppState = Depends(get_state)) -> FileResponse:
    a = state.db.get_alert(alert_id)
    path = a and a.get("crop_path" if kind == "crop" else "snapshot_path")
    if not path:
        raise HTTPException(404, "Snapshot not found")
    p = Path(path).resolve()
    if not p.is_file() or state.cfg.alerts_dir.resolve() not in p.parents:
        raise HTTPException(404, "Snapshot not found")
    return FileResponse(p, media_type="image/jpeg")


# =========================================================================== analytics
@router.get("/analytics", dependencies=VIEWER)
def analytics(start: Optional[str] = None, end: Optional[str] = None, camera_id: Optional[str] = None,
              cls: Optional[str] = Query(None, alias="class"), model_version: Optional[str] = None,
              state: AppState = Depends(get_state)) -> dict[str, Any]:
    for v in (start, end):
        if v:
            try:
                dt.datetime.fromisoformat(v)
            except ValueError as exc:
                raise HTTPException(422, f"invalid ISO date: {v}") from exc
    return {"start": start, "end": end, "camera_id": camera_id,
            **state.db.analytics(start, end, camera_id, cls, model_version)}


# =========================================================================== models
@router.get("/models", dependencies=VIEWER)
def list_models(request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    import json as _json
    models = state.registry.sync()
    latest_eval: dict[str, dict[str, Any]] = {}
    for j in state.db.list_jobs(500):           # newest first
        if j["kind"] == "evaluation" and j["status"] == "done":
            full = state.db.get_job(j["id"]) or {}
            r = full.get("result") or {}
            v = r.get("model_version")
            if v and v not in latest_eval:
                latest_eval[v] = {"id": j["id"], "dataset": r.get("dataset"), "split": r.get("split"),
                                  "finished_at": j["finished_at"], "overall": r.get("overall")}
    for m in models:
        p = Path(m["path"])
        m["file_exists"] = p.is_file()
        meta_f = p.parent / "model_meta.json"
        meta = _json.loads(meta_f.read_text()) if meta_f.is_file() else {}
        m["architecture"] = Path(str(meta.get("base_model") or "unknown")).stem
        m["train_imgsz"] = meta.get("imgsz")
        m["experiment_id"] = meta.get("experiment_id")
        m["dataset"] = "data.yaml" if (p.parent / "data.yaml").is_file() else None
        if m["dataset"]:
            try:
                import yaml
                m["dataset"] = Path(str((yaml.safe_load((p.parent / "data.yaml").read_text()) or {}).get("path", ""))).name or None
            except Exception:  # noqa: BLE001
                pass
        m["size_mb"] = round(p.stat().st_size / 2**20, 1) if p.is_file() else None
        m["latest_evaluation"] = latest_eval.get(m["version"])
    if not full_details(request):                 # non-admins (login on): path inside the models folder only
        for m in models:
            p = Path(m["path"])
            try:
                m["path"] = p.resolve().relative_to(state.cfg.models_dir.resolve()).as_posix()
            except ValueError:
                m["path"] = p.name
    return {"models": models, "active": state.engine.version if state.engine else None}


@router.post("/models/activate", dependencies=ADMIN)
def activate_model(body: ActivateModel, state: AppState = Depends(get_state)) -> dict[str, Any]:
    try:
        m = state.activate_model(body.version)
    except KeyError as exc:
        raise HTTPException(404, f"Model {body.version} not found") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not load model {body.version}: {exc}. Previous model remains active.") from exc
    return {"active": m["version"]}


# =========================================================================== settings
@router.get("/settings", dependencies=VIEWER)
def get_settings(state: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"settings": state.settings().model_dump(), "defaults": DetectionSettings().model_dump(),
            "overrides": state.global_settings_json()}


@router.put("/settings", dependencies=ADMIN)
def put_settings(patch: dict[str, Any], state: AppState = Depends(get_state)) -> dict[str, Any]:
    try:
        s = state.save_global_settings(patch)
    except ValidationError as exc:
        raise HTTPException(422, [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]) from exc
    return {"settings": s.model_dump()}


def _override_patch(state: AppState, cam: Optional[dict[str, Any]], confidence: Optional[float],
                    image_size: Optional[int], tiling: str) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    if confidence is not None:
        cur = state.settings(cam).confidence
        patch["confidence"] = {"confirmed_threshold": confidence,
                               "candidate_threshold": min(cur.candidate_threshold, confidence),
                               "weak_threshold": min(cur.weak_threshold, confidence)}
    if image_size:
        patch["inference"] = {"image_size": image_size}
    if tiling == "on":
        patch["tiling"] = {"enabled": True, "mode": "always"}
    elif tiling == "off":
        patch["tiling"] = {"enabled": False}
    return patch


def _settings_override(state: AppState, cam: Optional[dict[str, Any]], confidence: Optional[float],
                       image_size: Optional[int], tiling: str) -> DetectionSettings:
    try:
        return state.settings_with(cam, _override_patch(state, cam, confidence, image_size, tiling))
    except ValidationError as exc:
        raise HTTPException(422, f"invalid settings: {exc.errors()[0]['msg']}") from exc


def _engine(state: AppState):
    if state.engine is None:
        raise HTTPException(503, state.engine_error or "Model not loaded")
    return state.engine


# ================================================================ dashboard import rows (four-screenshot jobs)
_SV_JOB_RE = re.compile(r"^[A-Za-z0-9_\-]{1,60}$")


@router.get("/detect/video/{job_id}/report", dependencies=VIEWER)
def video_dashboard_report(job_id: str, request: Request,
                           format: Literal["table", "csv", "csv_full"] = "table", per_object: bool = False,
                           authorization: Optional[str] = Header(default=None),
                           x_admin_token: Optional[str] = Header(default=None),
                           state: AppState = Depends(get_state)) -> Any:
    """Dashboard import rows of a finished four-screenshot Video Analysis (backend/services/dashboard_rows.py).
    ``table`` (JSON, viewer) and the two CSVs - ``csv`` = importer columns, ``csv_full`` = + store_name,source_file -
    come from the same build_dashboard_rows(). CSV downloads keep the CSV rule (admin when login is on)."""
    from backend.services import dashboard_rows as dbr
    if format != "table":
        require_admin_here(request, authorization, x_admin_token, state, off_mode="open")
    if not _SV_JOB_RE.match(job_id):
        raise HTTPException(404, "Analysis not found")
    jobs = state.db._query("SELECT * FROM sampled_jobs WHERE id=?", (job_id,))
    if not jobs:
        raise HTTPException(404, "Analysis not found")
    if jobs[0].get("state") != "done":
        raise HTTPException(409, "The analysis has not completed yet")
    rows = dbr.build_dashboard_rows(jobs[0], query=state.db._query, view_quality=state.settings().view_quality,
                                    per_object=per_object)
    if format == "table":
        return dbr.table(rows)
    full = format == "csv_full"
    name = f"{job_id}_dashboard_import{'_full' if full else ''}{'_objects' if per_object else ''}.csv"
    return Response(dbr.to_csv(rows, full=full).encode("utf-8"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
