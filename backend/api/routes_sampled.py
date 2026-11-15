"""Four-screenshot video analysis API (backend/sampling). Every route declares exactly one permission
(backend/api/authz.py): uploads, runs and ROI confirmation = OPERATOR; reads = VIEWER; CSV downloads =
ADMIN_IF_LOGIN (same rule as the existing CSV export)."""
from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Any, Literal, Optional

import cv2
from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.api.authz import ADMIN_IF_LOGIN, OPERATOR, VIEWER, message_filter, require_admin_here
from backend.api.deps import VIDEO_EXTS, get_state, safe_display_name, save_upload
from backend.database.db import utcnow
from backend.detection.filters import ROIZone
from backend.sampling import report
from backend.sampling.media import Unreadable, new_media_id, probe, sha256_of
from backend.sampling.reports import Reports, public_job
from backend.sampling.runner import DuplicateJob, JobError
from backend.services.state import AppState

router = APIRouter(prefix="/api/sampled")
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def _runner(state: AppState):
    return state.sampled


def _reports(state: AppState) -> Reports:
    return Reports(state.sampled.repo)


def _csv(rows: list[dict[str, Any]], name: str) -> StreamingResponse:
    return StreamingResponse(iter([report.to_csv(rows)]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


# =========================================================================== media
@router.post("/media", status_code=201, dependencies=OPERATOR)
async def upload_media(request: Request, file: UploadFile = File(...), state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Receive a CCTV video, check it is readable and return its metadata and a preview frame."""
    tmp = await save_upload(file, VIDEO_EXTS, state.cfg.max_video_mb, state.cfg.media_dir)
    mid = new_media_id()
    path = tmp.with_name(f"{mid}{tmp.suffix}")
    os.replace(tmp, path)
    name = safe_display_name(file.filename)
    try:
        meta, preview = await asyncio.to_thread(probe, path, name)
        sha = await asyncio.to_thread(sha256_of, path)
    except Unreadable as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from None
    except Exception:
        path.unlink(missing_ok=True)
        raise
    cv2.imwrite(str(state.cfg.media_dir / f"{mid}_preview.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 88])
    repo = _runner(state).repo
    row = {"id": mid, "created_at": utcnow(), "path": str(path), "sha256": sha, "size_bytes": path.stat().st_size, **meta}
    repo.insert_media(row)
    earlier = repo.media_by_sha(sha, exclude=mid)
    prev = []
    for m in earlier[-5:]:
        prev.append({"media_id": m["id"], "received_at": m["created_at"], "filename": m["filename"],
                     "jobs": [{"id": j["id"], "state": j["state"], "created_at": j["created_at"]}
                              for j in repo.jobs("WHERE j.media_id=?", (m["id"],), limit=5)]})
    return {**_public_media(row), "previously_received": prev}


def _public_media(m: dict[str, Any]) -> dict[str, Any]:
    return {k: m.get(k) for k in ("id", "created_at", "filename", "size_bytes", "width", "height", "fps", "frames",
                                  "duration_sec", "suggested_recording_start")} | {
        "preview_url": f"/api/sampled/media/{m['id']}/preview.jpg"}


@router.get("/media/{media_id}", dependencies=VIEWER)
def get_media(media_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    m = _runner(state).repo.media(media_id)
    if not m:
        raise HTTPException(404, "Video not found")
    return _public_media(m)


@router.get("/media/{media_id}/preview.jpg", dependencies=VIEWER)
def media_preview(media_id: str, state: AppState = Depends(get_state)) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{16}", media_id):
        raise HTTPException(404, "Not found")
    p = state.cfg.media_dir / f"{media_id}_preview.jpg"
    if not p.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(p, media_type="image/jpeg")


# =========================================================================== jobs
class JobIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    media_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    store_id: Optional[str] = Field(None, max_length=64)
    camera_id: Optional[str] = Field(None, max_length=40)
    recording_start: Optional[str] = Field(None, max_length=40)
    seed: Optional[int] = None
    roi_mode: Literal["auto", "camera", "manual"] = "auto"
    auto_accept: bool = False
    create_alerts: bool = True
    model_version: Optional[str] = Field(None, max_length=80)
    new_run: bool = False


@router.post("/jobs", status_code=202, dependencies=OPERATOR)
def create_job(body: JobIn, request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    if body.model_version:
        try:
            state.get_engine(body.model_version)
        except KeyError:
            raise HTTPException(422, f"Unknown model version '{body.model_version}'") from None
    elif state.engine is None:
        raise HTTPException(503, "No detection model is loaded; the analysis cannot run.")
    try:
        job, reused = _runner(state).create(body.media_id, **body.model_dump(exclude={"media_id"}))
    except KeyError:
        raise HTTPException(404, "Video not found (upload it again).") from None
    except DuplicateJob as exc:
        raise HTTPException(409, {"message": str(exc), "job_id": exc.job_id}) from None
    except JobError as exc:
        raise HTTPException(422, str(exc)) from None
    return {"job_id": job["id"], "reused": reused, "state": job["state"]}


@router.get("/jobs", dependencies=VIEWER)
def list_jobs(request: Request, limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0),
              state: AppState = Depends(get_state)) -> dict[str, Any]:
    mf = message_filter(request)
    return _reports(state).history(limit=limit, offset=offset, error=mf.error)


@router.get("/jobs/{job_id}", dependencies=VIEWER)
def get_job(job_id: str, request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    r = _runner(state)
    job = r.repo.job(job_id)
    if not job:
        raise HTTPException(404, "Analysis not found")
    return public_job(job, r.repo.media(job["media_id"]), r.repo.frames(job_id), message_filter(request).error)


class RoiConfirmIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    zones: list[ROIZone] = Field(default_factory=list, max_length=20)
    frame_zones: dict[int, list[ROIZone]] = Field(default_factory=dict)
    skip_frames: list[int] = Field(default_factory=list, max_length=4)
    whole_frame: bool = False
    save_for_camera: bool = False


@router.put("/jobs/{job_id}/roi", dependencies=OPERATOR)
def confirm_roi(job_id: str, body: RoiConfirmIn, request: Request,
                authorization: Optional[str] = Header(default=None), x_admin_token: Optional[str] = Header(default=None),
                state: AppState = Depends(get_state)) -> dict[str, Any]:
    r = _runner(state)
    job = r.repo.job(job_id)
    if not job:
        raise HTTPException(404, "Analysis not found")
    zones = [z.model_dump() for z in body.zones]
    if body.save_for_camera:
        cam_id = job.get("camera_id")
        if not cam_id:
            raise HTTPException(422, "Enter a Camera ID for this analysis to save the ROI for that camera.")
        if body.whole_frame or not any(z["mode"] == "include" for z in zones):
            raise HTTPException(422, "Only an ROI with an outdoor area can be saved for a camera.")
        if state.db.get_camera(cam_id):           # a registered camera's ROI is admin configuration (as PUT /roi)
            require_admin_here(request, authorization, x_admin_token, state, off_mode="open")
    try:
        out = r.confirm_roi(job_id, zones, frame_zones={k: [z.model_dump() for z in v] for k, v in body.frame_zones.items()},
                            skip_frames=body.skip_frames, whole_frame=body.whole_frame)
    except JobError as exc:
        raise HTTPException(422, str(exc)) from None
    if body.save_for_camera:
        cam = state.db.get_camera(job["camera_id"])
        if cam:
            cam["roi"] = zones
            state.db.upsert_camera(cam)
            state.sync_cameras()
        else:
            state.db.set_setting(f"roi:{job['camera_id']}", zones)
        state.db.log_event("info", "roi", f"ROI saved ({len(zones)} zone(s))", job["camera_id"])
    return {"job_id": out["id"], "state": out["state"], "stage": out["stage"]}


@router.post("/jobs/{job_id}/cancel", dependencies=OPERATOR)
def cancel_job(job_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    try:
        job = _runner(state).cancel(job_id)
    except KeyError:
        raise HTTPException(404, "Analysis not found") from None
    except JobError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"job_id": job["id"], "state": job["state"]}


FRAME_KINDS = {"original": "original_path", "annotated": "annotated_path"}


@router.get("/jobs/{job_id}/frames/{idx}/{kind}.jpg", dependencies=VIEWER)
def frame_image(job_id: str, idx: int, kind: str, state: AppState = Depends(get_state)) -> FileResponse:
    r = _runner(state)
    job, f = r.repo.job(job_id), r.repo.frame(job_id, idx)
    if not job or not f:
        raise HTTPException(404, "Not found")
    root = Path(job["output_dir"]).resolve()
    if kind == "proposal":
        p = root / "roi" / f"proposal_s{idx}.jpg"
    elif kind in FRAME_KINDS and f.get(FRAME_KINDS[kind]):
        p = Path(f[FRAME_KINDS[kind]])
    else:
        raise HTTPException(404, "Not found")
    p = p.resolve()
    if root not in p.parents or not p.is_file():          # never serve anything outside the job folder
        raise HTTPException(404, "Image not available")
    return FileResponse(p, media_type="image/jpeg")


@router.get("/jobs/{job_id}/report.csv", dependencies=ADMIN_IF_LOGIN)
def job_csv(job_id: str, state: AppState = Depends(get_state)) -> StreamingResponse:
    job = _runner(state).repo.job(job_id)
    if not job or not job.get("csv_path") or not Path(job["csv_path"]).is_file():
        raise HTTPException(404, "No CSV report for this analysis yet")
    # served through the formula filter: reports written before the protection existed are made safe on the way
    # out (the stored file is never rewritten)
    text = report.sanitize_csv_text(Path(job["csv_path"]).read_text(encoding="utf-8", newline=""))
    return StreamingResponse(iter([text]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{Path(job["csv_path"]).name}"'})


# =========================================================================== results dashboard
@router.get("/reports/overview", dependencies=VIEWER)
def overview(request: Request, state: AppState = Depends(get_state)) -> dict[str, Any]:
    return _reports(state).overview(message_filter(request).error)


@router.get("/reports/months/{month}.csv", dependencies=ADMIN_IF_LOGIN)
def month_csv(month: str, by: Literal["completion", "recording"] = "completion",
              state: AppState = Depends(get_state)) -> StreamingResponse:
    if not MONTH_RE.match(month):
        raise HTTPException(422, "month must look like 2026-11")
    rows = _reports(state).month_rows(month, by, state.cfg.storage_dir)
    suffix = "_by_recording_date" if by == "recording" else ""
    return _csv(rows, f"litter_report_{month}{suffix}.csv")


@router.get("/reports/months/{month}", dependencies=VIEWER)
def month(month: str, request: Request, by: Literal["completion", "recording"] = "completion",
          state: AppState = Depends(get_state)) -> dict[str, Any]:
    if not MONTH_RE.match(month):
        raise HTTPException(422, "month must look like 2026-11")
    out = _reports(state).month(month, by, message_filter(request).error)
    if out is None:
        raise HTTPException(404, "No analyses in this month")
    return out


@router.get("/reports/history", dependencies=VIEWER)
def history(request: Request, store: str = "", camera: str = "", start: str = "", end: str = "",
            status: str = Query("", alias="state"), q: str = Query("", max_length=80),
            limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
            state: AppState = Depends(get_state)) -> dict[str, Any]:
    return _reports(state).history(store=store, camera=camera, start=start, end=end, state=status, q=q,
                                   limit=limit, offset=offset, error=message_filter(request).error)


@router.get("/reports/detections", dependencies=VIEWER)
def detections(store: str = "", camera: str = "", cls: str = Query("", alias="class"), start: str = "", end: str = "",
               limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
               state: AppState = Depends(get_state)) -> dict[str, Any]:
    return _reports(state).detections(store=store, camera=camera, cls=cls, start=start, end=end, limit=limit, offset=offset)


@router.get("/reports/hourly", dependencies=VIEWER)
def hourly(store: str = "", camera: str = "", date: str = "", state: AppState = Depends(get_state)) -> dict[str, Any]:
    return _reports(state).hourly(store=store, camera=camera, date=date)


@router.get("/reports/all.csv", dependencies=ADMIN_IF_LOGIN)
def all_csv(state: AppState = Depends(get_state)) -> StreamingResponse:
    return _csv(_reports(state).all_rows(state.cfg.storage_dir), f"litter_report_all_{time.strftime('%Y%m%d_%H%M%S')}.csv")
