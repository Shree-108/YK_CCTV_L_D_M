"""Keyboard review of mined detections (phase 2 item 6) - backend/services/labeling.py over storage/review/<video_id>.

Permissions: reading the queue and images = VIEWER; labelling (autosave), true class and drawn "missed" boxes = OPERATOR.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.api.authz import OPERATOR, VIEWER, actor_of
from backend.api.deps import get_state
from backend.services.labeling import LabelError, ReviewStore
from backend.services.state import AppState

router = APIRouter(prefix="/api/labeling")


class LabelIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["", "TP", "FP_shadow", "FP_carton", "FP_stock", "FP_puddle", "FP_other"]
    whole_group: bool = True
    true_class: Optional[str] = Field(None, max_length=60)


class MissedIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_file: str = Field(pattern=r"^frames/f\d{7}\.jpg$")
    bbox: list[float] = Field(min_length=4, max_length=4)
    cls: str = Field(alias="class", min_length=1, max_length=60)


def _store(state: AppState) -> ReviewStore:
    return ReviewStore(state.cfg.storage_dir / "review")


def _guard(fn: Any) -> Any:
    try:
        return fn()
    except FileNotFoundError:
        raise HTTPException(404, "Review folder not found") from None
    except KeyError:
        raise HTTPException(404, "Detection not found") from None
    except LabelError as exc:
        raise HTTPException(422, str(exc)) from None


@router.get("/videos", dependencies=VIEWER)
def videos(state: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"videos": _store(state).sessions()}


@router.get("/videos/{video_id}", dependencies=VIEWER)
def items(video_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    return _guard(lambda: _store(state).items(video_id))


@router.put("/videos/{video_id}/labels/{target}", dependencies=OPERATOR)
def set_label(video_id: str, target: str, body: LabelIn, state: AppState = Depends(get_state)) -> dict[str, Any]:
    st = _store(state)
    out = _guard(lambda: st.set_label(video_id, target, body.label, whole_group=body.whole_group))
    if body.true_class is not None:
        out["true_class"] = _guard(lambda: st.set_true_class(video_id, out["changed"], body.true_class))
    return out


@router.post("/videos/{video_id}/missed", status_code=201, dependencies=OPERATOR)
def add_missed(video_id: str, body: MissedIn, request: Request, state: AppState = Depends(get_state)) -> dict:
    return _guard(lambda: _store(state).add_missed(video_id, body.frame_file, body.bbox, body.cls,
                                                   actor_of(request) or ""))


@router.delete("/videos/{video_id}/missed/{missed_id}", dependencies=OPERATOR)
def delete_missed(video_id: str, missed_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    if not _guard(lambda: _store(state).delete_missed(video_id, missed_id)):
        raise HTTPException(404, "Missed box not found")
    return {"deleted": missed_id}


@router.get("/videos/{video_id}/image", dependencies=VIEWER)
def image(video_id: str, path: str = Query(..., max_length=200), state: AppState = Depends(get_state)) -> FileResponse:
    p = _guard(lambda: _store(state).image_path(video_id, path))
    if p is None:
        raise HTTPException(404, "Image not found")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
