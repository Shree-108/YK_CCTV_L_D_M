"""Results view API: one row per camera check with the importer CSV fields (backend/services/results_view.py).
Permissions (backend/api/authz.py): reads = VIEWER; the CSV download = ADMIN_IF_LOGIN, like every CSV export."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response

from backend.api.authz import ADMIN_IF_LOGIN, VIEWER
from backend.api.deps import get_state
from backend.services import garbage_export as ge
from backend.services import results_view as rv
from backend.services.state import AppState

router = APIRouter(prefix="/api/results")
JOB_RE = re.compile(r"^[A-Za-z0-9_\-]{1,60}$")
DAY_RE = r"^\d{4}-\d{2}-\d{2}$"


def _filters(store: Optional[str], camera: Optional[str], status: Optional[str], litter_class: Optional[str],
             min_confidence: Optional[float], date_from: Optional[str], date_to: Optional[str],
             job_id: Optional[str]) -> rv.Filters:
    return rv.Filters(store or None, camera or None, status or None, litter_class or None, min_confidence,
                      date_from or None, date_to or None, job_id or None)


def _classes(state: AppState) -> Optional[set[str]]:
    return set(getattr(state.engine, "class_names", []) or []) or None


@router.get("/checks", dependencies=VIEWER)
def checks(store: Optional[str] = Query(None, max_length=64), camera: Optional[str] = Query(None, max_length=40),
           status: Optional[Literal["clean", "litter_detected", "view_blocked"]] = None,
           litter_class: Optional[str] = Query(None, max_length=60),
           min_confidence: Optional[float] = Query(None, ge=0, le=100),
           date_from: Optional[str] = Query(None, pattern=DAY_RE), date_to: Optional[str] = Query(None, pattern=DAY_RE),
           job_id: Optional[str] = Query(None, max_length=60),
           limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
           state: AppState = Depends(get_state)) -> dict[str, Any]:
    flt = _filters(store, camera, status, litter_class, min_confidence, date_from, date_to, job_id)
    return rv.view(state.db._query, ge.settings_from_config(state.cfg), flt, limit=limit, offset=offset,
                   allowed_classes=_classes(state))


@router.get("/checks.csv", dependencies=ADMIN_IF_LOGIN)
def checks_csv(store: Optional[str] = Query(None, max_length=64), camera: Optional[str] = Query(None, max_length=40),
               status: Optional[Literal["clean", "litter_detected", "view_blocked"]] = None,
               litter_class: Optional[str] = Query(None, max_length=60),
               min_confidence: Optional[float] = Query(None, ge=0, le=100),
               date_from: Optional[str] = Query(None, pattern=DAY_RE),
               date_to: Optional[str] = Query(None, pattern=DAY_RE),
               job_id: Optional[str] = Query(None, max_length=60),
               state: AppState = Depends(get_state)) -> Response:
    """Importer format (ai_garbage_detection, 9 columns, UTF-8 without BOM): only import-ready rows; how many were
    left out is in the X-Rows-Excluded header."""
    flt = _filters(store, camera, status, litter_class, min_confidence, date_from, date_to, job_id)
    text, n, left_out = rv.importer_csv(state.db._query, ge.settings_from_config(state.cfg), flt,
                                        allowed_classes=_classes(state))
    name = f"ai_garbage_detection_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(text.encode("utf-8"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"', "X-Rows-Written": str(n),
                             "X-Rows-Excluded": str(left_out),
                             "Access-Control-Expose-Headers": "X-Rows-Written, X-Rows-Excluded"})


@router.get("/checks/{job_id}/{idx}/image.jpg", dependencies=VIEWER)
def check_image(job_id: str, idx: int, state: AppState = Depends(get_state)) -> FileResponse:
    if not JOB_RE.match(job_id) or not 1 <= idx <= ge.BUNDLE_IMAGES:
        raise HTTPException(404, "Not found")
    p = rv.image_path(state.db._query, job_id, idx)
    roots = [Path(state.cfg.garbage_export_dir).resolve(), Path(state.cfg.sampled_dir).resolve()]
    if p is None or not any(r in p.resolve().parents for r in roots):
        raise HTTPException(404, "Image not found")
    return FileResponse(p, media_type="image/jpeg")
