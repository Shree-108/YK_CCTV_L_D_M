"""Operations/lab endpoints: datasets, experiments, evaluation, test lab, hard examples,
alert timeline, exports, system health, search, live event stream (SSE)."""
from __future__ import annotations

import asyncio
import csv
import io
import json
import shutil
import time
from pathlib import Path
from typing import Any, Literal, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.api.deps import IMAGE_EXTS, get_state, require_admin, save_upload
from backend.database.db import utcnow
from backend.services.datasets import DatasetError
from backend.services.evaluation import match_image
from backend.services.state import AppState

router = APIRouter(prefix="/api")
ADMIN = [Depends(require_admin)]


def _ds_call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except DatasetError as exc:
        raise HTTPException(404 if "not found" in str(exc) else 400, str(exc)) from exc


# =========================================================================== auth
@router.get("/auth/check", dependencies=ADMIN)
def auth_check() -> dict[str, Any]:
    return {"admin": True}


# =========================================================================== system health
@router.get("/system/health")
def system_health(state: AppState = Depends(get_state)) -> dict[str, Any]:
    import psutil

    now = utcnow()
    comps: list[dict[str, Any]] = []

    def comp(name: str, status: str, detail: str, **extra: Any) -> None:
        comps.append({"component": name, "status": status, "detail": detail, "checked_at": now, **extra})

    comp("backend", "operational", f"uptime {round(time.time() - state.started)} s")
    try:
        t0 = time.perf_counter()
        state.db.get_setting("__ping__")
        comp("database", "operational", "SQLite (WAL) reachable", latency_ms=round((time.perf_counter() - t0) * 1000, 1),
             size_mb=round(state.db.db_size_bytes() / 2**20, 2))
    except Exception as exc:  # noqa: BLE001
        comp("database", "offline", str(exc))
    eng = state.engine
    comp("inference", "operational" if eng else "offline",
         f"{eng.version} on {eng.device}" if eng else (state.engine_error or "no model"))
    workers = state.cameras.workers
    cams = state.db.list_cameras()
    enabled = [c for c in cams if c["enabled"]]
    online = [w for w in workers.values() if w.status.state == "online"]
    cam_status = ("operational" if enabled and len(online) == len(enabled) else
                  "degraded" if online else "offline" if enabled else "not_configured")
    comp("camera_services", cam_status, f"{len(online)}/{len(enabled)} enabled cameras online",
         cameras=[{"id": c["id"], "state": workers[c["id"]].status.state if c["id"] in workers else
                   ("disabled" if not c["enabled"] else "stopped"),
                   "last_error": workers[c["id"]].status.last_error if c["id"] in workers else None} for c in cams])
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    comp("cpu", "degraded" if cpu > 90 else "operational", f"{cpu:.0f}% used, {psutil.cpu_count()} logical cores",
         percent=cpu)
    comp("memory", "degraded" if mem.percent > 90 else "operational",
         f"{mem.percent:.0f}% of {mem.total / 2**30:.1f} GB used", percent=mem.percent)
    du = shutil.disk_usage(state.cfg.storage_dir)
    pct = du.used / du.total * 100
    comp("storage", "degraded" if pct > 90 else "operational",
         f"{du.free / 2**30:.1f} GB free of {du.total / 2**30:.0f} GB", percent=round(pct, 1))
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated(0) / 2**20
            total = torch.cuda.get_device_properties(0).total_memory / 2**20
            comp("gpu", "operational", f"{torch.cuda.get_device_name(0)}: {used:.0f}/{total:.0f} MB allocated")
        else:
            comp("gpu", "not_available", "No CUDA GPU detected; inference runs on CPU")
    except ImportError:
        comp("gpu", "not_available", "torch not installed")
    configured = [n for n in state.notifier.status() if n["configured"]]
    comp("notifications", "operational" if configured else "not_configured",
         ", ".join(n["provider"] for n in configured) or "no provider credentials in environment")
    overall = ("offline" if any(c["status"] == "offline" and c["component"] in ("backend", "database", "inference")
                                for c in comps)
               else "degraded" if any(c["status"] in ("degraded", "offline") for c in comps) else "operational")
    return {"overall": overall, "checked_at": now, "components": comps}


# =========================================================================== live event stream (SSE)
@router.get("/stream")
async def stream(request: Request, state: AppState = Depends(get_state)) -> StreamingResponse:
    """Server-sent events: `summary` every 2 s, `alert` for each new alert, `event` for system events."""
    async def gen():
        last_alert = (state.db.list_alerts(limit=1)[0] or [{"id": 0}])[0]["id"]
        last_event = (state.db.list_events(limit=1) or [{"id": 0}])[0]["id"]
        while not await request.is_disconnected():
            rows, _ = state.db.list_alerts(limit=20)
            for a in reversed([r for r in rows if r["id"] > last_alert]):
                last_alert = a["id"]
                yield f"event: alert\ndata: {json.dumps(_alert_brief(a))}\n\n"
            for e in reversed([e for e in state.db.list_events(limit=20) if e["id"] > last_event]):
                last_event = e["id"]
                yield f"event: event\ndata: {json.dumps(e)}\n\n"
            cams = {cid: {"state": w.status.state, "fps": w.status.fps, "inference_ms": w.status.inference_ms,
                          "summary": w.status.summary} for cid, w in state.cameras.workers.items()}
            yield f"event: summary\ndata: {json.dumps({'ts': utcnow(), 'cameras': cams})}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _alert_brief(a: dict[str, Any]) -> dict[str, Any]:
    return {k: a.get(k) for k in ("id", "ts", "camera_id", "class", "confidence", "severity", "status", "track_id")}


# =========================================================================== search
@router.get("/search")
def search(q: str = Query(..., min_length=1, max_length=80), state: AppState = Depends(get_state)) -> dict[str, Any]:
    ql = q.lower().strip()
    cams = [{"id": c["id"], "name": c["name"], "location": c["location"]} for c in state.db.list_cameras()
            if ql in f"{c['id']} {c['name']} {c['location']}".lower()][:8]
    alerts: list[dict[str, Any]] = []
    if ql.lstrip("#").isdigit():
        a = state.db.get_alert(int(ql.lstrip("#")))
        alerts = [_alert_brief(a)] if a else []
    else:
        rows, _ = state.db.list_alerts(limit=200)
        alerts = [_alert_brief(a) for a in rows if ql in f"{a['class']} {a['camera_id']} {a['severity']}".lower()][:8]
    models = [{"version": m["version"], "name": m["name"], "active": m["active"]} for m in state.db.list_models()
              if ql in f"{m['version']} {m['name']}".lower()][:8]
    exps = [{"id": e["id"], "name": e.get("name")} for e in state.experiments.list()
            if ql in f"{e['id']} {e.get('model', '')} {e.get('dataset', '')}".lower()][:8]
    dss = [{"name": d["name"]} for d in state.datasets.list() if ql in d["name"].lower()][:8]
    return {"query": q, "cameras": cams, "alerts": alerts, "models": models, "experiments": exps, "datasets": dss}


# =========================================================================== alerts extras
@router.get("/alerts/{alert_id}/timeline")
def alert_timeline(alert_id: int, state: AppState = Depends(get_state)) -> dict[str, Any]:
    a = state.db.get_alert(alert_id)
    if not a:
        raise HTTPException(404, "Alert not found")
    rows = (state.db.track_timeline(a["camera_id"], a["session_id"], a["track_id"])
            if a.get("session_id") and a.get("track_id") is not None else [])
    return {"alert_id": alert_id, "observations": rows,
            "note": None if rows else "No per-frame observations stored for this alert (image upload or "
                                      "throttled live logging)."}


def _csv_response(rows: list[dict[str, Any]], name: str) -> StreamingResponse:
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/export/{table}.csv")
def export_csv(table: Literal["alerts", "detections"], start: Optional[str] = None, end: Optional[str] = None,
               camera_id: Optional[str] = None, state: AppState = Depends(get_state)) -> StreamingResponse:
    rows = state.db.export_rows(table, start, end, camera_id)
    for r in rows:
        r.pop("snapshot_path", None)
        r.pop("crop_path", None)
    return _csv_response(rows, f"{table}_{time.strftime('%Y%m%d_%H%M%S')}.csv")


# =========================================================================== cameras extras
@router.get("/cameras/{cam_id}/live")
def camera_live(cam_id: str, since: int = Query(0, ge=0), state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Latest processed frame: raw JPEG (base64) + the detections computed on exactly that frame.
    Pass ``since=<seq>`` to receive only metadata when no newer frame exists."""
    import base64
    w = state.cameras.workers.get(cam_id)
    if not w:
        raise HTTPException(404, "Camera is not running")
    meta, raw = w.latest_live()
    cam = state.db.get_camera(cam_id) or {}
    fresh = meta is not None and raw is not None and meta["seq"] > since
    return {"camera_id": cam_id, "state": w.status.state, "frame": meta if fresh else None,
            "image": base64.b64encode(raw).decode() if fresh else None, "roi": cam.get("roi", []),
            "fps": w.status.fps, "inference_ms": w.status.inference_ms, "last_error": w.status.last_error}


@router.post("/cameras/{cam_id}/restart", dependencies=ADMIN)
def restart_camera(cam_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    cam = state.db.get_camera(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    w = state.cameras.workers.pop(cam_id, None)
    if w:
        w.stop()
    state.db.log_event("info", "camera", "stream restart requested", cam_id)
    state.sync_cameras()
    return {"restarted": cam_id, "running": cam_id in state.cameras.workers}


# =========================================================================== datasets
@router.get("/datasets")
def datasets(state: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"datasets": state.datasets.list()}


@router.get("/datasets/{name}")
def dataset_stats(name: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    return _ds_call(state.datasets.stats, name)


@router.get("/datasets/{name}/images")
def dataset_images(name: str, split: Optional[Literal["train", "val", "test"]] = None,
                   cls: Optional[str] = Query(None, alias="class"),
                   size: Optional[Literal["small", "medium", "large"]] = None,
                   status: Optional[Literal["annotated", "empty", "negative", "issues"]] = None,
                   offset: int = Query(0, ge=0), limit: int = Query(60, ge=1, le=500),
                   state: AppState = Depends(get_state)) -> dict[str, Any]:
    return _ds_call(state.datasets.list_images, name, split, cls, size, status, offset, limit)


@router.get("/datasets/{name}/image")
def dataset_image(name: str, id: str, state: AppState = Depends(get_state)) -> FileResponse:
    ref = _ds_call(state.datasets.resolve_image, name, id)
    return FileResponse(ref.path, headers={"Cache-Control": "max-age=300"})


@router.get("/datasets/{name}/annotation")
def dataset_annotation(name: str, id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    d = _ds_call(state.datasets.image_detail, name, id)
    d["reviews"] = state.db.list_reviews(name, id)
    return d


class LabelBox(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cls: int = Field(ge=0)
    cx: float = Field(ge=0, le=1)
    cy: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)


class LabelsIn(BaseModel):
    id: str
    boxes: list[LabelBox]


@router.put("/datasets/{name}/annotation", dependencies=ADMIN)
def save_annotation(name: str, body: LabelsIn, state: AppState = Depends(get_state)) -> dict[str, Any]:
    res = _ds_call(state.datasets.write_labels, name, body.id, [b.model_dump() for b in body.boxes])
    state.db.log_event("info", "dataset", f"labels updated: {name}/{body.id} ({res['objects']} objects)")
    return res


class NewVersion(BaseModel):
    name: str = Field(min_length=1, max_length=64)


@router.post("/datasets/{name}/versions", dependencies=ADMIN, status_code=201)
def dataset_version(name: str, body: NewVersion, state: AppState = Depends(get_state)) -> dict[str, Any]:
    res = _ds_call(state.datasets.create_version, name, body.name)
    state.db.log_event("info", "dataset", f"dataset version {body.name} created from {name}")
    return res


@router.get("/datasets/{name}/export", dependencies=ADMIN)
def dataset_export(name: str, state: AppState = Depends(get_state)) -> FileResponse:
    p = _ds_call(state.datasets.export_zip, name, state.cfg.storage_dir / "exports")
    return FileResponse(p, media_type="application/zip", filename=p.name)


# =========================================================================== experiments
@router.get("/experiments")
def experiments(state: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"experiments": state.experiments.list()}


@router.get("/experiments/{exp_id}")
def experiment(exp_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    try:
        return state.experiments.detail(exp_id)
    except KeyError as exc:
        raise HTTPException(404, "Experiment not found") from exc


class RegisterIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_id: str
    version: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")
    name: Optional[str] = Field(None, max_length=120)
    notes: str = Field("", max_length=1000)


@router.post("/models/register", dependencies=ADMIN, status_code=201)
def register_model(body: RegisterIn, state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Copy an experiment's weights into a new immutable model version (not activated)."""
    try:
        run = state.experiments.resolve(body.experiment_id)
    except KeyError as exc:
        raise HTTPException(404, "Experiment not found") from exc
    data_yaml = None
    try:
        import yaml
        args = yaml.safe_load((run / "args.yaml").read_text()) if (run / "args.yaml").is_file() else {}
        cand = Path(str(args.get("data", "")))
        data_yaml = cand if cand.is_file() else None
    except Exception:  # noqa: BLE001 - dataset yaml is optional metadata
        pass
    try:
        meta = state.registry.register_run(run, body.version, body.name or body.version, body.notes, data_yaml,
                                           {"experiment_id": body.experiment_id})
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    state.db.log_event("info", "models", f"registered model {body.version} from {body.experiment_id}")
    return meta


@router.get("/experiments/{exp_id}/artifacts/{name}")
def experiment_artifact(exp_id: str, name: str, state: AppState = Depends(get_state)) -> FileResponse:
    try:
        return FileResponse(state.experiments.artifact(exp_id, name))
    except KeyError as exc:
        raise HTTPException(404, "Artifact not found") from exc


# =========================================================================== evaluation
class EvalIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_version: Optional[str] = None
    dataset: str
    split: Literal["train", "val", "test"] = "test"
    compare_tiling: bool = False
    operating_threshold: Optional[float] = Field(None, gt=0, lt=1)


@router.post("/evaluations", status_code=202)
def start_evaluation(body: EvalIn, state: AppState = Depends(get_state)) -> dict[str, Any]:
    _ds_call(state.datasets.path, body.dataset)
    if body.model_version and body.model_version not in {m["version"] for m in state.db.list_models()}:
        raise HTTPException(404, f"model {body.model_version} not found")
    job = state.submit_evaluation(body.model_version, body.dataset, body.split, body.compare_tiling,
                                  body.operating_threshold)
    return {"job_id": job, "status_url": f"/api/jobs/{job}"}


@router.get("/evaluations")
def evaluations(state: AppState = Depends(get_state)) -> dict[str, Any]:
    out = []
    for j in state.db.list_jobs(500):
        if j["kind"] == "evaluation":
            full = state.db.get_job(j["id"]) or {}
            out.append({**j, "summary": full.get("result")})
    return {"evaluations": out}


@router.get("/evaluations/{job_id}")
def evaluation(job_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    p = state.cfg.storage_dir / "evaluations" / f"{job_id}.json"
    if not job_id.replace("_", "").isalnum() or not p.is_file():
        raise HTTPException(404, "Evaluation not found (still running, failed, or unknown id)")
    return json.loads(p.read_text(encoding="utf-8"))


# =========================================================================== test lab
class TestLabIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    image: str
    versions: list[str] = Field(min_length=1, max_length=4)
    confidence: float = Field(0.25, gt=0, lt=1)
    iou: float = Field(0.5, gt=0, lt=1)
    image_size: Optional[int] = Field(None, ge=160, le=4096)
    tiling: Literal["auto", "on", "off"] = "off"


def _run_models(state: AppState, img: np.ndarray, versions: list[str], conf: float, image_size: Optional[int],
                tiling: str, gt: Optional[list[dict[str, Any]]], iou: float) -> list[dict[str, Any]]:
    patch: dict[str, Any] = {"confidence": {"weak_threshold": min(conf, 0.3), "candidate_threshold": min(conf, 0.45),
                                            "confirmed_threshold": conf}}
    if image_size:
        patch["inference"] = {"image_size": image_size}
    s = state.settings_with(None, patch)
    out = []
    for v in versions:
        try:
            eng = state.get_engine(v)
        except KeyError:
            out.append({"model_version": v, "error": "model not found"})
            continue
        t0 = time.perf_counter()
        dets, info = eng.detect(img, s, force_tiling={"on": True, "off": False}.get(tiling))
        ms = (time.perf_counter() - t0) * 1000
        preds = [{"cls_id": d.cls_id, "class": d.cls_name, "confidence": round(d.confidence, 4),
                  "bbox": [round(x, 1) for x in d.bbox]} for d in dets if d.confidence >= conf]
        res: dict[str, Any] = {"model_version": v, "device": eng.device, "latency_ms": round(ms, 1),
                               "fps": round(1000 / ms, 2) if ms else None, "tiled": info.tiled, "tiles": info.tiles,
                               "detections": preds,
                               "confidence_histogram": np.histogram([p["confidence"] for p in preds],
                                                                    bins=np.linspace(0, 1, 11))[0].tolist()}
        if gt is not None:
            same = [g for g in gt if g["class"] in eng.class_names]
            gt_m = [{**g, "cls_id": eng.class_names.index(g["class"])} for g in same]
            pm, gm = match_image(gt_m, preds, iou)
            for p, m in zip(preds, pm):
                p["match"] = "tp" if m >= 0 else "fp"
                p["gt_index"] = gt_m[m]["index"] if m >= 0 else None
            res["missed_gt"] = [gt_m[i]["index"] for i, m in enumerate(gm) if m < 0]
            res["counts"] = {"ground_truth": len(gt), "predicted": len(preds), "true_positive": sum(m >= 0 for m in pm),
                             "false_positive": sum(m < 0 for m in pm), "missed": sum(m < 0 for m in gm)}
        out.append(res)
    return out


@router.post("/testlab/run")
async def testlab_run(body: TestLabIn, state: AppState = Depends(get_state)) -> dict[str, Any]:
    detail = _ds_call(state.datasets.image_detail, body.dataset, body.image)
    ref = _ds_call(state.datasets.resolve_image, body.dataset, body.image)
    img = cv2.imread(str(ref.path))
    results = await asyncio.to_thread(_run_models, state, img, body.versions, body.confidence, body.image_size,
                                      body.tiling, detail["ground_truth"], body.iou)
    return {"image": {k: detail[k] for k in ("id", "width", "height", "split", "negative")},
            "ground_truth": detail["ground_truth"], "results": results}


@router.post("/testlab/upload")
async def testlab_upload(file: UploadFile = File(...), versions: str = Form(...),
                         confidence: float = Form(0.25, gt=0, lt=1), image_size: Optional[int] = Form(None),
                         tiling: Literal["auto", "on", "off"] = Form("off"),
                         state: AppState = Depends(get_state)) -> dict[str, Any]:
    vs = [v for v in versions.split(",") if v][:4]
    if not vs:
        raise HTTPException(422, "select at least one model")
    path = await save_upload(file, IMAGE_EXTS, state.cfg.max_image_mb, state.cfg.uploads_dir)
    try:
        img = cv2.imread(str(path))
        if img is None:
            raise HTTPException(400, "Could not decode image")
        results = await asyncio.to_thread(_run_models, state, img, vs, confidence, image_size, tiling, None, 0.5)
    finally:
        path.unlink(missing_ok=True)
    return {"image": {"width": img.shape[1], "height": img.shape[0]}, "ground_truth": None, "results": results}


class ReviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    image: str
    model_version: str
    verdict: Literal["true_positive", "false_positive", "false_negative", "uncertain"]
    cls: Optional[str] = None
    confidence: Optional[float] = None
    bbox: list[float] = Field(min_length=4, max_length=4)
    notes: Optional[str] = Field(None, max_length=500)


@router.post("/testlab/reviews", status_code=201)
def add_review(body: ReviewIn, state: AppState = Depends(get_state)) -> dict[str, Any]:
    _ds_call(state.datasets.resolve_image, body.dataset, body.image)
    x1, y1, x2, y2 = body.bbox
    rid = state.db.insert_review({"created_at": utcnow(), "dataset": body.dataset, "split": None, "image": body.image,
                                  "model_version": body.model_version, "verdict": body.verdict, "class": body.cls,
                                  "confidence": body.confidence, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                  "notes": body.notes})
    if body.verdict in ("false_positive", "false_negative"):
        state.db.insert_hard_examples([{"created_at": utcnow(), "kind": body.verdict, "source": "manual",
                                        "dataset": body.dataset, "image": body.image, "class": body.cls,
                                        "confidence": body.confidence, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                        "model_version": body.model_version}])
    return {"id": rid}


# =========================================================================== hard examples
@router.get("/hard-examples")
def hard_examples(kind: Optional[str] = None, status: Optional[str] = None, offset: int = Query(0, ge=0),
                  limit: int = Query(48, ge=1, le=200), state: AppState = Depends(get_state)) -> dict[str, Any]:
    rows, total = state.db.list_hard_examples(kind, status, limit, offset)
    return {"items": rows, "total": total, "counts": state.db.hard_example_counts()}


class HardUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_status: Literal["pending", "approved", "rejected", "ignored"]
    notes: Optional[str] = Field(None, max_length=500)


@router.patch("/hard-examples/{hid}")
def update_hard_example(hid: int, body: HardUpdate, state: AppState = Depends(get_state)) -> dict[str, Any]:
    if not state.db.update_hard_example(hid, **body.model_dump(exclude_none=True)):
        raise HTTPException(404, "Hard example not found")
    return state.db.get_hard_example(hid)  # type: ignore[return-value]
