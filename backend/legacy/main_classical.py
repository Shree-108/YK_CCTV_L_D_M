"""
main.py — Sightline Vision API (Litter Detection)

Classical-ML edition: NO YOLO / NO torch / NO weight downloads.
Detection = OpenCV region proposals + handcrafted features +
scikit-learn RandomForest (see litter_ml.py), trained by
trainlittermaxaccuracy.py on the YOLO-format dataset.

API schema is unchanged so the Next.js frontend keeps working:
  GET  /health  /model  /classes  /cameras  /alerts  /detections  /stats
  POST /analyze  /alerts/{id}/acknowledge  /reload-model
"""
from __future__ import annotations

import base64
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import db
import litter_ml
import video_pipeline
from litter_ml import load_config

_detector: litter_ml.LitterDetector | None = None


def load_model(force: bool = False):
    """Return the cached ML detector (kept the old name for compatibility)."""
    global _detector
    if _detector is None or force:
        _detector = litter_ml.load_detector(force=force)
    return _detector


def is_litter_model() -> bool:
    det = load_model()
    return bool(det is not None and det.trained and len(det.class_names) > 0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    load_model()
    yield


app = FastAPI(
    title="Sightline Vision API — Litter Detection (ML)",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------------
# Inference helpers
# ----------------------------------------------------------------------------------
def _run_inference_image(img_bgr: np.ndarray) -> list[dict]:
    det = load_model()
    if img_bgr is None or img_bgr.size == 0:
        return []
    return det.detect(img_bgr)


def _run_inference_video(path: str, max_frames: int = 8) -> list[dict]:
    """Sample frames evenly, detect per frame, pool + NMS."""
    det = load_model()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idxs = np.linspace(0, max(0, total - 1), num=min(max_frames, max(1, total)),
                           dtype=int).tolist() if total > 0 else list(range(max_frames))
        pooled: list[dict] = []
        frame_no = 0
        want = set(idxs)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if total <= 0 and frame_no >= max_frames:
                break
            if total > 0 and frame_no not in want:
                frame_no += 1
                continue
            for d in det.detect(frame):
                pooled.append(d)
            frame_no += 1
            if frame_no > max(total, max_frames) + 5:
                break
        if not pooled:
            return []
        boxes = np.array([d["box"] for d in pooled], dtype=np.float32)
        scores = np.array([d["confidence"] for d in pooled], dtype=np.float32)
        keep = litter_ml.nms(boxes, scores, float(det.cfg.get("iou_threshold", 0.5)))
        return [pooled[i] for i in keep[: int(det.cfg.get("max_detections", 100))]]
    finally:
        cap.release()


def _annotate_image(img_bgr: np.ndarray, detections: list[dict]) -> str:
    """Draw bounding boxes on image and return as base64 JPEG."""
    cfg = load_config()
    colors = cfg.get("priority_colors", {
        "critical": "#f87171",
        "high": "#fb923c",
        "medium": "#fbbf24",
        "low": "#34d399",
    })

    def hex_to_bgr(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return (b, g, r)

    annotated = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["box"])
        color = hex_to_bgr(colors.get(det["priority"], "#67e8f9"))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class'].replace('_', ' ')} {det['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode annotated image")
    return base64.b64encode(buf.tobytes()).decode()


# ----------------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------------
@app.get("/health")
def health():
    det = load_model()
    return {
        "status": "ok",
        "model_loaded": det is not None and det.ready,
        "model_path": det.model_id if det else "none",
        "litter_model": is_litter_model(),
        "engine": "ml-classical-rf",
        "trained": bool(det and det.trained),
        "classes": det.class_names if det else [],
        "class_count": len(det.class_names) if det else 0,
    }


@app.get("/model")
def model_metadata():
    det = load_model()
    return {
        "available": det is not None and det.ready,
        "path": det.model_id if det else "none",
        "litter_model": is_litter_model(),
        "engine": "ml-classical-rf",
        "trained": bool(det and det.trained),
        "classes": det.class_names if det else [],
        "config": det.cfg if det else load_config(),
    }


@app.post("/reload-model")
def reload_model():
    det = load_model(force=True)
    return {
        "reloaded": True,
        "model_path": det.model_id,
        "trained": det.trained,
        "class_count": len(det.class_names),
    }


@app.get("/classes")
def classes():
    det = load_model()
    cfg = det.cfg
    result = []
    for name in cfg.get("class_names", det.class_names):
        result.append({
            "name": name,
            "label": name.replace("_", " ").title(),
            "priority": cfg.get("priority_map", {}).get(name, "medium"),
            "bin": cfg.get("bin_map", {}).get(name, "general"),
            "color": cfg.get("priority_colors", {}).get(
                cfg.get("priority_map", {}).get(name, "medium"), "#67e8f9"),
        })
    return {"classes": result}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    camera_id: str = Query(default="upload"),
    mode: str = Query(default="fast", description="fast or accurate"),
):
    allowed = {
        "image/jpeg", "image/png", "image/webp", "image/gif",
        "video/mp4", "video/quicktime",
    }
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}",
        )

    det = load_model()
    suffix = Path(file.filename or "media").suffix or ".jpg"
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        if mode not in ("fast", "accurate"):
            raise HTTPException(status_code=400, detail="mode must be fast or accurate")
        is_video = bool(file.content_type and file.content_type.startswith("video/"))
        is_gif = file.content_type == "image/gif"
        extra: dict = {}

        if is_video or (is_gif and len(raw) > 200_000):
            # maximum-accuracy video path: tiles/TTA/quality/temporal/flow/WBF
            vres = video_pipeline.process_video(det, tmp_path, mode=mode)
            detections = vres["detections"]
            annotated_b64 = None
            extra = {"tracks": vres["tracks"],
                     "unconfirmed_tracks": vres["unconfirmed_tracks"],
                     "quality_log": vres["quality_log"],
                     "timing": vres["timing"],
                     "frames_sampled": vres["frames_sampled"],
                     "video_mode": vres["mode"]}
        else:
            img_arr = np.frombuffer(raw, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise HTTPException(status_code=400, detail="Could not decode image")
            if mode == "accurate":
                detections, info = video_pipeline.process_image(det, img_bgr, "accurate")
                extra = {"pipeline_info": info}
            else:
                detections = _run_inference_image(img_bgr)
            annotated_b64 = _annotate_image(img_bgr, detections)

        db.insert_detections(camera_id, file.filename, detections,
                             method=det.engine_name)

        priority_counts: dict[str, int] = {}
        for d in detections:
            priority_counts[d["priority"]] = priority_counts.get(d["priority"], 0) + 1

        return {
            "filename": file.filename,
            "camera_id": camera_id,
            "detections": detections,
            "count": len(detections),
            "priority_summary": priority_counts,
            "annotated_image": annotated_b64,
            "model_path": det.model_id,
            "litter_model": is_litter_model(),
            "engine": det.engine_name,
            "mode": mode,
            **extra,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/cameras")
def cameras():
    return {
        "cameras": [
            {"id": "north-gate", "name": "North gate", "status": "online", "location": "Main entrance"},
            {"id": "loading-dock", "name": "Loading dock", "status": "online", "location": "East wing"},
            {"id": "parking-b", "name": "Parking lot B", "status": "online", "location": "South lot"},
            {"id": "warehouse-a4", "name": "Warehouse aisle 4", "status": "online", "location": "Warehouse"},
        ]
    }


@app.get("/alerts")
def get_alerts(
    limit: int = Query(default=50, le=200),
    include_acknowledged: bool = Query(default=False),
):
    alerts = db.get_alerts(limit=limit, include_acknowledged=include_acknowledged)
    return {"alerts": alerts, "count": len(alerts)}


@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int):
    success = db.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"acknowledged": True, "alert_id": alert_id}


@app.get("/detections")
def get_detections(
    limit: int = Query(default=50, le=500),
    camera_id: str | None = Query(default=None),
):
    rows = db.get_recent_detections(limit=limit, camera_id=camera_id)
    return {"detections": rows, "count": len(rows)}


@app.get("/stats")
def stats(hours: int = Query(default=24, le=720)):
    return db.get_stats(hours=hours)
