"""
Mainv0scaffold.py — minimal reference API using the classical ML detector.

Same engine as main.py (litter_ml.py: OpenCV proposals + RandomForest),
but with no database, no priorities, no auth — just health + analyze.
Kept for smoke-testing the detector in isolation. Do not run alongside
main.py on the same port.

  uvicorn Mainv0scaffold:app --port 8001
"""
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import cv2
import numpy as np

import litter_ml

app = FastAPI(title="Sightline Vision API (ML scaffold)", version="2.0.0")


@app.get("/health")
def health():
    det = litter_ml.load_detector()
    return {
        "status": "ok",
        "engine": "ml-classical-rf",
        "model_loaded": det.ready,
        "trained": det.trained,
        "model_path": det.model_id,
        "classes": det.class_names,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/png", "image/webp", "video/mp4", "video/quicktime"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=415, detail="Unsupported media type")

    det = litter_ml.load_detector()
    suffix = Path(file.filename or "media").suffix or ".jpg"
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(raw)
        temp_path = temp.name
    try:
        if file.content_type.startswith("video/"):
            cap = cv2.VideoCapture(temp_path)
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                return JSONResponse(status_code=400,
                                    content={"detail": "Could not read video frame"})
            dets = det.detect(frame)
        else:
            img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return JSONResponse(status_code=400,
                                    content={"detail": "Could not decode image"})
            dets = det.detect(img)
        return {
            "filename": file.filename,
            "engine": "ml-classical-rf",
            "detections": dets,
            "count": len(dets),
        }
    finally:
        Path(temp_path).unlink(missing_ok=True)
