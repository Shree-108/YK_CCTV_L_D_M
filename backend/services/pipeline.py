"""The per-stream detection pipeline and the image/video entry points.

FRAME -> YOLO (optional tiles) -> merge/NMS -> filters (conf tier, class, size, ROI)
      -> tracker (ByteTrack-style) -> temporal confirmation -> alert manager
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from backend.alerts.manager import AlertManager, ClassMetadata, ConfirmedEvent
from backend.core.settings import DetectionSettings
from backend.database.db import Database, utcnow
from backend.detection.engine import YoloEngine
from backend.detection.filters import ROISet, apply_filters, confirm_single_image
from backend.detection.types import CANDIDATE, CONFIRMED, REJECTED, WEAK, Detection, InferenceInfo
from backend.services.visualization import annotate, annotate_litter
from backend.tracking.tracker import LitterTracker

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    detections: list[Detection]
    summary: dict[str, int]
    new_alerts: list[dict[str, Any]]
    inference_ms: float
    tiled: bool
    events: list[dict[str, Any]] = field(default_factory=list)


def summarize(dets: list[Detection], tracker: LitterTracker, new_alerts: int) -> dict[str, int]:
    """Per-frame counters. confirmed_litter/active_tracks count objects visible now (<=2 missed frames),
    not tracks the tracker is merely remembering."""
    c = Counter(d.status for d in dets)
    visible = tracker.visible_tracks()
    return {"raw_detections": len(dets), "rejected": c[REJECTED], "weak": c[WEAK], "candidates": c[CANDIDATE],
            "confirmed_in_frame": c[CONFIRMED], "confirmed_litter": sum(1 for t in visible if t.confirmed),
            "active_tracks": len(visible), "tracks_in_memory": len(tracker.tracks), "new_alerts": new_alerts}


def roi_window(shape: tuple[int, ...], s: DetectionSettings, roi: Optional[ROISet]) -> Optional[tuple[int, int, int, int]]:
    """Pixel rectangle the model should look at, or None for the whole frame. Only used when the ROI
    covers clearly less than the frame; the ROI filter still decides what is inside the polygon."""
    f = s.filters
    if not (f.roi_enabled and f.roi_crop_inference and roi is not None):
        return None
    h, w = shape[:2]
    box = roi.crop_box(w, h, f.roi_crop_pad_frac)
    if box is None or (box[2] - box[0]) < 32 or (box[3] - box[1]) < 32:
        return None
    if (box[2] - box[0]) * (box[3] - box[1]) > 0.9 * w * h:
        return None
    return box


def detect_in_roi(engine: YoloEngine, frames: list[np.ndarray], s: DetectionSettings, roi: Optional[ROISet],
                  force_tiling: Optional[bool] = None) -> list[tuple[list[Detection], Any]]:
    """engine.detect_many() on the ROI crop only (fewer pixels, and small litter gets more model
    resolution); boxes are mapped back to full-frame coordinates."""
    if not frames:
        return []
    box = roi_window(frames[0].shape, s, roi)
    if box is None:
        return engine.detect_many(frames, s, force_tiling)
    x1, y1, x2, y2 = box
    out = engine.detect_many([np.ascontiguousarray(f[y1:y2, x1:x2]) for f in frames], s, force_tiling)
    for dets, _ in out:
        for d in dets:
            d.bbox = [d.bbox[0] + x1, d.bbox[1] + y1, d.bbox[2] + x1, d.bbox[3] + y1]
    return out


class StreamPipeline:
    """Stateful pipeline for one stream (live camera or one video job)."""

    def __init__(self, engine: YoloEngine, settings: DetectionSettings, roi: Optional[ROISet],
                 alerts: Optional[AlertManager], db: Optional[Database], camera_id: str,
                 source_type: str = "live", job_id: Optional[str] = None, location: str = "") -> None:
        self.engine, self.settings, self.roi = engine, settings, roi
        self.alerts, self.db = alerts, db
        self.camera_id, self.source_type, self.job_id, self.location = camera_id, source_type, job_id, location
        self.session_id = uuid.uuid4().hex[:12]
        self.tracker = LitterTracker(settings)
        self._first_seen: dict[int, str] = {}
        self._last_raw: Optional[list[Detection]] = None
        self._last_thumb: Optional[np.ndarray] = None
        self._last_infer_t: Optional[float] = None
        self.frames_reused = 0
        self._last_log = 0.0

    def update_settings(self, settings: DetectionSettings, roi: Optional[ROISet]) -> None:
        self.settings, self.roi = settings, roi
        self.tracker.update_settings(settings)

    def _raw_detect(self, frame: np.ndarray, s: DetectionSettings, video_time: Optional[float],
                    force_tiling: Optional[bool], allow_reuse: bool) -> tuple[list[Detection], Any, bool]:
        """Run the model, or - for a fixed camera whose view has not changed - reuse the previous raw
        detections. Reuse is bounded by ``max_reuse_sec`` so the model re-checks the scene regularly."""
        v = s.video
        thumb = cv2.cvtColor(cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.int16)
        if (allow_reuse and v.skip_static_frames and self._last_raw is not None and self._last_thumb is not None
                and video_time is not None and self._last_infer_t is not None
                and video_time - self._last_infer_t < v.max_reuse_sec
                and float(np.mean(np.abs(thumb - self._last_thumb))) < v.static_diff_threshold):
            dets = [Detection(list(d.bbox), d.cls_id, d.cls_name, d.confidence, source=d.source) for d in self._last_raw]
            return dets, InferenceInfo(), True
        dets, info = detect_in_roi(self.engine, [frame], s, self.roi, force_tiling)[0]
        self._last_raw = [Detection(list(d.bbox), d.cls_id, d.cls_name, d.confidence, source=d.source) for d in dets]
        self._last_thumb, self._last_infer_t = thumb, video_time
        return dets, info, False

    def plan_reuse(self, frame: np.ndarray, video_time: float) -> bool:
        """Decide (in frame order) whether this frame can reuse the previous model output. Frames that
        need the model become the new reference, so a batch can be planned before running inference."""
        v = self.settings.video
        thumb = cv2.cvtColor(cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.int16)
        if (v.skip_static_frames and self._last_thumb is not None and self._last_infer_t is not None
                and video_time - self._last_infer_t < v.max_reuse_sec
                and float(np.mean(np.abs(thumb - self._last_thumb))) < v.static_diff_threshold):
            return True
        self._last_thumb, self._last_infer_t = thumb, video_time
        return False

    def process(self, frame: np.ndarray, frame_idx: int, video_time: Optional[float] = None,
                annotate_fn: Optional[Callable[[np.ndarray, list[Detection]], np.ndarray]] = None,
                force_tiling: Optional[bool] = None, allow_reuse: bool = False,
                precomputed: Optional[tuple[list[Detection], Any]] = None, reuse_previous: bool = False) -> FrameResult:
        s = self.settings
        if precomputed is not None:
            dets, info = precomputed
            reused = False
            self._last_raw = [Detection(list(d.bbox), d.cls_id, d.cls_name, d.confidence, source=d.source) for d in dets]
        elif reuse_previous and self._last_raw is not None:
            dets = [Detection(list(d.bbox), d.cls_id, d.cls_name, d.confidence, source=d.source) for d in self._last_raw]
            info, reused = InferenceInfo(), True
        else:
            dets, info, reused = self._raw_detect(frame, s, video_time, force_tiling, allow_reuse)
        self.frames_reused += int(reused)
        apply_filters(dets, frame.shape, s, self.roi)
        track_events = self.tracker.update(dets, frame_idx, frame.shape)
        now = utcnow()
        for d in dets:
            if d.track_id is not None:
                self._first_seen.setdefault(d.track_id, now)

        new_alerts: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        annotated_cache: Optional[np.ndarray] = None
        for ev in track_events:
            t = ev.track
            events.append({"event": ev.kind, "track_id": t.track_id, "class": t.cls_name, "frame_idx": frame_idx,
                           "video_time_sec": video_time, "confidence": round(t.mean_hit_conf, 4)})
            if ev.kind == "confirmed" and self.alerts is not None:
                if annotated_cache is None:
                    annotated_cache = annotate_fn(frame, dets) if annotate_fn else annotate(frame, dets, self.roi)
                alert = self.alerts.on_confirmed(ConfirmedEvent(
                    camera_id=self.camera_id, session_id=self.session_id, track_id=t.track_id, cls_name=t.cls_name,
                    confidence=t.mean_hit_conf, bbox=[round(v, 1) for v in t.bbox], frame=frame,
                    annotated=annotated_cache, video_time_sec=video_time,
                    first_seen_ts=self._first_seen.get(t.track_id, now), hits=t.total_hits,
                    active_objects=sum(1 for x in self.tracker.visible_tracks() if x.confirmed), location=self.location, job_id=self.job_id,
                    source_type=self.source_type, model_version=self.engine.version,
                    live_track_ids=frozenset(x.track_id for x in self.tracker.active_tracks)), s)
                if alert:
                    new_alerts.append(alert)
                    events[-1]["alert_id"] = alert["id"]
            elif ev.kind == "lost" and self.alerts is not None:
                self.alerts.on_lost(self.session_id, t.track_id)

        self._log(dets, frame_idx, video_time)
        summary = summarize(dets, self.tracker, len(new_alerts))
        return FrameResult(dets, summary, new_alerts, info.inference_ms, info.tiled, events)

    def _log(self, dets: list[Detection], frame_idx: int, video_time: Optional[float]) -> None:
        if self.db is None or not dets:
            return
        if self.source_type == "live":
            interval = self.settings.video.live_log_interval_sec
            if time.monotonic() - self._last_log < interval:
                return
            self._last_log = time.monotonic()
        ts = utcnow()
        self.db.insert_detections([{
            "ts": ts, "camera_id": self.camera_id, "source_type": self.source_type, "job_id": self.job_id,
            "frame_idx": frame_idx, "video_time_sec": video_time, "class": d.cls_name,
            "confidence": round(d.confidence, 4), "x1": d.bbox[0], "y1": d.bbox[1], "x2": d.bbox[2],
            "y2": d.bbox[3], "status": d.status, "reason": d.reason, "track_id": d.track_id,
            "model_version": self.engine.version, "session_id": self.session_id} for d in dets])


def roi_is_active(roi: Optional[ROISet], s: DetectionSettings) -> bool:
    return bool(s.filters.roi_enabled and roi is not None and roi.includes)


def litter_summary(events: list[dict[str, Any]], frames: list[dict[str, Any]], roi: Optional[ROISet],
                   s: DetectionSettings) -> dict[str, Any]:
    """Plain-language results for a video: one entry per confirmed litter object (its class, its
    best/average confidence over the frames it was confirmed in, when it was first confirmed) plus
    per-class totals. Every confirmed object passed the ROI filter, so all of them are inside the ROI."""
    first = {e["track_id"]: e for e in events if e["event"] == "confirmed"}
    confs: dict[int, list[float]] = {}
    for f in frames:
        for d in f["detections"]:
            if d["status"] == CONFIRMED and d.get("track_id") in first:
                confs.setdefault(d["track_id"], []).append(d["confidence"])
    objects = []
    for tid, e in sorted(first.items(), key=lambda kv: kv[1]["video_time_sec"] or 0):
        c = confs.get(tid) or [e["confidence"]]
        objects.append({"track_id": tid, "class": e["class"], "first_seen_sec": e["video_time_sec"],
                        "max_confidence": round(max(c), 4), "avg_confidence": round(float(np.mean(c)), 4)})
    by_class: dict[str, dict[str, Any]] = {}
    for o in objects:
        b = by_class.setdefault(o["class"], {"class": o["class"], "count": 0, "max_confidence": 0.0, "_sum": 0.0,
                                             "timestamps_sec": []})
        b["count"] += 1
        b["max_confidence"] = max(b["max_confidence"], o["max_confidence"])
        b["_sum"] += o["avg_confidence"]
        b["timestamps_sec"].append(o["first_seen_sec"])
    classes = []
    for b in sorted(by_class.values(), key=lambda b: -b["count"]):
        b["avg_confidence"] = round(b.pop("_sum") / b["count"], 4)
        classes.append(b)
    ignored = sum(1 for f in frames for d in f["detections"] if d.get("reason") == "outside_roi")
    return {"total": len(objects), "roi_active": roi_is_active(roi, s), "inside_roi": len(objects),
            "outside_roi_ignored": ignored,
            "highest_confidence": max((o["max_confidence"] for o in objects), default=None),
            "average_confidence": round(float(np.mean([o["avg_confidence"] for o in objects])), 4) if objects else None,
            "classes": classes, "objects": objects}


def open_video_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    """H.264 (browser-playable) when the OpenCV build supports it, else MPEG-4 part 2."""
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if w.isOpened():
            return w, codec
        w.release()
    raise RuntimeError("no usable video codec for writing MP4")


# ---------------------------------------------------------------------- images
def detect_image(engine: YoloEngine, image: np.ndarray, settings: DetectionSettings,
                 roi: Optional[ROISet] = None, meta: Optional[ClassMetadata] = None,
                 db: Optional[Database] = None, alerts: Optional[AlertManager] = None,
                 camera_id: str = "upload", filename: str = "") -> dict[str, Any]:
    """Detect litter in one image. A still image has no temporal evidence, so
    'confirmed' here means: passed every filter AND confidence >= confirmed_threshold."""
    dets, info = detect_in_roi(engine, [image], settings, roi)[0]
    apply_filters(dets, image.shape, settings, roi)
    confirm_single_image(dets, settings)
    # give each individual object an index so the UI/alerts can reference it
    for i, d in enumerate(sorted(dets, key=lambda d: -d.confidence), start=1):
        d.track_id = i
    annotated = annotate_litter(image, dets, roi, roi_active=roi_is_active(roi, settings))
    alerts_out: list[dict[str, Any]] = []
    if db is not None:
        ts = utcnow()
        db.insert_detections([{
            "ts": ts, "camera_id": camera_id, "source_type": "image", "job_id": None, "frame_idx": 0,
            "video_time_sec": None, "class": d.cls_name, "confidence": round(d.confidence, 4),
            "x1": d.bbox[0], "y1": d.bbox[1], "x2": d.bbox[2], "y2": d.bbox[3], "status": d.status,
            "reason": d.reason, "track_id": None, "model_version": engine.version} for d in dets])
    if alerts is not None and settings.alerts.alert_on_image_uploads:
        session = uuid.uuid4().hex[:12]
        confirmed = [d for d in dets if d.status == CONFIRMED]
        for d in confirmed:
            a = alerts.on_confirmed(ConfirmedEvent(
                camera_id=camera_id, session_id=session, track_id=d.track_id or 0, cls_name=d.cls_name,
                confidence=d.confidence, bbox=[round(v, 1) for v in d.bbox], frame=image, annotated=annotated,
                video_time_sec=None, first_seen_ts=utcnow(), hits=1, active_objects=len(confirmed),
                job_id=f"image:{session}", source_type="image", model_version=engine.version,
                live_track_ids=frozenset(x.track_id for x in confirmed)), settings)
            if a:
                alerts_out.append(a)
    counts = Counter(d.status for d in dets)
    by_class = Counter(d.cls_name for d in dets if d.status == CONFIRMED)
    confirmed_confs = [d.confidence for d in dets if d.status == CONFIRMED]
    return {
        "detections": [d.to_dict(meta.priority(d.cls_name) if meta else None)
                       for d in sorted(dets, key=lambda d: -d.confidence)],
        "count": counts[CONFIRMED],
        "counts": {"confirmed": counts[CONFIRMED], "candidate": counts[CANDIDATE], "weak": counts[WEAK],
                   "rejected": counts[REJECTED], "raw": len(dets)},
        "by_class": dict(by_class),
        "roi_active": roi_is_active(roi, settings),
        "outside_roi_ignored": sum(1 for d in dets if d.reason == "outside_roi"),
        "highest_confidence": round(max(confirmed_confs), 4) if confirmed_confs else None,
        "average_confidence": round(float(np.mean(confirmed_confs)), 4) if confirmed_confs else None,
        "alerts": alerts_out,
        "inference_ms": round(info.inference_ms, 1),
        "tiled": info.tiled, "tiles": info.tiles,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "model_version": engine.version,
        "annotated": annotated,
    }


# ---------------------------------------------------------------------- videos
def detect_video(engine: YoloEngine, video_path: str | Path, out_dir: str | Path, settings: DetectionSettings,
                 roi: Optional[ROISet] = None, db: Optional[Database] = None,
                 alerts: Optional[AlertManager] = None, camera_id: str = "upload", job_id: Optional[str] = None,
                 progress: Optional[Callable[[float, dict[str, Any]], None]] = None) -> dict[str, Any]:
    """Process a video file: annotated MP4 + detections.json + events.json + stats.json (+ alert snapshots)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video (corrupt file or unsupported codec)")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not (0 < fps <= 240):
            fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(settings.video.max_video_seconds * fps)
        step = max(1, int(round(fps / settings.video.sample_fps)))
        pipe = StreamPipeline(engine, settings, roi, alerts, db, camera_id, "video", job_id)
        writer, codec = open_video_writer(out_dir / "annotated.mp4",
                                          fps if settings.video.annotate_all_frames else fps / step, (w, h))
        frames_json: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        new_alerts: list[dict[str, Any]] = []
        peak = Counter()
        inf_ms: list[float] = []
        last: Optional[FrameResult] = None
        idx, t0 = 0, time.perf_counter()
        tiled = None if settings.video.tiling else False
        batch = max(1, settings.video.batch_size) if engine.device.startswith("cuda") and not settings.video.tiling else 1
        pending: list[tuple[int, float, np.ndarray, bool]] = []   # (idx, video_time, frame, is_sample)
        roi_on = roi_is_active(roi, settings)

        def flush() -> None:
            nonlocal last
            samples = [p for p in pending if p[3]]
            reuse = {p[0]: pipe.plan_reuse(p[2], p[1]) for p in samples}
            need = [p for p in samples if not reuse[p[0]]]
            results = dict(zip([p[0] for p in need], detect_in_roi(engine, [p[2] for p in need], settings, roi, tiled)))
            for i, vt, frame, is_sample in pending:
                if is_sample:
                    last = pipe.process(frame, i, vt, precomputed=results.get(i), reuse_previous=reuse[i])
                    if last.inference_ms:
                        inf_ms.append(last.inference_ms)
                    frames_json.append({"frame_idx": i, "video_time_sec": round(vt, 3),
                                        "detections": [d.to_dict() for d in last.detections], "summary": last.summary})
                    events.extend(last.events)
                    new_alerts.extend(last.new_alerts)
                    for k, v in last.summary.items():
                        peak[k] = max(peak[k], v)
                if last is not None and (is_sample or settings.video.annotate_all_frames):
                    writer.write(annotate_litter(frame, last.detections, roi, roi_active=roi_on))
            pending.clear()

        while idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            is_sample = idx % step == 0
            if is_sample or (settings.video.annotate_all_frames and last is not None) or pending:
                pending.append((idx, idx / fps, frame, is_sample))
            # flush when the batch is full, just before the next sampled frame, so the non-sampled frames
            # in between are annotated with the latest result (carry-over between detection frames)
            if (idx + 1) % step == 0 and sum(1 for p in pending if p[3]) >= batch:
                flush()
            if progress and idx % 25 == 0:
                el = time.perf_counter() - t0
                frac = min(0.99, idx / total) if total else 0.0
                progress(frac, {"frames_read": idx, "frames_total": total, "frames_analysed": len(frames_json),
                                "processing_fps": round(len(frames_json) / el, 2) if el else None,
                                "eta_sec": round(el / frac - el, 1) if frac > 0.01 else None,
                                "detections": sum(len(f["detections"]) for f in frames_json),
                                "confirmed_objects": len({e["track_id"] for e in events if e["event"] == "confirmed"}),
                                "new_alerts": len(new_alerts)})
            idx += 1
        if pending:
            flush()
        writer.release()
    finally:
        cap.release()

    wall = time.perf_counter() - t0
    confirmed_ids = sorted({e["track_id"] for e in events if e["event"] == "confirmed"})
    all_dets = [d for f in frames_json for d in f["detections"]]
    litter = litter_summary(events, frames_json, roi, settings)
    stats = {
        "video": {"fps": fps, "frames": idx, "width": w, "height": h, "duration_sec": round(idx / fps, 2),
                  "output_codec": codec},
        "frames_analysed": len(frames_json), "sample_every_n_frames": step,
        "unique_confirmed_objects": len(confirmed_ids),
        "litter": litter,
        "confirmed_by_class": dict(Counter(e["class"] for e in events if e["event"] == "confirmed")),
        "new_alerts": len(new_alerts),
        "detections_by_status": dict(Counter(d["status"] for d in all_dets)),
        "rejection_reasons": dict(Counter(d.get("reason") for d in all_dets if d["status"] == REJECTED)),
        "peak_per_frame": dict(peak),
        "model_runs": len(inf_ms), "frames_reused_static": pipe.frames_reused, "batch_size": batch,
        "device": engine.device,
        "mean_inference_ms": round(float(np.mean(inf_ms)), 1) if inf_ms else None,
        "processing_fps": round(len(frames_json) / wall, 2) if wall else None,
        "wall_time_sec": round(wall, 1),
        "model_version": engine.version,
        "settings": {"confidence": settings.confidence.model_dump(), "tracking": settings.tracking.model_dump(),
                     "tiling": settings.tiling.model_dump()},
    }
    (out_dir / "detections.json").write_text(json.dumps(frames_json), encoding="utf-8")
    (out_dir / "events.json").write_text(json.dumps(events, indent=1), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if progress:
        progress(1.0, {"frames_read": idx, "frames_total": total, "frames_analysed": len(frames_json)})
    return {"stats": stats, "events": events, "alerts": new_alerts,
            "files": {k: str(out_dir / f) for k, f in [("video", "annotated.mp4"), ("detections", "detections.json"),
                                                       ("events", "events.json"), ("stats", "stats.json")]}}
