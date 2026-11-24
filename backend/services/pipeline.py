"""The per-stream detection pipeline and the image/video entry points.

FRAME -> YOLO on the FULL frame (optional overlapping tiles for small objects) -> merge/NMS
      -> filters (confidence tier, class, size, aspect) -> tracker (ByteTrack-style) -> temporal confirmation
      -> litter events / alert manager

Monitored region (owner brief 2026-11-24, backend/services/roi_service.py): when a caller passes an ROI context,
detection runs only inside the automatic pavement region (uniform gate, masked crop, bottom-centre + overlap rule).
Without one (``roi=None``) the frame is processed whole, exactly as before. Inside the region every detection above
``confidence.weak_threshold`` is kept, tagged with a status and shown.

Terms used in results (backend/services/litter_reports.py):
  observation   one detection in one processed frame
  tracked object one tracker track (the same object across frames)
  litter event  one DETECTED object: a still image's detection above ``confirmed_threshold``, or a video track that
                passed temporal confirmation. Repeated observations of that object never create new events.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from backend.alerts.manager import AlertManager, ClassMetadata, ConfirmedEvent
from backend.core.settings import DetectionSettings
from backend.database.db import Database, utcnow
from backend.detection.engine import YoloEngine
from backend.detection.filters import apply_filters, confirm_single_image
from backend.detection.types import CANDIDATE, CONFIRMED, REJECTED, WEAK, Detection, InferenceInfo
from backend.services.litter_reports import DETECTED, TRACKED, detection_status, iso, local_now
from backend.services.roi_service import (NO_LITTER_MSG, RoiContext, StaticCandidates, resolve_candidates,
                                          roi_detect)
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
    roi: Optional[dict[str, Any]] = None          # roi_service block of this frame (None: no ROI)


def summarize(dets: list[Detection], tracker: LitterTracker, new_alerts: int) -> dict[str, int]:
    """Per-frame counters. confirmed_litter/active_tracks count objects visible now (<=2 missed frames),
    not tracks the tracker is merely remembering."""
    c = Counter(d.status for d in dets)
    visible = tracker.visible_tracks()
    return {"raw_detections": len(dets), "rejected": c[REJECTED], "weak": c[WEAK], "candidates": c[CANDIDATE],
            "confirmed_in_frame": c[CONFIRMED], "confirmed_litter": sum(1 for t in visible if t.confirmed),
            "active_tracks": len(visible), "tracks_in_memory": len(tracker.tracks), "new_alerts": new_alerts}


def _copy(dets: list[Detection]) -> list[Detection]:
    return [Detection(list(d.bbox), d.cls_id, d.cls_name, d.confidence, source=d.source) for d in dets]


def _copy_block(block: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    return json.loads(json.dumps(block)) if block is not None else None


def pending_block(camera_id: str) -> dict[str, Any]:
    """Live stream whose monitored region is still being computed: nothing is scanned meanwhile."""
    return {"applied": False, "source": "pending", "camera_id": camera_id, "status": "roi_pending",
            "warning": "Monitored region is being computed: detection starts when it is ready",
            "candidates": [], "polygons": [], "remark_tags": ["roi_pending"], "detector_run": False}


def confirming_boxes(dets: list[Detection], s: DetectionSettings) -> list[list[float]]:
    """Detector boxes that confirm a colour candidate: kept by the filters and at/above the confirmed threshold."""
    return [d.bbox for d in dets if d.status in (CANDIDATE, CONFIRMED)
            and d.confidence >= s.confidence.confirmed_threshold]


EventSink = Callable[[str, Any, "StreamPipeline", Optional[np.ndarray], list[Detection]], None]


class StreamPipeline:
    """Stateful pipeline for one stream (live camera or one video job). ``event_sink(kind, track, pipeline, frame,
    dets)`` is called for every tracker event ("confirmed" / "lost"); live cameras use it to record litter events."""

    def __init__(self, engine: YoloEngine, settings: DetectionSettings, alerts: Optional[AlertManager],
                 db: Optional[Database], camera_id: str, source_type: str = "live", job_id: Optional[str] = None,
                 location: str = "", event_sink: Optional[EventSink] = None, roi: Any = None) -> None:
        self.engine, self.settings = engine, settings
        self.alerts, self.db = alerts, db
        self.camera_id, self.source_type, self.job_id, self.location = camera_id, source_type, job_id, location
        self.event_sink = event_sink
        self.session_id = uuid.uuid4().hex[:12]
        self.tracker = LitterTracker(settings)
        self._first_seen: dict[int, str] = {}
        self._last_raw: Optional[list[Detection]] = None
        self._last_thumb: Optional[np.ndarray] = None
        self._last_infer_t: Optional[float] = None
        self.frames_reused = 0
        self._last_log = 0.0
        # ROI: a fixed RoiContext (video) or a provider with .update(frame) -> RoiContext|None and .pending (live)
        self.roi = roi
        self._last_block: Optional[dict[str, Any]] = None
        self._static = StaticCandidates()
        self.roi_counts: Counter = Counter()
        # pilot stationary rule: (first time seen, first box) per track; confirmed tracks waiting to be still
        self._first_pos: dict[int, tuple[float, list[float]]] = {}
        self._await_still: set[int] = set()

    def roi_for(self, frame: np.ndarray) -> tuple[Optional[RoiContext], bool]:
        """(context, pending). pending: a live ROI is still being built -> detection waits."""
        if self.roi is None or isinstance(self.roi, RoiContext):
            return self.roi, False
        ctx = self.roi.update(frame)
        return ctx, bool(getattr(self.roi, "pending", False))

    def update_settings(self, settings: DetectionSettings) -> None:
        self.settings = settings
        self.tracker.update_settings(settings)

    def _raw_detect(self, frame: np.ndarray, s: DetectionSettings, video_time: Optional[float],
                    force_tiling: Optional[bool], allow_reuse: bool
                    ) -> tuple[list[Detection], Any, bool, Optional[dict[str, Any]]]:
        """Run the model, or - for a fixed camera whose view has not changed - reuse the previous raw
        detections. Reuse is bounded by ``max_reuse_sec`` so the model re-checks the scene regularly."""
        v = s.video
        thumb = cv2.cvtColor(cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.int16)
        if (allow_reuse and v.skip_static_frames and self._last_raw is not None and self._last_thumb is not None
                and video_time is not None and self._last_infer_t is not None
                and video_time - self._last_infer_t < v.max_reuse_sec
                and float(np.mean(np.abs(thumb - self._last_thumb))) < v.static_diff_threshold):
            return _copy(self._last_raw), InferenceInfo(), True, _copy_block(self._last_block)
        ctx, pending = self.roi_for(frame)
        if pending:
            return [], InferenceInfo(), False, pending_block(self.camera_id)
        dets, info, block = roi_detect(self.engine, frame, s, ctx, force_tiling)
        self._last_raw, self._last_block = _copy(dets), _copy_block(block)
        self._last_thumb, self._last_infer_t = thumb, video_time
        return dets, info, False, block

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
            dets, info = precomputed[0], precomputed[1]
            block = precomputed[2] if len(precomputed) > 2 else None
            reused = False
            self._last_raw, self._last_block = _copy(dets), _copy_block(block)
        elif reuse_previous and self._last_raw is not None:
            dets, info, reused, block = _copy(self._last_raw), InferenceInfo(), True, _copy_block(self._last_block)
        else:
            dets, info, reused, block = self._raw_detect(frame, s, video_time, force_tiling, allow_reuse)
        self.frames_reused += int(reused)
        apply_filters(dets, frame.shape, s)
        if block is not None:
            # colour candidates: confirmed by a strong detector box in this frame, or by staying static
            resolve_candidates(block, confirming_boxes(dets, s))
            self._static.update(block, video_time if video_time is not None else time.monotonic(), s.roi)
            self.roi_counts[block.get("status") or "none"] += 1
            self.roi_counts["possible_candidates"] += sum(1 for c in block.get("candidates") or []
                                                          if c["state"] == "possible")
        track_events = self.tracker.update(dets, frame_idx, frame.shape)
        now = utcnow()
        t_now = video_time if video_time is not None else time.monotonic()
        for d in dets:
            if d.track_id is not None:
                self._first_seen.setdefault(d.track_id, now)
                self._first_pos.setdefault(d.track_id, (t_now, list(d.bbox)))
        in_roi = None if block is None else bool(block.get("applied"))

        new_alerts: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        annotated_cache: list[Optional[np.ndarray]] = [None]

        def raise_alert(t: Any) -> Optional[dict[str, Any]]:
            if annotated_cache[0] is None:
                annotated_cache[0] = annotate_fn(frame, dets) if annotate_fn else annotate(frame, dets)
            return self.alerts.on_confirmed(ConfirmedEvent(
                camera_id=self.camera_id, session_id=self.session_id, track_id=t.track_id, cls_name=t.cls_name,
                confidence=t.mean_hit_conf, bbox=[round(v, 1) for v in t.bbox], frame=frame,
                annotated=annotated_cache[0], video_time_sec=video_time,
                first_seen_ts=self._first_seen.get(t.track_id, now), hits=t.total_hits,
                active_objects=sum(1 for x in self.tracker.visible_tracks() if x.confirmed), location=self.location,
                job_id=self.job_id, source_type=self.source_type, model_version=self.engine.version,
                live_track_ids=frozenset(x.track_id for x in self.tracker.active_tracks), in_roi=in_roi), s)

        a = s.alerts
        wait_still = a.pilot_mode and a.stationary_sec > 0 and self.source_type in ("video", "live")
        for ev in track_events:
            t = ev.track
            events.append({"event": ev.kind, "track_id": t.track_id, "class": t.cls_name, "frame_idx": frame_idx,
                           "video_time_sec": video_time, "confidence": round(t.mean_hit_conf, 4)})
            if self.event_sink is not None:
                try:
                    self.event_sink(ev.kind, t, self, frame, dets)
                except Exception:  # noqa: BLE001 - recording an event must never stop detection
                    log.exception("event sink failed for %s track %s", self.camera_id, t.track_id)
            if ev.kind == "confirmed" and self.alerts is not None:
                if wait_still:
                    self._await_still.add(t.track_id)      # pilot: alert once it has been stationary long enough
                    continue
                alert = raise_alert(t)
                if alert:
                    new_alerts.append(alert)
                    events[-1]["alert_id"] = alert["id"]
            elif ev.kind == "lost":
                self._await_still.discard(t.track_id)
                if self.alerts is not None:
                    self.alerts.on_lost(self.session_id, t.track_id)
        if self._await_still and self.alerts is not None:
            visible = {x.track_id: x for x in self.tracker.visible_tracks(grace=0)}
            for tid in sorted(self._await_still):
                t = visible.get(tid)
                first = self._first_pos.get(tid)
                if t is None or first is None or t_now - first[0] < a.stationary_sec:
                    continue
                fb, b = first[1], t.bbox
                diag = max(1.0, float(np.hypot(fb[2] - fb[0], fb[3] - fb[1])))
                shift = float(np.hypot((b[0] + b[2] - fb[0] - fb[2]) / 2, (b[1] + b[3] - fb[1] - fb[3]) / 2))
                if shift > a.stationary_max_shift * diag:
                    self._first_pos[tid] = (t_now, list(b))   # it moved: the stationary clock restarts here
                    continue
                self._await_still.discard(tid)
                alert = raise_alert(t)
                events.append({"event": "stationary", "track_id": tid, "class": t.cls_name, "frame_idx": frame_idx,
                               "video_time_sec": video_time, "stationary_sec": round(t_now - first[0], 1)})
                if alert:
                    new_alerts.append(alert)
                    events[-1]["alert_id"] = alert["id"]

        self._log(dets, frame_idx, video_time)
        summary = summarize(dets, self.tracker, len(new_alerts))
        if block is not None and block.get("applied") and block.get("status") in ("checked", "clean", "litter"):
            # video: litter = a tracker-confirmed object visible now (the temporal rule), not one strong frame
            block["status"] = "litter" if summary["confirmed_litter"] else "clean"
            block["message"] = None if summary["confirmed_litter"] else (block.get("message") or NO_LITTER_MSG)
        return FrameResult(dets, summary, new_alerts, info.inference_ms, info.tiled, events, block)

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


def open_video_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    """H.264 (browser-playable) when the OpenCV build supports it, else MPEG-4 part 2."""
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if w.isOpened():
            return w, codec
        w.release()
    raise RuntimeError("no usable video codec for writing MP4")


def _with_status(d: Detection, meta: Optional[ClassMetadata]) -> dict[str, Any]:
    out = d.to_dict(meta.priority(d.cls_name) if meta else None)
    out["detection_status"] = detection_status(d.status)
    return out


# ---------------------------------------------------------------------- images
def detect_image(engine: YoloEngine, image: np.ndarray, settings: DetectionSettings,
                 meta: Optional[ClassMetadata] = None, db: Optional[Database] = None,
                 alerts: Optional[AlertManager] = None, camera_id: str = "upload", filename: str = "",
                 job_id: Optional[str] = None, *, source_type: str = "image", frame_idx: int = 0,
                 video_time_sec: Optional[float] = None, session_id: Optional[str] = None,
                 create_alerts: Optional[bool] = None, location: str = "",
                 header: tuple[str, ...] = (), roi: Optional[RoiContext] = None) -> dict[str, Any]:
    """Detect litter in one image (full frame). A still image has no temporal evidence, so DETECTED here means:
    passed every filter AND confidence >= confirmed_threshold; other kept detections are UNDER_REVIEW.
    ``job_id`` groups images for alert de-duplication: uploads get a unique one (different recordings),
    scheduled screenshots of one fixed camera share one so persisting litter updates its alert.

    Keyword-only extras (defaults = the image-upload behaviour): ``source_type``/``frame_idx``/``video_time_sec`` are
    recorded on the detection rows; for any source other than "image" the ``job_id`` is recorded too (sampled video
    screenshots link every row to their job). ``session_id`` fixes the alert session, ``create_alerts`` overrides
    ``alerts.alert_on_image_uploads``. ``header``: extra HUD lines on the annotated image. ``roi``: the monitored
    region (roi_service.RoiService.resolve); None = full frame, as before."""
    dets, info, block = roi_detect(engine, image, settings, roi)
    apply_filters(dets, image.shape, settings)
    confirm_single_image(dets, settings)
    resolve_candidates(block, [d.bbox for d in dets if d.status == CONFIRMED])
    # give each individual object an index so the UI/alerts can reference it
    for i, d in enumerate(sorted(dets, key=lambda d: -d.confidence), start=1):
        d.track_id = i
    annotated = annotate_litter(image, dets, header, roi=block if settings.roi.draw_overlay else None,
                                roi_text=block)
    alerts_out: list[dict[str, Any]] = []
    linked: dict[int, int] = {}                        # object index -> alert it was counted on (incl. merges)
    if db is not None:
        ts = utcnow()
        db.insert_detections([{
            "ts": ts, "camera_id": camera_id, "source_type": source_type,
            "job_id": None if source_type == "image" else job_id, "frame_idx": frame_idx,
            "video_time_sec": video_time_sec, "session_id": session_id if source_type != "image" else None,
            "class": d.cls_name, "confidence": round(d.confidence, 4),
            "x1": d.bbox[0], "y1": d.bbox[1], "x2": d.bbox[2], "y2": d.bbox[3], "status": d.status,
            "reason": d.reason, "track_id": None, "model_version": engine.version} for d in dets])
    want_alerts = settings.alerts.alert_on_image_uploads if create_alerts is None else create_alerts
    if alerts is not None and want_alerts:
        session = session_id or uuid.uuid4().hex[:12]
        confirmed = [d for d in dets if d.status == CONFIRMED]
        for d in confirmed:
            a = alerts.on_confirmed(ConfirmedEvent(
                camera_id=camera_id, session_id=session, track_id=d.track_id or 0, cls_name=d.cls_name,
                confidence=d.confidence, bbox=[round(v, 1) for v in d.bbox], frame=image, annotated=annotated,
                video_time_sec=video_time_sec, first_seen_ts=utcnow(), hits=1, active_objects=len(confirmed),
                location=location, job_id=job_id or f"image:{session}", source_type=source_type,
                model_version=engine.version,
                live_track_ids=frozenset(x.track_id for x in confirmed),
                in_roi=None if block is None else bool(block.get("applied"))), settings)
            if a:
                alerts_out.append(a)
            held = alerts.linked_alert(session, d.track_id or 0)
            if held is not None:
                linked[d.track_id or 0] = held
    counts = Counter(d.status for d in dets)
    by_class = Counter(d.cls_name for d in dets if d.status == CONFIRMED)
    confirmed_confs = [d.confidence for d in dets if d.status == CONFIRMED]
    return {
        "detections": [_with_status(d, meta) for d in sorted(dets, key=lambda d: -d.confidence)],
        "count": counts[CONFIRMED],
        "counts": {"confirmed": counts[CONFIRMED], "candidate": counts[CANDIDATE], "weak": counts[WEAK],
                   "rejected": counts[REJECTED], "raw": len(dets)},
        "by_class": dict(by_class),
        "highest_confidence": round(max(confirmed_confs), 4) if confirmed_confs else None,
        "average_confidence": round(float(np.mean(confirmed_confs)), 4) if confirmed_confs else None,
        "alerts": alerts_out,
        "linked_alerts": linked,
        "inference_ms": round(info.inference_ms, 1),
        "tiled": info.tiled, "tiles": info.tiles,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "model_version": engine.version,
        **({"roi": block} if block is not None else {}),     # no key at all when ROI is off
        "annotated": annotated,
    }


# ---------------------------------------------------------------------- videos
def video_litter(frames: list[dict[str, Any]], confirmations: dict[int, int]) -> dict[str, Any]:
    """Observations, tracked objects and litter events of an analysed video.

    ``frames``: detections.json content (per processed frame: frame_idx, video_time_sec, processed_at, detections).
    ``confirmations``: track_id -> frame_idx at which the tracker confirmed it (one litter event per such track).

    Every detection becomes one observation. For a track that became an event, the observation in the confirmation
    frame is DETECTED and every other observation of that track (before or after) is TRACKED, so
    ``observation_count`` == the event's rows in the detection CSV == its share of ``total_frame_observations``.
    Detections of other tracks are UNDER_REVIEW, filtered ones REJECTED."""
    per_track: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    observations: list[dict[str, Any]] = []
    for f in frames:
        for d in f["detections"]:
            tid = d.get("track_id")
            if d["status"] != REJECTED and tid is not None:
                per_track[tid].append((f, d))
            ev = tid is not None and tid in confirmations and d["status"] != REJECTED
            status = ((DETECTED if f["frame_idx"] == confirmations[tid] and d["status"] == CONFIRMED else TRACKED)
                      if ev else detection_status(d["status"]))
            observations.append({"frame_number": f["frame_idx"], "frame_timestamp": f["video_time_sec"],
                                 "processed_at": f.get("processed_at"), "track_id": tid, "class": d["class"],
                                 "confidence": d["confidence"], "status": d["status"], "reason": d.get("reason"),
                                 "detection_status": status, "bbox": d["bbox"], "event_track": ev})
    tracks: dict[int, dict[str, Any]] = {}
    for tid, obs in per_track.items():
        confs = [d["confidence"] for _, d in obs]
        first, last = obs[0][0], obs[-1][0]
        at_conf = next((d for f, d in obs if f["frame_idx"] == confirmations.get(tid)), obs[-1][1])
        cls = (at_conf["class"] if tid in confirmations
               else Counter(d["class"] for _, d in obs).most_common(1)[0][0])
        tracks[tid] = {"track_id": tid, "class": cls, "is_event": tid in confirmations,
                       "first_seen_sec": first["video_time_sec"], "last_seen_sec": last["video_time_sec"],
                       "first_processed_at": first.get("processed_at"), "last_processed_at": last.get("processed_at"),
                       "duration_sec": round(last["video_time_sec"] - first["video_time_sec"], 3),
                       "observation_count": len(obs), "max_confidence": round(max(confs), 4),
                       "mean_confidence": round(float(np.mean(confs)), 4), "bbox": at_conf["bbox"],
                       "confirmed_frame": confirmations.get(tid)}
    events = sorted((t for t in tracks.values() if t["is_event"]), key=lambda t: (t["first_seen_sec"], t["track_id"]))
    by_class: dict[str, dict[str, Any]] = {}
    for t in events:
        b = by_class.setdefault(t["class"], {"class": t["class"], "events": 0, "observations": 0})
        b["events"] += 1
        b["observations"] += t["observation_count"]
    return {"unique_events": len(events), "frame_observations": sum(t["observation_count"] for t in events),
            "tracked_objects": len(tracks), "under_review_objects": len(tracks) - len(events),
            "by_class": sorted(by_class.values(), key=lambda b: (-b["events"], b["class"])),
            "events": events, "tracks": tracks, "observations": observations}


def _fourcc(cap: cv2.VideoCapture) -> str:
    v = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    s = "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))
    return s if v and s.isprintable() and s.strip() else "unknown"


def detect_video(engine: YoloEngine, video_path: str | Path, out_dir: str | Path, settings: DetectionSettings,
                 db: Optional[Database] = None, alerts: Optional[AlertManager] = None, camera_id: str = "upload",
                 job_id: Optional[str] = None, progress: Optional[Callable[[float, dict[str, Any]], None]] = None,
                 evidence_dir: Optional[Path] = None, header: tuple[str, ...] = (),
                 roi: Optional[RoiContext] = None) -> dict[str, Any]:
    """Process a video file: annotated MP4 + detections.json + events.json + stats.json
    (+ alert snapshots, + one evidence JPEG per litter event in ``evidence_dir`` as track_<id>.jpg).
    ``roi``: monitored region (inference only inside it); None = full frame.
    Returns stats, tracker events, alerts, files, ``validation`` and ``litter`` (video_litter())."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if evidence_dir is not None:
        Path(evidence_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video (corrupt file or unsupported codec)")
    try:
        fps_raw = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fps = fps_raw if 0 < fps_raw <= 240 else 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        validation: dict[str, Any] = {"width": w, "height": h, "fps": round(fps, 3), "codec": _fourcc(cap),
                                      "frames_reported": total, "fps_reported": round(fps_raw, 3), "warnings": []}
        if fps != fps_raw:
            validation["warnings"].append(f"The file reports an invalid frame rate ({fps_raw}); 25 fps was assumed, "
                                          "so video timestamps may be wrong.")
        if w <= 0 or h <= 0:
            raise ValueError("The video reports no picture size (damaged file or unsupported codec)")
        max_frames = int(settings.video.max_video_seconds * fps)
        step = max(1, int(round(fps / settings.video.sample_fps)))
        pipe = StreamPipeline(engine, settings, alerts, db, camera_id, "video", job_id, roi=roi)
        draw_roi_on = settings.roi.draw_overlay
        writer, codec = open_video_writer(out_dir / "annotated.mp4",
                                          fps if settings.video.annotate_all_frames else fps / step, (w, h))
        frames_json: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        new_alerts: list[dict[str, Any]] = []
        confirmations: dict[int, int] = {}
        peak: Counter = Counter()
        inf_ms: list[float] = []
        last: Optional[FrameResult] = None
        size_changes = 0
        idx, t0 = 0, time.perf_counter()
        tiled = None if settings.video.tiling else False
        batch = max(1, settings.video.batch_size) if engine.device.startswith("cuda") and not settings.video.tiling else 1
        pending: list[tuple[int, float, np.ndarray, bool]] = []   # (idx, video_time, frame, is_sample)

        def flush() -> None:
            nonlocal last
            samples = [p for p in pending if p[3]]
            reuse = {p[0]: pipe.plan_reuse(p[2], p[1]) for p in samples}
            need = [p for p in samples if not reuse[p[0]]]
            if roi is None:
                results = dict(zip([p[0] for p in need], engine.detect_many([p[2] for p in need], settings, tiled)))
            else:   # ROI crop per frame (the gate may skip the model on a uniform region)
                results = {p[0]: roi_detect(engine, p[2], settings, roi, tiled) for p in need}
            for i, vt, frame, is_sample in pending:
                if is_sample:
                    last = pipe.process(frame, i, vt, precomputed=results.get(i), reuse_previous=reuse[i])
                    if last.inference_ms:
                        inf_ms.append(last.inference_ms)
                    fj = {"frame_idx": i, "video_time_sec": round(vt, 3), "processed_at": iso(local_now()),
                          "detections": [d.to_dict() for d in last.detections], "summary": last.summary}
                    if last.roi is not None:
                        fj["roi"] = {k: last.roi.get(k) for k in ("status", "message", "candidates", "detector_run",
                                                                   "dropped_outside")}
                    frames_json.append(fj)
                    for e in last.events:
                        if e["event"] == "confirmed":
                            confirmations[e["track_id"]] = i
                            if evidence_dir is not None:
                                img = annotate_litter(frame, last.detections,
                                                      (*header, f"Video time {vt:.1f} s  frame {i}"),
                                                      label_rejected=False, roi=last.roi if draw_roi_on else None,
                                                      roi_text=last.roi)
                                cv2.imwrite(str(Path(evidence_dir) / f"track_{e['track_id']}.jpg"), img,
                                            [cv2.IMWRITE_JPEG_QUALITY, 88])
                    events.extend(last.events)
                    new_alerts.extend(last.new_alerts)
                    for k, v in last.summary.items():
                        peak[k] = max(peak[k], v)
                if last is not None and (is_sample or settings.video.annotate_all_frames):
                    writer.write(annotate_litter(frame, last.detections, header, label_rejected=False,
                                                 roi=last.roi if draw_roi_on else None, roi_text=last.roi))
            pending.clear()

        while idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != w or frame.shape[0] != h:     # a stream that changes resolution mid-file
                size_changes += 1
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
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
                                "confirmed_objects": len(confirmations), "new_alerts": len(new_alerts)})
            idx += 1
        if pending:
            flush()
        writer.release()
    finally:
        cap.release()

    if idx == 0:
        raise ValueError("No frame could be decoded from the video (damaged file or unsupported codec)")
    validation.update(frames_read=idx, duration_sec=round(idx / fps, 2),
                      truncated=bool(total and idx < max_frames and idx < total - max(2, int(0.01 * total))),
                      limited_to_sec=settings.video.max_video_seconds if idx >= max_frames else None,
                      resized_frames=size_changes)
    if validation["truncated"]:
        validation["warnings"].append(f"Only {idx} of the {total} frames the file reports could be decoded: the video "
                                      "is damaged or truncated. The rest was NOT analysed.")
    if validation["limited_to_sec"]:
        validation["warnings"].append(f"Only the first {settings.video.max_video_seconds:.0f} s were analysed "
                                      "(video.max_video_seconds).")
    if size_changes:
        validation["warnings"].append(f"{size_changes} frame(s) had a different resolution and were resized.")

    wall = time.perf_counter() - t0
    all_dets = [d for f in frames_json for d in f["detections"]]
    litter = video_litter(frames_json, confirmations)
    stats = {
        "video": {"fps": fps, "frames": idx, "width": w, "height": h, "duration_sec": round(idx / fps, 2),
                  "output_codec": codec},
        "frames_analysed": len(frames_json), "sample_every_n_frames": step,
        "sample_interval_sec": round(step / fps, 3),
        "unique_confirmed_objects": litter["unique_events"],
        "frame_observations": litter["frame_observations"],
        "confirmed_by_class": {b["class"]: b["events"] for b in litter["by_class"]},
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
        **({} if roi is None else {"roi": {**roi.summary(), "frames_by_status": {
            k: v for k, v in pipe.roi_counts.items() if k != "possible_candidates"},
            "possible_candidates": pipe.roi_counts.get("possible_candidates", 0),
            "static_confirmed": sum(1 for f in frames_json for c in (f.get("roi") or {}).get("candidates") or []
                                    if c.get("confirmed_by") == "static")}}),
        "settings": {"confidence": settings.confidence.model_dump(), "tracking": settings.tracking.model_dump(),
                     "tiling": settings.tiling.model_dump(), "video_tiling": settings.video.tiling,
                     "inference": settings.inference.model_dump()},
    }
    (out_dir / "detections.json").write_text(json.dumps(frames_json), encoding="utf-8")
    (out_dir / "events.json").write_text(json.dumps(events, indent=1), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if progress:
        progress(1.0, {"frames_read": idx, "frames_total": total, "frames_analysed": len(frames_json)})
    return {"stats": stats, "events": events, "alerts": new_alerts, "validation": validation, "litter": litter,
            "files": {k: str(out_dir / f) for k, f in [("video", "annotated.mp4"), ("detections", "detections.json"),
                                                       ("events", "events.json"), ("stats", "stats.json")]}}
