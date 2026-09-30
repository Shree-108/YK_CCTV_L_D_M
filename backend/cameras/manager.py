"""Multi-camera workers: capture thread -> bounded queue -> processing thread.

* Each camera is fully independent: its own threads, its own pipeline/tracker.
  An exception or disconnect in one camera never touches the others.
* Capture keeps reading at source speed and drops stale frames (queue size 1),
  so inference always sees the latest frame and memory stays bounded.
* Disconnects trigger reconnection with exponential backoff (1s .. 30s).
* Only the most recent annotated JPEG is kept per camera (for MJPEG/preview).
"""
from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

from backend.detection.filters import ROISet
from backend.services.pipeline import StreamPipeline
from backend.services.visualization import annotate_litter

log = logging.getLogger(__name__)
OPEN_TIMEOUT_MS = int(os.getenv("CAMERA_OPEN_TIMEOUT_MS", "5000"))
READ_TIMEOUT_MS = int(os.getenv("CAMERA_READ_TIMEOUT_MS", "5000"))
CRED_RE = re.compile(r"(\w+://)([^:/@\s]+):([^@/\s]+)@")


def mask_source(src: str) -> str:
    """Hide credentials in URLs for API responses and logs."""
    return CRED_RE.sub(r"\1\2:***@", src or "")


def resolve_source(src: str) -> Any:
    """'env:VAR' -> value of VAR (keeps RTSP passwords out of the DB); digits -> webcam index."""
    if src.startswith("env:"):
        val = os.getenv(src[4:])
        if not val:
            raise ValueError(f"environment variable {src[4:]} is not set")
        src = val
    if src.isdigit():
        return int(src)
    if "://" not in src:
        # relative file paths are relative to the project, not the server's working directory
        from pathlib import Path

        from backend.core.config import PROJECT_ROOT
        p = Path(src)
        if not p.is_absolute() and (PROJECT_ROOT / p).exists():
            return str(PROJECT_ROOT / p)
    return src


def camera_source(cam: dict[str, Any]) -> Any:
    """Capture source for a camera row. 'cctv:<id>' cameras get their authenticated RTSP URL built in
    memory from rtsp_json + environment credentials (see backend.cameras.cctv)."""
    from backend.cameras.cctv import camera_url, is_cctv

    return camera_url(cam) if is_cctv(cam) else resolve_source(cam["source"])


def open_capture(src: Any) -> cv2.VideoCapture:
    if isinstance(src, int):
        return cv2.VideoCapture(src)
    if isinstance(src, str) and src.lower().startswith(("rtsp://", "rtsps://")):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        # explicit timeouts: without them an unreachable camera blocks ~30 s per attempt
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                                                      cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(src)


@dataclass
class CameraStatus:
    state: str = "stopped"          # stopped | connecting | online | reconnecting | error | disabled
    last_error: Optional[str] = None
    reconnects: int = 0
    frames_captured: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    fps: float = 0.0                # processed frames per second
    capture_fps: float = 0.0        # frames per second received from the source
    inference_ms: float = 0.0
    last_frame_ts: Optional[float] = None
    summary: dict[str, int] = field(default_factory=dict)
    resolution: Optional[tuple[int, int]] = None


class CameraWorker:
    def __init__(self, cam: dict[str, Any], pipeline_factory: Callable[[dict[str, Any]], StreamPipeline],
                 process_fps: float, on_event: Callable[[str, str, str], None],
                 capture_factory: Callable[[Any], cv2.VideoCapture] = open_capture,
                 max_backoff: float = 30.0) -> None:
        self.cam = cam
        self.id = cam["id"]
        self.pipeline_factory = pipeline_factory
        self.pipeline: Optional[StreamPipeline] = None
        self.process_fps = process_fps
        self.on_event = on_event
        self.capture_factory = capture_factory
        self.max_backoff = max_backoff
        self.status = CameraStatus()
        self._q: "queue.Queue[tuple[np.ndarray, float]]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._jpeg: Optional[bytes] = None
        self._raw_jpeg: Optional[bytes] = None
        self._live: Optional[dict[str, Any]] = None
        self._seq = 0
        self._jpeg_lock = threading.Lock()
        self._times: deque = deque(maxlen=30)
        self._cap_times: deque = deque(maxlen=50)
        self._threads: list[threading.Thread] = []

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._stop.clear()
        self.status.state = "connecting"
        self._threads = [threading.Thread(target=self._capture_loop, name=f"cap-{self.id}", daemon=True),
                         threading.Thread(target=self._process_loop, name=f"proc-{self.id}", daemon=True)]
        for t in self._threads:
            t.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout)
        self.status.state = "stopped"

    @property
    def latest_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._jpeg

    def latest_live(self) -> tuple[Optional[dict[str, Any]], Optional[bytes]]:
        """(metadata with detections, raw JPEG) of the same processed frame."""
        with self._jpeg_lock:
            return self._live, self._raw_jpeg

    # ---------------------------------------------------------------- capture
    def _capture_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            cap = None
            try:
                src = camera_source(self.cam)
                cap = self.capture_factory(src)
                if cap is None or not cap.isOpened():
                    raise ConnectionError("could not open source")
                if self.status.state in ("reconnecting", "error"):
                    self.on_event("info", self.id, "camera reconnected")
                self.status.state, self.status.last_error, backoff = "online", None, 1.0
                is_file = isinstance(src, str) and os.path.isfile(src)
                src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                fails = 0
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        if is_file:  # demo/file cameras loop instead of "disconnecting"
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            fails += 1
                            if fails > 3:
                                raise ConnectionError("file source unreadable")
                            continue
                        raise ConnectionError("stream read failed")
                    fails = 0
                    self.status.frames_captured += 1
                    self._cap_times.append(time.monotonic())
                    if len(self._cap_times) > 1:
                        self.status.capture_fps = round((len(self._cap_times) - 1) / max(
                            self._cap_times[-1] - self._cap_times[0], 1e-6), 1)
                    skip = self.pipeline.settings.video.live_frame_skip if self.pipeline else 0
                    if skip and self.status.frames_captured % (skip + 1):
                        continue   # FRAME_SKIP: only every (skip+1)-th frame is offered to the model
                    try:
                        self._q.put_nowait((frame, time.time()))
                    except queue.Full:
                        try:
                            self._q.get_nowait()          # drop the stale frame
                            self.status.frames_dropped += 1
                        except queue.Empty:
                            pass
                        self._q.put_nowait((frame, time.time()))
                    if is_file:
                        time.sleep(1.0 / max(1.0, min(src_fps, 60.0)))  # emulate real-time
            except Exception as exc:  # noqa: BLE001 - isolate every failure to this camera
                if self._stop.is_set():
                    break
                from backend.cameras.cctv import sanitize
                exc = sanitize(str(exc))   # never let a URL with credentials reach logs/UI/DB
                was_online = self.status.state == "online"
                self.status.state, self.status.last_error = "reconnecting", exc
                self.status.capture_fps = 0.0
                self.status.reconnects += 1
                if was_online or self.status.reconnects == 1:
                    self.on_event("warning", self.id, f"camera disconnected: {exc}; retrying")
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.max_backoff)
            finally:
                if cap is not None:
                    cap.release()

    # ---------------------------------------------------------------- processing
    def _process_loop(self) -> None:
        interval = 1.0 / self.process_fps if self.process_fps > 0 else 0
        frame_idx = 0
        while not self._stop.is_set():
            try:
                frame, ts = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            t_start = time.monotonic()
            try:
                if self.pipeline is None:
                    self.pipeline = self.pipeline_factory(self.cam)
                res = self.pipeline.process(frame, frame_idx)
                frame_idx += 1
                self._times.append(time.monotonic())
                st = self.status
                st.frames_processed += 1
                st.inference_ms = round(res.inference_ms, 1)
                st.summary = res.summary
                st.last_frame_ts = ts
                st.resolution = (frame.shape[1], frame.shape[0])
                if len(self._times) > 1:
                    st.fps = round((len(self._times) - 1) / max(self._times[-1] - self._times[0], 1e-6), 2)
                from backend.services.pipeline import roi_is_active
                img = annotate_litter(frame, res.detections, self.pipeline.roi,
                                      roi_active=roi_is_active(self.pipeline.roi, self.pipeline.settings),
                                      extra=[f"LIVE  {self.cam.get('name', self.id)}",
                                             f"FPS {st.fps:.1f}  |  {st.inference_ms:.0f} ms"])
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                ok_raw, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok and ok_raw:
                    with self._jpeg_lock:
                        self._jpeg, self._raw_jpeg = buf.tobytes(), raw.tobytes()
                        self._seq += 1
                        # raw frame + the detections computed on exactly that frame (client-side overlay)
                        self._live = {"seq": self._seq, "ts": ts, "width": frame.shape[1], "height": frame.shape[0],
                                      "detections": [d.to_dict() for d in res.detections],
                                      "summary": res.summary, "inference_ms": st.inference_ms, "fps": st.fps,
                                      "model_version": self.pipeline.engine.version}
            except Exception as exc:  # noqa: BLE001
                log.exception("camera %s processing error", self.id)
                self.status.last_error = f"processing: {exc}"
                self.on_event("error", self.id, f"processing error: {exc}")
                self._stop.wait(1.0)
            spent = time.monotonic() - t_start
            if interval > spent:
                self._stop.wait(interval - spent)


class CameraManager:
    def __init__(self, pipeline_factory: Callable[[dict[str, Any]], StreamPipeline],
                 process_fps: Callable[[dict[str, Any]], float], on_event: Callable[[str, str, str], None],
                 capture_factory: Callable[[Any], cv2.VideoCapture] = open_capture) -> None:
        self.workers: dict[str, CameraWorker] = {}
        self.pipeline_factory, self.process_fps = pipeline_factory, process_fps
        self.on_event, self.capture_factory = on_event, capture_factory
        self._lock = threading.Lock()

    def sync(self, cameras: list[dict[str, Any]]) -> None:
        """Start enabled cameras, stop disabled/removed ones, restart changed ones."""
        with self._lock:
            wanted = {c["id"]: c for c in cameras if c.get("enabled")}
            for cid in list(self.workers):
                w = self.workers[cid]
                if cid not in wanted or wanted[cid] != w.cam:
                    w.stop()
                    del self.workers[cid]
            for cid, cam in wanted.items():
                if cid not in self.workers:
                    w = CameraWorker(cam, self.pipeline_factory, self.process_fps(cam), self.on_event,
                                     self.capture_factory)
                    self.workers[cid] = w
                    w.start()

    def refresh_settings(self, settings_for: Callable[[dict[str, Any]], tuple[Any, ROISet]]) -> None:
        for w in list(self.workers.values()):
            if w.pipeline is not None:
                w.pipeline.update_settings(*settings_for(w.cam))

    def stop_all(self) -> None:
        with self._lock:
            for w in self.workers.values():
                w.stop()
            self.workers.clear()

    def status(self, cam_id: str) -> Optional[CameraStatus]:
        w = self.workers.get(cam_id)
        return w.status if w else None
