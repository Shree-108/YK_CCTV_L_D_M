"""Random-interval screenshot capture from live cameras.

Per camera: wait a random time between ``min_minutes`` and ``max_minutes``, grab ONE frame, run the
camera's model (outdoor litter or indoor uniform) through the existing ``detect_image`` pipeline
(ROI, confidence tiers, alert de-duplication), and store original + annotated image + result.

* One background thread for all cameras; a per-camera lock means a scheduled capture and
  "Capture Now" never overlap.
* Schedule state lives in the settings table (``schedule:<camera_id>``), so a restart resumes it.
  A capture missed while the server was down runs ONCE shortly after start-up (no catch-up burst).
* Frames come from the running live worker when the stream is connected (no second RTSP session -
  NVRs limit concurrent sessions); otherwise a short-lived connection with retries is opened.
* Nothing here ever logs or stores a stream URL; errors pass through ``cctv.sanitize``.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import random
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.cameras import cctv
from backend.database.db import utcnow

if TYPE_CHECKING:
    from backend.services.state import AppState

log = logging.getLogger(__name__)
_rng = random.SystemRandom()


class CaptureError(Exception):
    """A camera problem with a safe, credential-free code + message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


class ScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_minutes: float = Field(120, ge=1, le=24 * 60)
    max_minutes: float = Field(180, ge=1, le=24 * 60)
    task: Literal["litter", "uniform"] = "litter"        # outdoor litter | indoor uniform
    model_version: Optional[str] = None                  # None: active litter model / configured uniform model
    retry_minutes: float = Field(10, ge=1, le=240)       # after a failed capture
    attempts: int = Field(3, ge=1, le=10)                 # connection attempts per capture
    timeout_sec: float = Field(30, ge=5, le=300)          # hard limit per attempt (grabber process is killed)
    backoff_sec: float = Field(2, ge=0, le=60)            # 2 s, 4 s, 8 s ... between attempts

    @model_validator(mode="after")
    def _order(self) -> "ScheduleConfig":
        if self.max_minutes < self.min_minutes:
            raise ValueError("max_minutes must be >= min_minutes")
        return self


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(t: Optional[dt.datetime]) -> Optional[str]:
    return t.isoformat(timespec="seconds") if t else None


class CaptureScheduler:
    def __init__(self, state: "AppState", tick_sec: float = 1.0) -> None:
        self.state = state
        self.tick = tick_sec
        self._locks: dict[str, threading.Lock] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._guard = threading.Lock()
        self._inflight: set[str] = set()     # cameras with a scheduled capture thread already started

    # ------------------------------------------------------------------ persistence
    def _key(self, cam_id: str) -> str:
        return f"schedule:{cam_id}"

    def get(self, cam_id: str) -> dict[str, Any]:
        raw = self.state.db.get_setting(self._key(cam_id)) or {}
        return {"state": raw.get("state", "stopped"), "config": ScheduleConfig.model_validate(raw.get("config") or {}).model_dump(),
                "next_run_at": raw.get("next_run_at"), "last_run_at": raw.get("last_run_at"),
                "last_status": raw.get("last_status"), "last_error": raw.get("last_error"),
                "last_success_at": raw.get("last_success_at"),
                "consecutive_failures": raw.get("consecutive_failures", 0)}

    def _save(self, cam_id: str, sched: dict[str, Any]) -> dict[str, Any]:
        self.state.db.set_setting(self._key(cam_id), sched)
        return sched

    def status(self, cam_id: str) -> dict[str, Any]:
        s = self.get(cam_id)
        s["busy"] = self._lock(cam_id).locked()
        if s["next_run_at"] and s["state"] == "running":
            s["seconds_to_next"] = max(0, round((dt.datetime.fromisoformat(s["next_run_at"]) - _now()).total_seconds()))
        s["model_available"], s["model_note"] = self._model_check(s["config"])
        return s

    def _lock(self, cam_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(cam_id, threading.Lock())

    def _next(self, cfg: dict[str, Any], minutes: Optional[float] = None) -> str:
        m = minutes if minutes is not None else _rng.uniform(cfg["min_minutes"], cfg["max_minutes"])
        return _iso(_now() + dt.timedelta(minutes=m))  # type: ignore[return-value]

    # ------------------------------------------------------------------ controls
    def configure(self, cam_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
        s = self.get(cam_id)
        s["config"] = ScheduleConfig.model_validate({**s["config"], **cfg}).model_dump()
        if s["state"] == "running":   # new interval applies from now
            s["next_run_at"] = self._next(s["config"])
        return self._save(cam_id, s)

    def start(self, cam_id: str) -> dict[str, Any]:
        s = self.get(cam_id)
        s.update(state="running", next_run_at=self._next(s["config"]), consecutive_failures=0)
        self.state.db.log_event("info", "schedule", f"capture schedule started (next {s['next_run_at']})", cam_id)
        return self._save(cam_id, s)

    def pause(self, cam_id: str) -> dict[str, Any]:
        s = self.get(cam_id)
        if s["state"] == "running":
            s["state"] = "paused"
            self.state.db.log_event("info", "schedule", "capture schedule paused", cam_id)
        return self._save(cam_id, s)

    def resume(self, cam_id: str) -> dict[str, Any]:
        s = self.get(cam_id)
        if s["state"] != "paused":
            return s
        nxt = s.get("next_run_at")
        if not nxt or dt.datetime.fromisoformat(nxt) <= _now():
            nxt = self._next(s["config"])
        s.update(state="running", next_run_at=nxt)
        self.state.db.log_event("info", "schedule", "capture schedule resumed", cam_id)
        return self._save(cam_id, s)

    def stop(self, cam_id: str) -> dict[str, Any]:
        s = self.get(cam_id)
        s.update(state="stopped", next_run_at=None)
        self.state.db.log_event("info", "schedule", "capture schedule stopped", cam_id)
        return self._save(cam_id, s)

    # ------------------------------------------------------------------ background loop
    def start_loop(self) -> None:
        self._recover()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="capture-scheduler", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(5)

    def _recover(self) -> None:
        """After a restart: interrupted captures are marked, missed captures run once soon."""
        for r in self.state.db._query("SELECT id FROM captures WHERE status='running'"):
            self.state.db.update_capture(r["id"], status="interrupted", error="server restarted during capture")
        for row in self.state.db._query("SELECT key FROM settings WHERE key LIKE 'schedule:%'"):
            cam_id = row["key"].split(":", 1)[1]
            s = self.get(cam_id)
            if s["state"] == "running" and (not s["next_run_at"] or dt.datetime.fromisoformat(s["next_run_at"]) <= _now()):
                s["next_run_at"] = self._next(s["config"], minutes=_rng.uniform(1, 3))
                s["recovered"] = True
                self._save(cam_id, s)
                self.state.db.log_event("info", "schedule", "missed capture while offline: running once shortly", cam_id)

    def _loop(self) -> None:
        while not self._stop.wait(self.tick):
            try:
                for row in self.state.db._query("SELECT key, value_json FROM settings WHERE key LIKE 'schedule:%'"):
                    cam_id = row["key"].split(":", 1)[1]
                    s = json.loads(row["value_json"])
                    if s.get("state") != "running" or not s.get("next_run_at"):
                        continue
                    if dt.datetime.fromisoformat(s["next_run_at"]) > _now():
                        continue
                    busy = self._lock(cam_id).locked()        # takes _guard itself: call it outside the block below
                    with self._guard:
                        if busy or cam_id in self._inflight:
                            continue
                        self._inflight.add(cam_id)
                    trigger = "recovery" if s.get("recovered") else "scheduled"
                    threading.Thread(target=self._run_scheduled, args=(cam_id, trigger), daemon=True,
                                     name=f"capture-{cam_id}").start()
            except Exception:  # noqa: BLE001 - the scheduler must never die
                log.exception("capture scheduler tick failed")

    def _run_scheduled(self, cam_id: str, trigger: str) -> None:
        try:
            self._run_scheduled_inner(cam_id, trigger)
        finally:
            with self._guard:
                self._inflight.discard(cam_id)

    def _run_scheduled_inner(self, cam_id: str, trigger: str) -> None:
        res = self.capture(cam_id, trigger)
        if res is None:      # someone else holds the lock (manual capture) - next tick retries
            return
        s = self.get(cam_id)
        if s["state"] != "running":   # stopped/paused while capturing
            return
        ok = res["status"] in ("done", "skipped")
        s["consecutive_failures"] = 0 if ok else s.get("consecutive_failures", 0) + 1
        s["next_run_at"] = self._next(s["config"]) if ok else self._next(s["config"], minutes=s["config"]["retry_minutes"])
        s.pop("recovered", None)
        self._save(cam_id, s)

    # ------------------------------------------------------------------ one capture
    def _model_check(self, cfg: dict[str, Any]) -> tuple[bool, Optional[str]]:
        if cfg["task"] == "litter":
            return (self.state.engine is not None or bool(cfg.get("model_version"))), None
        models = {m["version"]: m for m in self.state.db.list_models()}
        ver = cfg.get("model_version")
        if ver and ver in models:
            return True, None
        return False, ("Uniform model not installed: register the uniform model and set it as this camera's "
                       "model version. Captures are saved but not analysed.")

    def _grab(self, cam: dict[str, Any], cfg: dict[str, Any], out_path: Path) -> np.ndarray:
        """One valid frame, written to ``out_path``. Raises CaptureError (never hangs the API process)."""
        from backend.cameras.grabber import frame_problem, grab_frame
        from backend.cameras.manager import camera_source

        w = self.state.cameras.workers.get(cam["id"])
        if w is not None and w.status.state == "online":   # live stream already open: no second RTSP session
            _, raw = w.latest_live()
            if raw is not None:
                frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if frame_problem(frame) is None:
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    return frame
        try:
            src = camera_source(cam)
        except ValueError as exc:            # configuration problem: retrying will not help
            raise CaptureError("invalid_config", str(exc)) from None
        is_file = isinstance(src, str) and Path(src).is_file()
        last = {"code": "no_frame", "message": "no frame"}
        for i in range(cfg["attempts"]):
            res = grab_frame(src, out_path, timeout=cfg["timeout_sec"], random_seek=is_file)
            if res.get("ok"):
                frame = cv2.imread(str(out_path))
                if frame_problem(frame) is None:
                    return frame
                res = {"code": "invalid_frame", "message": frame_problem(frame)}
            last = res
            log.info("event=capture_attempt camera=%s attempt=%d/%d code=%s", cam["id"], i + 1, cfg["attempts"], res.get("code"))
            if i + 1 < cfg["attempts"] and not self._stop.is_set():
                self._stop.wait(min(cfg["backoff_sec"] * 2 ** i, 60))      # exponential backoff
        raise CaptureError(last.get("code", "error"), cctv.sanitize(str(last.get("message", ""))))

    def capture(self, cam_id: str, trigger: str = "manual") -> Optional[dict[str, Any]]:
        """Run one capture now. Returns the capture record, or None if one is already running."""
        lock = self._lock(cam_id)
        if not lock.acquire(blocking=False):
            return None
        try:
            return self._capture_locked(cam_id, trigger)
        finally:
            lock.release()

    def _capture_locked(self, cam_id: str, trigger: str) -> dict[str, Any]:
        from backend.services.pipeline import detect_image

        st = self.state
        cam = st.db.get_camera(cam_id)
        sched = self.get(cam_id)
        cfg = sched["config"]
        cap_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        out = st.cfg.storage_dir / "captures" / cam_id / cap_id
        t0 = time.perf_counter()
        st.db.insert_capture({"id": cap_id, "camera_id": cam_id, "ts": utcnow(), "trigger": trigger, "task": cfg["task"],
                              "status": "running"})
        result: dict[str, Any] = {"id": cap_id, "camera_id": cam_id, "trigger": trigger, "task": cfg["task"]}
        try:
            if cam is None:
                raise ValueError("camera not found")
            out.mkdir(parents=True, exist_ok=True)
            frame = self._grab(cam, cfg, out / "original.jpg")
            fields: dict[str, Any] = {"original_path": str(out / "original.jpg")}
            available, note = self._model_check(cfg)
            if not available:
                fields.update(status="skipped", error=note)
            else:
                engine = st.get_engine(cfg.get("model_version"))
                settings = st.settings(cam)
                # a scheduled screenshot is 2-3 h after the last one: litter that is still there must update
                # the existing alert (occurrences+1), not create a new one every capture
                window = max(settings.alerts.dedup_window_sec, cfg["max_minutes"] * 60 * 2)
                settings = settings.model_copy(update={"alerts": settings.alerts.model_copy(update={
                    "dedup_window_sec": window, "alert_on_image_uploads": True})})
                roi = st.roi_for(cam, cam_id) if cfg["task"] == "litter" else None
                res = detect_image(engine, frame, settings, roi, st.meta, st.db, st.alerts, cam_id, f"capture {cap_id}",
                                   job_id=f"schedule:{cam_id}")
                cv2.imwrite(str(out / "annotated.jpg"), res.pop("annotated"), [cv2.IMWRITE_JPEG_QUALITY, 90])
                fields.update(status="done", model_version=res["model_version"], annotated_path=str(out / "annotated.jpg"),
                              counts={**res["counts"], "by_class": res["by_class"],
                                      "highest_confidence": res.get("highest_confidence"),
                                      "average_confidence": res.get("average_confidence"),
                                      "outside_roi_ignored": res.get("outside_roi_ignored", 0)},
                              detections=res["detections"], alert_ids=[a["id"] for a in res["alerts"]])
                (out / "result.json").write_text(json.dumps({k: v for k, v in res.items()}, indent=1), encoding="utf-8")
        except CaptureError as exc:        # camera problem: recorded, inference skipped, never raised
            status = "invalid_frame" if exc.code == "invalid_frame" else "failed"
            fields = {"status": status, "error": f"{exc.code}: {exc.message}"}
            st.db.log_event("warning", "schedule", f"capture {status}: {exc.code} - {exc.message}", cam_id)
        except Exception as exc:  # noqa: BLE001 - e.g. model error: still never crash the scheduler
            log.exception("event=capture_error camera=%s capture=%s", cam_id, cap_id)
            fields = {"status": "failed", "error": cctv.sanitize(str(exc) or type(exc).__name__)}
            st.db.log_event("warning", "schedule", f"capture failed: {fields['error']}", cam_id)
        fields["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        st.db.update_capture(cap_id, **fields)
        sched = self.get(cam_id)
        sched.update(last_run_at=utcnow(), last_status=fields["status"], last_error=fields.get("error"))
        if fields["status"] in ("done", "skipped"):
            sched["last_success_at"] = sched["last_run_at"]
        log.info("event=capture camera=%s capture=%s trigger=%s status=%s duration_ms=%s litter=%s",
                 cam_id, cap_id, trigger, fields["status"], fields["duration_ms"],
                 (fields.get("counts") or {}).get("confirmed"))
        self._save(cam_id, sched)
        return {**result, **(st.db.get_capture(cap_id) or {})}
