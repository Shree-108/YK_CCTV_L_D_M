"""Application container: wires config, DB, engine, alerts, cameras and jobs."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from backend.alerts.manager import AlertManager, ClassMetadata
from backend.alerts.notifier import Notifier
from backend.cameras.manager import CameraManager, open_capture
from backend.core.config import AppConfig
from backend.core.settings import DetectionSettings, merged_settings
from backend.database.db import Database, utcnow
from backend.detection.engine import YoloEngine, hardware_profile
from backend.detection.filters import ROISet
from backend.services.model_registry import ModelRegistry
from backend.services.pipeline import StreamPipeline, detect_video

log = logging.getLogger(__name__)


class AppState:
    def __init__(self, cfg: AppConfig, capture_factory=open_capture) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self.started = time.time()
        self.db = Database(cfg.db_path)
        self.meta = ClassMetadata(cfg.class_metadata_path)
        self.registry = ModelRegistry(self.db, cfg.yolo_models_dir)
        self.registry.sync()
        self.engine: Optional[YoloEngine] = None
        self.engine_error: Optional[str] = None
        self._engine_lock = threading.Lock()
        self.notifier = Notifier(self._notify_prefs,
                                 on_error=lambda p, e: self.db.log_event("error", "notifier", f"{p}: {e}"),
                                 on_delivered=lambda p, a: self.db.log_event(
                                     "info", "notifier", f"alert {a.get('id')} delivered via {p}", a.get("camera_id")))
        self.alerts = AlertManager(self.db, cfg.alerts_dir, self.meta, self.notifier)
        self.jobs = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-job")
        # jobs are in-process threads: any job still queued/running belongs to a previous server process
        for j in self.db._query("SELECT id FROM jobs WHERE status IN ('queued','running')", ()):
            self.db.update_job(j["id"], status="failed", finished_at=utcnow(),
                               error="Interrupted: the server restarted during processing. Please upload the video again.")
        self.cameras = CameraManager(self._make_pipeline, self._live_fps, self._camera_event, capture_factory)
        self._hw: Optional[dict[str, Any]] = None
        self._pool: dict[str, YoloEngine] = {}
        from backend.core.config import PROJECT_ROOT
        from backend.services.datasets import DatasetService
        from backend.services.experiments import ExperimentService
        self.datasets = DatasetService(cfg.dataset_dir)
        self.experiments = ExperimentService([PROJECT_ROOT / "runs", PROJECT_ROOT / "backend" / "runs",
                                              cfg.storage_dir / "runs"], PROJECT_ROOT)
        self.load_engine()
        from backend.cameras.scheduler import CaptureScheduler
        self.scheduler = CaptureScheduler(self)
        self.scheduler.start_loop()

    # ------------------------------------------------------------------ settings
    def global_settings_json(self) -> dict[str, Any]:
        return self.db.get_setting("global", {}) or {}

    def settings(self, camera: Optional[dict[str, Any]] = None) -> DetectionSettings:
        return merged_settings(self.global_settings_json(), (camera or {}).get("settings"))

    def roi_for(self, camera: Optional[dict[str, Any]], camera_id: Optional[str] = None) -> Optional[ROISet]:
        """A registered camera's ROI; for a context without a camera row (e.g. "upload") the ROI saved
        with PUT /api/roi/{camera_id}."""
        if camera:
            return ROISet.from_json(camera.get("roi"))
        return self.upload_roi(camera_id) if camera_id else None

    def upload_roi(self, camera_id: str) -> Optional[ROISet]:
        raw = self.db.get_setting(f"roi:{camera_id}")
        return ROISet.from_json(raw) if raw else None

    def save_global_settings(self, patch: dict[str, Any]) -> DetectionSettings:
        from backend.core.settings import deep_merge
        new_json = deep_merge(self.global_settings_json(), patch)
        s = merged_settings(new_json)  # validates; raises on error
        self.db.set_setting("global", new_json)
        if self.engine:
            self.engine.update_settings(s)
        self.cameras.refresh_settings(lambda cam: (self.settings(cam), self.roi_for(cam)))
        self.db.log_event("info", "settings", "global detection settings updated", data=patch)
        return s

    # ------------------------------------------------------------------ engine
    def load_engine(self) -> None:
        m = self.registry.active()
        if m is None:
            self.engine_error = ("No YOLO model registered. Put a trained model in models/litter_yolo/<version>/ "
                                 "or run training/register_model.py")
            log.error(self.engine_error)
            return
        try:
            self.engine = YoloEngine(m["path"], self.settings(), m["version"])
            self.engine_error = None
        except Exception as exc:  # noqa: BLE001
            self.engine_error = f"Failed to load model {m['version']}: {exc}"
            log.exception(self.engine_error)

    def activate_model(self, version: str) -> dict[str, Any]:
        models = {m["version"]: m for m in self.db.list_models()}
        if version not in models:
            raise KeyError(version)
        path = models[version]["path"]
        with self._engine_lock:
            if self.engine is None:
                self.engine = YoloEngine(path, self.settings(), version)
            else:
                self.engine.swap(path, version)  # raises -> old model stays active
            self.db.set_active_model(version)
            self.engine_error = None
        self.db.log_event("info", "models", f"activated model {version}")
        # models may ship recommended thresholds (e.g. open-vocabulary scores are calibrated lower)
        import json as _json
        meta_f = Path(path).with_name("model_meta.json")
        rec = _json.loads(meta_f.read_text()).get("recommended_settings") if meta_f.is_file() else None
        if rec:
            self.save_global_settings(rec)
            self.db.log_event("info", "models", f"applied recommended settings of {version}", data=rec)
        return models[version]

    def require_engine(self) -> YoloEngine:
        if self.engine is None:
            raise RuntimeError(self.engine_error or "model not loaded")
        return self.engine

    def get_engine(self, version: Optional[str] = None) -> YoloEngine:
        """Active engine, or a cached engine for another registered version (Test Lab / evaluation).
        Non-active engines are kept in a small LRU so comparing models doesn't reload weights every call."""
        if not version or (self.engine is not None and version == self.engine.version):
            return self.require_engine()
        with self._engine_lock:
            if version in self._pool:
                self._pool[version] = self._pool.pop(version)   # LRU touch
                return self._pool[version]
            models = {m["version"]: m for m in self.db.list_models()}
            if version not in models:
                raise KeyError(version)
            eng = YoloEngine(models[version]["path"], self.settings(), version)
            self._pool[version] = eng
            while len(self._pool) > 3:
                self._pool.pop(next(iter(self._pool)))
            return eng

    def settings_with(self, camera: Optional[dict[str, Any]], patch: Optional[dict[str, Any]]) -> DetectionSettings:
        """Camera/global settings plus a per-request override (validated)."""
        from backend.core.settings import deep_merge
        base = deep_merge(self.global_settings_json(), (camera or {}).get("settings") or {})
        return merged_settings(deep_merge(base, patch or {}))

    # ------------------------------------------------------------------ evaluation jobs
    def submit_evaluation(self, version: Optional[str], dataset: str, split: str, compare_tiling: bool,
                          operating_threshold: Optional[float]) -> str:
        import json as _json

        from backend.services.evaluation import run_evaluation
        job_id = f"eval_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        out = self.cfg.storage_dir / "evaluations"
        out.mkdir(parents=True, exist_ok=True)
        self.db.create_job(job_id, "evaluation", None, f"{dataset}/{split}", str(out))

        def run() -> None:
            self.db.update_job(job_id, status="running")
            try:
                eng = self.get_engine(version)
                res = run_evaluation(eng, self.datasets, dataset, split, self.settings(), self.meta.report_groups,
                                     compare_tiling, True, operating_threshold,
                                     progress=lambda p: self.db.update_job(job_id, progress=round(p, 3)))
                hard = res.pop("_hard")
                for h in hard:
                    h["source"] = f"evaluation:{job_id}"
                self.db.insert_hard_examples(hard)
                res["id"] = job_id
                (out / f"{job_id}.json").write_text(_json.dumps(res), encoding="utf-8")
                self.db.update_job(job_id, status="done", finished_at=utcnow(), progress=1.0,
                                   result={"model_version": res["model_version"], "dataset": dataset, "split": split,
                                           "overall": res["overall"], "tiling_comparison": res["tiling_comparison"]})
                self.db.log_event("info", "evaluation", f"evaluation {job_id} finished for {res['model_version']}")
            except Exception as exc:  # noqa: BLE001
                log.exception("evaluation %s failed", job_id)
                self.db.update_job(job_id, status="failed", finished_at=utcnow(), error=str(exc))
                self.db.log_event("error", "evaluation", f"evaluation {job_id} failed: {exc}")

        self.jobs.submit(run)
        return job_id

    @property
    def hardware(self) -> dict[str, Any]:
        if self._hw is None:
            self._hw = hardware_profile()
        return self._hw

    # ------------------------------------------------------------------ cameras
    def _make_pipeline(self, cam: dict[str, Any]) -> StreamPipeline:
        return StreamPipeline(self.require_engine(), self.settings(cam), self.roi_for(cam), self.alerts, self.db,
                              cam["id"], "live", None, cam.get("location", ""))

    def _live_fps(self, cam: dict[str, Any]) -> float:
        return self.settings(cam).video.live_process_fps

    def _camera_event(self, level: str, cam_id: str, msg: str) -> None:
        self.db.log_event(level, "camera", msg, cam_id)

    def _notify_prefs(self, cam_id: str) -> dict[str, Any]:
        cam = self.db.get_camera(cam_id)
        return (cam or {}).get("notify", {})

    def sync_cameras(self) -> None:
        if self.cfg.start_cameras and self.engine is not None:
            self.cameras.sync(self.db.list_cameras())

    # ------------------------------------------------------------------ video jobs
    def submit_video(self, path: Path, camera_id: str, original_name: str,
                     overrides: Optional[dict[str, Any]] = None, model_version: Optional[str] = None,
                     use_roi: bool = True) -> str:
        job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        out_dir = self.cfg.videos_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        self.db.create_job(job_id, "video", camera_id, original_name, str(out_dir))
        cam = self.db.get_camera(camera_id)

        def run() -> None:
            self.db.update_job(job_id, status="running")
            try:
                res = detect_video(self.get_engine(model_version), path, out_dir, self.settings_with(cam, overrides),
                                   self.roi_for(cam, camera_id) if use_roi else None,
                                   self.db, self.alerts, camera_id, job_id,
                                   progress=lambda p, info: self.db.update_job(job_id, progress=round(p, 3),
                                                                               result={"progress": info}))
                self.db.update_job(job_id, status="done", finished_at=utcnow(), progress=1.0,
                                   result={"stats": res["stats"], "events": res["events"],
                                           "alert_ids": [a["id"] for a in res["alerts"]]})
            except Exception as exc:  # noqa: BLE001
                log.exception("video job %s failed", job_id)
                self.db.update_job(job_id, status="failed", finished_at=utcnow(), error=str(exc))
            finally:
                Path(path).unlink(missing_ok=True)   # uploads are temporary; outputs are kept

        self.jobs.submit(run)
        return job_id

    def shutdown(self) -> None:
        self.scheduler.shutdown()
        self.cameras.stop_all()
        self.jobs.shutdown(wait=False, cancel_futures=True)
