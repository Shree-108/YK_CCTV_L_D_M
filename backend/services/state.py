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
        from backend.services.roi_service import RoiService
        # automatic monitored region (pavement ROI): config/camera_roi.json, reference frames in storage/auto_roi
        self.roi = RoiService(cfg.roi_store_path, cfg.storage_dir / "auto_roi", enabled=cfg.roi_mode == "on",
                              legacy_path=cfg.storage_dir / "camera_roi.json")
        self.roi_warmup: dict[str, Any] = {"state": "not started"}
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
        from backend.sampling.runner import SampledRunner
        self.sampled = SampledRunner(self)
        from backend.cameras.scheduler import CaptureScheduler
        self.scheduler = CaptureScheduler(self)
        self.scheduler.start_loop()

    # ------------------------------------------------------------------ settings
    def global_settings_json(self) -> dict[str, Any]:
        return self.db.get_setting("global", {}) or {}

    def settings(self, camera: Optional[dict[str, Any]] = None) -> DetectionSettings:
        return merged_settings(self.global_settings_json(), (camera or {}).get("settings"))

    def save_global_settings(self, patch: dict[str, Any]) -> DetectionSettings:
        from backend.core.settings import deep_merge
        new_json = deep_merge(self.global_settings_json(), patch)
        s = merged_settings(new_json)  # validates; raises on error
        self.db.set_setting("global", new_json)
        if self.engine:
            self.engine.update_settings(s)
        self.cameras.refresh_settings(self.settings)
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
        from backend.services.live_events import LiveEventRecorder
        from backend.services.roi_service import LiveRoi
        roi = LiveRoi(self.roi, cam["id"], lambda: self.settings(self.db.get_camera(cam["id"]) or cam)) \
            if self.roi.enabled else None
        return StreamPipeline(self.require_engine(), self.settings(cam), self.alerts, self.db, cam["id"], "live", None,
                              cam.get("location", ""), event_sink=LiveEventRecorder(self.db, cam["id"]), roi=roi)

    # ------------------------------------------------------------------ monitored region (ROI)
    def roi_for_frames(self, camera_id: Optional[str], frames: Any, settings: DetectionSettings,
                       request_polygons: Any = None) -> Any:
        """RoiContext for one image / a few frames (cached per camera; None when ROI is off)."""
        return self.roi.resolve(camera_id, frames, settings, request_polygons)

    def warmup_roi(self) -> None:
        """App start (background thread): load the segmenter and build missing/stale ROIs of configured cameras that
        are not streaming (streaming cameras build theirs from the live frames)."""
        from backend.services.roi_service import frames_from_source

        def frames_for(cam: dict[str, Any]) -> list[Any]:
            if cam["id"] in self.cameras.workers:
                return []
            return frames_from_source(cam, self.settings(cam).roi)

        self.roi_warmup = {"state": "running"}
        try:
            cams = [c for c in self.db.list_cameras() if c.get("enabled", True) and c["id"] not in self.cameras.workers]
            res = self.roi.warmup(self.settings(), cams, frames_for)
            self.roi_warmup = {"state": "done", **res}
            if res.get("error"):
                self.db.log_event("error", "roi", f"ROI model not available: {res['error']}")
            else:
                self.db.log_event("info", "roi", f"ROI ready ({res.get('model')}); {len(res['cameras'])} camera(s)"
                                  " checked at start-up")
        except Exception as exc:  # noqa: BLE001 - start-up must never fail because of the ROI
            log.exception("ROI warm-up failed")
            self.roi_warmup = {"state": "failed", "error": str(exc)}

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
    def submit_video(self, path: Path, ident: Any, original_name: str,
                     overrides: Optional[dict[str, Any]] = None, model_version: Optional[str] = None,
                     recording_start: Optional[Any] = None, roi_polygons: Any = None) -> str:
        """Full-video analysis on the full frame with tracking. ``ident``: litter_reports.Identity (resolved and
        validated by the caller). Writes the standard outputs (backend/services/analysis_outputs.py); a failed job
        gets status FAILED and no CSV."""
        from backend.services import analysis_outputs as ao
        from backend.services import video_evidence as ve

        job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        out_dir = self.cfg.videos_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        camera_id = ident.camera_id if ident.camera_id != "UNKNOWN" else "upload"
        self.db.create_job(job_id, "video", camera_id, original_name, str(out_dir))
        cam = self.db.get_camera(camera_id)

        def run() -> None:
            self.db.update_job(job_id, status="running")
            aid = paths = started = None
            settings = self.settings_with(cam, overrides)
            summary = {"confidence": settings.confidence.model_dump(), "sample_fps": settings.video.sample_fps,
                       "video_tiling": settings.video.tiling, "image_size": settings.inference.image_size,
                       "tracking": settings.tracking.model_dump(), "overrides": overrides or {}}
            try:
                engine = self.get_engine(model_version)
                aid, paths, started = ao.start_video(self.db, self.cfg.storage_dir, ident, job_id, engine.version,
                                                     recording_start)
                # monitored region: ~30 frames of the first ~10 s, majority vote (cached per camera)
                self.db.update_job(job_id, result={"progress": {"stage": "monitored region"}, "analysis_id": aid})
                roi = self.roi.for_video(camera_id, path, settings, roi_polygons)
                res = detect_video(engine, path, out_dir, settings, self.db, self.alerts, camera_id, job_id,
                                   progress=lambda p, info: self.db.update_job(job_id, progress=round(p, 3),
                                                                               result={"progress": info,
                                                                                       "analysis_id": aid}),
                                   evidence_dir=paths.sub("annotated"),
                                   header=ao.header_lines(ident, started), roi=roi)
                # four evidence screenshots: needs the uploaded file, so before keep_original() moves it
                ev_root = ve.evidence_dir(self.cfg.storage_dir, aid)
                shots = ve.build_screenshots(engine=engine, settings=settings, video_path=Path(path), res=res,
                                             analysis_id=aid, out_dir=ev_root, header=ao.header_lines(ident, started),
                                             recording_start=recording_start, seed=settings.video.screenshot_seed,
                                             roi=roi)
                ref = ao.keep_original(Path(path), paths, original_name, settings.video.keep_original_video)
                out = ao.finish_video(self.db, aid, paths, ident, job_id=job_id, res=res,
                                      recording_start=recording_start, started=started, video_ref=ref,
                                      settings_summary=summary, screenshots=shots, evidence_root=ev_root,
                                      candidate_threshold=settings.confidence.candidate_threshold,
                                      review_min_observations=settings.tracking.min_confirmations)
                self.db.update_job(job_id, status="done", finished_at=utcnow(), progress=1.0,
                                   result={"stats": res["stats"], "events": res["events"],
                                           "alert_ids": [a["id"] for a in res["alerts"]],
                                           **({"roi": res["stats"]["roi"]} if res["stats"].get("roi") else {}),
                                           "identity": ident.as_dict(), "video_validation": res["validation"], **out})
                self.db.log_event("info", "video", f"video analysis {job_id} finished: {out['litter']['unique_events']}"
                                  f" litter event(s), {out['litter']['frame_observations']} observation(s)",
                                  camera_id)
                # ai_garbage_detection import CSV; only when LDM_GARBAGE_EXPORT=on. Never lets an export problem reach
                # the except below, which would mark this finished job as failed.
                try:
                    from backend.services.export_guard import safe_auto_export
                    safe_auto_export(self, kind="video", source_id=aid, out_dir=ev_root)
                except Exception:  # noqa: BLE001
                    log.exception("ai_garbage_detection export guard failed for %s", aid)
            except Exception as exc:  # noqa: BLE001
                log.exception("video job %s failed", job_id)
                if aid is not None:
                    ao.fail(self.db, aid, paths, ident, getattr(self.engine, "version", ""), str(exc),
                            {"analysis_id": aid, "kind": "video", "job_id": job_id})
                self.db.update_job(job_id, status="failed", finished_at=utcnow(), error=str(exc),
                                   result={"identity": ident.as_dict(),
                                           "analysis": {"analysis_id": aid, "processing_status": "FAILED"}})
            finally:
                Path(path).unlink(missing_ok=True)   # moved into LDM_OUTPUT/.../input when kept

        self.jobs.submit(run)
        return job_id

    def shutdown(self) -> None:
        self.scheduler.shutdown()
        self.cameras.stop_all()
        self.jobs.shutdown(wait=False, cancel_futures=True)
