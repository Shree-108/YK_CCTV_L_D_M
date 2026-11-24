"""Background execution of four-screenshot analyses (the quick check; the full-video tracking analysis is the
primary video workflow).

One worker thread with its own queue (independent of the full-video job executor). Stages, in order:

    received -> validating -> sampling -> analysing -> generating_images -> generating_csv -> completed

States: queued | running | done | failed | cancelled | interrupted. (``awaiting_roi`` only appears on jobs stored
before ROI was removed from the application, 2026-11-19.)
A job is never run twice at the same time: one active job per uploaded video (HTTP 409 otherwise).
A server restart marks queued/running jobs ``interrupted``.

Modes (screenshot positions only - detection is identical, on the FULL frame, without any region):
  manual  four seeded random screenshots, one per quarter of the video
  auto    zero-click: fixed screenshots at 10/35/65/90 %; only one active automatic job per video *content* (SHA-256):
          a repeated upload of the same file while it is being analysed returns the running job.

Every completed job gets reports/detection_report.csv, reports/analysis_summary.csv/json (backend/sampling/
detection_report.py) and the standard litter outputs (backend/services/analysis_outputs.quick_analysis: detection
CSV, summary CSV, metadata, litter events), written before the job is marked done.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import cv2

from backend.alerts.manager import compute_severity
from backend.database.db import utcnow
from backend.sampling import SAMPLE_COUNT, detection_report, report
from backend.sampling.extract import Extracted, Stopped, extract
from backend.sampling.plan import (STRATEGY_FIXED, STRATEGY_RANDOM, Sample, new_seed, parse_recording_start,
                                   plan_fixed, plan_samples)
from backend.sampling.repo import SampledRepo
from backend.services import analysis_outputs, litter_store
from backend.services.pipeline import detect_image

if TYPE_CHECKING:
    from backend.services.state import AppState

log = logging.getLogger(__name__)

STAGES = ["received", "validating", "sampling", "analysing", "generating_images", "generating_csv", "completed"]
ACTIVE = ("queued", "running", "awaiting_roi")          # awaiting_roi: legacy jobs only (can still be cancelled)
ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")
STORE_RE = re.compile(r"^[A-Za-z0-9._ \-]{1,64}$")
REPORT_FILES = ("detection_report.csv", "analysis_summary.csv", "analysis_summary.json")


class DuplicateJob(Exception):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"An analysis of this video is already in progress ({job_id}).")
        self.job_id = job_id


class JobError(ValueError):
    """A request that cannot be accepted (HTTP 4xx with this message)."""


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", Path(name).stem)[:40] or "video"


class SampledRunner:
    def __init__(self, state: "AppState") -> None:
        self.state = state
        self.repo = SampledRepo(state.db)
        self._q: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._lock = threading.Lock()
        self._cancel: set[str] = set()
        for j in self.repo.jobs_in_states(("queued", "running")):
            self.repo.update_job(j["id"], state="interrupted", finished_at=utcnow(),
                                 error="Interrupted: the server restarted during processing. Start a new analysis.")
        self._thread = threading.Thread(target=self._loop, name="sampled-worker", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ public API
    def create(self, media_id: str, *, store_id: Optional[str] = None, camera_id: Optional[str] = None,
               recording_start: Optional[str] = None, seed: Optional[int] = None, create_alerts: bool = True,
               model_version: Optional[str] = None, new_run: bool = False,
               mode: str = "manual") -> tuple[dict[str, Any], bool]:
        """(job, reused). Raises JobError (bad input), DuplicateJob (409), KeyError (unknown media).
        ``mode="auto"``: zero-click job (fixed screenshots)."""
        if mode not in ("manual", "auto"):
            raise JobError("mode must be manual or auto.")
        media = self.repo.media(media_id)
        if media is None or not Path(media["path"]).is_file():
            raise KeyError(media_id)
        store_id, camera_id = (store_id or "").strip() or None, (camera_id or "").strip() or None
        if store_id and not STORE_RE.match(store_id):
            raise JobError("Store ID may contain letters, digits, spaces, '.', '-' and '_' (max 64).")
        if camera_id and not ID_RE.match(camera_id):
            raise JobError("Camera ID may contain letters, digits, '-' and '_' (max 40).")
        try:                       # a registered camera's store wins; a conflicting store is refused (never mixed)
            ident = litter_store.resolve_identity(self.state.db, store_id, camera_id)
        except litter_store.IdentityError as exc:
            raise JobError(str(exc)) from None
        if ident.store_id != "UNKNOWN":
            store_id = ident.store_id
        try:
            start = parse_recording_start(recording_start)
        except ValueError:
            raise JobError("Recording start must be a date and time such as 2026-10-29T18:30:09.") from None
        if seed is not None and not (0 < seed < 2 ** 31):
            raise JobError("seed must be between 1 and 2147483647.")
        auto = mode == "auto"
        with self._lock:
            active = self.repo.active_job_for_media(media_id)
            if active:
                raise DuplicateJob(active["id"])
            if auto:                              # same file content uploaded again while it is being analysed
                same = self.repo.active_auto_job_for_sha(media["sha256"])
                if same:
                    raise DuplicateJob(same["id"])
            if seed is not None and not new_run and not auto:
                same = self.repo.jobs("WHERE j.media_id=? AND j.seed=? AND j.state='done' AND IFNULL(j.store_id,'')=? "
                                      "AND IFNULL(j.camera_id,'')=? AND IFNULL(j.recording_start,'')=?",
                                      (media_id, seed, store_id or "", camera_id or "",
                                       start.isoformat() if start else ""), limit=1)
                if same:
                    return same[0], True
            seed = 0 if auto else (seed or new_seed())          # 0 = no random draw (fixed positions)
            try:
                samples = (plan_fixed(media["frames"], media["fps"]) if auto
                           else plan_samples(media["frames"], media["fps"], SAMPLE_COUNT, seed))
            except ValueError as exc:
                raise JobError(str(exc)) from None
            job_id = f"sv_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            out = self.state.cfg.sampled_dir / job_id
            out.mkdir(parents=True, exist_ok=False)          # a job folder is never reused
            self.repo.insert_job({
                "id": job_id, "media_id": media_id, "created_at": utcnow(), "stage": "received", "state": "queued",
                "store_id": store_id, "camera_id": camera_id, "recording_start": start.isoformat() if start else None,
                "seed": seed, "sample_count": SAMPLE_COUNT, "roi_mode": "none",
                "auto_accept": 0, "create_alerts": int(create_alerts),
                "model_version": model_version, "output_dir": str(out), "roi_status": "not_used",
                "mode": mode, "sampling": STRATEGY_FIXED if auto else STRATEGY_RANDOM})
            self.repo.insert_frames([{"job_id": job_id, "idx": s.idx, "frame_idx": s.frame,
                                      "video_time_sec": s.seconds, "status": "planned"} for s in samples])
            self.repo.update_job(job_id, counts={"plan": [s.as_dict() for s in samples]})
        self.state.db.log_event("info", "sampled", f"{'automatic ' if auto else ''}four-screenshot analysis "
                                f"{job_id} queued", camera_id)
        self._q.put(("start", job_id))
        return self.repo.job(job_id), False   # type: ignore[return-value]

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.repo.job(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["state"] not in ACTIVE:
                raise JobError(f"This analysis has already finished (state: {job['state']}).")
            self._cancel.add(job_id)
            if job["state"] in ("queued", "awaiting_roi"):
                self.repo.update_job(job_id, state="cancelled", finished_at=utcnow(), error="Cancelled by the operator.")
        return self.repo.job(job_id)  # type: ignore[return-value]

    def detail(self, job_id: str) -> Optional[dict[str, Any]]:
        job = self.repo.job(job_id)
        if job is None:
            return None
        job["media"] = self.repo.media(job["media_id"])
        job["frames"] = self.repo.frames(job_id)
        return job

    def wait_idle(self, timeout: float = 60.0) -> None:
        """Tests: block until the queue is drained."""
        end = time.time() + timeout
        while time.time() < end:
            if self._q.unfinished_tasks == 0:
                return
            time.sleep(0.05)
        raise TimeoutError("sampled worker still busy")

    # ------------------------------------------------------------------ worker
    def _loop(self) -> None:
        while True:
            kind, job_id = self._q.get()
            try:
                if job_id in self._cancel and (j := self.repo.job(job_id)) and j["state"] == "cancelled":
                    continue
                (self._start if kind == "start" else self._analyse)(job_id)
            except Stopped:
                self.repo.update_job(job_id, state="cancelled", finished_at=utcnow(), error="Cancelled by the operator.")
            except Exception as exc:  # noqa: BLE001
                log.exception("sampled job %s failed", job_id)
                self.repo.update_job(job_id, state="failed", finished_at=utcnow(), error=str(exc))
                self.state.db.log_event("error", "sampled", f"four-screenshot analysis {job_id} failed: {exc}")
            finally:
                self._cancel.discard(job_id)
                self._q.task_done()

    def _stage(self, job_id: str, stage: str, **extra: Any) -> None:
        self.repo.update_job(job_id, stage=stage, **extra)

    def _stop(self, job_id: str) -> Callable[[], bool]:
        return lambda: job_id in self._cancel

    def _check_cancel(self, job_id: str) -> None:
        """Stop at a stage boundary if the operator cancelled while the previous stage was running."""
        if job_id in self._cancel:
            raise Stopped()

    def _start(self, job_id: str) -> None:
        job = self.repo.job(job_id)
        if job is None or job["state"] != "queued":
            return
        media = self.repo.media(job["media_id"])
        self._stage(job_id, "validating", state="running", started_at=utcnow())
        if media is None or not Path(media["path"]).is_file():
            raise RuntimeError("The uploaded video is no longer on disk.")
        cap = cv2.VideoCapture(media["path"])
        ok = cap.isOpened()
        cap.release()
        if not ok:
            raise RuntimeError("The video could not be opened (missing codec or damaged file).")

        self._stage(job_id, "sampling")
        frames = self.repo.frames(job_id)
        windows = {p["idx"]: tuple(p["segment"]) for p in (job.get("counts") or {}).get("plan") or []}
        samples = [Sample(f["idx"], f["frame_idx"], f["video_time_sec"], windows.get(f["idx"], (0, 0))) for f in frames]
        adjust = job.get("sampling") == STRATEGY_FIXED

        def saved(e: Extracted) -> None:
            moved = e.frame != e.sample.frame
            self.repo.update_frame(job_id, e.sample.idx, status=e.status, video_time_sec=e.seconds,
                                   original_path=str(e.path) if e.path else None, error=e.error,
                                   frame_idx=e.frame, planned_frame_idx=e.sample.frame if moved else None,
                                   sampling_note=e.note,
                                   outcome=None if e.status == "saved" else ("UNREADABLE" if e.status == "unreadable" else "FAILED"))

        results = extract(Path(media["path"]), media["fps"], samples, Path(job["output_dir"]) / "frames",
                          _safe_stem(media["filename"]), self._stop(job_id), saved, adjust=adjust)
        images = {e.sample.idx: e.image for e in results if e.status == "saved" and e.image is not None}
        del results                                       # release the decoded pictures early (low-RAM machines)
        if not images:
            raise RuntimeError("None of the four screenshots could be read from this video "
                               "(see the per-screenshot errors).")

        images.clear()
        self._check_cancel(job_id)
        self._analyse(job_id, internal=True)

    def _analyse(self, job_id: str, internal: bool = False) -> None:
        """``internal``: called by _start right after sampling (the job is already running)."""
        job = self.repo.job(job_id)
        if job is None or job["state"] != ("running" if internal else "queued"):
            return
        media = self.repo.media(job["media_id"])
        self._stage(job_id, "analysing", state="running")
        engine = self.state.get_engine(job.get("model_version"))
        cam = self.state.db.get_camera(job["camera_id"]) if job.get("camera_id") else None
        settings = self.state.settings_with(cam, None)
        meta = self.state.meta
        out = Path(job["output_dir"])
        (out / "annotated").mkdir(exist_ok=True)
        location = " · ".join(x for x in (job.get("store_id"), job.get("camera_id")) if x)
        totals = {"screenshots_planned": job["sample_count"], "screenshots_extracted": 0, "screenshots_analysed": 0,
                  "litter": 0, "by_class": {}, "alerts": 0, "outcomes": {}}
        annotated_frames: list[tuple[int, Any, Path]] = []
        for f in self.repo.frames(job_id):
            if job_id in self._cancel:
                raise Stopped()
            if f["status"] == "saved":
                totals["screenshots_extracted"] += 1
            if f["status"] != "saved":
                continue
            try:
                img = cv2.imread(f["original_path"])
                if img is None:
                    raise RuntimeError("the saved screenshot could not be read back")
                res = detect_image(engine, img, settings, meta, self.state.db,
                                   self.state.alerts if job["create_alerts"] else None,
                                   job.get("camera_id") or "upload", Path(f["original_path"]).name, job_id,
                                   source_type="sample", frame_idx=f["frame_idx"], video_time_sec=f["video_time_sec"],
                                   session_id=f"{job_id}:s{f['idx']}", create_alerts=bool(job["create_alerts"]),
                                   location=location,
                                   header=(f"Store: {job.get('store_id') or 'UNKNOWN'}  Camera: "
                                           f"{job.get('camera_id') or 'UNKNOWN'}",
                                           f"Screenshot {f['idx']}  video time {f['video_time_sec'] or 0:.1f} s"))
                annotated = res.pop("annotated")
                alerts = res.pop("alerts")
                res["alert_ids"] = [a["id"] for a in alerts]
                n = res["counts"]["confirmed"]
                alert_of = {a.get("track_id"): a["id"] for a in alerts}
                for d in res["detections"]:
                    if d.get("status") == "confirmed":        # existing severity rules (Settings -> severity)
                        d["severity"] = compute_severity(d["class"], d["confidence"], n, settings, meta)
                        if d.get("track_id") in alert_of:
                            d["alert_id"] = alert_of[d["track_id"]]
                outcome = "LITTER_DETECTED" if n else "NO_LITTER"
                self.repo.update_frame(job_id, f["idx"], outcome=outcome, detection_count=n, result=res, error=None)
                annotated_frames.append((f["idx"], annotated, Path(f["original_path"])))
                totals["screenshots_analysed"] += 1
                totals["alerts"] += len(alerts)
                totals["litter"] += n
                for k, v in res["by_class"].items():
                    totals["by_class"][k] = totals["by_class"].get(k, 0) + v
            except Exception as exc:  # noqa: BLE001 - one bad screenshot does not fail the others
                log.exception("sampled job %s screenshot %s failed", job_id, f["idx"])
                self.repo.update_frame(job_id, f["idx"], outcome="FAILED", error=str(exc))

        self._stage(job_id, "generating_images")
        for idx, img, orig in annotated_frames:
            p = out / "annotated" / f"{orig.stem}_annotated.jpg"
            if cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                self.repo.update_frame(job_id, idx, annotated_path=str(p))

        self._stage(job_id, "generating_csv")
        frames = self.repo.frames(job_id)
        for f in frames:
            totals["outcomes"][f.get("outcome") or "NONE"] = totals["outcomes"].get(f.get("outcome") or "NONE", 0) + 1
        # one estimated object per piece of litter seen in several screenshots (existing alert de-dup thresholds)
        objects = detection_report.link_objects(
            frames, iou_min=settings.alerts.dedup_iou, center_frac=settings.alerts.dedup_center_frac,
            similar=lambda a, b: meta.similar(a, b, settings.alerts.class_match))
        for f in frames:
            if f.get("result"):
                self.repo.update_frame(job_id, f["idx"], result=f["result"])     # now with object ids
        job = self.repo.job(job_id) or job
        job["model_version"] = engine.version
        rows = report.rows_for_job(job, media or {}, frames, self.state.cfg.storage_dir)
        csv_path = out / f"litter_report_{job_id}.csv"
        report.write_atomic(csv_path, report.to_csv(rows))
        prev = job.get("counts") or {}
        failed_all = totals["screenshots_analysed"] == 0 and any(f.get("outcome") == "FAILED" for f in frames)
        completed_at = utcnow()
        det_rows, summary = detection_report.build(job, media or {}, frames, objects,
                                                   completed_at=completed_at, failed=failed_all)
        rdir = out / "reports"
        report.write_atomic(rdir / "detection_report.csv",
                            detection_report.to_csv(det_rows, detection_report.DETECTION_FIELDS))
        report.write_atomic(rdir / "analysis_summary.csv",
                            detection_report.to_csv([summary], detection_report.SUMMARY_FIELDS))
        report.write_atomic(rdir / "analysis_summary.json", detection_report.summary_json(summary, objects, frames))
        ident = litter_store.resolve_identity(self.state.db, job.get("store_id"), job.get("camera_id"),
                                              recording_id=job_id)
        try:
            rec_start = parse_recording_start(job.get("recording_start"))
        except ValueError:
            rec_start = None
        standard = analysis_outputs.quick_analysis(
            self.state.db, self.state.cfg.storage_dir, job=job, frames=self.repo.frames(job_id), objects=objects,
            ident=ident, model_version=engine.version, recording_start=rec_start, failed=failed_all)
        self.repo.update_job(job_id, stage="completed" if not failed_all else "generating_csv",
                             state="failed" if failed_all else "done", finished_at=completed_at, csv_path=str(csv_path),
                             model_version=engine.version,
                             counts={**totals, "plan": prev.get("plan"), "summary": summary,
                                     "standard_outputs": standard, "identity": ident.as_dict()},
                             error="No screenshot could be analysed (see the per-screenshot errors)." if failed_all else None)
        self.state.db.log_event("info", "sampled", f"four-screenshot analysis {job_id} finished: "
                                f"{totals['litter']} litter detection(s)", job.get("camera_id"))
        if not failed_all:      # ai_garbage_detection import CSV; only when LDM_GARBAGE_EXPORT=on
            try:                # never lets an export problem reach the job's error handler (job stays "done")
                from backend.services.export_guard import safe_auto_export
                safe_auto_export(self.state, kind="sampled", source_id=job_id, out_dir=rdir)
            except Exception:  # noqa: BLE001
                log.exception("ai_garbage_detection export guard failed for %s", job_id)
            # result bundle: 4 annotated images + import CSV + manifest (always; after the flat export above, so a
            # check that export already wrote is not written again). Never fails the job.
            try:
                from backend.services.export_guard import safe_result_bundle
                safe_result_bundle(self.state, job_id=job_id, out_dir=rdir)
            except Exception:  # noqa: BLE001
                log.exception("result bundle guard failed for %s", job_id)
