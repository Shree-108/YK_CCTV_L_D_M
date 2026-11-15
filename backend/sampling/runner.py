"""Background execution of four-screenshot analyses.

One worker thread with its own queue (independent of the full-video job executor). Stages, in order:

    received -> validating -> sampling -> identifying_roi -> roi_confirmation (pauses: state awaiting_roi)
             -> analysing -> generating_images -> generating_csv -> completed

States: queued | running | awaiting_roi | done | failed | cancelled | interrupted.
A job is never run twice at the same time: one active job per uploaded video (HTTP 409 otherwise).
A server restart marks queued/running jobs ``interrupted``; jobs waiting for ROI confirmation keep waiting
(their screenshots are on disk).
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

from backend.database.db import utcnow
from backend.detection.filters import ROISet, ROIZone
from backend.sampling import SAMPLE_COUNT, report
from backend.sampling.extract import Extracted, Stopped, extract
from backend.sampling.plan import Sample, new_seed, parse_recording_start, plan_samples
from backend.sampling.repo import SampledRepo
from backend.sampling.roi_proposal import AUTO_ACCEPT, Proposal, Segmenter, propose
from backend.services.pipeline import detect_image

if TYPE_CHECKING:
    from backend.services.state import AppState

log = logging.getLogger(__name__)

STAGES = ["received", "validating", "sampling", "identifying_roi", "roi_confirmation", "analysing",
          "generating_images", "generating_csv", "completed"]
ACTIVE = ("queued", "running", "awaiting_roi")
ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")
STORE_RE = re.compile(r"^[A-Za-z0-9._ \-]{1,64}$")


class DuplicateJob(Exception):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"An analysis of this video is already in progress ({job_id}).")
        self.job_id = job_id


class JobError(ValueError):
    """A request that cannot be accepted (HTTP 4xx with this message)."""


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", Path(name).stem)[:40] or "video"


class SampledRunner:
    def __init__(self, state: "AppState", segmenter: Callable[[], Optional[Segmenter]] = lambda: None) -> None:
        self.state = state
        self.repo = SampledRepo(state.db)
        self._segmenter = segmenter
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
               recording_start: Optional[str] = None, seed: Optional[int] = None, roi_mode: str = "auto",
               auto_accept: bool = False, create_alerts: bool = True, model_version: Optional[str] = None,
               new_run: bool = False) -> tuple[dict[str, Any], bool]:
        """(job, reused). Raises JobError (bad input), DuplicateJob (409), KeyError (unknown media)."""
        media = self.repo.media(media_id)
        if media is None or not Path(media["path"]).is_file():
            raise KeyError(media_id)
        store_id, camera_id = (store_id or "").strip() or None, (camera_id or "").strip() or None
        if store_id and not STORE_RE.match(store_id):
            raise JobError("Store ID may contain letters, digits, spaces, '.', '-' and '_' (max 64).")
        if camera_id and not ID_RE.match(camera_id):
            raise JobError("Camera ID may contain letters, digits, '-' and '_' (max 40).")
        try:
            start = parse_recording_start(recording_start)
        except ValueError:
            raise JobError("Recording start must be a date and time such as 2026-10-29T18:30:09.") from None
        if roi_mode not in ("auto", "camera", "manual"):
            raise JobError("roi_mode must be auto, camera or manual.")
        if seed is not None and not (0 < seed < 2 ** 31):
            raise JobError("seed must be between 1 and 2147483647.")
        with self._lock:
            active = self.repo.active_job_for_media(media_id)
            if active:
                raise DuplicateJob(active["id"])
            if seed is not None and not new_run:
                same = self.repo.jobs("WHERE j.media_id=? AND j.seed=? AND j.state='done' AND IFNULL(j.store_id,'')=? "
                                      "AND IFNULL(j.camera_id,'')=? AND IFNULL(j.recording_start,'')=?",
                                      (media_id, seed, store_id or "", camera_id or "",
                                       start.isoformat() if start else ""), limit=1)
                if same:
                    return same[0], True
            seed = seed or new_seed()
            try:
                samples = plan_samples(media["frames"], media["fps"], SAMPLE_COUNT, seed)
            except ValueError as exc:
                raise JobError(str(exc)) from None
            job_id = f"sv_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            out = self.state.cfg.sampled_dir / job_id
            out.mkdir(parents=True, exist_ok=False)          # a job folder is never reused
            self.repo.insert_job({
                "id": job_id, "media_id": media_id, "created_at": utcnow(), "stage": "received", "state": "queued",
                "store_id": store_id, "camera_id": camera_id, "recording_start": start.isoformat() if start else None,
                "seed": seed, "sample_count": SAMPLE_COUNT, "roi_mode": roi_mode, "auto_accept": int(auto_accept),
                "create_alerts": int(create_alerts), "model_version": model_version, "output_dir": str(out),
                "roi_status": "pending"})
            self.repo.insert_frames([{"job_id": job_id, "idx": s.idx, "frame_idx": s.frame,
                                      "video_time_sec": s.seconds, "status": "planned"} for s in samples])
            self.repo.update_job(job_id, counts={"plan": [s.as_dict() for s in samples]})
        self.state.db.log_event("info", "sampled", f"four-screenshot analysis {job_id} queued", camera_id)
        self._q.put(("start", job_id))
        return self.repo.job(job_id), False   # type: ignore[return-value]

    def confirm_roi(self, job_id: str, zones: list[dict], *, frame_zones: Optional[dict[int, list[dict]]] = None,
                    skip_frames: Optional[list[int]] = None, whole_frame: bool = False) -> dict[str, Any]:
        job = self.repo.job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["state"] != "awaiting_roi":
            raise JobError(f"This analysis is not waiting for ROI confirmation (state: {job['state']}).")
        frames = {f["idx"]: f for f in self.repo.frames(job_id)}
        skip = set(skip_frames or [])
        frame_zones = {int(k): v for k, v in (frame_zones or {}).items()}
        bad = (skip | set(frame_zones)) - set(frames)
        if bad:
            raise JobError(f"Unknown screenshot number(s): {sorted(bad)}.")
        clean = [ROIZone(**z).model_dump() for z in zones]
        per = {i: [ROIZone(**z).model_dump() for z in zs] for i, zs in frame_zones.items()}
        has_include = lambda zs: any(z["mode"] == "include" for z in zs)   # noqa: E731
        usable = [i for i, f in frames.items() if f["status"] == "saved" and i not in skip]
        if not usable:
            raise JobError("Every readable screenshot was skipped: nothing would be analysed.")
        if not whole_frame:
            missing = [i for i in usable if not has_include(per.get(i, clean))]
            if missing:
                raise JobError("Draw the outdoor ground area (an include polygon) for screenshot(s) "
                               f"{missing}, skip them, or explicitly choose to analyse the whole frame.")
        proposal = (job.get("proposal") or {}).get("proposal") or {}
        prefill = (job.get("proposal") or {}).get("prefill") or []
        source = ("whole_frame" if whole_frame else
                  (job.get("proposal") or {}).get("prefill_source") if (clean == prefill and not per) else "manual")
        with self._lock:
            for i, f in frames.items():
                if i in skip:
                    self.repo.update_frame(job_id, i, roi=None, roi_status="uncertain")
                elif i in per:
                    self.repo.update_frame(job_id, i, roi=per[i], roi_status="confirmed")
                elif f["status"] == "saved":
                    self.repo.update_frame(job_id, i, roi=None, roi_status="confirmed")
            self.repo.update_job(job_id, roi=[] if whole_frame else clean, roi_source=source, roi_status="confirmed",
                                 roi_confidence=proposal.get("confidence") if source == "auto" else None,
                                 state="queued", stage="analysing")
        self.state.db.log_event("info", "sampled", f"ROI confirmed for analysis {job_id} ({source})", job.get("camera_id"))
        self._q.put(("analyse", job_id))
        return self.repo.job(job_id)  # type: ignore[return-value]

    def cancel(self, job_id: str) -> dict[str, Any]:
        # Decided under the lock: the worker moves a job from running to awaiting_roi under the same lock after a
        # last cancel check (_await_roi), so a cancel can never fall between the two and be lost.
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

    def _await_roi(self, job_id: str) -> None:
        """running -> awaiting_roi, unless a cancel arrived first (checked under the lock cancel() also takes)."""
        with self._lock:
            self._check_cancel(job_id)
            self._stage(job_id, "roi_confirmation", state="awaiting_roi")

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
        samples = [Sample(f["idx"], f["frame_idx"], f["video_time_sec"], (0, 0)) for f in frames]

        def saved(e: Extracted) -> None:
            self.repo.update_frame(job_id, e.sample.idx, status=e.status, video_time_sec=e.seconds,
                                   original_path=str(e.path) if e.path else None, error=e.error,
                                   outcome=None if e.status == "saved" else ("UNREADABLE" if e.status == "unreadable" else "FAILED"))

        results = extract(Path(media["path"]), media["fps"], samples, Path(job["output_dir"]) / "frames",
                          _safe_stem(media["filename"]), self._stop(job_id), saved)
        images = {e.sample.idx: e.image for e in results if e.status == "saved" and e.image is not None}
        if not images:
            raise RuntimeError("None of the four screenshots could be read from this video "
                               "(see the per-screenshot errors).")

        self._check_cancel(job_id)
        self._stage(job_id, "identifying_roi")
        try:
            proposal = propose(images, self._segmenter())
        except Exception as exc:  # noqa: BLE001 - a failing segmenter must not block manual confirmation
            log.exception("ROI proposal failed for %s", job_id)
            from backend.sampling.roi_proposal import unavailable
            proposal = unavailable(f"Automatic ROI identification failed: {exc}")
        self._check_cancel(job_id)          # ROI identification is the slow step: honour a cancel sent during it
        self._save_proposal_images(job, proposal, images)
        prefill, prefill_source = self._prefill(job, proposal)
        status = proposal.status if proposal.available else "pending"
        self.repo.update_job(job_id, proposal={"proposal": proposal.as_dict(), "prefill": prefill,
                                               "prefill_source": prefill_source},
                             roi_status="proposed" if status == "proposed" else "uncertain" if status == "uncertain" else "pending",
                             roi_confidence=proposal.confidence)
        # per-frame proposals (moved camera) stay in the proposal record: only a confirmation writes frame ROIs
        if (job["auto_accept"] and job["roi_mode"] == "auto" and proposal.status == "proposed"
                and (proposal.confidence or 0) >= AUTO_ACCEPT):
            self._await_roi(job_id)
            self.confirm_roi(job_id, proposal.zones)
            self.repo.update_job(job_id, roi_source="auto", roi_confidence=proposal.confidence)
            return
        self._await_roi(job_id)
        self.state.db.log_event("info", "sampled", f"analysis {job_id} is waiting for ROI confirmation", job.get("camera_id"))

    def _prefill(self, job: dict[str, Any], proposal: Proposal) -> tuple[list[dict], Optional[str]]:
        saved = self.state.roi_for(self.state.db.get_camera(job["camera_id"]) if job.get("camera_id") else None,
                                   job.get("camera_id") or "upload")
        saved_zones = [z.model_dump() for z in saved.zones] if saved and saved.zones else []
        auto = proposal.zones if proposal.available and proposal.static_view else []
        order = {"auto": [(auto, "auto"), (saved_zones, "camera")],
                 "camera": [(saved_zones, "camera"), (auto, "auto")],
                 "manual": [(saved_zones, "camera")]}[job["roi_mode"]]
        for zones, src in order:
            if zones:
                return zones, src
        return [], None

    def _save_proposal_images(self, job: dict[str, Any], p: Proposal, images: dict[int, Any]) -> None:
        """Draw the proposed polygons on each screenshot (evidence for the operator and for later audits)."""
        if not p.available:
            return
        out = Path(job["output_dir"]) / "roi"
        out.mkdir(exist_ok=True)
        per = {f.idx: f.zones for f in p.frames}
        for i, img in images.items():
            roi = ROISet.from_json(p.zones if p.static_view else per.get(i, []))
            from backend.services.visualization import annotate_litter
            vis = annotate_litter(img, [], roi, roi_active=bool(roi and roi.zones))
            cv2.imwrite(str(out / f"proposal_s{i}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])

    def _analyse(self, job_id: str) -> None:
        job = self.repo.job(job_id)
        if job is None or job["state"] != "queued":
            return
        media = self.repo.media(job["media_id"])
        self._stage(job_id, "analysing", state="running")
        engine = self.state.get_engine(job.get("model_version"))
        cam = self.state.db.get_camera(job["camera_id"]) if job.get("camera_id") else None
        settings = self.state.settings_with(cam, None)
        meta = self.state.meta
        out = Path(job["output_dir"])
        (out / "annotated").mkdir(exist_ok=True)
        whole = job.get("roi_source") == "whole_frame"
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
            if f.get("roi_status") == "uncertain":
                self.repo.update_frame(job_id, f["idx"], outcome="ROI_UNCERTAIN")
                continue
            try:
                img = cv2.imread(f["original_path"])
                if img is None:
                    raise RuntimeError("the saved screenshot could not be read back")
                zones = None if whole else (f.get("roi") or job.get("roi") or [])
                roi = ROISet.from_json(zones) if zones else None
                res = detect_image(engine, img, settings, roi, meta, self.state.db,
                                   self.state.alerts if job["create_alerts"] else None,
                                   job.get("camera_id") or "upload", Path(f["original_path"]).name, job_id,
                                   source_type="sample", frame_idx=f["frame_idx"], video_time_sec=f["video_time_sec"],
                                   session_id=f"{job_id}:s{f['idx']}", create_alerts=bool(job["create_alerts"]),
                                   location=location)
                annotated = res.pop("annotated")
                alerts = res.pop("alerts")
                res["alert_ids"] = [a["id"] for a in alerts]
                n = res["counts"]["confirmed"]
                outcome = "LITTER_DETECTED" if n else "NO_LITTER"
                self.repo.update_frame(job_id, f["idx"], outcome=outcome, detection_count=n, result=res, error=None)
                annotated_frames.append((f["idx"], annotated, Path(f["original_path"])))
                totals["screenshots_analysed"] += 1
                totals["litter"] += n
                totals["alerts"] += len(alerts)
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
        job = self.repo.job(job_id) or job
        rows = report.rows_for_job(job, media or {}, frames, self.state.cfg.storage_dir)
        csv_path = out / f"litter_report_{job_id}.csv"
        report.write_atomic(csv_path, report.to_csv(rows))
        plan = (job.get("counts") or {}).get("plan")
        failed_all = totals["screenshots_analysed"] == 0 and any(f.get("outcome") == "FAILED" for f in frames)
        self.repo.update_job(job_id, stage="completed" if not failed_all else "generating_csv",
                             state="failed" if failed_all else "done", finished_at=utcnow(), csv_path=str(csv_path),
                             model_version=engine.version, counts={**totals, "plan": plan},
                             error="No screenshot could be analysed (see the per-screenshot errors)." if failed_all else None)
        self.state.db.log_event("info", "sampled", f"four-screenshot analysis {job_id} finished: "
                                f"{totals['litter']} litter detection(s)", job.get("camera_id"))
