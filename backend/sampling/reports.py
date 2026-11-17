"""Results Dashboard data for the four-screenshot analysis: summary, latest analysis, monthly reports, history and
hourly summary. Everything is computed from the sampled_* tables; nothing is estimated.

Time rules:
  * video timestamp      = position in the file (always known)
  * recording timestamp  = recording start + video timestamp, only when the operator supplied the start
  * monthly "completion" = local month in which the analysis finished; "recording" = month of the recording
  * hourly summary       = recording time only; analyses without a recording start are counted as excluded
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Optional

from backend.sampling import report
from backend.sampling.plan import clock_time, hhmmss, parse_recording_start
from backend.sampling.repo import SampledRepo, _decode

ANALYSED = ("LITTER_DETECTED", "NO_LITTER")
FAILED_STATES = ("failed", "interrupted")
ACTIVE_STATES = ("queued", "running", "awaiting_roi")


def _local(ts: Optional[str]) -> Optional[dt.datetime]:
    if not ts:
        return None
    d = dt.datetime.fromisoformat(ts)
    return d.astimezone() if d.tzinfo else d


def _start(job: dict[str, Any]) -> Optional[dt.datetime]:
    try:
        return parse_recording_start(job.get("recording_start"))
    except ValueError:
        return None


# ---------------------------------------------------------------------- public shapes
def public_frame(job: dict[str, Any], f: dict[str, Any]) -> dict[str, Any]:
    base = f"/api/sampled/jobs/{job['id']}/frames/{f['idx']}"
    clock = clock_time(_start(job), f.get("video_time_sec"))
    res = f.get("result") or {}
    proposal = ((job.get("proposal") or {}).get("proposal") or {})
    return {
        "idx": f["idx"], "frame_idx": f["frame_idx"], "video_time_sec": f.get("video_time_sec"),
        "video_timestamp": hhmmss(f.get("video_time_sec"), millis=True),
        "recording_timestamp": clock.isoformat(timespec="seconds") if clock else None,
        "status": f["status"], "outcome": f.get("outcome"), "roi_status": f.get("roi_status"), "roi": f.get("roi"),
        "detection_count": f.get("detection_count", 0), "error": f.get("error"),
        "original_url": f"{base}/original.jpg" if f.get("original_path") else None,
        "annotated_url": f"{base}/annotated.jpg" if f.get("annotated_path") else None,
        "proposal_url": f"{base}/proposal.jpg" if proposal.get("available") and f.get("original_path") else None,
        "detections": res.get("detections", []), "counts": res.get("counts"), "by_class": res.get("by_class", {}),
        "alert_ids": res.get("alert_ids", []), "inference_ms": res.get("inference_ms"),
        "image_size": res.get("image_size"), "outside_roi_ignored": res.get("outside_roi_ignored"),
        "roi_counts": res.get("roi_counts"),
        "planned_frame_idx": f.get("planned_frame_idx"), "sampling_note": f.get("sampling_note"),
        "download_original_url": f"{base}/original.jpg?download=1" if f.get("original_path") else None,
        "download_annotated_url": f"{base}/annotated.jpg?download=1" if f.get("annotated_path") else None,
        # pavement-only mode
        "pavement": res.get("pavement"),
        "diagnostic_url": f"{base}/diagnostic.jpg" if res.get("diagnostic_path") else None,
    }


def lifecycle(job: dict[str, Any]) -> str:
    """One operator-facing status derived from (state, stage). The stored state/stage values are unchanged."""
    state, stage = job.get("state"), job.get("stage")
    if state == "done":
        return "COMPLETED"
    if state == "cancelled":
        return "CANCELLED"
    if state in ("failed", "interrupted"):
        return "FAILED"
    if state == "awaiting_roi":
        return "AWAITING_ROI"                     # operator workflow only; automatic jobs never pause
    if state == "queued":
        return "QUEUED"
    if stage == "sampling":
        return "EXTRACTING_SCREENSHOTS"
    if stage in ("analysing", "generating_images"):
        return "RUNNING_DETECTION"
    if stage == "generating_csv":
        return "GENERATING_REPORT"
    return "PROCESSING"                           # received, validating, identifying_roi


def report_urls(job: dict[str, Any]) -> Optional[dict[str, str]]:
    """Download links of reports/ files that exist (they are written before a job is marked done)."""
    from pathlib import Path
    rdir = Path(job.get("output_dir") or "") / "reports"
    if not job.get("output_dir") or not (rdir / "detection_report.csv").is_file():
        return None
    base = f"/api/sampled/jobs/{job['id']}/reports"
    return {"detection_csv": f"{base}/detection_report.csv", "summary_csv": f"{base}/analysis_summary.csv",
            "summary_json": f"{base}/analysis_summary.json"}


def public_job(job: dict[str, Any], media: Optional[dict[str, Any]], frames: list[dict[str, Any]],
               error: Callable[[Optional[str]], Optional[str]] = lambda t: t) -> dict[str, Any]:
    j = {k: v for k, v in job.items() if k not in ("output_dir", "csv_path")}
    j["error"] = error(job.get("error"))
    j["media"] = None if media is None else {k: media.get(k) for k in (
        "id", "filename", "size_bytes", "width", "height", "fps", "frames", "duration_sec", "created_at",
        "suggested_recording_start")}
    j["video_name"] = (media or {}).get("filename")
    j["frames"] = [{**public_frame(job, f), "error": error(f.get("error"))} for f in frames]
    j["csv_url"] = f"/api/sampled/jobs/{job['id']}/report.csv" if job.get("csv_path") else None
    j["lifecycle"] = lifecycle(job)
    j["reports"] = report_urls(job)
    counts = job.get("counts") or {}
    j["roi_policy"] = counts.get("roi_policy")
    j["summary"] = counts.get("summary")
    return j


# ---------------------------------------------------------------------- dashboard
class Reports:
    def __init__(self, repo: SampledRepo) -> None:
        self.repo = repo
        self.db = repo.db

    def _all(self, where: str = "", params: tuple = ()) -> list[dict[str, Any]]:
        return self.repo.jobs(where, params, limit=100000)

    def _frames(self, job_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {j: [] for j in job_ids}
        if not job_ids:
            return out
        for chunk in range(0, len(job_ids), 500):
            ids = job_ids[chunk:chunk + 500]
            for f in self.db._query(f"SELECT * FROM sampled_frames WHERE job_id IN ({','.join('?' * len(ids))}) "
                                    "ORDER BY job_id, idx", ids):
                out[f["job_id"]].append(_decode(f))   # type: ignore[arg-type]
        return out

    @staticmethod
    def _totals(jobs: list[dict[str, Any]], frames: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        by_class: dict[str, int] = {}
        litter = extracted = analysed = 0
        for j in jobs:
            for f in frames.get(j["id"], []):
                extracted += f["status"] == "saved"
                if f.get("outcome") in ANALYSED:
                    analysed += 1
                    litter += f.get("detection_count") or 0
                    for k, v in ((f.get("result") or {}).get("by_class") or {}).items():
                        by_class[k] = by_class.get(k, 0) + v
        return {
            "analyses": len(jobs),
            "analyses_completed": sum(j["state"] == "done" for j in jobs),
            "analyses_failed": sum(j["state"] in FAILED_STATES for j in jobs),
            "analyses_cancelled": sum(j["state"] == "cancelled" for j in jobs),
            "analyses_in_progress": sum(j["state"] in ACTIVE_STATES for j in jobs),
            "awaiting_roi": sum(j["state"] == "awaiting_roi" for j in jobs),
            "screenshots_extracted": extracted, "screenshots_analysed": analysed, "total_litter": litter,
            "categories": dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
        }

    def overview(self, error: Callable[[Optional[str]], Optional[str]]) -> dict[str, Any]:
        jobs = self._all()
        frames = self._frames([j["id"] for j in jobs])
        summary = {"videos_received": self.repo.count_media(), **self._totals(jobs, frames)}
        done = [j for j in jobs if j["state"] == "done"]
        latest = done[0] if done else None
        newest = jobs[0] if jobs else None

        def full(j: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
            if j is None:
                return None
            return public_job(self.repo.job(j["id"]) or j, self.repo.media(j["media_id"]), frames.get(j["id"], []), error)

        months_c = sorted({_local(j["finished_at"]).strftime("%Y-%m") for j in jobs   # type: ignore[union-attr]
                           if j["state"] in ("done", *FAILED_STATES) and j.get("finished_at")}, reverse=True)
        months_r = sorted({s.strftime("%Y-%m") for j in jobs if (s := _start(j))}, reverse=True)
        return {"summary": summary, "latest": full(latest),
                "newest_other": full(newest) if newest and (latest is None or newest["id"] != latest["id"]) else None,
                "months": {"completion": months_c, "recording": months_r},
                "grouping_rules": {
                    "completion": "Month in which the analysis finished (server local time).",
                    "recording": "Month of the recording start entered for the video; analyses without one are excluded."}}

    def month(self, month: str, by: str, error: Callable[[Optional[str]], Optional[str]]) -> Optional[dict[str, Any]]:
        dt.datetime.strptime(month, "%Y-%m")       # ValueError -> 422 in the route
        jobs = [j for j in self._all() if self._month_of(j, by) == month]
        if not jobs:
            return None
        frames = self._frames([j["id"] for j in jobs])
        return {"month": month, "by": by, **self._totals(jobs, frames),
                "analyses_list": [self._row(j, frames.get(j["id"], []), error) for j in jobs],
                "csv_url": f"/api/sampled/reports/months/{month}.csv?by={by}"}

    def month_rows(self, month: str, by: str, storage_dir) -> list[dict[str, Any]]:
        jobs = [j for j in self._all("WHERE j.state='done'") if self._month_of(j, by) == month]
        frames = self._frames([j["id"] for j in jobs])
        rows: list[dict[str, Any]] = []
        for j in sorted(jobs, key=lambda x: x["created_at"]):
            rows += report.rows_for_job(self.repo.job(j["id"]) or j, self.repo.media(j["media_id"]) or {},
                                        frames.get(j["id"], []), storage_dir)
        return rows

    def all_rows(self, storage_dir) -> list[dict[str, Any]]:
        jobs = self._all("WHERE j.state='done'")
        frames = self._frames([j["id"] for j in jobs])
        rows: list[dict[str, Any]] = []
        for j in sorted(jobs, key=lambda x: x["created_at"]):
            rows += report.rows_for_job(self.repo.job(j["id"]) or j, self.repo.media(j["media_id"]) or {},
                                        frames.get(j["id"], []), storage_dir)
        return rows

    @staticmethod
    def _month_of(j: dict[str, Any], by: str) -> Optional[str]:
        if by == "recording":
            s = _start(j)
            return s.strftime("%Y-%m") if s else None
        if j["state"] not in ("done", *FAILED_STATES) or not j.get("finished_at"):
            return None
        return _local(j["finished_at"]).strftime("%Y-%m")   # type: ignore[union-attr]

    @staticmethod
    def _row(j: dict[str, Any], frames: list[dict[str, Any]], error: Callable[[Optional[str]], Optional[str]]) -> dict[str, Any]:
        return {"id": j["id"], "video_name": j.get("video_name"), "store_id": j.get("store_id"),
                "camera_id": j.get("camera_id"), "recording_start": j.get("recording_start"),
                "created_at": j["created_at"], "finished_at": j.get("finished_at"), "state": j["state"],
                "stage": j["stage"], "error": error(j.get("error")), "duration_sec": j.get("duration_sec"),
                "roi_status": j.get("roi_status"), "roi_source": j.get("roi_source"),
                "screenshots_extracted": sum(f["status"] == "saved" for f in frames),
                "screenshots_analysed": sum(f.get("outcome") in ANALYSED for f in frames),
                "litter": sum((f.get("detection_count") or 0) for f in frames if f.get("outcome") in ANALYSED),
                "timestamps": [hhmmss(f.get("video_time_sec")) for f in frames],
                "thumbs": [f"/api/sampled/jobs/{j['id']}/frames/{f['idx']}/{'annotated' if f.get('annotated_path') else 'original'}.jpg"
                           for f in frames if f.get("original_path")],
                "csv_url": f"/api/sampled/jobs/{j['id']}/report.csv" if j.get("csv_path") else None}

    def history(self, *, store: str = "", camera: str = "", start: str = "", end: str = "", state: str = "",
                q: str = "", limit: int = 50, offset: int = 0,
                error: Callable[[Optional[str]], Optional[str]] = lambda t: t) -> dict[str, Any]:
        where, p = ["1=1"], []
        if store:
            where.append("j.store_id=?"); p.append(store)
        if camera:
            where.append("j.camera_id=?"); p.append(camera)
        if start:
            where.append("j.created_at>=?"); p.append(start)
        if end:
            where.append("j.created_at<=?"); p.append(end)
        if state:
            where.append("j.state=?"); p.append(state)
        if q:
            where.append("m.filename LIKE ?"); p.append(f"%{q}%")
        w = "WHERE " + " AND ".join(where)
        jobs = self.repo.jobs(w, tuple(p), limit, offset)
        frames = self._frames([j["id"] for j in jobs])
        stores = [r["store_id"] for r in self.db._query("SELECT DISTINCT store_id FROM sampled_jobs WHERE store_id IS NOT NULL ORDER BY 1")]
        cams = [r["camera_id"] for r in self.db._query("SELECT DISTINCT camera_id FROM sampled_jobs WHERE camera_id IS NOT NULL ORDER BY 1")]
        return {"total": self.repo.count_jobs(w, tuple(p)), "stores": stores, "cameras": cams,
                "analyses": [self._row(j, frames.get(j["id"], []), error) for j in jobs]}

    def detections(self, *, store: str = "", camera: str = "", cls: str = "", start: str = "", end: str = "",
                   limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """Confirmed litter detections from the four-screenshot analyses, with evidence links."""
        # screenshots without an approved ROI (ROI_NOT_CONFIGURED / ROI_FRAME_MISMATCH) are review material only
        where, p = ["d.source_type='sample'", "d.status='confirmed'",
                    "IFNULL(f.outcome,'') NOT IN ('ROI_NOT_CONFIGURED','ROI_FRAME_MISMATCH')"], []
        if store:
            where.append("j.store_id=?"); p.append(store)
        if camera:
            where.append("j.camera_id=?"); p.append(camera)
        if cls:
            where.append("d.class=?"); p.append(cls)
        if start:
            where.append("d.ts>=?"); p.append(start)
        if end:
            where.append("d.ts<=?"); p.append(end)
        w = " AND ".join(where)
        base = ("FROM detections d JOIN sampled_jobs j ON j.id=d.job_id JOIN sampled_media m ON m.id=j.media_id "
                "LEFT JOIN sampled_frames f ON f.job_id=d.job_id AND f.frame_idx=d.frame_idx")
        rows = self.db._query(f"SELECT d.id, d.ts, d.class, d.confidence, d.x1, d.y1, d.x2, d.y2, d.frame_idx, "
                              f"d.video_time_sec, d.job_id, d.model_version, j.store_id, j.camera_id, j.recording_start, "
                              f"m.filename AS video_name, f.idx AS screenshot_idx, f.annotated_path IS NOT NULL AS has_annotated "
                              f"{base} WHERE {w} ORDER BY d.id DESC LIMIT ? OFFSET ?", (*p, limit, offset))
        total = int(self.db._one(f"SELECT COUNT(*) AS n {base} WHERE {w}", p)["n"])   # type: ignore[index]
        for r in rows:
            s = _start(r)
            c = clock_time(s, r.get("video_time_sec"))
            r["video_timestamp"] = hhmmss(r.get("video_time_sec"), millis=True)
            r["recording_timestamp"] = c.isoformat(timespec="seconds") if c else None
            kind = "annotated" if r.pop("has_annotated") else "original"
            r["evidence_url"] = (f"/api/sampled/jobs/{r['job_id']}/frames/{r['screenshot_idx']}/{kind}.jpg"
                                 if r.get("screenshot_idx") else None)
        classes = [r["class"] for r in self.db._query(
            "SELECT DISTINCT class FROM detections WHERE source_type='sample' AND status='confirmed' ORDER BY 1")]
        return {"total": total, "classes": classes, "detections": rows}

    def hourly(self, *, store: str = "", camera: str = "", date: str = "") -> dict[str, Any]:
        jobs = [j for j in self._all("WHERE j.state='done'")
                if (not store or j.get("store_id") == store) and (not camera or j.get("camera_id") == camera)]
        with_time = [j for j in jobs if _start(j)]
        frames = self._frames([j["id"] for j in with_time])
        buckets: dict[tuple, dict[str, Any]] = {}
        for j in with_time:
            for f in frames.get(j["id"], []):
                c = clock_time(_start(j), f.get("video_time_sec"))
                if c is None or (date and c.date().isoformat() != date):
                    continue
                h = c.replace(minute=0, second=0, microsecond=0)
                key = (h.isoformat(), j.get("store_id") or "", j.get("camera_id") or "")
                b = buckets.setdefault(key, {"hour_start": h.isoformat(timespec="minutes"),
                                             "hour_end": (h + dt.timedelta(hours=1)).isoformat(timespec="minutes"),
                                             "store_id": key[1], "camera_id": key[2], "analyses": set(),
                                             "screenshots": 0, "analysed": 0, "with_litter": 0, "litter": 0,
                                             "by_class": {}})
                b["analyses"].add(j["id"])
                b["screenshots"] += 1
                if f.get("outcome") in ANALYSED:
                    b["analysed"] += 1
                    n = f.get("detection_count") or 0
                    b["litter"] += n
                    b["with_litter"] += n > 0
                    for k, v in ((f.get("result") or {}).get("by_class") or {}).items():
                        b["by_class"][k] = b["by_class"].get(k, 0) + v
        rows = [{**b, "analyses": len(b["analyses"])} for _, b in sorted(buckets.items())]
        return {"available": bool(with_time), "basis": "recording_time", "rows": rows,
                "excluded_analyses": len(jobs) - len(with_time),
                "note": ("Hours are real recording time (recording start + position in the video). Analyses without "
                         "a recording start are not included, because their video-relative times are not clock times.")}
