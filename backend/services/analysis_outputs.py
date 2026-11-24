"""Turns one analysis (image, full video, four-screenshot quick check) into its stored outputs:

    analyses row (status RUNNING -> COMPLETED_* / FAILED) + litter_events rows
    <storage>/LDM_OUTPUT/<STORE>/<YYYY>/<MM>/<DD>/<CAMERA>/
        input/       original image / original video (or a reference to it)
        annotated/   annotated image, annotated video, one evidence JPEG per litter event
        detections/  <stem>_detections.csv
        reports/     <stem>_summary.csv
        metadata/    <stem>_metadata.json

Nothing here re-runs the model or changes a status: rows, events and the annotated image all come from the same
detection result. Formats: backend/services/litter_reports.py.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import cv2

from backend.database.db import Database
from backend.services import daily_report as dr
from backend.services import litter_reports as lr
from backend.services import litter_store as ls
from backend.services import video_evidence as ve

log = logging.getLogger(__name__)

RECORDING_TIME, PROCESSING_TIME = "RECORDING_TIME", "PROCESSING_TIME"
FILE_KINDS = ("original", "annotated", "detections_csv", "summary_csv", "metadata", "video", "report_csv",
              "daily_report_csv", *ve.SCREENSHOT_KINDS, "screenshots_json")
MEDIA_TYPES = {"detections_csv": "text/csv; charset=utf-8", "summary_csv": "text/csv; charset=utf-8",
               "report_csv": "text/csv; charset=utf-8", "daily_report_csv": "text/csv; charset=utf-8","screenshots_json": "application/json",
               "metadata": "application/json", "video": "video/mp4"}


def output_root(storage_dir: Path) -> Path:
    return Path(storage_dir) / "LDM_OUTPUT"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def header_lines(ident: lr.Identity, when: dt.datetime) -> tuple[str, ...]:
    return (f"Store: {ident.store_id}  Camera: {ident.camera_id}", f"Analysed: {when:%Y-%m-%d %H:%M:%S}")


def urls_for(analysis_id: str, files: dict[str, str]) -> dict[str, str]:
    return {k: f"/api/analyses/{analysis_id}/files/{k}" for k in files if k in FILE_KINDS}


def _events_with_ids(analysis_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**e, "event_id": f"{analysis_id}-E{i:03d}"} for i, e in enumerate(events, start=1)]


# ----------------------------------------------------------------------------------------------- image
def image_analysis(db: Database, storage_dir: Path, *, result: dict[str, Any], image_bytes_path: Path,
                   original_name: str, ident: lr.Identity, settings_summary: dict[str, Any], save: bool = True,
                   source_type: str = "image") -> dict[str, Any]:
    """Outputs of one single-image analysis (``result`` = pipeline.detect_image()). ``save`` False: the rows and
    counts are computed but nothing is written (no analysis row, no files, no events)."""
    when = lr.local_now()
    aid = lr.new_analysis_id("img", when)
    ident.recording_id = ident.recording_id or aid
    events, observations = [], []
    for d in result["detections"]:
        eid = None
        if d["detection_status"] == lr.DETECTED:
            eid = f"{aid}-E{len(events) + 1:03d}"
            events.append({"event_id": eid, "track_id": d.get("track_id"), "class": d["class"],
                           "max_confidence": d["confidence"], "mean_confidence": d["confidence"], "bbox": d["bbox"],
                           "first_seen": lr.iso(when), "last_seen": lr.iso(when), "observation_count": 1,
                           "duration_sec": 0.0})
            d["event_id"] = eid
        observations.append({"timestamp": lr.iso(when), "frame_timestamp": None, "frame_number": 0,
                             "event_id": eid, "track_id": d.get("track_id"), "class": d["class"],
                             "confidence": d["confidence"], "detection_status": d["detection_status"],
                             "bbox": d["bbox"], "image_path": ""})
    status = lr.processing_status(False, len(events))
    out: dict[str, Any] = {"analysis_id": aid if save else None, "processing_status": status,
                           "unique_events": len(events), "frame_observations": lr.count_observations(observations),
                           "timestamp": lr.iso(when), "timestamp_source": PROCESSING_TIME, "urls": {}}
    if not save:
        return out
    paths = lr.output_paths(output_root(storage_dir), ident, when, aid)
    ls.start_analysis(db, aid, "image", ident, job_id=None, model_version=result["model_version"],
                      output_dir=str(paths.base), started=when)
    try:
        ext = Path(original_name).suffix.lower() or ".jpg"
        original = paths.sub("input") / f"{paths.stem}_original{ext}"
        shutil.copyfile(image_bytes_path, original)
        annotated = paths.sub("annotated") / f"{paths.stem}_annotated.jpg"
        if not cv2.imwrite(str(annotated), result["annotated"], [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError("the annotated image could not be written")
        for o in observations:
            o["image_path"] = str(annotated)
        for e in events:
            e["image_path"] = str(annotated)
        files = lr.write_reports(paths, observations=observations, events=events, ident=ident,
                                 model_version=result["model_version"], status=status, when=when,
                                 metadata={"analysis_id": aid, "kind": "image", "source_file": original_name,
                                           "original_sha256": _sha(original), "image_size": result["image_size"],
                                           "timestamp_source": PROCESSING_TIME, "settings": settings_summary,
                                           "counts": result["counts"], "inference_ms": result["inference_ms"],
                                           "tiled": result["tiled"],
                                           **({"roi": result["roi"]} if result.get("roi") else {})})
        files.update(original=str(original), annotated=str(annotated))
        ls.insert_events(db, aid, source_type, ident, events, job_id=None, model_version=result["model_version"],
                         timestamp_source=PROCESSING_TIME)
        ls.finish_analysis(db, aid, status, unique_events=len(events), frame_observations=out["frame_observations"],
                           frames_processed=1, files=files)
    except Exception as exc:
        ls.finish_analysis(db, aid, lr.FAILED, error=str(exc))
        raise
    out["urls"] = urls_for(aid, files)
    out["output_dir"] = str(paths.base)
    return out


# ----------------------------------------------------------------------------------------------- video
def start_video(db: Database, storage_dir: Path, ident: lr.Identity, job_id: str, model_version: str,
                recording_start: Optional[dt.datetime]) -> tuple[str, lr.OutputPaths, dt.datetime]:
    when = lr.local_now()
    aid = lr.new_analysis_id("vid", when)
    ident.recording_id = ident.recording_id or job_id
    paths = lr.output_paths(output_root(storage_dir), ident, recording_start or when, aid)
    ls.start_analysis(db, aid, "video", ident, job_id=job_id, model_version=model_version, output_dir=str(paths.base),
                      started=when)
    return aid, paths, when


def keep_original(video_path: Path, paths: lr.OutputPaths, original_name: str, keep: bool) -> dict[str, Any]:
    """Preserve the uploaded video in input/ (moved, not copied: the upload is a temporary file) - or, with
    ``keep`` False, record only its name, size and checksum."""
    ref = {"original_name": original_name, "size_bytes": video_path.stat().st_size, "sha256": _sha(video_path)}
    if keep:
        dest = paths.sub("input") / f"{paths.stem}_original{video_path.suffix.lower() or '.mp4'}"
        shutil.move(str(video_path), dest)
        ref["stored_path"] = str(dest)
    (paths.sub("input") / f"{paths.stem}_video_reference.json").write_text(json.dumps(ref, indent=2), "utf-8")
    return ref


def finish_video(db: Database, aid: str, paths: lr.OutputPaths, ident: lr.Identity, *, job_id: str,
                 res: dict[str, Any], recording_start: Optional[dt.datetime], started: dt.datetime,
                 video_ref: dict[str, Any], settings_summary: dict[str, Any],
                 screenshots: Optional[dict[str, Any]] = None, evidence_root: Optional[Path] = None,
                 candidate_threshold: Optional[float] = None, review_min_observations: int = 1) -> dict[str, Any]:
    """Events + observations of a finished video (pipeline.detect_video result) -> reports and DB rows.
    With ``evidence_root`` (storage/analysis_outputs/<id>): also the six-column litter report snapshot and the review
    candidates (tracks >= ``candidate_threshold`` that never became events); ``screenshots``: the
    video_evidence.build_screenshots() document, whose files are recorded on the analysis."""
    lit = res["litter"]
    tsrc = RECORDING_TIME if recording_start else PROCESSING_TIME

    def at(sec: Optional[float], processed: Optional[str]) -> str:
        if recording_start is not None and sec is not None:
            return lr.iso(recording_start + dt.timedelta(seconds=sec))
        return processed or ""

    evidence = paths.sub("annotated")
    events = []
    for e in _events_with_ids(aid, lit["events"]):
        src = evidence / f"track_{e['track_id']}.jpg"
        img = evidence / f"{paths.stem}_{e['event_id'].rsplit('-', 1)[1]}_evidence.jpg"
        if src.is_file():
            src.replace(img)
        events.append({**e, "first_seen": at(e["first_seen_sec"], e["first_processed_at"]),
                       "last_seen": at(e["last_seen_sec"], e["last_processed_at"]),
                       "image_path": str(img) if img.is_file() else ""})
    by_track = {e["track_id"]: e for e in events}
    observations = []
    for o in lit["observations"]:
        ev = by_track.get(o["track_id"]) if o["event_track"] else None
        tr = lit["tracks"].get(o["track_id"]) if o["track_id"] is not None and not ev else None
        observations.append({**o, "timestamp": at(o["frame_timestamp"], o["processed_at"]),
                             "event_id": ev["event_id"] if ev else None,
                             "image_path": ev["image_path"] if ev else str(paths.sub("annotated") /
                                                                           f"{paths.stem}_annotated.mp4"),
                             **({"track_observations": tr["observation_count"],
                                 "track_first_seen": at(tr["first_seen_sec"], tr["first_processed_at"]),
                                 "track_last_seen": at(tr["last_seen_sec"], tr["last_processed_at"])} if tr else {})})
    status = lr.processing_status(False, len(events))
    annotated_video = paths.sub("annotated") / f"{paths.stem}_annotated.mp4"
    if Path(res["files"]["video"]).is_file():
        shutil.copyfile(res["files"]["video"], annotated_video)
    model = res["stats"]["model_version"]
    files = lr.write_reports(paths, observations=observations, events=events, ident=ident, model_version=model,
                             status=status, when=recording_start or started,
                             metadata={"analysis_id": aid, "kind": "video", "job_id": job_id,
                                       "timestamp_source": tsrc,
                                       "recording_start": lr.iso(recording_start) if recording_start else None,
                                       "video": video_ref, "validation": res["validation"],
                                       "sampling": {"interval_sec": res["stats"]["sample_interval_sec"],
                                                    "every_n_frames": res["stats"]["sample_every_n_frames"],
                                                    "frames_analysed": res["stats"]["frames_analysed"]},
                                       "tracked_objects": lit["tracked_objects"],
                                       "under_review_objects": lit["under_review_objects"],
                                       "settings": settings_summary, "stats": res["stats"]})
    files["video"] = str(annotated_video)
    if video_ref.get("stored_path"):
        files["original"] = video_ref["stored_path"]
    report: Optional[dict[str, Any]] = None
    if evidence_root is not None and candidate_threshold is not None:
        root = Path(evidence_root)
        root.mkdir(parents=True, exist_ok=True)
        review = ve.review_rows(lit, candidate_threshold, at, review_min_observations)
        review_path = root / "review_candidates.json"
        lr.write_atomic(review_path, json.dumps(review, indent=2))
        rows = ve.report_rows(ident.store_id, ident.camera_id,
                              [{**e, "review_status": lr.DETECTED} for e in events], review)
        # No per-video CSV file: the rows go into the day's consolidated CSV (appended below, once the DB rows are
        # stored). report_csv (this analysis only) is rebuilt from the DB on download (routes); daily_report_csv is
        # the whole day's file. root = <storage>/analysis_outputs/<aid>.
        day = lr.local_now().date()
        daily_path = dr.path_for(root.parent.parent, day)
        files.update(review_candidates=str(review_path), report_csv=str(daily_path),
                     daily_report_csv=str(daily_path))
        report = {"columns": ve.REPORT_COLUMNS, "rows": rows, "counts": ve.counts(rows),
                  "url": f"/api/analyses/{aid}/files/report_csv"}
    shots_out: list[dict[str, Any]] = []
    if screenshots:
        files["screenshots_json"] = str(Path(screenshots["screenshots"][0]["path"]).parent / "screenshots.json")
        for sh in screenshots["screenshots"]:
            files[sh["kind"]] = sh["path"]
            shots_out.append({k: v for k, v in sh.items() if k != "path"}
                             | {"url": f"/api/analyses/{aid}/files/{sh['kind']}"})
    ls.insert_events(db, aid, "video", ident, events, job_id=job_id, model_version=model, timestamp_source=tsrc)
    ls.finish_analysis(db, aid, status, unique_events=len(events), frame_observations=lit["frame_observations"],
                       frames_processed=res["stats"]["frames_analysed"], files=files)
    if report is not None:
        report["daily"] = dr.append(daily_path.parents[2], day, report["rows"])
    public_events =[{k: e[k] for k in ("event_id", "track_id", "class", "first_seen", "last_seen", "first_seen_sec",
                                        "last_seen_sec", "duration_sec", "observation_count", "max_confidence",
                                        "mean_confidence", "bbox")}
                     | {"detection_status": lr.DETECTED,
                        "evidence_url": f"/api/jobs/{job_id}/evidence/{e['event_id']}.jpg" if e["image_path"] else None}
                     for e in events]
    return {"analysis": {"analysis_id": aid, "processing_status": status, "unique_events": len(events),
                         "frame_observations": lit["frame_observations"], "timestamp_source": tsrc,
                         "urls": urls_for(aid, files), "output_dir": str(paths.base),
                         "evidence_dir": str(evidence_root) if evidence_root is not None else None},
            "screenshots": shots_out,
            "screenshot_selection": ({k: screenshots[k] for k in ("seed", "method", "duration_sec", "frames_read")}
                                     if screenshots else None),
            "report": report,
            "litter": {"unique_events": len(events), "frame_observations": lit["frame_observations"],
                       "tracked_objects": lit["tracked_objects"], "under_review_objects": lit["under_review_objects"],
                       "by_class": lit["by_class"], "events": public_events}}


def fail(db: Database, aid: str, paths: lr.OutputPaths, ident: lr.Identity, model_version: str, error: str,
         metadata: dict[str, Any]) -> None:
    try:
        lr.write_failure(paths, ident=ident, model_version=model_version, error=error, metadata=metadata)
    except Exception:  # noqa: BLE001 - the DB status below is what the UI shows
        log.exception("could not write the failure metadata of %s", aid)
    ls.finish_analysis(db, aid, lr.FAILED, error=error, files={"metadata": str(paths.metadata)})


# ----------------------------------------------------------------------------------------------- quick check
def quick_analysis(db: Database, storage_dir: Path, *, job: dict[str, Any], frames: list[dict[str, Any]],
                   objects: dict[str, dict[str, Any]], ident: lr.Identity, model_version: str,
                   recording_start: Optional[dt.datetime], failed: bool) -> dict[str, Any]:
    """Standard outputs of a four-screenshot job. Screenshots are seconds-to-minutes apart, so the object linking of
    backend/sampling/detection_report.link_objects (an ESTIMATE) decides which detections are the same object; one
    DETECTED object = one event, its other confirmed sightings are TRACKED."""
    when = lr.local_now()
    aid = lr.new_analysis_id("qck", when)
    ident.recording_id = ident.recording_id or job["id"]
    paths = lr.output_paths(output_root(storage_dir), ident, recording_start or when, aid)
    ls.start_analysis(db, aid, "quick_check", ident, job_id=job["id"], model_version=model_version,
                      output_dir=str(paths.base), started=when)
    if failed:
        fail(db, aid, paths, ident, model_version, job.get("error") or "no screenshot could be analysed",
             {"analysis_id": aid, "kind": "quick_check", "job_id": job["id"]})
        return {"analysis_id": aid, "processing_status": lr.FAILED, "urls": {}}
    tsrc = RECORDING_TIME if recording_start else PROCESSING_TIME

    def at(f: dict[str, Any]) -> str:
        if recording_start is not None and f.get("video_time_sec") is not None:
            return lr.iso(recording_start + dt.timedelta(seconds=f["video_time_sec"]))
        return lr.iso(when)

    seen: dict[str, dict[str, Any]] = {}
    observations: list[dict[str, Any]] = []
    for f in sorted(frames, key=lambda x: x["idx"]):
        res = f.get("result") or {}
        for d in res.get("detections") or []:
            oid = d.get("object_id") if d.get("status") == "confirmed" else None
            if oid:
                first = oid not in seen
                if first:
                    seen[oid] = {"event_id": f"{aid}-E{len(seen) + 1:03d}", "track_id": None, "class": d["class"],
                                 "confs": [], "bbox": d["bbox"], "first_seen": at(f), "last_seen": at(f),
                                 "first_seen_sec": f.get("video_time_sec"), "image_path": f.get("annotated_path") or ""}
                ev = seen[oid]
                ev["confs"].append(d["confidence"])
                ev["last_seen"], ev["last_seen_sec"] = at(f), f.get("video_time_sec")
                status = lr.DETECTED if first else lr.TRACKED
            else:
                status = lr.detection_status(d.get("status", ""))
            observations.append({"timestamp": at(f), "frame_timestamp": f.get("video_time_sec"),
                                 "frame_number": f.get("frame_idx"), "event_id": seen[oid]["event_id"] if oid else None,
                                 "track_id": None, "class": d["class"], "confidence": d["confidence"],
                                 "detection_status": status, "bbox": d["bbox"],
                                 "image_path": f.get("annotated_path") or ""})
    events = []
    for ev in seen.values():
        c = ev.pop("confs")
        ev.update(observation_count=len(c), max_confidence=round(max(c), 4), mean_confidence=round(sum(c) / len(c), 4),
                  duration_sec=round((ev.get("last_seen_sec") or 0) - (ev.get("first_seen_sec") or 0), 3))
        events.append(ev)
    status = lr.processing_status(False, len(events))
    files = lr.write_reports(paths, observations=observations, events=events, ident=ident, model_version=model_version,
                             status=status, when=recording_start or when,
                             metadata={"analysis_id": aid, "kind": "quick_check", "job_id": job["id"],
                                       "timestamp_source": tsrc, "screenshots": len(frames),
                                       "object_linking": "estimated across screenshots (alert de-dup rule)",
                                       "estimated_objects": len(objects)})
    ls.insert_events(db, aid, "sample", ident, events, job_id=job["id"], model_version=model_version,
                     timestamp_source=tsrc)
    ls.finish_analysis(db, aid, status, unique_events=len(events), frame_observations=lr.count_observations(observations),
                       frames_processed=len(frames), files=files)
    return {"analysis_id": aid, "processing_status": status, "unique_events": len(events),
            "frame_observations": lr.count_observations(observations), "urls": urls_for(aid, files)}


# ----------------------------------------------------------------------------------------------- files
def analysis_file(db: Database, analysis_id: str, kind: str) -> Optional[Path]:
    """A stored file of an analysis, only from the paths recorded on its row (never from a request path)."""
    if not lr.ANALYSIS_ID_RE.match(analysis_id) or kind not in FILE_KINDS:
        return None
    a = ls.get_analysis(db, analysis_id)
    p = Path(a["files"].get(kind) or "") if a else None
    return p if p and p.is_file() else None
