"""Visual evidence + six-column litter report of a full-video analysis (owner brief 2026-11-20).

After the normal full-video pass (every frame at the sampling interval, tracking, unique events - unchanged), this
module adds:

  storage/analysis_outputs/<analysis_id>/
      screenshots/screenshot_01.png .. screenshot_04.png   exactly FOUR annotated frames
      screenshots/screenshots.json                          frame index, video time and timestamp of each
      litter_report.csv                                     snapshot at completion (six columns, see REPORT_COLUMNS)
      review_candidates.json                                the UNDER_REVIEW rows of that report

Screenshot selection (``select_frames``): the decoded duration is split into four equal segments and one frame is
drawn at random (seeded ``random.Random``) from the ANALYSED frames of each segment, so the four are distinct, spread
over the recording and never just the start. The boxes on a screenshot are the pipeline's own result for that frame
(same detections, same track ids), so a screenshot can never show litter that the CSV/DB does not contain. Only when a
video has fewer than four analysed frames are other decoded frames used; the model is run on those and every box is
drawn as UNDER REVIEW (no temporal confirmation exists for them) and they are flagged ``analysed_frame: false``.

What is drawn: boxes of litter events (solid, ``Class 87%``) and of review candidates (dashed amber, ``REVIEW``).
Weak-only tracks and filter rejections are not drawn. A frame without a litter-event box gets a "No litter detected"
banner - nothing is ever added to make a screenshot look populated.

Report rows (``report_rows``): one row per unique object, never per frame observation.
  * DETECTED / VERIFIED - the litter_events rows of the analysis, read from the DB with their CURRENT operator review
    (VERIFIED and RESOLVED -> VERIFIED; operator-REJECTED false positives are left out);
  * UNDER_REVIEW        - tracks that never passed temporal confirmation but reached ``candidate_threshold`` at least
    once AND were seen in at least ``tracking.min_confirmations`` analysed frames (existing settings; nothing new is
    tuned). Litter is static: on real CCTV (2026-11-20) 203 of 235 such tracks were single-frame flickers on passing
    vehicles' lights, which are not uncertain litter. Weak-only tracks are not reported.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
import random
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from backend.core.csv_safety import safe_rows
from backend.core.settings import DetectionSettings
from backend.detection.filters import apply_filters
from backend.detection.types import CANDIDATE, CONFIRMED, REJECTED, Detection
from backend.services import litter_reports as lr
from backend.services.roi_service import NO_LITTER_MSG, resolve_candidates, roi_detect
from backend.services.visualization import FONT, annotate_litter

log = logging.getLogger(__name__)

N_SCREENSHOTS = 4
REPORT_COLUMNS = ["store_id", "camera_id", "timestamp", "litter_class", "confidence_score", "detection_status"]
REPORT_STATUSES = (lr.DETECTED, lr.UNDER_REVIEW, lr.VERIFIED)
EVIDENCE_DIR = "analysis_outputs"
SCREENSHOT_KINDS = tuple(f"screenshot_{i:02d}" for i in range(1, N_SCREENSHOTS + 1))


def evidence_dir(storage_dir: Path, analysis_id: str) -> Path:
    return Path(storage_dir) / EVIDENCE_DIR / analysis_id


def default_seed(analysis_id: str) -> int:
    """Stable per analysis (not Python's randomised hash): the same analysis id always picks the same frames."""
    return zlib.crc32(analysis_id.encode("utf-8"))


def report_time(iso_ts: Optional[str]) -> str:
    """ISO timestamp -> ``YYYY-MM-DD HH:MM:SS`` (the report format); empty stays empty - never invented."""
    if not iso_ts:
        return ""
    return dt.datetime.fromisoformat(iso_ts).replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")


def score(v: float) -> str:
    return f"{float(v):.2f}"


# ----------------------------------------------------------------------------------------------- selection
def select_frames(analysed: Sequence[tuple[int, float]], frames_read: int, fps: float, seed: int,
                  n: int = N_SCREENSHOTS) -> list[tuple[int, bool]]:
    """Pick ``n`` distinct frame indices spread over the video. ``analysed``: (frame_idx, video_time_sec) of every
    analysed frame. Returns [(frame_idx, analysed_frame)] sorted by time.

    One random analysed frame per equal-length segment of the decoded duration; a segment with no analysed frame (very
    long sampling interval) takes a random unused analysed frame; only if there are fewer than ``n`` analysed frames
    are random other decoded frames (indices < ``frames_read``) added."""
    if frames_read < n:
        raise ValueError(f"The video has only {frames_read} decodable frame(s); {n} distinct screenshots need at "
                         f"least {n}.")
    rng = random.Random(seed)
    duration = frames_read / fps
    seg = duration / n
    pool = sorted({int(i) for i, _ in analysed if 0 <= int(i) < frames_read})
    chosen: list[int] = []
    for k in range(n):
        lo, hi = k * seg, (k + 1) * seg
        cands = [i for i in pool if lo <= i / fps < hi and i not in chosen]
        if cands:
            chosen.append(rng.choice(cands))
    rest = [i for i in pool if i not in chosen]
    while len(chosen) < n and rest:
        chosen.append(rest.pop(rng.randrange(len(rest))))
    analysed_set = set(chosen)
    if len(chosen) < n:
        # fewer analysed frames than screenshots: random other decoded frames, one per segment where possible
        for k in range(n):
            if len(chosen) >= n:
                break
            lo, hi = int(k * frames_read / n), max(int((k + 1) * frames_read / n), int(k * frames_read / n) + 1)
            free = [i for i in range(lo, min(hi, frames_read)) if i not in chosen]
            if free:
                chosen.append(rng.choice(free))
        free = [i for i in range(frames_read) if i not in chosen]
        while len(chosen) < n:
            chosen.append(free.pop(rng.randrange(len(free))))
    return sorted((i, i in analysed_set) for i in chosen)


def capture(video_path: Path, indices: Sequence[int], size: tuple[int, int]) -> dict[int, np.ndarray]:
    """Decode exactly the frames ``indices`` by reading the file sequentially - the same frame numbering as the
    analysis pass (seeking by frame number is not exact for every codec). Frames are resized to ``size`` as the
    analysis did for a resolution change."""
    want = set(indices)
    last = max(want)
    out: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    try:
        idx = 0
        while idx <= last:
            if not cap.grab():
                break
            if idx in want:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    if (frame.shape[1], frame.shape[0]) != size:
                        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    out[idx] = frame
            idx += 1
    finally:
        cap.release()
    return out


# ----------------------------------------------------------------------------------------------- tracks
def review_tracks(litter: dict[str, Any], thr: float, min_obs: int = 1) -> dict[int, dict[str, Any]]:
    """Tracks that are NOT litter events but reached the candidate threshold ``thr`` and were seen in at least
    ``min_obs`` analysed frames: reported as UNDER_REVIEW."""
    return {tid: t for tid, t in litter["tracks"].items()
            if not t["is_event"] and t["max_confidence"] >= thr and t["observation_count"] >= min_obs}


def _dets_for_frame(frame_dets: list[dict[str, Any]], event_tracks: set[int],
                    review: set[int]) -> list[Detection]:
    """The pipeline's detections of one analysed frame, re-tiered for the evidence view: event track -> CONFIRMED
    (solid), review candidate -> CANDIDATE (dashed REVIEW); everything else (weak-only tracks, rejected) omitted."""
    out = []
    for d in frame_dets:
        tid = d.get("track_id")
        if d.get("status") == REJECTED or tid is None:
            continue
        if tid in event_tracks:
            status = CONFIRMED
        elif tid in review:
            status = CANDIDATE
        else:
            continue
        out.append(Detection(list(d["bbox"]), -1, d["class"], float(d["confidence"]), status=status, track_id=tid))
    return out


def _banner(img: np.ndarray, text: str) -> None:
    h, w = img.shape[:2]
    fs = float(np.clip(w / 900, 0.7, 1.8))
    thick = max(2, int(round(fs * 2)))
    (tw, th), base = cv2.getTextSize(text, FONT, fs, thick)
    x, y = (w - tw) // 2, h - th - base - int(24 * fs)
    pad = int(12 * fs)
    cv2.rectangle(img, (x - pad, y - pad), (x + tw + pad, y + th + base + pad), (30, 120, 30), -1)
    cv2.putText(img, text, (x, y + th), FONT, fs, (255, 255, 255), thick, cv2.LINE_AA)


def render(frame: np.ndarray, dets: list[Detection], header: Sequence[str], analysed: bool,
           roi: Optional[dict[str, Any]] = None, draw_roi: bool = True) -> np.ndarray:
    """``roi``: roi_service block of this frame (monitored region drawn when ``draw_roi``)."""
    img = annotate_litter(frame, dets, header, show_rejected=False, label_rejected=False,
                          roi=roi if draw_roi else None, roi_text=roi)
    if not any(d.status == CONFIRMED for d in dets):
        n_review = sum(1 for d in dets if d.status == CANDIDATE)
        base = NO_LITTER_MSG if roi and roi.get("applied") else "No litter detected"
        _banner(img, base + (f" - {n_review} under review" if n_review else ""))
    if not analysed:
        h = img.shape[0]
        cv2.putText(img, "Frame outside the sampled analysis: boxes are unconfirmed", (12, h - 12), FONT,
                    float(np.clip(img.shape[1] / 1600, 0.5, 1.0)), (0, 215, 255), 1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------------------------- build
def build_screenshots(*, engine: Any, settings: DetectionSettings, video_path: Path, res: dict[str, Any],
                      analysis_id: str, out_dir: Path, header: Sequence[str],
                      recording_start: Optional[dt.datetime], seed: Optional[int] = None,
                      roi: Any = None) -> dict[str, Any]:
    """Write the four screenshots + screenshots.json for a finished detect_video() result ``res``. Must run while
    the uploaded video still exists (before it is moved into LDM_OUTPUT/.../input)."""
    out_dir = Path(out_dir) / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = default_seed(analysis_id) if seed is None else int(seed)
    video = res["stats"]["video"]
    fps, frames_read = float(video["fps"]), int(video["frames"])
    size = (int(video["width"]), int(video["height"]))
    frames = json.loads(Path(res["files"]["detections"]).read_text(encoding="utf-8"))
    by_idx = {int(f["frame_idx"]): f for f in frames}
    lit = res["litter"]
    event_ids = {e["track_id"]: f"{analysis_id}-E{i:03d}" for i, e in enumerate(lit["events"], start=1)}
    review = set(review_tracks(lit, settings.confidence.candidate_threshold, settings.tracking.min_confirmations))

    picks = select_frames([(f["frame_idx"], f["video_time_sec"]) for f in frames], frames_read, fps, seed)
    images = capture(video_path, [i for i, _ in picks], size)
    if len(images) < len(picks):
        # a frame that decoded in the analysis pass but not now: replace it once with another analysed frame
        missing = [i for i, _ in picks if i not in images]
        spare = [i for i in sorted(by_idx) if i not in {p for p, _ in picks} and i < frames_read]
        rng = random.Random(seed + 1)
        repl = rng.sample(spare, min(len(spare), len(missing)))
        images.update(capture(video_path, repl, size))
        picks = sorted([p for p in picks if p[0] in images] + [(i, True) for i in repl if i in images])
    if len(picks) != N_SCREENSHOTS:
        raise RuntimeError(f"only {len(picks)} of {N_SCREENSHOTS} screenshot frames could be decoded")

    shots = []
    for n, (idx, analysed) in enumerate(picks, start=1):
        frame = images[idx]
        vt = round(idx / fps, 3)
        block = None
        if analysed:
            f = by_idx[idx]
            dets = _dets_for_frame(f["detections"], set(event_ids), review)
            processed = f.get("processed_at")
            if roi is not None:
                block = {**roi.summary(), **(f.get("roi") or {})}
        else:
            raw, _, block = roi_detect(engine, frame, settings, roi, False if not settings.video.tiling else None)
            apply_filters(raw, frame.shape, settings)
            dets = [Detection(d.bbox, d.cls_id, d.cls_name, d.confidence, status=CANDIDATE)
                    for d in raw if d.status != REJECTED and d.confidence >= settings.confidence.candidate_threshold]
            resolve_candidates(block, [])          # no temporal confirmation outside the sampled analysis
            processed = lr.iso(lr.local_now())
        if block is not None and block.get("applied") and block.get("status") in ("checked", "litter", "clean"):
            block["status"] = "litter" if any(d.status == CONFIRMED for d in dets) else "clean"
        when = recording_start + dt.timedelta(seconds=vt) if recording_start else None
        ts = lr.iso(when) if when else processed
        clock = f"{int(vt // 60)}:{vt % 60:04.1f}"
        hud = (*header, f"Screenshot {n} of {N_SCREENSHOTS} - video time {clock} (frame {idx})",
               *( [f"Recorded: {report_time(ts)}"] if when else []))
        img = render(frame, dets, hud, analysed, block, settings.roi.draw_overlay)
        name = f"screenshot_{n:02d}.png"
        if not cv2.imwrite(str(out_dir / name), img):
            raise RuntimeError(f"could not write {name}")
        litter = [d for d in dets if d.status == CONFIRMED]
        shots.append({
            "index": n, "kind": SCREENSHOT_KINDS[n - 1], "file": name, "path": str(out_dir / name),
            "frame_idx": idx, "video_time_sec": vt, "timestamp": report_time(ts),
            "timestamp_source": "RECORDING_TIME" if when else "PROCESSING_TIME", "analysed_frame": analysed,
            "litter_detected": len(litter),
            "under_review": sum(1 for d in dets if d.status == CANDIDATE),
            **({} if block is None else {"roi": {k: block.get(k) for k in (
                "applied", "source", "confidence", "needs_review", "approved", "status", "message", "warning",
                "candidates", "remark_tags", "polygons", "camera_id")}}),
            "detections": [{"class": d.cls_name, "confidence": round(d.confidence, 4),
                            "bbox": [round(v, 1) for v in d.bbox],
                            "detection_status": lr.DETECTED if d.status == CONFIRMED else lr.UNDER_REVIEW,
                            "event_id": event_ids.get(d.track_id) if d.status == CONFIRMED else None}
                           for d in sorted(dets, key=lambda d: -d.confidence)],
        })
    doc = {"analysis_id": analysis_id, "seed": seed, "method": "one random analysed frame per quarter of the "
           "decoded duration (random.Random(seed))", "frames_read": frames_read, "fps": fps,
           "duration_sec": round(frames_read / fps, 3), "screenshots": shots}
    (out_dir / "screenshots.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


# ----------------------------------------------------------------------------------------------- report
def review_rows(lit: dict[str, Any], thr: float, at: Any, min_obs: int = 1) -> list[dict[str, Any]]:
    """UNDER_REVIEW entries (one per review track). ``at(sec, processed_at)`` -> ISO timestamp."""
    out = []
    for tid, t in sorted(review_tracks(lit, thr, min_obs).items(), key=lambda kv: (kv[1]["first_seen_sec"], kv[0])):
        out.append({"track_id": tid, "class": t["class"], "confidence": t["max_confidence"],
                    "first_seen": at(t["first_seen_sec"], t["first_processed_at"]),
                    "first_seen_sec": t["first_seen_sec"], "observation_count": t["observation_count"]})
    return out


def _event_status(review_status: str) -> Optional[str]:
    if review_status == lr.REJECTED_S:
        return None                       # an operator marked it a false positive: never reported as litter
    if review_status in (lr.VERIFIED, lr.RESOLVED):
        return lr.VERIFIED
    return lr.DETECTED


def report_rows(store_id: str, camera_id: str, events: Sequence[dict[str, Any]],
                review: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Six-column rows. ``events``: litter_events DB rows (class, max_confidence, first_seen, review_status);
    ``review``: review_candidates entries."""
    rows = []
    for e in events:
        st = _event_status(e.get("review_status") or lr.DETECTED)
        if st:
            rows.append({"store_id": store_id or lr.UNKNOWN, "camera_id": camera_id or lr.UNKNOWN,
                         "timestamp": report_time(e["first_seen"]), "litter_class": e["class"],
                         "confidence_score": score(e["max_confidence"]), "detection_status": st})
    for r in review:
        rows.append({"store_id": store_id or lr.UNKNOWN, "camera_id": camera_id or lr.UNKNOWN,
                     "timestamp": report_time(r["first_seen"]), "litter_class": r["class"],
                     "confidence_score": score(r["confidence"]), "detection_status": lr.UNDER_REVIEW})
    return sorted(rows, key=lambda r: (r["timestamp"], r["detection_status"] == lr.UNDER_REVIEW))


def report_csv(rows: Sequence[dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=REPORT_COLUMNS, extrasaction="raise", lineterminator="\n")
    w.writeheader()
    w.writerows(safe_rows(rows))
    return buf.getvalue()


def live_report(db: Any, analysis: dict[str, Any]) -> str:
    """The report of a stored analysis built NOW from its DB events (current review status) and its saved review
    candidates - so the download always matches the database."""
    events = db._query("SELECT class, max_confidence, first_seen, review_status FROM litter_events "
                       "WHERE analysis_id=? ORDER BY first_seen, id", (analysis["id"],))
    p = Path((analysis.get("files") or {}).get("review_candidates") or "")
    review = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []
    return report_csv(report_rows(analysis["store_id"], analysis["camera_id"], events, review))


def counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    c = Counter(r["detection_status"] for r in rows)
    return {"rows": len(rows), **{s: c.get(s, 0) for s in REPORT_STATUSES},
            "classes": sorted({r["litter_class"] for r in rows if r["detection_status"] != lr.UNDER_REVIEW})}
