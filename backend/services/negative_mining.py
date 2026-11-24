"""Negative mining: run the detector over a video and save EVERY detection for human review.

Owner request 2026-11-24. Output (one folder per video under storage/review/):

    storage/review/<video_id>/
        frames/f<frame_idx>.jpg        full frames that had a detection, plus a capped sample of clean frames
        crops/<det_id>.jpg             padded crop of each detection
        detections.csv                 one row per detection - the reviewer fills ``label`` (and optionally true_class)
        frames.csv                     one row per sampled frame (n_detections, saved, clean)
        manifest.json                  model, thresholds, sampling, video duration, ROI used for in_roi

Labels: TP / FP_shadow / FP_carton / FP_stock / FP_puddle / FP_other. ``true_class`` lets a reviewer correct the
class of a TP (empty = the detector's class). training/negative_mining/build_dataset.py turns the labelled folders
into a YOLO dataset; training/negative_mining/report_metrics.py reports precision / recall / false alerts per hour.

The detector is passed in as a function (frame -> detections), so mining is independent of the engine and testable.
A labelled detections.csv is never overwritten unless ``force=True``.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import cv2
import numpy as np

from backend.detection.types import Detection

LABELS = ("TP", "FP_shadow", "FP_carton", "FP_stock", "FP_puddle", "FP_other")
NEGATIVE_LABELS = {"FP_shadow": "fp_shadow", "FP_carton": "fp_carton", "FP_stock": "fp_stock",
                   "FP_puddle": "fp_puddle"}           # become explicit negative classes; FP_other gets no box
DET_COLUMNS = ["det_id", "video_id", "frame_idx", "timestamp_sec", "frame_file", "crop_file", "class", "conf",
               "x1", "y1", "x2", "y2", "frame_w", "frame_h", "status", "filter_reason", "in_roi", "label",
               "true_class", "notes"]
FRAME_COLUMNS = ["video_id", "frame_idx", "timestamp_sec", "frame_file", "n_detections", "saved", "clean"]
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")

DetectFn = Callable[[np.ndarray], list[Detection]]


def video_id_for(path: Path) -> str:
    """Stable id: file stem (sanitised) + 8 hex of the content hash of its first MiB (renames keep the hash)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(1 << 20))
    stem = re.sub(r"[^A-Za-z0-9_\-]", "_", Path(path).stem)[:60] or "video"
    return f"{stem}_{h.hexdigest()[:8]}"


def crop_box(img: np.ndarray, bbox: Sequence[float], pad: float = 0.25, min_side: int = 48) -> np.ndarray:
    """Crop with context padding (reviewers need the surroundings to tell a shadow from a wrapper)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox[:4]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    px, py = max(bw * pad, (min_side - bw) / 2), max(bh * pad, (min_side - bh) / 2)
    a, b = int(max(0, math.floor(x1 - px))), int(max(0, math.floor(y1 - py)))
    c, d = int(min(w, math.ceil(x2 + px))), int(min(h, math.ceil(y2 + py)))
    return img[b:max(b + 1, d), a:max(a + 1, c)]


@dataclass
class MiningResult:
    video_id: str
    out_dir: Path
    sampled_frames: int
    detections: int
    frames_with_detections: int
    clean_frames_saved: int
    duration_sec: float


def _labelled(csv_path: Path) -> bool:
    if not csv_path.is_file():
        return False
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return any((r.get("label") or "").strip() for r in csv.DictReader(fh))


def mine_video(video_path: Path, detect: DetectFn, review_root: Path, *, video_id: Optional[str] = None,
               sample_every_sec: float = 1.0, max_clean_frames: int = 60, roi: Optional[dict[str, Any]] = None,
               meta: Optional[dict[str, Any]] = None, force: bool = False,
               progress: Optional[Callable[[int, int], None]] = None) -> MiningResult:
    """Sample one frame every ``sample_every_sec`` (sequential read - HEVC seeking is unreliable), detect, save."""
    from backend.services.auto_roi import detection_in_roi

    video_path = Path(video_path)
    vid = video_id or video_id_for(video_path)
    if not VIDEO_ID_RE.match(vid):
        raise ValueError(f"invalid video_id {vid!r}")
    out = Path(review_root) / vid
    if _labelled(out / "detections.csv") and not force:
        raise FileExistsError(f"{out / 'detections.csv'} already has labels; pass force=True to overwrite it")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * sample_every_sec)))
    expected = max(1, (total + step - 1) // step) if total else 0
    clean_stride = max(1, math.ceil(expected / max_clean_frames)) if (expected and max_clean_frames) else 1
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    det_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    idx = sampled = with_dets = clean_saved = clean_seen = 0
    try:
        while True:
            if not cap.grab():
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    sampled += 1
                    ts = round(idx / fps, 3)
                    h, w = frame.shape[:2]
                    dets = detect(frame)
                    fname = f"f{idx:07d}.jpg"
                    saved = False
                    if dets:
                        with_dets += 1
                        cv2.imwrite(str(out / "frames" / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        saved = True
                        for k, d in enumerate(dets):
                            det_id = f"{vid}_f{idx:07d}_d{k:02d}"
                            cv2.imwrite(str(out / "crops" / f"{det_id}.jpg"), crop_box(frame, d.bbox),
                                        [cv2.IMWRITE_JPEG_QUALITY, 92])
                            det_rows.append({
                                "det_id": det_id, "video_id": vid, "frame_idx": idx, "timestamp_sec": ts,
                                "frame_file": f"frames/{fname}", "crop_file": f"crops/{det_id}.jpg",
                                "class": d.cls_name, "conf": round(float(d.confidence), 4),
                                "x1": round(d.bbox[0], 1), "y1": round(d.bbox[1], 1),
                                "x2": round(d.bbox[2], 1), "y2": round(d.bbox[3], 1), "frame_w": w, "frame_h": h,
                                "status": d.status, "filter_reason": d.reason or "",
                                "in_roi": "" if roi is None else int(detection_in_roi(d.bbox, w, h, roi)),
                                "label": "", "true_class": "", "notes": ""})
                    else:
                        if clean_seen % clean_stride == 0 and clean_saved < max_clean_frames:
                            cv2.imwrite(str(out / "frames" / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                            saved, clean_saved = True, clean_saved + 1
                        clean_seen += 1
                    frame_rows.append({"video_id": vid, "frame_idx": idx, "timestamp_sec": ts,
                                       "frame_file": f"frames/{fname}" if saved else "", "n_detections": len(dets),
                                       "saved": int(saved), "clean": int(not dets)})
                    if progress:
                        progress(sampled, expected)
            idx += 1
    finally:
        cap.release()
    duration = round(idx / fps, 3)
    write_csv(out / "detections.csv", DET_COLUMNS, det_rows)
    write_csv(out / "frames.csv", FRAME_COLUMNS, frame_rows)
    manifest = {"video_id": vid, "video_file": video_path.name, "fps": fps, "frames_total": idx,
                "duration_sec": duration, "sample_every_sec": sample_every_sec, "sampled_frames": sampled,
                "detections": len(det_rows), "frames_with_detections": with_dets, "clean_frames_saved": clean_saved,
                "max_clean_frames": max_clean_frames, "labels": list(LABELS), "roi": roi,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **(meta or {})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return MiningResult(vid, out, sampled, len(det_rows), with_dets, clean_saved, duration)


# ---------------------------------------------------------------------- CSV io
def write_csv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def normalise_label(raw: Optional[str]) -> Optional[str]:
    """Case/spacing-tolerant label parse ("fp shadow" -> FP_shadow). Empty -> None; unknown -> ValueError."""
    s = (raw or "").strip()
    if not s:
        return None
    key = re.sub(r"[\s\-]+", "_", s).lower()
    for lab in LABELS:
        if lab.lower() == key:
            return lab
    raise ValueError(f"unknown label {raw!r}; allowed: {', '.join(LABELS)}")


def load_review(video_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    video_dir = Path(video_dir)
    manifest = json.loads((video_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest, read_csv(video_dir / "detections.csv"), read_csv(video_dir / "frames.csv")


def review_dirs(roots: Iterable[Path]) -> list[Path]:
    """Every <video_id> folder (has manifest.json + detections.csv) under the given roots (or the folders themselves)."""
    out: list[Path] = []
    for r in roots:
        r = Path(r)
        if (r / "manifest.json").is_file() and (r / "detections.csv").is_file():
            out.append(r)
            continue
        out += sorted(p for p in r.iterdir() if p.is_dir() and (p / "manifest.json").is_file()
                      and (p / "detections.csv").is_file()) if r.is_dir() else []
    return out
