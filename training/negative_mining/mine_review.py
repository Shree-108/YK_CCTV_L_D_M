"""Run the detector on a video and save every detection for review (backend/services/negative_mining.py).

    python -m training.negative_mining.mine_review storage/media/<video>.mp4 \
        [--review-root storage/review] [--weights models/litter_yolo/v3_store_cctv/best.pt] \
        [--every 1.0] [--roi auto|none|<roi.json>|camera:<CAM_ID>] [--max-clean 60] [--force]

Detection settings = the app's global settings from the database (read-only), else the defaults - the same
thresholds/tiling the app uses. Every detection the model returns (>= weak_threshold) is saved, including ones the
size/aspect filters reject (status/filter_reason columns), so the reviewer sees exactly what the detector produced.

--roi auto computes the auto ROI from the median of 7 frames of this video (needs_review - it only fills the in_roi
column, nothing is dropped) and writes roi_overlay.jpg into the review folder.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2

from backend.core.config import AppConfig
from backend.core.settings import DetectionSettings, merged_settings
from backend.detection.engine import YoloEngine
from backend.detection.filters import apply_filters
from backend.services import auto_roi as ar
from backend.services.negative_mining import mine_video, video_id_for

DEFAULT_WEIGHTS = Path("models/litter_yolo/v3_store_cctv/best.pt")


def app_settings(db_path: Path) -> tuple[DetectionSettings, str]:
    try:
        c = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        row = c.execute("SELECT value_json FROM settings WHERE key='global'").fetchone()
        c.close()
        return merged_settings(json.loads(row[0]) if row else {}), ("database" if row else "defaults")
    except sqlite3.Error:
        return merged_settings({}), "defaults (database unreadable)"


def resolve_roi(spec: str, video: Path, out_dir: Path, storage_dir: Path) -> Optional[dict[str, Any]]:
    if spec == "none":
        return None
    if spec.startswith("camera:"):
        rec, src = ar.store_for(storage_dir).resolve(None, spec.split(":", 1)[1])
        if rec is None:
            raise SystemExit(f"no ROI saved for {spec} (and no global ROI)")
        return {**rec, "resolved_from": src}
    if spec != "auto":
        return json.loads(Path(spec).read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(video))
    frames = ar.sample_frames(cap, 7)
    cap.release()
    res = ar.compute_auto_roi(frames)
    if res.record is None:
        print(f"auto ROI: {res.note} - in_roi left empty")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "roi_overlay.jpg"), ar.render_overlay(res.reference, res.record))
    cv2.imwrite(str(out_dir / "roi_overlay_frame.jpg"), ar.render_overlay(frames[len(frames) // 2], res.record))
    return res.record


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--review-root", type=Path, default=None, help="default: <storage>/review")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--every", type=float, default=1.0, help="seconds between sampled frames")
    ap.add_argument("--max-clean", type=int, default=60, help="max clean (no-detection) frames kept as background")
    ap.add_argument("--roi", default="auto")
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an already labelled detections.csv")
    a = ap.parse_args(argv)

    cfg = AppConfig()
    root = a.review_root or (cfg.storage_dir / "review")
    settings, origin = app_settings(cfg.db_path)
    meta_file = a.weights.with_name("model_meta.json")
    version = json.loads(meta_file.read_text(encoding="utf-8")).get("version", a.weights.parent.name) \
        if meta_file.is_file() else a.weights.parent.name
    engine = YoloEngine(a.weights, settings, version)
    vid = a.video_id or video_id_for(a.video)
    roi = resolve_roi(a.roi, a.video, root / vid, cfg.storage_dir)

    def detect(frame):
        dets, _ = engine.detect(frame, settings)
        return apply_filters(dets, frame.shape, settings)

    t0 = time.time()
    last = [0.0]

    def progress(i: int, n: int) -> None:
        if time.time() - last[0] > 15:
            last[0] = time.time()
            print(f"  {i}/{n or '?'} frames, {time.time() - t0:.0f} s", flush=True)

    c = settings.confidence
    meta = {"model_version": version, "weights": a.weights.as_posix(), "class_names": engine.class_names,
            "settings_origin": origin,
            "thresholds": {"weak_threshold": c.weak_threshold, "candidate_threshold": c.candidate_threshold,
                           "confirmed_threshold": c.confirmed_threshold},
            "tiling": settings.tiling.model_dump(), "image_size": settings.inference.image_size,
            "roi_spec": a.roi}
    try:
        res = mine_video(a.video, detect, root, video_id=vid, sample_every_sec=a.every, max_clean_frames=a.max_clean,
                         roi=roi, meta=meta, force=a.force, progress=progress)
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"video_id": res.video_id, "out": str(res.out_dir), "sampled_frames": res.sampled_frames,
                      "detections": res.detections, "frames_with_detections": res.frames_with_detections,
                      "clean_frames_saved": res.clean_frames_saved, "duration_sec": res.duration_sec,
                      "elapsed_sec": round(time.time() - t0, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
