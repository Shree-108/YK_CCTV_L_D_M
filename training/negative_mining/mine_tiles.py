"""Mine a video at LOW confidence with the app's own detector path, for keyboard review (phase 2 item 6).

    python -m training.negative_mining.mine_tiles storage/media/<video>.mp4 [--camera CAM_STREET_01] \
        [--conf 0.10] [--every 1.0] [--cap 800] [--crop-side 256] [--review-root storage/review] [--force]

Same path as inference (backend/services/roi_service.roi_detect): monitored region (the camera's saved region, else
computed for this video and not saved) -> uniform gate -> masked ROI crop -> detector at roi.image_size with the
inference tiles -> boxes mapped back -> bottom-centre / overlap rule. Only the confidence floor is lowered to
``--conf`` (default 0.10), so every box between 0.10 and the app's 0.30 weak threshold is recorded too
(``below_030`` = 1). The size / aspect filters only set ``status`` / ``filter_reason``; nothing is dropped.

Crops are cut at NATIVE pixel scale (never resized) - the scale the inference tiles see - as a square window of
``max(--crop-side, 1.6 x the box)`` around the box; ``crop_x/y/w/h`` record the window in frame pixels.
``group_id`` chains the same box (IoU >= 0.6, class-agnostic) through consecutive sampled frames, so the review page
can label a static object once. At most ``--cap`` crops are kept per class: one per group first (spread over the
confidence range), then other members; the rest keep their CSV row with ``capped`` = 1 and no crop.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from backend.core.config import AppConfig
from backend.detection.engine import YoloEngine
from backend.detection.filters import apply_filters
from backend.detection.tiling import iou_matrix
from backend.services import roi_service as rs
from backend.services.negative_mining import (DET_COLUMNS, FRAME_COLUMNS, LABELS, VIDEO_ID_RE, _labelled,
                                              video_id_for, write_csv)
from training.negative_mining.mine_review import DEFAULT_WEIGHTS, app_settings

EXTRA_COLUMNS = ["below_030", "group_id", "crop_x", "crop_y", "crop_w", "crop_h", "capped"]
COLUMNS = DET_COLUMNS + EXTRA_COLUMNS
APP_WEAK = 0.30


def crop_window(bbox: Sequence[float], w: int, h: int, side: int) -> tuple[int, int, int, int]:
    """Square window (x, y, cw, ch) of max(side, 1.6 x box) native pixels centred on the box, shifted inside."""
    x1, y1, x2, y2 = bbox[:4]
    s = int(math.ceil(max(side, 1.6 * max(x2 - x1, y2 - y1))))
    s = min(s, w, h) if min(w, h) > 0 else s
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x = int(round(min(max(0, cx - s / 2), max(0, w - s))))
    y = int(round(min(max(0, cy - s / 2), max(0, h - s))))
    return x, y, min(s, w - x), min(s, h - y)


def assign_groups(prev: list[tuple[str, list[float]]], boxes: list[list[float]], next_id: int, vid: str,
                  thr: float = 0.6) -> tuple[list[str], int]:
    """Greedy IoU chaining to the previous sampled frame's boxes -> group ids for ``boxes``."""
    out = [""] * len(boxes)
    if prev and boxes:
        m = iou_matrix(np.array(boxes, np.float32), np.array([b for _, b in prev], np.float32))
        used: set[int] = set()
        for i, j in sorted(((i, j) for i in range(len(boxes)) for j in range(len(prev))),
                           key=lambda ij: -m[ij[0], ij[1]]):
            if m[i, j] < thr:
                break
            if out[i] or j in used:
                continue
            out[i], _ = prev[j][0], used.add(j)
    for i in range(len(boxes)):
        if not out[i]:
            out[i] = f"{vid}_g{next_id:05d}"
            next_id += 1
    return out, next_id


def select_capped(rows: list[dict[str, Any]], cap: int) -> set[str]:
    """det_ids that KEEP their crop: per class, one representative per group (highest confidence) spread evenly over
    the confidence range, then other members spread evenly, up to ``cap``."""
    keep: set[str] = set()
    by_cls: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cls.setdefault(r["class"], []).append(r)

    def spread(items: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        if n >= len(items):
            return items
        items = sorted(items, key=lambda r: r["conf"])
        idx = np.linspace(0, len(items) - 1, n).round().astype(int)
        return [items[i] for i in sorted(set(idx.tolist()))]

    for cls_rows in by_cls.values():
        if len(cls_rows) <= cap:
            keep |= {r["det_id"] for r in cls_rows}
            continue
        reps: dict[str, dict[str, Any]] = {}
        for r in cls_rows:
            g = r["group_id"]
            if g not in reps or r["conf"] > reps[g]["conf"]:
                reps[g] = r
        chosen = spread(list(reps.values()), cap)
        left = cap - len(chosen)
        if left > 0:
            ids = {r["det_id"] for r in chosen}
            chosen += spread([r for r in cls_rows if r["det_id"] not in ids], left)
        keep |= {r["det_id"] for r in chosen}
    return keep


def resolve_ctx(camera: Optional[str], video: Path, settings: Any, cfg: AppConfig) -> Optional[rs.RoiContext]:
    svc = rs.RoiService(cfg.roi_store_path, enabled=True)
    s = settings.model_copy(update={"roi": settings.roi.model_copy(update={"enabled": True})})
    if camera:
        rec = svc.store.get(camera)
        if rec and rec.get("polygons"):
            return rs.RoiContext(camera, rec, rec.get("source", "auto"))
    frames = rs.video_frames(video, s.roi.vote_frames, None)
    return svc.resolve(None, frames, s, save=False)


def mine(video: Path, *, camera: Optional[str], conf: float, every: float, cap: int, crop_side: int, review_root: Path,
         video_id: Optional[str], weights: Path, force: bool, max_clean: int = 60) -> dict[str, Any]:
    cfg = AppConfig()
    base, origin = app_settings(cfg.db_path)
    settings = base.model_copy(update={"confidence": base.confidence.model_copy(update={"weak_threshold": conf})})
    meta_file = weights.with_name("model_meta.json")
    version = json.loads(meta_file.read_text(encoding="utf-8")).get("version", weights.parent.name) \
        if meta_file.is_file() else weights.parent.name
    engine = YoloEngine(weights, settings, version)
    vid = video_id or video_id_for(video)
    if not VIDEO_ID_RE.match(vid):
        raise ValueError(f"invalid video id {vid!r}")
    out = review_root / vid
    if _labelled(out / "detections.csv") and not force:
        raise FileExistsError(f"{out / 'detections.csv'} already has labels; pass --force to overwrite it")
    ctx = resolve_ctx(camera, video, settings, cfg)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    cap_ = cv2.VideoCapture(str(video))
    if not cap_.isOpened():
        raise ValueError(f"cannot open {video}")
    fps = float(cap_.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    total = int(cap_.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * every)))
    rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    prev: list[tuple[str, list[float]]] = []
    next_gid, idx, sampled, clean_saved, size = 1, 0, 0, 0, [0, 0]
    t0, last = time.time(), 0.0
    try:
        while cap_.grab():
            if idx % step == 0:
                ok, frame = cap_.retrieve()
                if ok and frame is not None:
                    sampled += 1
                    h, w = frame.shape[:2]
                    size = [w, h]
                    dets, _, block = rs.roi_detect(engine, frame, settings, ctx, people=None)
                    apply_filters(dets, frame.shape, settings)
                    fname = f"f{idx:07d}.jpg"
                    saved = False
                    if dets or clean_saved < max_clean:
                        cv2.imwrite(str(out / "frames" / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        saved = True
                        clean_saved += int(not dets)
                    gids, next_gid = assign_groups(prev, [list(d.bbox) for d in dets], next_gid, vid)
                    prev = list(zip(gids, [list(d.bbox) for d in dets]))
                    for k, (d, g) in enumerate(zip(dets, gids)):
                        det_id = f"{vid}_f{idx:07d}_d{k:02d}"
                        cx, cy, cw, ch = crop_window(d.bbox, w, h, crop_side)
                        cv2.imwrite(str(out / "crops" / f"{det_id}.jpg"), frame[cy:cy + ch, cx:cx + cw],
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                        rows.append({
                            "det_id": det_id, "video_id": vid, "frame_idx": idx, "timestamp_sec": round(idx / fps, 3),
                            "frame_file": f"frames/{fname}", "crop_file": f"crops/{det_id}.jpg", "class": d.cls_name,
                            "conf": round(float(d.confidence), 4), "x1": round(d.bbox[0], 1),
                            "y1": round(d.bbox[1], 1), "x2": round(d.bbox[2], 1), "y2": round(d.bbox[3], 1),
                            "frame_w": w, "frame_h": h, "status": d.status, "filter_reason": d.reason or "",
                            "in_roi": "" if block is None else int(bool(block.get("applied"))), "label": "",
                            "true_class": "", "notes": "", "below_030": int(d.confidence < APP_WEAK), "group_id": g,
                            "crop_x": cx, "crop_y": cy, "crop_w": cw, "crop_h": ch, "capped": 0})
                    frame_rows.append({"video_id": vid, "frame_idx": idx, "timestamp_sec": round(idx / fps, 3),
                                       "frame_file": f"frames/{fname}" if saved else "", "n_detections": len(dets),
                                       "saved": int(saved), "clean": int(not dets)})
                    if time.time() - last > 30:
                        last = time.time()
                        print(f"  {sampled}/{(total + step - 1) // step if total else '?'} frames, {len(rows)} boxes, "
                              f"{time.time() - t0:.0f} s", flush=True)
            idx += 1
    finally:
        cap_.release()
    keep = select_capped(rows, cap)
    for r in rows:
        if r["det_id"] not in keep:
            (out / r["crop_file"]).unlink(missing_ok=True)
            r.update(capped=1, crop_file="")
    write_csv(out / "detections.csv", COLUMNS, rows)
    write_csv(out / "frames.csv", FRAME_COLUMNS, frame_rows)
    c = settings.confidence
    by_cls: dict[str, dict[str, int]] = {}
    for r in rows:
        b = by_cls.setdefault(r["class"], {"boxes": 0, "below_030": 0, "crops_kept": 0, "groups": 0})
        b["boxes"] += 1
        b["below_030"] += r["below_030"]
        b["crops_kept"] += int(not r["capped"])
    for cls in by_cls:
        by_cls[cls]["groups"] = len({r["group_id"] for r in rows if r["class"] == cls})
    manifest = {
        "video_id": vid, "video_file": video.name, "video_path": str(video), "fps": fps, "frames_total": idx,
        "duration_sec": round(idx / fps, 3), "frame_size": size, "sample_every_sec": every,
        "sampled_frames": sampled, "detections": len(rows), "labels": list(LABELS),
        "miner": "mine_tiles", "mining_conf": conf, "app_weak_threshold": APP_WEAK, "cap_per_class": cap,
        "crop_side": crop_side, "crop_scale": "native (1.0) = the inference tile scale",
        "per_class": by_cls, "model_version": version, "weights": weights.as_posix(),
        "class_names": engine.class_names, "settings_origin": origin,
        "thresholds": {"weak_threshold": conf, "candidate_threshold": c.candidate_threshold,
                       "confirmed_threshold": c.confirmed_threshold},
        "tiling": settings.tiling.model_dump(), "image_size": settings.inference.image_size,
        "roi_image_size": settings.roi.image_size, "roi_tiling": settings.roi.tiling,
        "roi": None if ctx is None else {k: v for k, v in ctx.summary().items() if k in (
            "applied", "source", "camera_id", "confidence", "needs_review", "approved", "area_frac", "polygons")},
        "camera_id": camera, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--camera", default=None, help="use this camera's saved monitored region")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--every", type=float, default=1.0, help="seconds between sampled frames")
    ap.add_argument("--cap", type=int, default=800, help="max crops kept per class")
    ap.add_argument("--crop-side", type=int, default=256, help="min crop window (native pixels)")
    ap.add_argument("--review-root", type=Path, default=None, help="default: <storage>/review")
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    root = a.review_root or (AppConfig().storage_dir / "review")
    try:
        m = mine(a.video, camera=a.camera, conf=a.conf, every=a.every, cap=a.cap, crop_side=a.crop_side,
                 review_root=root, video_id=a.video_id, weights=a.weights, force=a.force)
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: m[k] for k in ("video_id", "sampled_frames", "detections", "per_class", "elapsed_sec")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
