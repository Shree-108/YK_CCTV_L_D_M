"""
video_pipeline.py — maximum-accuracy VIDEO litter detection.

Adapted to our stack: there is no YOLO best.pt here. The "ensemble" is votes
across (a) detector engines — tiny-CNN > RandomForest > heuristic, whichever are
loaded, (b) working scales, (c) tiles, and (d) TTA flips — merged with Weighted
Box Fusion instead of NMS. Temporal fusion then turns per-frame votes into
confirmed tracks. Every technique is toggleable via the VIDEO_DEFAULTS block /
litter_config.json "video" section, so the same code runs a fast path (single
engine, no TTA/tiles) or maximum-accuracy mode (everything on).

Why each technique helps VIDEO specifically (vs single images):
  * tiling      — tiny chips packets are unresolvable in a downscaled full frame
                  but clear inside a 1:1 tile; video frames are high-res.
  * TTA flip    — a second independent vote on the same frame; cheap vs re-encode.
  * quality gate— video has blurry/exposed frames; they must count LESS, not equal.
  * temporal    — N independent looks at one stationary object: quality-weighted
                  confidence + median consensus box + majority-vote label beats
                  any single frame and kills single-frame false positives.
  * flow assist — bridges 1-2 frame detection gaps so one miss doesn't break a
                  track's persistence (the spam/flicker in the live view).
  * soft-NMS    — touching packets (explicit project requirement) both survive.
  * calibration — per-class thresholds (tuned on val) instead of one global knob.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

import litter_ml
from litter_ml import box_iou

VIDEO_DEFAULTS = {
    "enable_tta": False,          # 2x cost, typically +1-3% — toggle
    "enable_tiling": True,        # highest leverage for tiny packets
    "tile_size": 640,
    "tile_overlap": 0.25,
    "tile_min_ratio": 1.8,        # tile only if width/working_side > this
    "working_scales": [960],      # extra scales = extra votes (e.g. [960, 1280])
    "enable_quality_gate": True,
    "blur_threshold": 60.0,       # Laplacian variance below => blurry
    "dark_frac": 0.02,            # mean gray below => near-black/IR failure
    "bright_frac": 0.98,          # mean gray above => glare
    "enable_temporal": True,
    "iou_match": 0.30,            # track match IoU (static CCTV => generous)
    "min_persistence": 3,         # hits needed to confirm a track
    "max_gap": 2,                 # frames a track survives unmatched
    "enable_flow": True,          # LK gap-fill (needs temporal)
    "use_soft_nms": True,
    "soft_nms_sigma": 0.5,
    "soft_nms_nt": 0.5,
    "wbf_iou": 0.55,
    "sample_fps": 2.0,
    "max_frames": 24,
}


def video_cfg() -> dict:
    cfg = litter_ml.load_config().get("video", {})
    out = dict(VIDEO_DEFAULTS)
    out.update(cfg if isinstance(cfg, dict) else {})
    return out


# ---------------------------------------------------------------------------
# Frame-quality gating — WHY video: frames vary in blur/exposure; weight them.
# ---------------------------------------------------------------------------

def frame_quality(img_bgr: np.ndarray, vcfg: dict) -> dict:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean()) / 255.0
    reasons = []
    usable = True
    if blur < float(vcfg["blur_threshold"]):
        usable = False
        reasons.append(f"blurry({blur:.0f})")
    if mean < float(vcfg["dark_frac"]):
        usable = False
        reasons.append("near-black")
    elif mean > float(vcfg["bright_frac"]):
        usable = False
        reasons.append("glare")
    # quality weight in [0.15, 1.0]: blurry frames count less, never zero
    weight = float(np.clip(blur / max(1.0, 3.0 * float(vcfg["blur_threshold"])), 0.15, 1.0))
    if not usable:
        weight = min(weight, 0.35)
    return {"blur": round(blur, 1), "exposure": round(mean, 3),
            "usable": usable, "reasons": reasons, "weight": round(weight, 3)}


# ---------------------------------------------------------------------------
# Tiled inference — WHY video: full-frame downscale erases tiny packets; 1:1
# tiles preserve them. Highest-leverage step for high-res CCTV.
# ---------------------------------------------------------------------------

def _tiles_for(w: int, h: int, tile: int, overlap: float):
    step = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), step))
    ys = list(range(0, max(1, h - tile + 1), step))
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    return [(x, y) for y in ys for x in xs]


def detect_tiled(det, img_bgr, vcfg, raw: bool) -> list[dict]:
    h, w = img_bgr.shape[:2]
    work_side = max(det.cfg.get("image_size", 960), 1)
    if max(w, h) / max(1, work_side) < float(vcfg["tile_min_ratio"]):
        return []
    out = []
    for x, y in _tiles_for(w, h, int(vcfg["tile_size"]), float(vcfg["tile_overlap"])):
        tile = img_bgr[y:y + int(vcfg["tile_size"]), x:x + int(vcfg["tile_size"])]
        for d in det.detect(tile, raw=raw):
            b = d["box"]
            out.append({**d, "box": [b[0] + x, b[1] + y, b[2] + x, b[3] + y]})
    return out


# ---------------------------------------------------------------------------
# Fusion math: WBF + Soft-NMS
# ---------------------------------------------------------------------------

def weighted_box_fusion(dets: list[dict], iou_thr: float = 0.55) -> list[dict]:
    """Cluster same-label boxes by IoU; average coords/conf weighted by conf.

    Unlike NMS (keep-best, delete rest), WBF keeps evidence from every vote, so
    a box found confidently at only one scale/tile survives with averaged coords.
    """
    if not dets:
        return []
    by_label: dict[str, list[dict]] = {}
    for d in dets:
        by_label.setdefault(d["class"], []).append(d)
    fused = []
    for label, items in by_label.items():
        boxes = np.array([d["box"] for d in items], np.float32)
        confs = np.array([d["confidence"] for d in items], np.float32)
        used = np.zeros(len(items), bool)
        order = np.argsort(-confs)
        for i in order:
            if used[i]:
                continue
            ious = box_iou(boxes[i:i + 1], boxes)[0]
            cluster = [j for j in order
                       if not used[j] and ious[j] >= iou_thr] or [i]
            for j in cluster:
                used[j] = True
            w = confs[cluster]
            w = w / w.sum()
            avg = (boxes[cluster] * w[:, None]).sum(0)
            fused.append({**items[i], "box": [round(float(v), 2) for v in avg],
                          "confidence": round(float(confs[cluster].mean()), 4),
                          "votes": len(cluster)})
    return fused


def soft_nms(dets: list[dict], sigma: float = 0.5, nt: float = 0.5,
             score_thr: float = 0.05) -> list[dict]:
    """Gaussian Soft-NMS: decay (don't delete) overlapping boxes.

    WHY: touching/overlapping chips packets must BOTH survive; hard NMS deletes
    the lower one. Decay lets genuinely separate objects stay above threshold.
    """
    if not dets:
        return []
    dets = [dict(d) for d in dets]
    for d in dets:
        d.setdefault("votes", 1)
    out = []
    while dets:
        dets.sort(key=lambda d: -d["confidence"])
        best = dets.pop(0)
        out.append(best)
        if not dets:
            break
        boxes = np.array([d["box"] for d in dets], np.float32)
        ious = box_iou(np.array([best["box"]], np.float32), boxes)[0]
        keep = []
        for d, iou in zip(dets, ious):
            if iou > nt:
                d["confidence"] = round(float(d["confidence"] * np.exp(-(iou ** 2) / sigma)), 4)
            if d["confidence"] >= score_thr:
                keep.append(d)
        dets = keep
    return out


# ---------------------------------------------------------------------------
# Per-frame detection: ensemble votes -> WBF -> soft-NMS -> calibrated gates
# ---------------------------------------------------------------------------

def _raw_votes(det, frame: np.ndarray, vcfg: dict, max_side: int) -> list[dict]:
    votes = det.detect(frame, raw=True, working_max_side=max_side)
    if vcfg["enable_tta"]:
        # WHY video: TTA = a second independent vote on the SAME frame without
        # needing another (possibly blurry) frame. ~2x cost, +1-3% typical gain.
        flip = cv2.flip(frame, 1)
        w = frame.shape[1]
        for d in det.detect(flip, raw=True, working_max_side=max_side):
            b = d["box"]
            votes.append({**d, "box": [w - b[2], b[1], w - b[0], b[3]]})
    if vcfg["enable_tiling"]:
        votes += detect_tiled(det, frame, vcfg, raw=True)
    return votes


def detect_frame(det, frame: np.ndarray, vcfg: dict) -> tuple[list[dict], dict]:
    t0 = time.time()
    votes = []
    for side in vcfg.get("working_scales", [960]):
        votes += _raw_votes(det, frame, vcfg, int(side))
    t_vote = time.time() - t0
    fused = weighted_box_fusion(votes, float(vcfg["wbf_iou"]))
    if vcfg["use_soft_nms"]:
        fused = soft_nms(fused, float(vcfg["soft_nms_sigma"]), float(vcfg["soft_nms_nt"]))
    # calibrated gates: per-class tuned threshold + global anti-spam floor + size
    floor = float(det.cfg.get("global_floor", 0.0))
    min_area = float(det.cfg.get("min_box_area", 0.0))
    gated = []
    for d in fused:
        thr = max(litter_ml.threshold_for(det.cfg, d["class"]), floor)
        if d["confidence"] < thr:
            continue
        area = (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
        if area < min_area:
            continue
        d["area"] = round(float(area), 2)
        gated.append(d)
    gated.sort(key=lambda d: -d["confidence"])
    gated = gated[: int(det.cfg.get("max_detections", 100))]
    return gated, {"votes": len(votes), "fused": len(fused),
                   "vote_s": round(t_vote, 3)}


# ---------------------------------------------------------------------------
# Temporal fusion — WHY video: N looks at one stationary object beat one look.
# IoU tracker + quality-weighted confidence + median consensus box +
# confidence-weighted majority-vote label. Feeds the same alert dicts as before.
# ---------------------------------------------------------------------------

class Track:
    _next = 1

    def __init__(self, det: dict, frame_idx: int, quality: dict):
        self.id = Track._next
        Track._next += 1
        self.hits = [(frame_idx, det, quality)]
        self.misses = 0
        self.last_box = list(det["box"])

    def predict_box(self):
        return self.last_box


def _match(tracks: list[Track], dets: list[dict], iou_thr: float):
    if not tracks or not dets:
        return {}, set(range(len(dets)))
    tb = np.array([t.predict_box() for t in tracks], np.float32)
    db = np.array([d["box"] for d in dets], np.float32)
    iou = box_iou(tb, db)
    pairs, used_t, used_d = {}, set(), set()
    flat = sorted([(iou[i, j], i, j) for i in range(len(tracks))
                   for j in range(len(dets))], reverse=True)
    for v, i, j in flat:
        if v < iou_thr or i in used_t or j in used_d:
            continue
        # same-label preferred, but allow cross-label (vote decides later)
        pairs[i] = j
        used_t.add(i)
        used_d.add(j)
    return pairs, set(range(len(dets))) - used_d


def _lk_shift(prev_gray, gray, box) -> list[float] | None:
    """Sparse LK optical flow on box corners+center: propagate through gaps.

    WHY video: a single missed frame shouldn't break persistence; flow gives a
    plausible carried-forward position instead of trusting detector silence.
    """
    try:
        x1, y1, x2, y2 = box
        pts = np.array([[[x1, y1]], [[x2, y1]], [[x1, y2]], [[x2, y2]],
                        [[(x1 + x2) / 2, (y1 + y2) / 2]]], np.float32)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None,
                                              winSize=(21, 21), maxLevel=3)
        if nxt is None or st is None:
            return None
        d = (nxt - pts).reshape(-1, 2)[st.reshape(-1) == 1]
        if len(d) == 0:
            return None
        dx, dy = float(np.median(d[:, 0])), float(np.median(d[:, 1]))
        if abs(dx) + abs(dy) > 60:  # implausible jump => camera cut, don't carry
            return None
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
    except Exception:
        return None


def fuse_track(track: Track) -> dict:
    """Quality-weighted mean conf + median consensus box + voted label."""
    bbb = []
    confs, weights, labels = [], [], []
    for _, det, q in track.hits:
        if det.get("propagated"):
            continue  # flow-filled frames don't vote on geometry
        bbb.append(det["box"])
        confs.append(det["confidence"])
        weights.append(q.get("weight", 1.0))
        labels.append(det["class"])
    if not bbb:  # all-propagated track: fall back to last real hit
        _, det, q = track.hits[-1]
        bbb, confs, weights, labels = [det["box"]], [det["confidence"]], [1.0], [det["class"]]
    w = np.array(weights, np.float32)
    conf = float((np.array(confs) * w).sum() / w.sum())
    consensus = [round(float(v), 2) for v in np.median(np.array(bbb, np.float32), axis=0)]
    tally: dict[str, float] = {}
    for lab, c in zip(labels, confs):
        tally[lab] = tally.get(lab, 0.0) + c
    label = max(tally, key=tally.get)
    return {"label": label, "confidence": round(conf, 4), "box": consensus}


def process_video(det, path: str, vcfg: dict | None = None,
                  mode: str = "accurate") -> dict:
    vcfg = dict(vcfg or video_cfg())
    if mode == "fast":  # fast path: single engine-scale, no TTA/tiles/flow
        vcfg.update(enable_tta=False, enable_tiling=False,
                    working_scales=[960], enable_flow=False, use_soft_nms=False)
    t_all = time.time()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / float(vcfg["sample_fps"]))))
    frames, quality_log = [], []
    idx = 0
    while len(frames) < int(vcfg["max_frames"]):
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
            quality_log.append({"frame": idx, **frame_quality(frame, vcfg)})
        idx += 1
    cap.release()

    tracks: list[Track] = []
    per_frame_counts = []
    timing = {"frames": len(frames), "vote_s": 0.0}
    prev_gray = None
    for fpos, (frame, q) in enumerate(zip(frames, quality_log)):
        dets, info = detect_frame(det, frame, vcfg)
        timing["vote_s"] += info["vote_s"]
        # down-weight (don't drop): blurry frames contribute weak votes
        if not q["usable"]:
            for d in dets:
                d["confidence"] = round(d["confidence"] * q["weight"], 4)
        pairs, unmatched = _match(tracks, dets, float(vcfg["iou_match"]))
        matched_d = set(pairs.values())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for ti, di in pairs.items():
            tracks[ti].hits.append((fpos, dets[di], q))
            tracks[ti].misses = 0
            tracks[ti].last_box = list(dets[di]["box"])
        for ti, t in enumerate(tracks):
            if ti in pairs:
                continue
            t.misses += 1
            if (vcfg["enable_flow"] and prev_gray is not None
                    and 0 < t.misses <= int(vcfg["max_gap"])):
                carried = _lk_shift(prev_gray, gray, t.last_box)
                if carried is not None:
                    last_det = t.hits[-1][1]
                    t.hits.append((fpos, {**last_det, "box": carried,
                                          "confidence": round(last_det["confidence"] * 0.8, 4),
                                          "propagated": True}, q))
                    t.last_box = carried
        for di in sorted(set(range(len(dets))) - matched_d):
            tracks.append(Track(dets[di], fpos, q))
        tracks = [t for t in tracks if t.misses <= int(vcfg["max_gap"])]
        per_frame_counts.append(len(dets))
        prev_gray = gray

    confirmed, unconfirmed = [], []
    for t in tracks:
        real_hits = sum(1 for _, d, _ in t.hits if not d.get("propagated"))
        (confirmed if real_hits >= int(vcfg["min_persistence"]) else unconfirmed).append(t)

    detections = []
    for t in confirmed:
        f = fuse_track(t)
        area = (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1])
        detections.append({
            "class": f["label"], "confidence": f["confidence"],
            "priority": litter_ml.priority_of(det.cfg, f["label"]),
            "bin": litter_ml.bin_for(det.cfg, f["label"]),
            "box": f["box"], "area": round(float(area), 2),
            "track_id": t.id, "frames_seen": len(t.hits),
        })
    detections.sort(key=lambda d: -d["confidence"])
    timing["total_s"] = round(time.time() - t_all, 3)
    timing["per_frame_s"] = round(timing["total_s"] / max(1, len(frames)), 3)
    return {
        "detections": detections,
        "count": len(detections),
        "tracks": [{"track_id": t.id, "hits": len(t.hits),
                    "confirmed": t in confirmed} for t in tracks],
        "unconfirmed_tracks": len(unconfirmed),
        "quality_log": quality_log,
        "per_frame_counts": per_frame_counts,
        "timing": timing,
        "mode": mode,
        "frames_sampled": len(frames),
    }


def process_image(det, img_bgr: np.ndarray, mode: str = "fast") -> tuple[list[dict], dict]:
    """Accurate image path: tiles + TTA + WBF + soft-NMS + calibrated gates."""
    vcfg = video_cfg()
    if mode == "fast":
        return det.detect(img_bgr), {"mode": "fast"}
    dets, info = detect_frame(det, img_bgr, vcfg)
    info["mode"] = "accurate"
    return dets, info
