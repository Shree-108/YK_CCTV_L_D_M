"""Automatic ground ROI (floor / road / sidewalk) per camera, proposed by a segmentation model.

Owner request 2026-11-24 (reverses the 2026-11-19 "no ROI" decision for this feature only).

Pipeline
  1. Frames    one frame, or the per-pixel MEDIAN of N frames of a camera. The median removes people and
               vehicles that pass through, so the ground behind them is visible.
  2. Segment   a semantic segmentation model (default: UperNet ConvNeXt-tiny, ADE20K, local weights in
               models/roi_segmenter/upernet-convnext-tiny) gives per-pixel class probabilities.
               ground = P(floor + road + sidewalk + earth + path + ...)
               remove = P(person + box + signboard + shelf + case + counter + vehicles + ...)
               Cartons in this deployment are labelled "box" or "signboard" (printed logos), so both are in the
               remove set. Bag, bottle and the like are NOT removed: they can be the litter itself.
  3. Clean     ground AND NOT dilated(remove) -> opening (cut thin bridges) -> closing (fill cracks/shadows)
               -> largest connected component -> small holes filled, large holes kept (removed objects).
  4. Polygon   outer contour simplified to <= MAX_POINTS points plus hole polygons, normalised to 0..1.
  5. Save      storage/camera_roi.json, source="auto", needs_review=true. A person approves, edits or clears it.

Precedence when an ROI is needed: request polygon > camera ROI > global ROI > none (full frame).
A record with needs_review=true is still returned by ``resolve``; callers decide whether an unreviewed ROI may
filter anything. Nothing in the detection pipeline is filtered by this module.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ADE20K label names (the segmenter's id2label). Unknown names are ignored, so another ADE20K model also works.
GROUND_LABELS: tuple[str, ...] = ("floor", "road", "sidewalk", "earth", "path", "land", "dirt track", "grass",
                                  "sand", "field")
REMOVE_LABELS: tuple[str, ...] = (
    "person",                                                               # people
    "box", "signboard", "trade name",                                       # cartons (printed cartons -> signboard)
    "shelf", "case", "cabinet", "counter", "countertop", "chest of drawers",  # stock / store fixtures
    "bookcase", "refrigerator", "buffet", "basket", "barrel", "ashcan",
    "car", "truck", "van", "bus", "minibike", "bicycle",                    # parked vehicles
)

MAX_POINTS = 24
MIN_REGION = 0.02          # largest ground component must cover >= 2 % of the frame
MIN_HOLE = 0.004           # holes smaller than 0.4 % of the frame are filled (cracks, shadows, litter itself)
DEFAULT_FRAMES = 5
SEGMENTER_DIR = Path(__file__).resolve().parents[2] / "models" / "roi_segmenter" / "upernet-convnext-tiny"
GLOBAL_KEY = "__global__"


# ====================================================================== segmentation
@dataclass
class SegResult:
    ground: np.ndarray                       # float32 HxW in [0,1]
    remove: np.ndarray                       # float32 HxW in [0,1]
    shares: dict[str, float] = field(default_factory=dict)   # share of the frame per argmax label (top 12)


class Segmenter(Protocol):
    name: str

    def segment(self, bgr: np.ndarray) -> SegResult: ...


class UperNetSegmenter:
    """ADE20K UperNet from local files (no download). CPU: ~5-10 s per 640-px frame, done once per camera."""

    def __init__(self, model_dir: Path | str = SEGMENTER_DIR, work_width: int = 640,
                 ground_labels: Sequence[str] = GROUND_LABELS, remove_labels: Sequence[str] = REMOVE_LABELS) -> None:
        import torch
        from transformers import UperNetForSemanticSegmentation

        self._torch = torch
        self.model_dir = Path(model_dir)
        if not (self.model_dir / "config.json").is_file():
            raise FileNotFoundError(f"segmentation model not found in {self.model_dir}")
        self.model = UperNetForSemanticSegmentation.from_pretrained(str(self.model_dir), local_files_only=True).eval()
        self.name = self.model_dir.name
        self.work_width = int(work_width)
        id2label = {int(k): v.strip() for k, v in self.model.config.id2label.items()}
        self.labels = id2label
        self.ground_idx = [i for i, n in id2label.items() if n in set(ground_labels)]
        self.remove_idx = [i for i, n in id2label.items() if n in set(remove_labels)]
        if not self.ground_idx:
            raise ValueError(f"{self.name} has none of the ground labels {ground_labels}")
        self._lock = threading.Lock()

    def segment(self, bgr: np.ndarray) -> SegResult:
        h, w = bgr.shape[:2]
        W = self.work_width
        H = max(32, int(round(h * W / w / 32)) * 32)
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        t = self._torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
        with self._lock, self._torch.no_grad():
            logits = self.model(pixel_values=t).logits[0]          # C x H x W (already at input size)
            prob = logits.softmax(0).numpy()
        ground = prob[self.ground_idx].sum(0)
        remove = prob[self.remove_idx].sum(0) if self.remove_idx else np.zeros_like(ground)
        lab = prob.argmax(0)
        ids, counts = np.unique(lab, return_counts=True)
        shares = {self.labels[int(i)]: round(float(c) / lab.size, 4)
                  for i, c in sorted(zip(ids, counts), key=lambda z: -z[1])[:12]}
        return SegResult(ground.astype(np.float32), remove.astype(np.float32), shares)


_SEGMENTER: Optional[Segmenter] = None
_SEG_LOCK = threading.Lock()


def get_segmenter() -> Segmenter:
    """Process-wide segmenter, loaded on first use (tests replace it with ``set_segmenter``)."""
    global _SEGMENTER
    with _SEG_LOCK:
        if _SEGMENTER is None:
            _SEGMENTER = UperNetSegmenter()
        return _SEGMENTER


def set_segmenter(seg: Optional[Segmenter]) -> None:
    global _SEGMENTER
    with _SEG_LOCK:
        _SEGMENTER = seg


# ====================================================================== frames
def median_frame(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Per-pixel median (moving people/vehicles disappear). Frames of another size are resized to the first."""
    frames = [f for f in frames if isinstance(f, np.ndarray) and f.ndim == 3 and f.size]
    if not frames:
        raise ValueError("no readable frame")
    if len(frames) == 1:
        return frames[0].copy()
    h, w = frames[0].shape[:2]
    stack = np.stack([f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)) for f in frames])
    return np.median(stack, axis=0).astype(np.uint8)


def sample_frames(cap: Any, n: int, live_gap_frames: int = 25, max_seconds: float = 60.0) -> list[np.ndarray]:
    """``n`` frames from an opened capture. Files: evenly spread over the whole video, read sequentially
    (seeking in HEVC CCTV exports returns broken frames). Live streams (no frame count): one frame every
    ``live_gap_frames`` frames, bounded by ``max_seconds``."""
    n = max(1, int(n))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out: list[np.ndarray] = []
    t0 = time.monotonic()
    if total > 0:
        want = set(np.linspace(0, total - 1, n).astype(int).tolist())
        i = 0
        while len(out) < len(want) and i <= max(want):
            if not cap.grab():
                break
            if i in want:
                ok, f = cap.retrieve()
                if ok and f is not None:
                    out.append(f)
            i += 1
        return out
    i = 0
    while len(out) < n and time.monotonic() - t0 < max_seconds:
        if not cap.grab():
            break
        if i % live_gap_frames == 0:
            ok, f = cap.retrieve()
            if ok and f is not None:
                out.append(f)
        i += 1
    return out


# ====================================================================== mask -> polygon
def _odd(n: int) -> int:
    return max(3, int(n) | 1)


def clean_mask(ground: np.ndarray, remove: np.ndarray, *, ground_thr: float = 0.5, remove_thr: float = 0.35,
               remove_dilate_frac: float = 0.012, close_frac: float = 0.03) -> np.ndarray:
    """Boolean mask of the largest clean ground region (see module docstring, step 3)."""
    h, w = ground.shape[:2]
    rm = (remove >= remove_thr).astype(np.uint8)
    if rm.any():
        rm = cv2.dilate(rm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(w * remove_dilate_frac),) * 2))
    m = ((ground >= ground_thr) & (rm == 0)).astype(np.uint8)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(w * 0.008),) * 2)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(w * close_frac),) * 2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    m[rm > 0] = 0                                   # closing must not grow back over a removed object
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros((h, w), bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < MIN_REGION * h * w:
        return np.zeros((h, w), bool)
    region = lab == best
    # fill small holes only (cracks, shadows, the litter itself); large holes are removed objects and stay out
    inv = (~region).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n2):
        x, y, bw, bh, area = st2[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches and area < MIN_HOLE * h * w:
            region[lab2 == i] = True
    return region


def _simplify(c: np.ndarray, max_points: int) -> np.ndarray:
    eps = 0.003 * cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, eps, True)
    while len(poly) > max_points:
        eps *= 1.3
        poly = cv2.approxPolyDP(c, eps, True)
    return poly.reshape(-1, 2)


def _norm(pts: np.ndarray, w: int, h: int) -> list[list[float]]:
    return [[round(min(1.0, max(0.0, float(x) / max(1, w - 1))), 4), round(min(1.0, max(0.0, float(y) / max(1, h - 1))), 4)]
            for x, y in pts]


def mask_to_polygon(mask: np.ndarray, max_points: int = MAX_POINTS
                    ) -> tuple[Optional[list[list[float]]], list[list[list[float]]]]:
    """(outer polygon, hole polygons), normalised to 0..1; (None, []) when there is no usable region."""
    h, w = mask.shape[:2]
    if not mask.any():
        return None, []
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, []
    hier = hier[0]
    outers = [i for i in range(len(contours)) if hier[i][3] < 0]
    o = max(outers, key=lambda i: cv2.contourArea(contours[i]))
    outer = _simplify(contours[o], max_points)
    if len(outer) < 3:
        return None, []
    holes = []
    for i in range(len(contours)):
        if hier[i][3] == o and cv2.contourArea(contours[i]) >= MIN_HOLE * h * w:
            hp = _simplify(contours[i], max(8, max_points // 2))
            if len(hp) >= 3:
                holes.append(_norm(hp, w, h))
    return _norm(outer, w, h), holes


def roi_mask(roi: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Boolean HxW mask of an ROI record (polygon minus holes)."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    poly = roi.get("polygon") or []
    if len(poly) >= 3:
        cv2.fillPoly(m, [_px(poly, w, h)], 1)
        for hole in roi.get("holes") or []:
            if len(hole) >= 3:
                cv2.fillPoly(m, [_px(hole, w, h)], 0)
    return m.astype(bool)


def _px(poly: Sequence[Sequence[float]], w: int, h: int) -> np.ndarray:
    return np.array([[round(x * (w - 1)), round(y * (h - 1))] for x, y in poly], np.int32)


def point_in_roi(x: float, y: float, roi: Optional[dict[str, Any]]) -> bool:
    """Normalised point inside the polygon and outside every hole. No ROI -> True (full frame)."""
    if not roi or not roi.get("polygon"):
        return True
    def inside(poly: Sequence[Sequence[float]]) -> bool:
        return cv2.pointPolygonTest(np.array(poly, np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0
    return inside(roi["polygon"]) and not any(inside(hh) for hh in roi.get("holes") or [] if len(hh) >= 3)


def detection_in_roi(bbox: Sequence[float], frame_w: int, frame_h: int, roi: Optional[dict[str, Any]]) -> bool:
    """Litter lies on the ground: test the bottom-centre of the box (its ground contact point)."""
    x1, _, x2, y2 = bbox[:4]
    return point_in_roi(((x1 + x2) / 2) / max(1, frame_w - 1), min(y2, frame_h - 1) / max(1, frame_h - 1), roi)


# ====================================================================== validation
def validate_polygon(poly: Any, *, what: str = "polygon") -> list[list[float]]:
    if not isinstance(poly, (list, tuple)) or len(poly) < 3:
        raise ValueError(f"{what} needs at least 3 points")
    if len(poly) > 200:
        raise ValueError(f"{what} has too many points (max 200)")
    out = []
    for p in poly:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"{what} points must be [x, y]")
        x, y = float(p[0]), float(p[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) or not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError(f"{what} points must be normalised to 0..1")
        out.append([round(x, 4), round(y, 4)])
    if abs(cv2.contourArea(np.array(out, np.float32))) < 1e-4:
        raise ValueError(f"{what} has no area")
    return out


# ====================================================================== compute
def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class AutoRoiResult:
    record: Optional[dict[str, Any]]        # None when no ground region was found
    reference: np.ndarray                   # the (median) frame the ROI was computed on
    mask: np.ndarray                        # cleaned ground mask at reference resolution
    note: str


def compute_auto_roi(frames: Sequence[np.ndarray] | np.ndarray, segmenter: Optional[Segmenter] = None
                     ) -> AutoRoiResult:
    """Run steps 1-4 on one frame or a list of frames. The record is NOT saved here."""
    frames = [frames] if isinstance(frames, np.ndarray) else list(frames)
    ref = median_frame(frames)
    seg = (segmenter or get_segmenter()).segment(ref)
    mask_small = clean_mask(seg.ground, seg.remove)
    h, w = ref.shape[:2]
    mask = cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    poly, holes = mask_to_polygon(mask_small)
    seg_name = getattr(segmenter or get_segmenter(), "name", "segmenter")
    base = {"frames_used": len(frames), "frame_size": [int(w), int(h)], "model": seg_name,
            "label_shares": seg.shares, "ground_share": round(float(mask.mean()), 4)}
    if poly is None:
        return AutoRoiResult(None, ref, mask, "No floor/road/sidewalk region large enough was found: draw the "
                                              "ground area manually or keep the full frame.")
    rec = {"polygon": poly, "holes": holes, "source": "auto", "needs_review": True, **base}
    return AutoRoiResult(rec, ref, mask, f"Ground area found ({base['ground_share']:.0%} of the frame). "
                                         "Needs review: approve, edit or clear it.")


# ====================================================================== overlay
def render_overlay(bgr: np.ndarray, roi: Optional[dict[str, Any]], title: str = "") -> np.ndarray:
    """Green fill inside the ROI, red outlines around holes (removed objects), status banner."""
    out = bgr.copy()
    h, w = out.shape[:2]
    lw = max(2, w // 400)
    if roi and roi.get("polygon"):
        m = roi_mask(roi, (h, w))
        tint = out.copy()
        tint[m] = (0.45 * tint[m] + 0.55 * np.array([60, 200, 60])).astype(np.uint8)
        out = cv2.addWeighted(tint, 0.85, out, 0.15, 0)
        cv2.polylines(out, [_px(roi["polygon"], w, h)], True, (40, 255, 40), lw, cv2.LINE_AA)
        for hole in roi.get("holes") or []:
            cv2.polylines(out, [_px(hole, w, h)], True, (40, 40, 255), lw, cv2.LINE_AA)
        for x, y in _px(roi["polygon"], w, h):
            cv2.circle(out, (int(x), int(y)), lw * 2 + 1, (255, 255, 255), -1, cv2.LINE_AA)
    if not title:
        if not roi:
            title = "NO ROI - full frame"
        else:
            title = (f"ROI {roi.get('source', '?').upper()} - "
                     f"{'NEEDS REVIEW' if roi.get('needs_review') else 'APPROVED'}")
    scale = max(0.6, w / 1600)
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, lw))
    cv2.rectangle(out, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(out, title, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 200, 255) if roi and roi.get("needs_review") else (255, 255, 255), max(1, lw), cv2.LINE_AA)
    return out


# ====================================================================== storage
class RoiStore:
    """storage/camera_roi.json: {"version": 1, "global": record|null, "cameras": {camera_id: record}}.
    Reference frames are kept next to it in storage/auto_roi/<key>.jpg."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.ref_dir = self.path.parent / "auto_roi"
        self._lock = threading.RLock()

    # ---- raw file
    def _read(self) -> dict[str, Any]:
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                doc.setdefault("cameras", {})
                doc.setdefault("global", None)
                return doc
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            log.error("camera_roi.json unreadable (%s); treating it as empty", exc)
        return {"version": 1, "global": None, "cameras": {}}

    def _write(self, doc: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def all(self) -> dict[str, Any]:
        with self._lock:
            return self._read()

    # ---- records
    def get(self, key: str) -> Optional[dict[str, Any]]:
        doc = self.all()
        return doc["global"] if key == GLOBAL_KEY else doc["cameras"].get(key)

    def _put(self, key: str, rec: Optional[dict[str, Any]]) -> None:
        with self._lock:
            doc = self._read()
            if key == GLOBAL_KEY:
                doc["global"] = rec
            elif rec is None:
                doc["cameras"].pop(key, None)
            else:
                doc["cameras"][key] = rec
            self._write(doc)

    def save_auto(self, key: str, result: AutoRoiResult) -> Optional[dict[str, Any]]:
        if result.record is None:
            return None
        self.ref_dir.mkdir(parents=True, exist_ok=True)
        ref = self.ref_dir / f"{_safe(key)}.jpg"
        cv2.imwrite(str(ref), result.reference, [cv2.IMWRITE_JPEG_QUALITY, 90])
        t = now_iso()
        rec = {**result.record, "created_at": t, "updated_at": t, "approved_at": None, "approved_by": None,
               "reference_image": ref.name}
        self._put(key, rec)
        return rec

    def approve(self, key: str, actor: str = "") -> dict[str, Any]:
        with self._lock:
            rec = self.get(key)
            if not rec:
                raise KeyError(key)
            rec = {**rec, "needs_review": False, "approved_at": now_iso(), "approved_by": actor or None,
                   "updated_at": now_iso()}
            self._put(key, rec)
            return rec

    def edit(self, key: str, polygon: Any, holes: Any = None, actor: str = "") -> dict[str, Any]:
        """A person drew/adjusted it: that is the review, so needs_review becomes false."""
        poly = validate_polygon(polygon)
        hs = [validate_polygon(h, what="hole") for h in (holes or [])]
        with self._lock:
            old = self.get(key) or {}
            t = now_iso()
            rec = {**old, "polygon": poly, "holes": hs,
                   "source": "auto_edited" if old.get("source", "").startswith("auto") else "manual",
                   "needs_review": False, "approved_at": t, "approved_by": actor or None, "updated_at": t}
            rec.setdefault("created_at", t)
            self._put(key, rec)
            return rec

    def clear(self, key: str) -> bool:
        with self._lock:
            existed = self.get(key) is not None
            self._put(key, None)
            (self.ref_dir / f"{_safe(key)}.jpg").unlink(missing_ok=True)
            return existed

    def reference_path(self, key: str) -> Optional[Path]:
        p = self.ref_dir / f"{_safe(key)}.jpg"
        return p if p.is_file() else None

    # ---- precedence
    def resolve(self, request_polygon: Any = None, camera_id: Optional[str] = None,
                request_holes: Any = None) -> tuple[Optional[dict[str, Any]], str]:
        """request > camera > global > none. Returns (record, "request"|"camera"|"global"|"none")."""
        if request_polygon:
            return ({"polygon": validate_polygon(request_polygon),
                     "holes": [validate_polygon(h, what="hole") for h in (request_holes or [])],
                     "source": "request", "needs_review": False}, "request")
        doc = self.all()
        if camera_id and doc["cameras"].get(camera_id):
            return doc["cameras"][camera_id], "camera"
        if doc.get("global"):
            return doc["global"], "global"
        return None, "none"


def _safe(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:64] or "_"


def store_for(storage_dir: Path) -> RoiStore:
    return RoiStore(Path(storage_dir) / "camera_roi.json")


def frames_from_capture(open_fn: Callable[[], Any], n: int) -> list[np.ndarray]:
    cap = open_fn()
    try:
        if not cap.isOpened():
            return []
        return sample_frames(cap, n)
    finally:
        cap.release()
