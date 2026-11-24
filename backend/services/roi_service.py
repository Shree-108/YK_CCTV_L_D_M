"""Automatic monitored region (outdoor pavement ROI) for ANY camera angle, and litter detection restricted to it.

Owner brief 2026-11-24. It runs automatically wherever detection runs (image upload, video upload, four-screenshot
video analysis, scheduled captures, live streams), with no manual step.

1. Region      frames -> ADE20K semantic segmentation (default SegFormer-B2, ``settings.roi.model``) -> per pixel
               P(road + sidewalk + path + earth) >= threshold. "floor" is NEVER ground (it is the shop interior).
               Videos: ``vote_frames`` (40) frames spread over the WHOLE video; live streams: over ``vote_seconds``.
               Per-pixel MAJORITY VOTE over the frames in which the pixel is KNOWN: person / vehicle pixels
               (``unknown_classes``) are unknown in that frame, not non-ground, so people standing still do not
               cut holes into the region.
2. Clean-up    open -> close -> drop components below ``min_component_frac`` -> fill holes below ``fill_hole_frac``.
3. Polygons    findContours (outer + holes) -> approxPolyDP (epsilon = ``poly_epsilon_frac`` of the perimeter,
               raised until <= ``max_vertices``). Any number of polygons, any shape: no rectangle, no camera-angle
               assumption. Normalised 0..1.
4. Record      config/camera_roi.json {camera_id: {polygons, source, confidence, needs_review, approved,
               frame_signature,...}}. Every automatic or edited region stays ``needs_review`` until an operator
               approves it (phase 2).
               Area outside [min_area_frac, max_area_frac] -> low confidence + needs_review. The frame signature
               (normalised grey thumbnail) detects a moved camera -> the auto ROI is recomputed (a manual one is
               kept and flagged needs_review). Precedence: request > manual > auto > global.
5. Failure     no model / no region / error -> ``on_failure`` (default full_frame) with a visible warning and the
               remark tag ``roi_not_determined``. Never silent.
6. Detection   ``roi_detect``: uniform gate first (Lab, per-tile median/MAD) -> pixels outside the ROI filled with the
               ROI's median colour -> crop to the ROI bounding box -> detector (tiling, higher imgsz) -> boxes mapped
               back -> kept only if the bottom-centre is inside the ROI and >= ``min_mask_overlap`` of the box lies
               inside the mask. The shop interior is never given to the detector.
7. Candidates  colour (chroma) and weak (luminance) anomalies inside the ROI. A candidate alone NEVER gives
               litter_detected: it is CONFIRMED only when the detector confirms it (or, in video, it stays static for
               ``static_confirm_sec``); otherwise it is POSSIBLE (status stays clean, remark says so).
"""
from __future__ import annotations

import base64
import copy
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

from backend.core.config import PROJECT_ROOT

log = logging.getLogger(__name__)

GLOBAL_KEY = "__global__"
SOURCES = ("auto", "manual", "sam_click")
NO_CACHE_KEYS = {"", "upload", "UNKNOWN", None}          # one-off uploads: computed per image, never stored
MODELS_DIR = PROJECT_ROOT / "models" / "roi_segmenter"
HF_IDS = {"segformer-b2-finetuned-ade-512-512": "nvidia/segformer-b2-finetuned-ade-512-512",
          "segformer-b4-finetuned-ade-512-512": "nvidia/segformer-b4-finetuned-ade-512-512",
          "upernet-convnext-tiny": "openmmlab/upernet-convnext-tiny"}
ROI_RGB = (135, 206, 250)                                 # light sky blue
ROI_BGR = (ROI_RGB[2], ROI_RGB[1], ROI_RGB[0])
ROI_LABEL = "Monitored region"
NO_LITTER_MSG = "No litter found in the monitored region"
UNIFORM_MSG = "uniform region: no colour or texture anomalies"
PEOPLE_COCO = {0, 1, 2, 3, 5, 7}                          # person, bicycle, car, motorcycle, bus, truck


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ============================================================================================ segmentation
class Segmenter(Protocol):
    name: str

    def ground_prob(self, bgr: np.ndarray) -> np.ndarray:
        """float32 HxW (any size; resized by the caller) = P(pixel is outdoor ground)."""

    # optional: ground_and_unknown(bgr) -> (P(ground), P(person/vehicle) or None) - see segment()


def model_dir(name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else MODELS_DIR / name


class HFSegmenter:
    """Any Hugging Face ADE20K semantic segmentation model (SegFormer, UperNet, BEiT, DPT) from local files."""

    def __init__(self, name: str, ground_classes: Sequence[str], work_width: int = 640,
                 unknown_classes: Sequence[str] = ()) -> None:
        import torch
        from transformers import AutoModelForSemanticSegmentation

        self.dir = model_dir(name)
        if not (self.dir / "config.json").is_file():
            raise FileNotFoundError(
                f"ROI segmentation model '{name}' not found in {self.dir}. Fetch it once with "
                f"'python -m backend.services.roi_service fetch {name}' (deploy/start-ldm.ps1 does this).")
        self._torch = torch
        self.model = AutoModelForSemanticSegmentation.from_pretrained(str(self.dir), local_files_only=True).eval()
        self.name = self.dir.name
        self.work_width = int(work_width)
        id2label = {int(k): str(v).strip() for k, v in self.model.config.id2label.items()}
        wanted = {c.strip().lower() for c in ground_classes}
        if "floor" in wanted:
            raise ValueError("'floor' is the shop interior and can never be a ground class")
        self.ground_idx = [i for i, n in id2label.items() if n.lower() in wanted]
        if not self.ground_idx:
            raise ValueError(f"{self.name} has none of the ground classes {sorted(wanted)}")
        unk = {c.strip().lower() for c in unknown_classes} - wanted
        self.unknown_idx = [i for i, n in id2label.items() if n.lower() in unk]
        self._lock = threading.Lock()

    def ground_prob(self, bgr: np.ndarray) -> np.ndarray:
        return self.ground_and_unknown(bgr)[0]

    def ground_and_unknown(self, bgr: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        h, w = bgr.shape[:2]
        W = self.work_width
        H = max(32, int(round(h * W / w / 32)) * 32)
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = (x / 255 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        t = self._torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
        with self._lock, self._torch.no_grad():
            logits = self.model(pixel_values=t).logits
            if logits.shape[-2:] != (H, W):        # SegFormer predicts at 1/4 resolution
                logits = self._torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear",
                                                               align_corners=False)
            sm = logits[0].softmax(0)
            prob = sm[self.ground_idx].sum(0).numpy()
            unknown = sm[self.unknown_idx].sum(0).numpy().astype(np.float32) if self.unknown_idx else None
        return prob.astype(np.float32), unknown


_SEG: dict[tuple, Segmenter] = {}
_SEG_LOCK = threading.Lock()
_SEG_OVERRIDE: Optional[Segmenter] = None


def get_segmenter(r: Any) -> Segmenter:
    """Lazy, process-wide segmenter per (model, classes, width). Raises FileNotFoundError when weights are missing."""
    if _SEG_OVERRIDE is not None:
        return _SEG_OVERRIDE
    unknown = tuple(getattr(r, "unknown_classes", ()) or ())
    key = (r.model, tuple(r.ground_classes), r.work_width, unknown)
    with _SEG_LOCK:
        if key not in _SEG:
            t0 = time.perf_counter()
            _SEG[key] = HFSegmenter(r.model, r.ground_classes, r.work_width, unknown)
            log.info("ROI segmenter %s loaded in %.1f s (ground classes %s)", r.model, time.perf_counter() - t0,
                     ", ".join(r.ground_classes))
        return _SEG[key]


def set_segmenter(seg: Optional[Segmenter]) -> None:
    """Tests: replace the segmenter for every call (None restores the real one)."""
    global _SEG_OVERRIDE
    _SEG_OVERRIDE = seg


def fetch_model(name: str) -> Path:
    """Download a known model once into models/roi_segmenter/<name> (safetensors)."""
    d = model_dir(name)
    if (d / "config.json").is_file() and any(d.glob("*.safetensors")):
        return d
    repo = HF_IDS.get(name)
    if repo is None:
        raise ValueError(f"unknown model {name!r}; put the files in {d} yourself or use one of {sorted(HF_IDS)}")
    from huggingface_hub import hf_hub_download, list_repo_files
    from transformers import AutoModelForSemanticSegmentation

    files = list_repo_files(repo)
    weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    for f in ("config.json", "preprocessor_config.json", weights):
        if f in files:
            hf_hub_download(repo, f, local_dir=str(d))
    if weights != "model.safetensors":
        m = AutoModelForSemanticSegmentation.from_pretrained(str(d), local_files_only=True)
        m.save_pretrained(str(d), safe_serialization=True)
        (d / weights).unlink(missing_ok=True)
    return d


# ============================================================================================ masks and polygons
def _odd(n: float) -> int:
    return max(3, int(n) | 1)


def segment(seg: Any, bgr: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """(P(ground), P(person/vehicle) or None) - segmenters without ``ground_and_unknown`` have no unknown pixels."""
    fn = getattr(seg, "ground_and_unknown", None)
    return fn(bgr) if callable(fn) else (seg.ground_prob(bgr), None)


def vote(probs: Sequence[np.ndarray], threshold: float, majority: float,
         unknowns: Optional[Sequence[Optional[np.ndarray]]] = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel majority vote of per-frame ground masks over the frames in which the pixel is KNOWN (``unknowns``:
    per-frame bool masks of person/vehicle pixels, which vote neither way). A pixel unknown in every frame counts as
    not ground here; the hole filling of ``clean_mask`` closes it when it is surrounded by ground.
    Returns (mask bool, agreement float in [0,1])."""
    h, w = probs[0].shape[:2]
    votes = np.zeros((h, w), np.float32)
    known = np.zeros((h, w), np.float32)
    for i, p in enumerate(probs):
        if p.shape[:2] != (h, w):
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        u = unknowns[i] if unknowns is not None and i < len(unknowns) else None
        if u is None:
            k = np.ones((h, w), bool)
        else:
            if u.shape[:2] != (h, w):
                u = cv2.resize(u.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            k = ~u
        votes += (p >= threshold) & k
        known += k
    share = np.where(known > 0, votes / np.maximum(known, 1), 0.0).astype(np.float32)
    return share >= majority, share


def clean_mask(mask: np.ndarray, r: Any) -> np.ndarray:
    """open -> close -> components >= min_component_frac -> holes < fill_hole_frac filled."""
    h, w = mask.shape[:2]
    m = mask.astype(np.uint8)
    if r.open_frac > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(w * r.open_frac),) * 2))
    if r.close_frac > 0:
        k = _odd(w * r.close_frac)
        pad = k
        mp = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_REPLICATE)   # closing must not erode at borders
        mp = cv2.morphologyEx(mp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m = mp[pad:-pad, pad:-pad]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(1.0, r.min_component_frac * h * w)
    region = keep[lab]
    inv = (~region).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small = np.zeros(n2, bool)
    for i in range(1, n2):
        x, y, bw, bh, area = st2[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        small[i] = not touches and area < r.fill_hole_frac * h * w
    region |= small[lab2]
    return region


def _simplify(c: np.ndarray, eps_frac: float, max_vertices: int) -> np.ndarray:
    eps = max(0.5, eps_frac * cv2.arcLength(c, True))
    poly = cv2.approxPolyDP(c, eps, True)
    while len(poly) > max_vertices:
        eps *= 1.25
        poly = cv2.approxPolyDP(c, eps, True)
    return poly.reshape(-1, 2)


def _norm(pts: np.ndarray, w: int, h: int) -> list[list[float]]:
    return [[round(min(1.0, max(0.0, float(x) / max(1, w - 1))), 4),
             round(min(1.0, max(0.0, float(y) / max(1, h - 1))), 4)] for x, y in pts]


def mask_to_polygons(mask: np.ndarray, r: Any) -> list[dict[str, Any]]:
    """[{"points": [[x,y],...], "holes": [[[x,y],...], ...]}, ...] normalised to 0..1, largest first."""
    h, w = mask.shape[:2]
    if not mask.any():
        return []
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    hier = hier[0]
    min_area = max(9.0, r.min_component_frac * h * w)
    out = []
    for i, c in enumerate(contours):
        if hier[i][3] >= 0 or cv2.contourArea(c) < min_area:
            continue
        outer = _simplify(c, r.poly_epsilon_frac, r.max_vertices)
        if len(outer) < 3:
            continue
        holes = []
        j = hier[i][2]                          # first child, then its siblings
        while j >= 0:
            if cv2.contourArea(contours[j]) >= max(9.0, r.fill_hole_frac * h * w):
                hp = _simplify(contours[j], r.poly_epsilon_frac, r.max_vertices)
                if len(hp) >= 3:
                    holes.append(_norm(hp, w, h))
            j = hier[j][0]
        out.append((cv2.contourArea(c), {"points": _norm(outer, w, h), "holes": holes}))
    return [p for _, p in sorted(out, key=lambda z: -z[0])]


def _px(poly: Sequence[Sequence[float]], w: int, h: int) -> np.ndarray:
    return np.array([[round(x * (w - 1)), round(y * (h - 1))] for x, y in poly], np.int32)


def polygons_mask(polygons: Sequence[dict[str, Any]], shape: tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    for p in polygons or []:
        if len(p.get("points") or []) >= 3:
            cv2.fillPoly(m, [_px(p["points"], w, h)], 1)
    for p in polygons or []:
        for hole in p.get("holes") or []:
            if len(hole) >= 3:
                cv2.fillPoly(m, [_px(hole, w, h)], 0)
    return m.astype(bool)


def validate_polygons(polys: Any) -> list[dict[str, Any]]:
    """Request/manual input: a list of {"points": [...], "holes": [...]} or of bare point lists."""
    if not isinstance(polys, (list, tuple)) or not polys:
        raise ValueError("polygons must be a non-empty list")
    if len(polys) > 50:
        raise ValueError("too many polygons (max 50)")

    def pts(raw: Any, what: str) -> list[list[float]]:
        if not isinstance(raw, (list, tuple)) or not 3 <= len(raw) <= 500:
            raise ValueError(f"{what} needs 3-500 points")
        out = []
        for p in raw:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError(f"{what} points must be [x, y]")
            x, y = float(p[0]), float(p[1])
            if not (np.isfinite(x) and np.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"{what} points must be normalised to 0..1")
            out.append([round(x, 4), round(y, 4)])
        if abs(cv2.contourArea(np.array(out, np.float32))) < 1e-5:
            raise ValueError(f"{what} has no area")
        return out

    out = []
    for i, p in enumerate(polys):
        if isinstance(p, dict):
            out.append({"points": pts(p.get("points"), f"polygon {i + 1}"),
                        "holes": [pts(hh, f"hole of polygon {i + 1}") for hh in (p.get("holes") or [])]})
        else:
            out.append({"points": pts(p, f"polygon {i + 1}"), "holes": []})
    return out


# ============================================================================================ frame signature
SIG_W, SIG_H = 48, 27


def frame_signature(bgr: np.ndarray) -> str:
    """Normalised grey thumbnail (zero mean, unit std) quantised to int8, base64. Invariant to global brightness and
    contrast; changes when the camera points elsewhere."""
    g = cv2.cvtColor(cv2.resize(bgr, (SIG_W * 4, SIG_H * 4), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(cv2.GaussianBlur(g, (5, 5), 0), (SIG_W, SIG_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    z = (g - g.mean()) / (g.std() + 1e-6)
    return base64.b64encode(np.clip(np.round(z * 32), -127, 127).astype(np.int8).tobytes()).decode()


def signature_similarity(a: Optional[str], b: Optional[str]) -> Optional[float]:
    """Correlation (-1..1) of two signatures; None when either is missing/invalid."""
    try:
        x = np.frombuffer(base64.b64decode(a or ""), np.int8).astype(np.float32)
        y = np.frombuffer(base64.b64decode(b or ""), np.int8).astype(np.float32)
    except (ValueError, TypeError):
        return None
    if x.size != SIG_W * SIG_H or y.size != x.size:
        return None
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-6))


# ============================================================================================ building an ROI
@dataclass
class BuildResult:
    record: Optional[dict[str, Any]]          # None: no region could be made
    mask: np.ndarray                          # cleaned mask at work resolution
    reference: np.ndarray                     # median frame (reference image)
    note: str


def build_roi(frames: Sequence[np.ndarray], r: Any, segmenter: Optional[Segmenter] = None) -> BuildResult:
    """Steps 1-3 + confidence. ``frames``: 1 (image) or many (stream/video, majority vote)."""
    frames = [f for f in frames if isinstance(f, np.ndarray) and f.ndim == 3 and f.size]
    if not frames:
        raise ValueError("no readable frame for the ROI")
    seg = segmenter or get_segmenter(r)
    h0, w0 = frames[0].shape[:2]
    probs: list[np.ndarray] = []
    unknowns: list[Optional[np.ndarray]] = []
    for f in frames:
        p, u = segment(seg, f if f.shape[:2] == (h0, w0) else cv2.resize(f, (w0, h0)))
        probs.append(p)
        unknowns.append(None if u is None else u >= getattr(r, "unknown_prob", 0.5))
    ph, pw = probs[0].shape[:2]
    seen = [float(u.mean()) for u in unknowns if u is not None]
    unknown_share = float(np.mean(seen)) if seen else 0.0
    mask, share = vote(probs, r.ground_prob_threshold, r.vote_majority, unknowns)
    region = clean_mask(mask, r)
    polygons = mask_to_polygons(region, r)
    ref = (frames[0].copy() if len(frames) == 1
           else np.median(np.stack([f if f.shape[:2] == (h0, w0) else cv2.resize(f, (w0, h0)) for f in frames]),
                          axis=0).astype(np.uint8))
    if not polygons:
        return BuildResult(None, region, ref, "no outdoor ground (road / sidewalk / path / earth) found")
    poly_mask = polygons_mask(polygons, (ph, pw))
    area = float(poly_mask.mean())
    mean_p = float(np.mean([(p if p.shape[:2] == (ph, pw) else cv2.resize(p, (pw, ph)))[poly_mask].mean()
                            for p in probs]))
    agree = float(share[poly_mask].mean()) if poly_mask.any() else 0.0
    conf = round(max(0.0, min(1.0, mean_p * agree)), 3)
    needs_review, note = True, f"outdoor ground {area:.0%} of the frame; awaiting operator approval"
    if area < r.min_area_frac or area > r.max_area_frac:
        conf, needs_review = round(min(conf, 0.2), 3), True
        note = (f"monitored region is {area:.1%} of the frame (expected "
                f"{r.min_area_frac:.0%}-{r.max_area_frac:.0%}): low confidence, needs review")
    rec = {"polygons": polygons, "source": "auto", "confidence": conf, "needs_review": needs_review,
           "approved": False, "unknown_px_share": round(unknown_share, 4),
           "frame_signature": frame_signature(ref), "area_frac": round(area, 4), "frames_used": len(frames),
           "frame_size": [int(w0), int(h0)], "model": getattr(seg, "name", "segmenter"),
           "ground_classes": list(r.ground_classes), "note": note}
    return BuildResult(rec, region, ref, note)


# ============================================================================================ store
class RoiStore:
    """config/camera_roi.json = {camera_id | "__global__": record}. Writes are atomic (tmp + replace)."""

    def __init__(self, path: Path, legacy_path: Optional[Path] = None) -> None:
        self.path = Path(path)
        self.legacy_path = legacy_path
        self._lock = threading.RLock()

    def _read(self) -> dict[str, Any]:
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                return doc
            log.error("%s is not a JSON object; treating it as empty", self.path)
        except FileNotFoundError:
            return self._import_legacy()
        except (OSError, ValueError) as exc:
            log.error("%s unreadable (%s); treating it as empty", self.path, exc)
        return {}

    def _import_legacy(self) -> dict[str, Any]:
        """One-time import of storage/camera_roi.json (single polygon + holes) written by backend/services/auto_roi.py."""
        if not self.legacy_path or not Path(self.legacy_path).is_file():
            return {}
        try:
            old = json.loads(Path(self.legacy_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        doc: dict[str, Any] = {}
        items = list((old.get("cameras") or {}).items()) + ([(GLOBAL_KEY, old["global"])] if old.get("global") else [])
        for key, rec in items:
            if not rec or len(rec.get("polygon") or []) < 3:
                continue
            src = "manual" if rec.get("source") in ("manual", "auto_edited") else "auto"
            doc[key] = {"polygons": [{"points": rec["polygon"], "holes": rec.get("holes") or []}], "source": src,
                        "confidence": None, "needs_review": bool(rec.get("needs_review", src == "auto")),
                        "frame_signature": None, "created_at": rec.get("created_at"),
                        "updated_at": rec.get("updated_at"), "imported_from": str(self.legacy_path)}
        if doc:
            self._write(doc)
            log.info("imported %d ROI record(s) from %s into %s", len(doc), self.legacy_path, self.path)
        return doc

    def _write(self, doc: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def all(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._read())

    def get(self, key: str) -> Optional[dict[str, Any]]:
        return self.all().get(key)

    def put(self, key: str, rec: Optional[dict[str, Any]]) -> None:
        with self._lock:
            doc = self._read()
            if rec is None:
                doc.pop(key, None)
            else:
                doc[key] = rec
            self._write(doc)

    def update(self, key: str, **fields: Any) -> Optional[dict[str, Any]]:
        with self._lock:
            doc = self._read()
            if key not in doc:
                return None
            doc[key] = {**doc[key], **fields, "updated_at": now_iso()}
            self._write(doc)
            return doc[key]


# ============================================================================================ resolved ROI
@dataclass
class RoiContext:
    """The ROI used for one analysis. ``record`` None = not determined (``warning`` says why)."""
    camera_id: str
    record: Optional[dict[str, Any]]
    source: str                                  # request | manual | auto | sam_click | global | none
    on_failure: str = "full_frame"
    warning: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    _masks: dict[tuple[int, int], np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def active(self) -> bool:
        return bool(self.record and self.record.get("polygons"))

    @property
    def polygons(self) -> list[dict[str, Any]]:
        return (self.record or {}).get("polygons") or []

    @property
    def needs_review(self) -> bool:
        """Phase 2: a region stays needs_review until an operator approves it (``approved``). Records written before
        approval existed (no ``approved`` key) keep their own flag; polygons sent with a request are trusted."""
        rec = self.record or {}
        if self.source == "request":
            return False
        if "approved" in rec:
            return bool(rec.get("needs_review")) or not rec.get("approved")
        return bool(rec.get("needs_review"))

    def mask(self, shape: tuple[int, ...]) -> np.ndarray:
        key = (int(shape[0]), int(shape[1]))
        if key not in self._masks:
            self._masks[key] = polygons_mask(self.polygons, key)
        return self._masks[key]

    def remark_tags(self) -> list[str]:
        tags = []
        if self.active:
            tags.append(f"roi={'global' if self.source == 'global' else self.source}")
            if self.needs_review:
                tags.append("needs_review")
        else:
            tags.append("roi_not_determined")
        return tags

    def summary(self) -> dict[str, Any]:
        rec = self.record or {}
        return {"applied": self.active, "source": self.source, "camera_id": self.camera_id,
                "confidence": rec.get("confidence"), "needs_review": self.needs_review,
                "approved": True if self.source == "request" else bool(rec.get("approved")),
                "approved_at": rec.get("approved_at"), "frame_size": rec.get("frame_size"),
                "area_frac": rec.get("area_frac"), "polygons": self.polygons,
                "on_failure": None if self.active else self.on_failure, "warning": self.warning,
                "notes": list(self.notes), "remark_tags": self.remark_tags()}


# ============================================================================================ service
class RoiService:
    """Resolves the ROI for a camera/frame with caching, precedence and camera-moved detection."""

    def __init__(self, store_path: Path, reference_dir: Optional[Path] = None, enabled: bool = True,
                 legacy_path: Optional[Path] = None) -> None:
        self.store = RoiStore(store_path, legacy_path)
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.enabled = enabled
        self._build_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self.last_error: Optional[str] = None

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._build_locks.setdefault(key, threading.Lock())

    # ---- main entry
    def resolve(self, camera_id: Optional[str], frames: Sequence[np.ndarray] | np.ndarray, settings: Any,
                request_polygons: Any = None, *, save: bool = True) -> Optional[RoiContext]:
        """None when ROI is switched off (LDM_ROI=off or settings.roi.enabled false): full-frame behaviour, no tags.
        Otherwise a RoiContext - possibly not ``active`` (then ``warning`` explains, and on_failure applies)."""
        r = settings.roi
        if not self.enabled or not r.enabled:
            return None
        cam = camera_id or "upload"
        frames = [frames] if isinstance(frames, np.ndarray) else [f for f in frames if f is not None]
        if request_polygons:
            return RoiContext(cam, {"polygons": validate_polygons(request_polygons), "source": "request",
                                    "confidence": 1.0, "needs_review": False}, "request", r.on_failure)
        doc = self.store.all()
        rec = doc.get(cam) if cam not in NO_CACHE_KEYS else None
        notes: list[str] = []
        if rec and rec.get("polygons"):
            sim = signature_similarity(rec.get("frame_signature"), frame_signature(frames[0])) if frames else None
            moved = sim is not None and sim < r.signature_threshold
            if rec.get("source") in ("manual", "sam_click"):
                if moved and not rec.get("needs_review"):
                    rec = self.store.update(cam, needs_review=True, review_reason="camera moved") or rec
                    log.warning("camera %s moved (signature %.2f): manual ROI kept, flagged needs_review", cam, sim)
                if moved:
                    notes.append("camera moved since the manual ROI was drawn")
                return RoiContext(cam, rec, rec.get("source", "manual"), r.on_failure, notes=notes)
            if not moved:
                return RoiContext(cam, rec, "auto", r.on_failure)
            notes.append(f"camera moved (signature {sim:.2f}): auto ROI recomputed")
            log.info("camera %s moved (signature %.2f < %.2f): recomputing its ROI", cam, sim, r.signature_threshold)
        # nothing usable for this camera -> compute now (unless the global ROI applies and there is nothing to build)
        ctx = self._compute(cam, frames, r, save=save and cam not in NO_CACHE_KEYS)
        ctx.notes = notes + ctx.notes
        if ctx.active:
            return ctx
        g = doc.get(GLOBAL_KEY)
        if g and g.get("polygons"):
            return RoiContext(cam, g, "global", r.on_failure, warning=ctx.warning,
                              notes=ctx.notes + ["camera ROI not determined: global ROI used"])
        return ctx

    def _compute(self, cam: str, frames: Sequence[np.ndarray], r: Any, save: bool) -> RoiContext:
        if not frames:
            return self._failed(cam, r, "no frame available to compute the ROI")
        with self._lock_for(cam):
            if save:                                   # another thread may have built it meanwhile
                rec = self.store.get(cam)
                if rec and rec.get("polygons") and rec.get("source") == "auto":
                    sim = signature_similarity(rec.get("frame_signature"), frame_signature(frames[0]))
                    if sim is not None and sim >= r.signature_threshold:
                        return RoiContext(cam, rec, "auto", r.on_failure)
            t0 = time.perf_counter()
            try:
                res = build_roi(frames, r)
            except FileNotFoundError as exc:
                return self._failed(cam, r, f"segmentation model missing: {exc}")
            except Exception as exc:  # noqa: BLE001 - never fail the analysis because of the ROI
                log.exception("ROI computation failed for %s", cam)
                return self._failed(cam, r, f"ROI computation failed: {exc}")
            took = time.perf_counter() - t0
            if res.record is None:
                return self._failed(cam, r, res.note)
            t = now_iso()
            rec = {**res.record, "created_at": t, "updated_at": t, "compute_sec": round(took, 1)}
            if save:
                if self.reference_dir is not None:
                    try:
                        self.reference_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(self.reference_dir / f"{safe_key(cam)}.jpg"), res.reference,
                                    [cv2.IMWRITE_JPEG_QUALITY, 90])
                        rec["reference_image"] = f"{safe_key(cam)}.jpg"
                    except OSError:
                        log.warning("could not save the ROI reference frame of %s", cam)
                self.store.put(cam, rec)
            log.info("ROI for %s: %d polygon(s), %.0f%% of the frame, confidence %s%s (%d frame(s), %.1f s)", cam,
                     len(rec["polygons"]), 100 * rec["area_frac"], rec["confidence"],
                     ", NEEDS REVIEW" if rec["needs_review"] else "", len(frames), took)
            return RoiContext(cam, rec, "auto", r.on_failure, notes=[rec["note"]] if rec["needs_review"] else [])

    def _failed(self, cam: str, r: Any, why: str) -> RoiContext:
        self.last_error = why
        action = "detection on the FULL frame" if r.on_failure == "full_frame" else "detection skipped"
        log.warning("ROI NOT DETERMINED for %s: %s -> %s (roi.on_failure=%s)", cam, why, action, r.on_failure)
        return RoiContext(cam, None, "none", r.on_failure, warning=f"Monitored region not determined ({why}): {action}")

    # ---- convenience
    def for_video(self, camera_id: Optional[str], path: Path | str, settings: Any,
                  request_polygons: Any = None) -> Optional[RoiContext]:
        r = settings.roi
        if not self.enabled or not r.enabled:
            return None
        frames = [] if request_polygons else video_frames(path, r.vote_frames,
                                                          None if r.video_whole else r.vote_seconds)
        return self.resolve(camera_id, frames, settings, request_polygons)

    def reference_path(self, key: str) -> Optional[Path]:
        if self.reference_dir is None:
            return None
        p = self.reference_dir / f"{safe_key(key)}.jpg"
        return p if p.is_file() else None

    def warmup(self, settings: Any, cameras: Sequence[dict[str, Any]] = (),
               frames_for: Optional[Callable[[dict[str, Any]], list[np.ndarray]]] = None) -> dict[str, Any]:
        """App start: load the segmenter (clear message if missing) and build missing/stale ROIs of configured
        cameras. Returns a small status dict (also logged)."""
        out: dict[str, Any] = {"model": settings.roi.model, "loaded": False, "cameras": {}}
        if not self.enabled or not settings.roi.enabled:
            out["skipped"] = "ROI disabled"
            return out
        try:
            get_segmenter(settings.roi)
            out["loaded"] = True
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            out["error"] = str(exc)
            log.error("=" * 78)
            log.error("ROI MODEL NOT AVAILABLE: %s", exc)
            log.error("Detection will follow roi.on_failure=%s until it is installed.", settings.roi.on_failure)
            log.error("=" * 78)
            return out
        for cam in cameras:
            if not cam.get("enabled", True) or frames_for is None:
                continue
            try:
                frames = frames_for(cam)
                ctx = self.resolve(cam["id"], frames, settings)
                out["cameras"][cam["id"]] = ctx.summary() if ctx else None
            except Exception as exc:  # noqa: BLE001
                log.warning("ROI warm-up for camera %s failed: %s", cam.get("id"), exc)
                out["cameras"][cam["id"]] = {"error": str(exc)}
        return out


def safe_key(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:64] or "_"


def video_frames(path: Path | str, n: int, seconds: Optional[float]) -> list[np.ndarray]:
    """``n`` frames spread evenly over the first ``seconds`` s of a video file, or over the WHOLE video when
    ``seconds`` is None (read sequentially: seeking in HEVC CCTV exports returns broken frames)."""
    cap = cv2.VideoCapture(str(path))
    out: list[np.ndarray] = []
    try:
        if not cap.isOpened():
            return out
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fps = fps if 0 < fps <= 240 else 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if seconds is None or seconds <= 0:
            last = total - 1 if total > 0 else int(10 * fps)
        else:
            last = int(min(seconds * fps, total - 1 if total > 0 else seconds * fps))
        want = sorted(set(np.linspace(0, max(0, last), max(1, n)).astype(int).tolist()))
        i = 0
        for target in want:
            while i < target:
                if not cap.grab():
                    return out
                i += 1
            ok, f = cap.read()
            i += 1
            if ok and f is not None:
                out.append(f)
    finally:
        cap.release()
    return out


def frames_from_source(cam: dict[str, Any], r: Any, max_wait: float = 60.0) -> list[np.ndarray]:
    """``vote_frames`` frames over ``vote_seconds`` from a camera's own source (warm-up of a non-streaming camera).
    Files are sampled like an uploaded video; live sources are read in real time."""
    from backend.cameras.manager import camera_source, open_capture

    src = camera_source(cam)
    if isinstance(src, str) and Path(src).is_file():
        return video_frames(src, r.vote_frames, None if getattr(r, "video_whole", True) else r.vote_seconds)
    cap = open_capture(src)
    out: list[np.ndarray] = []
    try:
        if not cap.isOpened():
            return out
        gap = r.vote_seconds / max(1, r.vote_frames)
        t0 = last = time.monotonic()
        last -= gap
        while len(out) < r.vote_frames and time.monotonic() - t0 < max_wait:
            if not cap.grab():
                break
            if time.monotonic() - last >= gap:
                ok, f = cap.retrieve()
                if ok and f is not None:
                    out.append(f)
                    last = time.monotonic()
    finally:
        cap.release()
    return out


class LiveRoi:
    """ROI of one live stream: cached record if the camera has not moved; otherwise ``vote_frames`` frames are
    collected over ``vote_seconds`` and the ROI is built in a background thread. ``recheck_sec`` re-runs the
    camera-moved check. ``current`` is None while nothing usable exists yet (detection waits: the shop interior
    is never scanned just because the region is still being computed)."""

    def __init__(self, service: RoiService, camera_id: str, settings_fn: Callable[[], Any]) -> None:
        self.service, self.camera_id, self.settings_fn = service, camera_id, settings_fn
        self.current: Optional[RoiContext] = None
        self.state = "init"                       # init | collecting | building | ready | failed
        self._frames: list[np.ndarray] = []
        self._last_take = float("-inf")
        self._last_check = 0.0
        self._thread: Optional[threading.Thread] = None

    def update(self, frame: np.ndarray, now: Optional[float] = None) -> Optional[RoiContext]:
        now = time.monotonic() if now is None else now
        s = self.settings_fn()
        r = s.roi
        if not self.service.enabled or not r.enabled:
            self.current, self.state = None, "off"
            return None
        if self.state == "init":
            rec = self.service.store.get(self.camera_id)
            sim = signature_similarity((rec or {}).get("frame_signature"), frame_signature(frame)) if rec else None
            if rec and rec.get("polygons") and (rec.get("source") != "auto" or (sim is not None and sim >= r.signature_threshold)):
                self.current = self.service.resolve(self.camera_id, [frame], s)
                self.state, self._last_check = "ready", now
            else:
                self.state, self._frames = "collecting", []
        if self.state == "ready" and now - self._last_check >= r.recheck_sec:
            self._last_check = now
            rec = (self.current.record if self.current else None) or {}
            sim = signature_similarity(rec.get("frame_signature"), frame_signature(frame))
            if sim is not None and sim < r.signature_threshold and rec.get("source") == "auto":
                log.info("live camera %s moved (signature %.2f): collecting frames for a new ROI", self.camera_id, sim)
                self.state, self._frames, self._last_take = "collecting", [], float("-inf")     # keep using the old ROI until the new one is ready
        if self.state == "collecting":
            gap = r.vote_seconds / max(1, r.vote_frames)
            if now - self._last_take >= gap:
                self._frames.append(frame.copy())
                self._last_take = now
            if len(self._frames) >= r.vote_frames:
                frames, self._frames = self._frames, []
                self.state = "building"
                self._thread = threading.Thread(target=self._build, args=(frames, s), daemon=True,
                                                name=f"roi-{self.camera_id}")
                self._thread.start()
        return self.current

    def _build(self, frames: list[np.ndarray], s: Any) -> None:
        old = self.service.store.get(self.camera_id)
        if old and old.get("source") == "auto":
            # the camera moved (or there was no valid record): force a rebuild
            self.service.store.update(self.camera_id, frame_signature=None)
        ctx = self.service.resolve(self.camera_id, frames, s)
        self.current = ctx
        self.state = "ready" if ctx is not None and (ctx.active or ctx.on_failure == "full_frame") else "failed"
        self._last_check = time.monotonic()

    @property
    def pending(self) -> bool:
        return self.current is None and self.state in ("init", "collecting", "building")


# ============================================================================================ gate + candidates
@dataclass
class Candidate:
    bbox: list[float]                 # full-frame pixels
    kind: str                         # chroma | luma
    area_px: int
    score: float                      # chroma distance or |L - median|
    state: str = "possible"           # possible | confirmed
    confirmed_by: Optional[str] = None   # detector | static

    def to_dict(self) -> dict[str, Any]:
        return {"bbox": [round(v, 1) for v in self.bbox], "kind": self.kind, "area_px": self.area_px,
                "score": round(self.score, 1), "state": self.state, "confirmed_by": self.confirmed_by}


@dataclass
class GateResult:
    uniform: bool
    candidates: list[Candidate]
    stats: dict[str, float]


def _tile_stats(L: np.ndarray, a: np.ndarray, b: np.ndarray, m: np.ndarray, T: int) -> dict[str, np.ndarray]:
    h, w = m.shape
    gh, gw = -(-h // T), -(-w // T)
    keys = ("L", "a", "b", "madC", "madL")
    out = {k: np.full((gh, gw), np.nan, np.float32) for k in keys}
    min_n = max(16, int(0.2 * T * T))
    for i in range(gh):
        for j in range(gw):
            sl = (slice(i * T, min(h, (i + 1) * T)), slice(j * T, min(w, (j + 1) * T)))
            sel = m[sl]
            if int(sel.sum()) < min_n:
                continue
            l, aa, bb = L[sl][sel], a[sl][sel], b[sl][sel]
            ml, ma, mb = np.median(l), np.median(aa), np.median(bb)
            out["L"][i, j], out["a"][i, j], out["b"][i, j] = ml, ma, mb
            out["madC"][i, j] = np.median(np.hypot(aa - ma, bb - mb))
            out["madL"][i, j] = np.median(np.abs(l - ml))
    if m.any():
        glob = {"L": np.median(L[m]), "a": np.median(a[m]), "b": np.median(b[m])}
        glob["madC"] = np.median(np.hypot(a[m] - glob["a"], b[m] - glob["b"]))
        glob["madL"] = np.median(np.abs(L[m] - glob["L"]))
    else:
        glob = {k: 0.0 for k in keys}
    for k in keys:
        grid = out[k]
        bad = np.isnan(grid)
        if bad.all():
            grid[:] = glob[k]
        elif bad.any():                           # nearest valid tile (keeps local lighting at the ROI edges)
            idx = cv2.distanceTransformWithLabels(bad.astype(np.uint8), cv2.DIST_L2, 3,
                                                  labelType=cv2.DIST_LABEL_PIXEL)[1]
            valid_vals = grid[~bad]
            lab_of_valid = idx[~bad]
            lut = np.zeros(int(idx.max()) + 1, np.float32)
            lut[lab_of_valid] = valid_vals
            grid[bad] = lut[idx[bad]]
        out[k] = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR) if (gh, gw) != (h, w) else grid
    return out


def _boxes_overlap(b: Sequence[float], boxes: Sequence[Sequence[float]]) -> bool:
    x1, y1, x2, y2 = b
    return any(not (x2 <= p[0] or x1 >= p[2] or y2 <= p[1] or y1 >= p[3]) for p in boxes)


def analyse_region(frame: np.ndarray, mask: np.ndarray, r: Any,
                   people: Optional[Callable[[np.ndarray], list[list[float]]] | list[list[float]]] = None) -> GateResult:
    """Uniform gate + colour/luminance candidates inside ``mask`` (bool, frame size). Shadows (luminance-only,
    large) are ignored; candidates on people/vehicles are dropped."""
    H, W = frame.shape[:2]
    s = min(1.0, r.gate_width / W)
    small = cv2.resize(frame, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else frame
    m = cv2.resize(mask.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    roi_area = int(m.sum())
    if roi_area < 64:
        return GateResult(False, [], {"roi_px": roi_area})
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    t = _tile_stats(L, a, b, m, r.tile_px)
    dC = np.hypot(a - t["a"], b - t["b"])
    dL = L - t["L"]
    inner = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0      # the ROI border is not an anomaly
    thrC = np.maximum(r.chroma_k * 1.4826 * t["madC"], r.chroma_min)
    chroma_px = (dC > thrC) & inner
    Lb = cv2.GaussianBlur(L, (3, 3), 0)
    mag = cv2.magnitude(cv2.Sobel(Lb, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(Lb, cv2.CV_32F, 0, 1, ksize=3)) / 4.0
    edges = (mag > r.edge_threshold) & inner
    thrL = np.maximum(r.luma_k * 1.4826 * t["madL"], r.luma_min)
    near_edge = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    luma_px = (np.abs(dL) > thrL) & inner & near_edge & ~chroma_px
    stats = {"roi_px": roi_area, "chroma_std": float(dC[m].std()), "l_std": float(dL[m].std()),
             "edge_density": float(edges.sum() / roi_area)}

    kernel = np.ones((3, 3), np.uint8)
    max_area = r.max_blob_frac * roi_area
    min_px = max(1, int(round(r.min_blob_px * s * s))) if s < 1 else r.min_blob_px
    cands: list[Candidate] = []
    for kind, px, score in (("chroma", chroma_px, dC), ("luma", luma_px, np.abs(dL))):
        mm = cv2.morphologyEx(px.astype(np.uint8), cv2.MORPH_OPEN, kernel) if kind == "chroma" else \
            cv2.morphologyEx(px.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        n, lab_i, st, _ = cv2.connectedComponentsWithStats(mm, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = (int(v) for v in st[i])
            if area < min_px or bw * bh > max_area or area > max_area:
                continue
            if kind == "luma" and (max(bw, bh) / max(1, min(bw, bh)) > 4 or area < 0.35 * bw * bh):
                continue                                  # thin/hollow shapes: joints, kerbs, shadow edges
            sc = float(score[lab_i == i].mean())
            cands.append(Candidate([x / s, y / s, (x + bw) / s, (y + bh) / s], kind, int(round(area / (s * s))), sc))
    weak = [c for c in cands if c.kind == "luma"]
    if len(weak) > r.max_weak_candidates:                 # many light/dark blobs = pavement texture (brick joints)
        cands = [c for c in cands if c.kind != "luma"]
        stats["weak_texture"] = len(weak)
    if cands and people is not None:
        boxes = people(frame) if callable(people) else people
        if boxes:
            cands = [c for c in cands if not _boxes_overlap(c.bbox, boxes)]
    uniform = (stats["chroma_std"] < r.gate_chroma_std and stats["l_std"] < r.gate_l_std and not cands
               and stats["edge_density"] < r.gate_edge_density)
    return GateResult(uniform, cands, {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()})


class PeopleDetector:
    """COCO YOLO (person, bicycle, car, motorcycle, bus, truck), loaded on first use; only run when candidates exist."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    def __call__(self, frame: np.ndarray, weights: str) -> list[list[float]]:
        p = Path(weights) if Path(weights).is_absolute() else PROJECT_ROOT / weights
        if not p.is_file():
            log.warning("people/vehicle model %s not found: candidates are not checked against people", p)
            return []
        with self._lock:
            if str(p) not in self._models:
                from ultralytics import YOLO
                self._models[str(p)] = YOLO(str(p))
            res = self._models[str(p)].predict(frame, imgsz=640, conf=0.35, verbose=False, device="cpu")[0]
        if res.boxes is None:
            return []
        return [b.tolist() for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int))
                if c in PEOPLE_COCO]


PEOPLE = PeopleDetector()


# ============================================================================================ ROI-restricted detection
def _overlap_frac(mask_ii: np.ndarray, box: Sequence[float], w: int, h: int) -> float:
    x1, y1 = int(max(0, np.floor(box[0]))), int(max(0, np.floor(box[1])))
    x2, y2 = int(min(w, np.ceil(box[2]))), int(min(h, np.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inside = mask_ii[y2, x2] - mask_ii[y1, x2] - mask_ii[y2, x1] + mask_ii[y1, x1]
    return float(inside) / float((x2 - x1) * (y2 - y1))


def keep_in_roi(dets: list, mask: np.ndarray, min_overlap: float) -> tuple[list, int]:
    """Keep a detection only if its bottom-centre is inside the mask AND >= min_overlap of the box is inside."""
    h, w = mask.shape[:2]
    ii = cv2.integral(mask.astype(np.uint8))
    kept, dropped = [], 0
    for d in dets:
        x1, _, x2, y2 = d.bbox[:4]
        cx, cy = int(np.clip(round((x1 + x2) / 2), 0, w - 1)), int(np.clip(round(y2) - 1, 0, h - 1))
        if mask[cy, cx] and _overlap_frac(ii, d.bbox, w, h) >= min_overlap:
            kept.append(d)
        else:
            dropped += 1
    return kept, dropped


def _crop_settings(settings: Any) -> Any:
    r = settings.roi
    inf = settings.inference.model_copy(update={"image_size": max(settings.inference.image_size, r.image_size)})
    return settings.model_copy(update={"inference": inf})


def roi_detect(engine: Any, frame: np.ndarray, settings: Any, ctx: Optional[RoiContext],
               force_tiling: Optional[bool] = None, people: Any = "auto") -> tuple[list, Any, Optional[dict[str, Any]]]:
    """Detect litter with the ROI rules. Returns (raw detections in full-frame pixels, InferenceInfo, roi block).
    ``ctx`` None -> the unchanged full-frame engine call and no block. ``force_tiling`` None -> settings.roi.tiling.
    The block's ``candidates`` are resolved later by ``resolve_candidates`` (after the filters set statuses)."""
    from backend.detection.types import InferenceInfo

    if ctx is None:
        dets, info = engine.detect_many([frame], settings, force_tiling)[0]
        return dets, info, None
    r = settings.roi
    block = ctx.summary()
    block.update(status="checked", gate=None, candidates=[], dropped_outside=0, detector_run=True, message=None)
    if not ctx.active:
        if ctx.on_failure == "skip":
            block.update(status="roi_not_determined", detector_run=False)
            return [], InferenceInfo(), block
        dets, info = engine.detect_many([frame], settings, force_tiling)[0]
        return dets, info, block
    h, w = frame.shape[:2]
    mask = ctx.mask(frame.shape)
    if not mask.any():
        block.update(status="clean", detector_run=False, message=NO_LITTER_MSG)
        return [], InferenceInfo(), block
    gate = None
    if r.gate_enabled:
        pfn = (lambda f: PEOPLE(f, r.people_model)) if (people == "auto" and r.exclude_people) else \
            (None if people == "auto" else people)
        gate = analyse_region(frame, mask, r, pfn)
        block["gate"] = {"uniform": gate.uniform, **gate.stats}
        block["candidates"] = [c.to_dict() for c in gate.candidates]
        if gate.uniform:
            block.update(status="clean", detector_run=False, message=UNIFORM_MSG)
            return [], InferenceInfo(), block
    ys, xs = np.nonzero(mask)
    pad = int(0.01 * max(h, w))
    x1, y1 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x2, y2 = min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad)
    crop = frame[y1:y2, x1:x2].copy()
    cm = mask[y1:y2, x1:x2]
    fill = np.median(frame[mask], axis=0).astype(np.uint8)
    crop[~cm] = fill                                       # the shop interior never reaches the detector
    s2 = _crop_settings(settings)
    tiled = force_tiling if force_tiling is not None else {"on": True, "off": False, "auto": None}[r.tiling]
    dets, info = engine.detect(crop, s2, tiled)
    for d in dets:
        d.bbox = [d.bbox[0] + x1, d.bbox[1] + y1, d.bbox[2] + x1, d.bbox[3] + y1]
    dets, dropped = keep_in_roi(dets, mask, r.min_mask_overlap)
    info.extra["roi_crop"] = [x1, y1, x2, y2]
    block["dropped_outside"] = dropped
    return dets, info, block


def resolve_candidates(block: Optional[dict[str, Any]], confirmed: Sequence[Sequence[float]]) -> None:
    """After filtering: a candidate overlapped by a confirming detector box (``confirmed``: xyxy boxes) is CONFIRMED
    by the detector; the others stay POSSIBLE. Sets the block's frame status and message."""
    if not block:
        return
    confirmed = [list(b) for b in confirmed]
    for c in block.get("candidates") or []:
        if c["state"] != "confirmed" and confirmed and _boxes_overlap(c["bbox"], confirmed):
            c.update(state="confirmed", confirmed_by="detector")
    if block.get("status") in ("clean", "roi_not_determined") and not confirmed:
        return
    if confirmed:
        block["status"], block["message"] = "litter", None
    else:
        block["status"] = "clean"
        if block.get("applied"):
            block["message"] = NO_LITTER_MSG


def roi_remarks(block: Optional[dict[str, Any]]) -> list[str]:
    """Remark parts for the dashboard / import CSV (the column set is unchanged)."""
    if not block:
        return []
    parts = list(block.get("remark_tags") or [])
    if block.get("status") == "clean" and block.get("message") == UNIFORM_MSG:
        parts.append(UNIFORM_MSG)
    # phase 2 (noise): colour / light-dark candidates are NOT listed in the remark. They stay in the job JSON
    # (block["candidates"]) for review and mining; the uniform gate is unchanged.
    return parts


class StaticCandidates:
    """Video/live: a POSSIBLE candidate that stays in place for ``static_confirm_sec`` becomes CONFIRMED (static).
    It is reported (block, remark, drawing) but never turned into a litter_detected row: that needs a model class."""

    def __init__(self) -> None:
        self.tracks: list[dict[str, Any]] = []

    def update(self, block: Optional[dict[str, Any]], t: Optional[float], r: Any) -> None:
        if not block or t is None:
            return
        cands = [c for c in block.get("candidates") or [] if c["kind"] == "chroma"]
        from backend.detection.tiling import iou_matrix
        alive = []
        if self.tracks and cands:
            ious = iou_matrix(np.array([c["bbox"] for c in cands], np.float32),
                              np.array([tr["bbox"] for tr in self.tracks], np.float32))
        used = set()
        for i, c in enumerate(cands):
            j = int(np.argmax(ious[i])) if self.tracks and cands else -1
            if j >= 0 and j not in used and ious[i, j] >= 0.3:
                tr = self.tracks[j]
                used.add(j)
                tr.update(last=t, bbox=c["bbox"])
            else:
                tr = {"bbox": c["bbox"], "first": t, "last": t}
            alive.append(tr)
            if c["state"] == "possible" and t - tr["first"] >= r.static_confirm_sec:
                c.update(state="confirmed", confirmed_by="static", static_sec=round(t - tr["first"], 1))
        # a track survives a short gap (the gate may call one frame uniform)
        alive += [tr for k, tr in enumerate(self.tracks) if k not in used and t - tr["last"] <= 3.0]
        self.tracks = alive


# ============================================================================================ drawing
def roi_line_px(width: int, frac: float = 0.004, min_px: int = 4) -> int:
    """Outline thickness on an annotated image: max(min_px, frac x image width)."""
    return max(int(min_px), int(round(frac * width)))


def draw_roi(img: np.ndarray, block: Optional[dict[str, Any]], *, draw_label: bool = True,
             outline_frac: float = 0.004, outline_min_px: int = 4, fill_alpha: float = 0.20) -> None:
    """Light-blue (RGB 135,206,250) polygons: outline max(4 px, 0.4 % of the width), 20 % translucent fill, label
    "Monitored region" with the font scaled to the image width; candidates as small boxes. In place."""
    if not block:
        return
    h, w = img.shape[:2]
    polys = block.get("polygons") or []
    if block.get("applied") and polys:
        m = polygons_mask(polys, (h, w))
        a = float(np.clip(fill_alpha, 0.0, 1.0))
        if m.any() and a > 0:
            img[m] = (img[m].astype(np.float32) * (1 - a) + np.array(ROI_BGR, np.float32) * a).astype(np.uint8)
        t = roi_line_px(w, outline_frac, outline_min_px)
        for p in polys:
            cv2.polylines(img, [_px(p["points"], w, h)], True, ROI_BGR, t, cv2.LINE_AA)
            for hole in p.get("holes") or []:
                cv2.polylines(img, [_px(hole, w, h)], True, ROI_BGR, t, cv2.LINE_AA)
        if draw_label:
            pts = _px(polys[0]["points"], w, h)
            x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
            fs = max(0.5, w / 1100.0)                       # 2560 px wide -> ~2.3, 1280 -> ~1.16
            ft = max(1, int(round(fs * 1.6)))
            pad = max(4, int(round(6 * fs)))
            label = ROI_LABEL + (" (needs review)" if block.get("needs_review") else "")
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
            x = int(np.clip(x + t + 4, 0, max(0, w - tw - 2 * pad)))
            y = int(np.clip(y + t + 4, 0, max(0, h - th - base - 2 * pad)))
            cv2.rectangle(img, (x, y), (x + tw + 2 * pad, y + th + base + 2 * pad), ROI_BGR, -1)
            cv2.putText(img, label, (x + pad, y + pad + th), cv2.FONT_HERSHEY_SIMPLEX, fs, (40, 40, 40), ft,
                        cv2.LINE_AA)
    for c in block.get("candidates") or []:
        x1, y1, x2, y2 = (int(round(v)) for v in c["bbox"])
        col = (0, 140, 255) if c.get("state") == "confirmed" else ROI_BGR
        cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), col, 1 if c.get("state") == "possible" else 2,
                      cv2.LINE_AA)


def banner_lines(block: Optional[dict[str, Any]]) -> list[str]:
    """Extra HUD lines for the annotated image."""
    if not block:
        return []
    out = []
    if block.get("warning"):
        out.append("WARNING: " + block["warning"])
    elif block.get("applied"):
        conf = block.get("confidence")
        out.append(f"Monitored region: {block.get('source')}"
                   + (f", confidence {conf:.2f}" if isinstance(conf, (int, float)) else "")
                   + (", NEEDS REVIEW" if block.get("needs_review") else ""))
    if block.get("status") == "clean" and block.get("applied"):
        out.append(NO_LITTER_MSG + (" (uniform)" if block.get("message") == UNIFORM_MSG else ""))
    possible = sum(1 for c in block.get("candidates") or [] if c.get("state") == "possible")
    if possible:
        out.append(f"Possible anomalies (not litter until confirmed): {possible}")
    return out


# ============================================================================================ CLI
def _main(argv: Sequence[str]) -> int:
    import argparse

    from backend.core.settings import RoiSettings

    ap = argparse.ArgumentParser(prog="python -m backend.services.roi_service")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the segmentation model once (default: settings.roi.model)")
    f.add_argument("model", nargs="?", default=RoiSettings().model)
    c = sub.add_parser("check", help="exit 0 when the model is installed, 1 otherwise")
    c.add_argument("model", nargs="?", default=RoiSettings().model)
    o = sub.add_parser("overlay", help="compute the ROI of an image/video and write an overlay JPEG")
    o.add_argument("source")
    o.add_argument("out")
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        print(f"ROI model ready: {fetch_model(args.model)}")
        return 0
    if args.cmd == "check":
        ok = (model_dir(args.model) / "config.json").is_file() and any(model_dir(args.model).glob("*.safetensors"))
        print(("ROI model present: " if ok else "ROI model MISSING: ") + str(model_dir(args.model)))
        return 0 if ok else 1
    from backend.core.settings import DetectionSettings
    s = DetectionSettings()
    src = Path(args.source)
    frames = ([cv2.imread(str(src))] if src.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
              else video_frames(src, s.roi.vote_frames, s.roi.vote_seconds))
    res = build_roi(frames, s.roi)
    ctx = RoiContext("cli", res.record, "auto" if res.record else "none")
    img = res.reference.copy()
    draw_roi(img, ctx.summary())
    cv2.imwrite(args.out, img)
    print(json.dumps({k: v for k, v in (res.record or {"note": res.note}).items() if k != "frame_signature"}))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
