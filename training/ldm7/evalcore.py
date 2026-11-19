"""Evaluation core shared by small_object_experiment.py and compare_models.py.

A *predictor* is any object with
    names: list[str]                                     the model's own class names (index = class id)
    __call__(image_bgr, imgsz, conf) -> np.ndarray (N, 6) rows x1, y1, x2, y2, confidence, class_id in the pixel
                                                          coordinates of the image it was given
so tests can use a fake predictor and real runs use UltralyticsPredictor (loaded lazily, CPU by default).

Tiling reuses backend.detection.tiling.tile_grid when importable (import only, the backend is never modified) and
falls back to an identical self-contained grid otherwise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from training.ldm7.common import read_image, size_bucket
from training.ldm7.schema import NUM_CLASSES

try:  # read-only reuse of the production tiler
    from backend.detection.tiling import tile_grid as _backend_tile_grid
except Exception:  # pragma: no cover - backend not importable
    _backend_tile_grid = None


def _local_tile_grid(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    if tile <= 0:
        raise ValueError("tile size must be positive")
    stride = max(1, int(round(tile * (1.0 - overlap))))

    def starts(length: int) -> list[int]:
        if length <= tile:
            return [0]
        s = list(range(0, length - tile, stride))
        s.append(length - tile)
        return sorted(set(s))

    return [(x, y, min(x + tile, width), min(y + tile, height)) for y in starts(height) for x in starts(width)]


def tile_grid(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    return (_backend_tile_grid or _local_tile_grid)(width, height, tile, overlap)


TILER_SOURCE = "backend.detection.tiling.tile_grid" if _backend_tile_grid else "training.ldm7.evalcore (local)"


# ---------------------------------------------------------------- geometry
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def nms_classwise(preds: np.ndarray, iou_thr: float) -> np.ndarray:
    if len(preds) == 0:
        return preds.reshape(0, 6)
    keep = []
    for c in np.unique(preds[:, 5]):
        idx = np.nonzero(preds[:, 5] == c)[0]
        idx = idx[np.argsort(-preds[idx, 4], kind="stable")]
        while len(idx):
            i = idx[0]
            keep.append(i)
            if len(idx) == 1:
                break
            ious = iou_matrix(preds[i:i + 1, :4], preds[idx[1:], :4])[0]
            idx = idx[1:][ious < iou_thr]
    keep = sorted(keep, key=lambda i: (-preds[i, 4], i))
    return preds[keep]


# ---------------------------------------------------------------- ground truth
def load_gt(label_path: Path, width: int, height: int) -> np.ndarray:
    """YOLO label file -> (N, 5) rows class, x1, y1, x2, y2 in pixels. Missing file = error (never 'clean')."""
    if not Path(label_path).is_file():
        raise FileNotFoundError(f"{label_path} missing: background images need an EMPTY label file")
    rows = []
    for ln in Path(label_path).read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if not p:
            continue
        if len(p) != 5:
            raise ValueError(f"{label_path}: bad line {ln!r}")
        c, cx, cy, w, h = int(float(p[0])), *map(float, p[1:])
        if not 0 <= c < NUM_CLASSES:
            raise ValueError(f"{label_path}: class id {c} outside the ldm7 schema")
        rows.append([c, (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def gt_bucket(gt_row: np.ndarray) -> str:
    return size_bucket(float((gt_row[3] - gt_row[1]) * (gt_row[4] - gt_row[2])))


def iter_labelled_images(images_dir: Path, labels_dir: Path):
    from training.ldm7.common import list_images
    for p in list_images(images_dir):
        img = read_image(p)
        if img is None:
            raise ValueError(f"{p} cannot be decoded")
        h, w = img.shape[:2]
        yield p, img, load_gt(Path(labels_dir) / f"{p.stem}.txt", w, h)


# ---------------------------------------------------------------- prediction
class UltralyticsPredictor:
    """Lazy Ultralytics YOLO wrapper (CPU unless device is given). Loaded only when a real run is requested."""

    def __init__(self, weights: Path, device: str = "cpu") -> None:
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.names = [self.model.names[i] for i in sorted(self.model.names)]
        self.device = device

    def __call__(self, image: np.ndarray, imgsz: int, conf: float) -> np.ndarray:
        r = self.model.predict(image, imgsz=int(imgsz), conf=float(conf), device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 6))
        b = r.boxes
        return np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1)


def predict_setting(predictor: Any, image: np.ndarray, setting: dict[str, Any], conf: float) -> np.ndarray:
    """Run one inference setting: full frame at imgsz, or overlapping tiles (+ optional full frame) merged by NMS."""
    h, w = image.shape[:2]
    if not setting.get("tiling"):
        return np.asarray(predictor(image, setting["imgsz"], conf), dtype=np.float64).reshape(-1, 6)
    parts = []
    if setting.get("include_full_frame", True):
        parts.append(np.asarray(predictor(image, setting["imgsz"], conf), dtype=np.float64).reshape(-1, 6))
    for x1, y1, x2, y2 in tile_grid(w, h, int(setting["tile_size"]), float(setting["overlap"])):
        side = int(np.ceil(max(x2 - x1, y2 - y1) / 32.0) * 32)       # tile inferred near native resolution
        d = np.asarray(predictor(image[y1:y2, x1:x2], side, conf), dtype=np.float64).reshape(-1, 6).copy()
        d[:, [0, 2]] += x1
        d[:, [1, 3]] += y1
        parts.append(d)
    allp = np.concatenate(parts, 0) if parts else np.zeros((0, 6))
    return nms_classwise(allp, float(setting.get("merge_iou", 0.5)))


def map_predictions(preds: np.ndarray, mapper: Sequence[Optional[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Model class ids -> ldm7 ids. Returns (mapped, unmapped); unmapped keeps the model's own class id."""
    if len(preds) == 0:
        return preds.reshape(0, 6), preds.reshape(0, 6)
    tgt = np.array([-1 if mapper[int(c)] is None else mapper[int(c)] for c in preds[:, 5]])
    mapped = preds[tgt >= 0].copy()
    mapped[:, 5] = tgt[tgt >= 0]
    return mapped, preds[tgt < 0].copy()


# ---------------------------------------------------------------- matching / metrics
def greedy_match(preds: np.ndarray, gts: np.ndarray, iou_thr: float, class_aware: bool) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one greedy matching in descending confidence. Returns (tp flag per pred, matched gt index or -1)."""
    order = np.argsort(-preds[:, 4], kind="stable") if len(preds) else np.zeros(0, int)
    tp = np.zeros(len(preds), bool)
    which = np.full(len(preds), -1)
    if len(preds) == 0 or len(gts) == 0:
        return tp, which
    ious = iou_matrix(preds[:, :4], gts[:, 1:5])
    used = np.zeros(len(gts), bool)
    for i in order:
        cand = ious[i].copy()
        cand[used] = -1
        if class_aware:
            cand[gts[:, 0] != preds[i, 5]] = -1
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            tp[i], which[i], used[j] = True, j, True
    return tp, which


def average_precision(confs: Sequence[float], tps: Sequence[bool], n_gt: int) -> Optional[float]:
    """All-point interpolated AP (VOC 2010+ / COCO-style area). None when there is no ground truth."""
    if n_gt == 0:
        return None
    if len(confs) == 0:
        return 0.0
    order = np.argsort(-np.asarray(confs, dtype=np.float64), kind="stable")
    tp = np.asarray(tps, dtype=np.float64)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def safe_div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else a / b
