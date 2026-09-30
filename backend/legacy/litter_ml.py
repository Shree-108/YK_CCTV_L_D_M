"""
litter_ml.py — classical + tiny-CNN litter detector (no YOLO, no downloads).

Pipeline:
  1. PROPOSE  candidate boxes with OpenCV: contour proposals at 1x AND 2x
             scale (tiny ~28px objects) + MSER blobs, merged with NMS.
  2. DESCRIBE each box (RF path) with handcrafted features v4: HSV histogram
             + thumbnail + gradient stats + Hu moments + orientation histogram.
  3. CLASSIFY with the best available engine (priority order):
             tiny-CNN (litter_cnn.pt, trained from scratch, no downloads) >
             RandomForest (litter_ml_model.pkl) >
             heuristic contour scorer (always works, never 503s).

Public API:
  load_detector() / reload_detector() -> LitterDetector
  LitterDetector.detect(img_bgr)      -> list[dict] (frontend schema)
  extract_features(crop_bgr)          -> np.ndarray (RF path, v4)
  propose_boxes(img_bgr)              -> [[x1,y1,x2,y2], ...]
  nms / box_iou helpers
  preprocess_crops_for_cnn(crops, mean, std) -> np.ndarray NCHW float32
  TinyLitterCNN                       -> torch module (import torch lazily)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).parent
RF_MODEL_PATH = HERE.parents[1] / "models" / "litter_ml_model.pkl"  # moved; never overwritten
CNN_MODEL_PATH = HERE / "litter_cnn.pt"
CFG_PATH = HERE / "litter_config.json"

FEATURE_VERSION = 4
WORKING_MAX_SIDE = 960
FIXED_CROP = (64, 64)
THUMB = (16, 16)
CNN_INPUT = 48  # square

BACKGROUND_LABEL = "__background__"


# ---------------------------------------------------------------------------
# Config helpers (same file main.py reads)
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CFG_PATH.exists():
        try:
            return json.loads(CFG_PATH.read_text())
        except Exception:
            pass
    return {
        "class_names": [],
        "priority_map": {},
        "bin_map": {},
        "confidence_threshold": 0.30,
        "class_thresholds": {},
        "iou_threshold": 0.5,
        "image_size": WORKING_MAX_SIDE,
        "max_detections": 100,
        "engine": "ml",
        "priority_colors": {
            "critical": "#f87171",
            "high": "#fb923c",
            "medium": "#fbbf24",
            "low": "#34d399",
        },
    }


def priority_of(cfg: dict, cls_name: str) -> str:
    return cfg.get("priority_map", {}).get(cls_name, "medium")


def bin_for(cfg: dict, cls_name: str) -> str:
    return cfg.get("bin_map", {}).get(cls_name, "general")


def threshold_for(cfg: dict, cls_name: str) -> float:
    try:
        per = cfg.get("class_thresholds", {}).get(cls_name, None)
        if per is not None:
            return float(per)
    except Exception:
        pass
    try:
        return float(cfg.get("confidence_threshold", 0.30))
    except Exception:
        return 0.30


# ---------------------------------------------------------------------------
# Feature extraction v4 — pure cv2/numpy, fixed length (RF path)
# ---------------------------------------------------------------------------

def _orientation_hist(gray64: np.ndarray) -> np.ndarray:
    """8-orientation magnitude-weighted histogram over a 4x4 grid = 128 dims."""
    gx = cv2.Sobel(gray64, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray64, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) * 180.0 / np.pi + 180.0)  # 0..360
    h, w = gray64.shape
    feats = []
    for i in range(4):
        for j in range(4):
            m = mag[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4]
            a = ang[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4]
            hist, _ = np.histogram(a, bins=8, range=(0, 360), weights=m)
            s = hist.sum()
            feats.append(hist / s if s > 0 else hist)
    return np.concatenate(feats).astype(np.float32)


def extract_features(crop_bgr: np.ndarray) -> np.ndarray:
    """530-dim feature vector for a BGR crop (v4)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros(feature_dim(), dtype=np.float32)
    img = cv2.resize(crop_bgr, FIXED_CROP, interpolation=cv2.INTER_LINEAR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    s = float(hist.sum())
    hist = hist / s if s > 0 else hist  # 128

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    thumb = thumb.flatten()  # 256

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    edges = cv2.Canny(gray, 60, 150)
    grad = np.array([float(mag.mean() / 255.0), float(mag.std() / 255.0),
                     float((edges > 0).mean())], dtype=np.float32)  # 3

    stats = np.array([
        float(hsv[:, :, 0].mean() / 180.0), float(hsv[:, :, 1].mean() / 255.0),
        float(hsv[:, :, 2].mean() / 255.0), float(hsv[:, :, 1].std() / 255.0),
        float(hsv[:, :, 2].std() / 255.0), float(img[:, :, 0].mean() / 255.0),
        float(img[:, :, 1].mean() / 255.0), float(img[:, :, 2].mean() / 255.0),
    ], dtype=np.float32)  # 8

    try:
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-9)
        hu = np.nan_to_num(hu, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    except Exception:
        hu = np.zeros(7, dtype=np.float32)  # 7

    ori = _orientation_hist(gray)  # 128

    return np.concatenate([hist, thumb, grad, stats, hu, ori]).astype(np.float32)


def feature_dim() -> int:
    return 128 + 256 + 3 + 8 + 7 + 128  # 530


# ---------------------------------------------------------------------------
# Region proposals — contours at 1x+2x plus MSER
# ---------------------------------------------------------------------------

def _boxes_from_contours(contours, img_area: int, min_area: float = 25.0) -> list[list[float]]:
    boxes: list[list[float]] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area or area > 0.15 * img_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 6 or h < 6:
            continue
        aspect = w / max(1.0, float(h))
        if aspect < 0.15 or aspect > 6.0:
            continue
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
    return boxes


def _contour_proposals(img_bgr: np.ndarray) -> list[list[float]]:
    h, w = img_bgr.shape[:2]
    img_area = float(max(1, h * w))
    smooth = cv2.bilateralFilter(img_bgr, 7, 60, 60)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    boxes: list[list[float]] = []
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes += _boxes_from_contours(cnts, img_area)

    try:
        adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 5)
        adap = cv2.morphologyEx(adap, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        adap = cv2.morphologyEx(adap, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts2, _ = cv2.findContours(adap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes += _boxes_from_contours(cnts2, img_area)
    except Exception:
        pass
    return boxes


def _mser_proposals(img_bgr: np.ndarray) -> list[list[float]]:
    """MSER blobs — strong on small high-contrast litter."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return []
    h, w = gray.shape
    img_area = float(max(1, h * w))
    boxes: list[list[float]] = []
    try:
        mser = cv2.MSER_create(_min_area=25, _max_area=int(0.10 * img_area),
                               _max_variation=0.35, _min_diversity=0.3)
        regions, _ = mser.detectRegions(gray)
        for pts in regions:
            x, y, bw, bh = cv2.boundingRect(pts)
            if bw < 6 or bh < 6:
                continue
            aspect = bw / max(1.0, float(bh))
            if aspect < 0.15 or aspect > 6.0:
                continue
            boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
    except Exception:
        pass
    return boxes


def _bgsub_proposals(img_bgr: np.ndarray) -> list[list[float]]:
    """Background-subtraction blobs: median-blur background estimate, Otsu diff.

    Catches pasted/rendered litter whose interior is uniform but differs from
    the local ground — the case Canny/MSER miss most often.
    """
    try:
        h, w = img_bgr.shape[:2]
        blur = cv2.medianBlur(img_bgr, 31)
        diff = cv2.absdiff(img_bgr, blur)
        g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = cv2.morphologyEx(t, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return _boxes_from_contours(cnts, float(h * w))
    except Exception:
        return []


def _border_objectness(img_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Cheap objectness: mean gradient magnitude on each box's border.

    True litter boxes sit on real luminance boundaries; large uniform false
    blobs score near zero however big they are. Vectorised-ish per box.
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    except Exception:
        return np.ones(len(boxes), dtype=np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mmax = float(mag.max())
    if mmax > 0:
        mag /= mmax
    H, W = gray.shape
    out = np.zeros(len(boxes), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(W, int(x2) + 1), min(H, int(y2) + 1)
        if ix2 - ix1 < 2 or iy2 - iy1 < 2:
            continue
        top = mag[iy1:iy1 + 2, ix1:ix2]
        bot = mag[max(iy1, iy2 - 2):iy2, ix1:ix2]
        lft = mag[iy1:iy2, ix1:ix1 + 2]
        rgt = mag[iy1:iy2, max(ix1, ix2 - 2):ix2]
        vals = np.concatenate([top.ravel(), bot.ravel(), lft.ravel(), rgt.ravel()])
        out[i] = float(vals.mean()) if vals.size else 0.0
    return out


def propose_boxes(img_bgr: np.ndarray, max_proposals: int = 350) -> list[list[float]]:
    """Working-scale proposals: contours + MSER + bg-subtraction @1x and @2x."""
    if img_bgr is None or img_bgr.size == 0:
        return []
    h, w = img_bgr.shape[:2]

    boxes = (_contour_proposals(img_bgr) + _mser_proposals(img_bgr)
             + _bgsub_proposals(img_bgr))

    # 2x upscale pass — tiny objects become segmentable (bounded cost)
    if h * w <= 1_100_000:
        try:
            big = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            for x1, y1, x2, y2 in (_contour_proposals(big) + _mser_proposals(big)
                                   + _bgsub_proposals(big)):
                boxes.append([x1 / 2.0, y1 / 2.0, x2 / 2.0, y2 / 2.0])
        except Exception:
            pass

    if not boxes:
        return []
    arr = np.array(boxes, dtype=np.float32)
    # clip + drop degenerate
    arr[:, 0::2] = np.clip(arr[:, 0::2], 0, w)
    arr[:, 1::2] = np.clip(arr[:, 1::2], 0, h)
    good = (arr[:, 2] - arr[:, 0] >= 6) & (arr[:, 3] - arr[:, 1] >= 6)
    arr = arr[good]
    if len(arr) == 0:
        return []
    scores = _border_objectness(img_bgr, arr)
    keep = nms(arr, scores, 0.6)
    arr = arr[keep]
    scores = scores[keep]
    order = np.argsort(-scores)[:max_proposals]
    arr = arr[order]
    out: list[list[float]] = []
    for x1, y1, x2, y2 in arr.tolist():
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0.0, x1 - 0.12 * bw); y1 = max(0.0, y1 - 0.12 * bh)
        x2 = min(float(w), x2 + 0.12 * bw); y2 = min(float(h), y2 + 0.12 * bh)
        if x2 - x1 >= 6 and y2 - y1 >= 6:
            out.append([x1, y1, x2, y2])
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    bb = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores).tolist()
    kept: list[int] = []
    while order:
        i = order.pop(0)
        kept.append(i)
        if not order:
            break
        rest = np.array(order)
        ious = box_iou(boxes[i:i + 1], boxes[rest])[0]
        order = [j for j, v in zip(order, ious.tolist()) if v <= iou_thresh]
    return kept


# ---------------------------------------------------------------------------
# Tiny CNN (PyTorch) — trained from scratch, no downloads
# ---------------------------------------------------------------------------

def get_cnn_module():
    """Lazily build the TinyLitterCNN class (torch import deferred)."""
    import torch
    import torch.nn as nn

    class TinyLitterCNN(nn.Module):
        def __init__(self, n_classes: int = 14):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 24
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 12
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),  # 128*4*4
            )
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(128 * 16, 256), nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, n_classes),
            )

        def forward(self, x):
            return self.head(self.features(x).flatten(1))

    return TinyLitterCNN


def preprocess_crops_for_cnn(crops: list[np.ndarray], mean: list[float],
                             std: list[float]) -> np.ndarray:
    """BGR crops -> NCHW float32 normalized array for the CNN."""
    arr = np.stack([
        cv2.resize(c if c is not None and c.size else np.zeros((8, 8, 3), np.uint8),
                   (CNN_INPUT, CNN_INPUT), interpolation=cv2.INTER_LINEAR)
        for c in crops
    ]).astype(np.float32) / 255.0  # NHWC RGB? still BGR order...
    arr = arr[:, :, :, ::-1]  # BGR -> RGB
    m = np.array(mean, np.float32).reshape(1, 1, 1, 3)
    s = np.array(std, np.float32).reshape(1, 1, 1, 3)
    arr = (arr - m) / np.maximum(s, 1e-6)
    return arr.transpose(0, 3, 1, 2)  # NCHW


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class LitterDetector:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.class_names: list[str] = list(self.cfg.get("class_names", []))
        self.cnn: Any = None
        self.cnn_meta: dict = {}
        self.pipeline: Any = None  # RF fallback
        self._load_models()

    def _load_models(self) -> None:
        self.cfg = load_config()
        self.class_names = list(self.cfg.get("class_names", []))
        # 1) tiny CNN
        if CNN_MODEL_PATH.exists():
            try:
                import torch
                bundle = torch.load(str(CNN_MODEL_PATH), map_location="cpu", weights_only=False)
                TinyLitterCNN = get_cnn_module()
                n_cls = int(bundle.get("n_classes", len(self.class_names) + 1))
                model = TinyLitterCNN(n_classes=n_cls)
                model.load_state_dict(bundle["state_dict"])
                model.eval()
                self.cnn = model
                self.cnn_meta = {k: v for k, v in bundle.items() if k != "state_dict"}
                names = bundle.get("class_names") or []
                if names:
                    self.class_names = list(names)
                print(f"[litter_ml] CNN loaded: {CNN_MODEL_PATH} "
                      f"(classes={len(self.class_names)}, acc={bundle.get('best_val_acc')})")
            except Exception as exc:
                print(f"[litter_ml] CNN load failed ({exc}) — trying RF")
                self.cnn = None
        # 2) RandomForest fallback
        if RF_MODEL_PATH.exists():
            try:
                import joblib
                bundle = joblib.load(str(RF_MODEL_PATH))
                if bundle.get("feature_version") == FEATURE_VERSION:
                    self.pipeline = bundle.get("pipeline")
                    names = bundle.get("class_names") or []
                    if names and self.cnn is None:
                        self.class_names = list(names)
                    print(f"[litter_ml] RF loaded: {RF_MODEL_PATH}")
                else:
                    print(f"[litter_ml] RF feature v{bundle.get('feature_version')} != "
                          f"v{FEATURE_VERSION} — ignoring stale model")
            except Exception as exc:
                print(f"[litter_ml] RF load failed: {exc}")
        if self.cnn is None and self.pipeline is None:
            print("[litter_ml] no trained model — heuristic mode")

    @property
    def ready(self) -> bool:
        return True

    @property
    def trained(self) -> bool:
        return self.cnn is not None or self.pipeline is not None

    @property
    def engine_name(self) -> str:
        if self.cnn is not None:
            return "tiny-cnn"
        if self.pipeline is not None:
            return "ml-classical-rf"
        return "heuristic-contour"

    @property
    def model_id(self) -> str:
        if self.cnn is not None:
            return str(CNN_MODEL_PATH)
        if self.pipeline is not None:
            return str(RF_MODEL_PATH)
        return "heuristic-contour"

    # -- classification backends -------------------------------------------
    def _classify_cnn(self, crops) -> tuple[list[int], list[float]] | None:
        if self.cnn is None or not self.class_names:
            return None
        try:
            import torch
            mean = self.cnn_meta.get("norm_mean", [0.485, 0.456, 0.406])
            std = self.cnn_meta.get("norm_std", [0.229, 0.224, 0.225])
            x = preprocess_crops_for_cnn(crops, mean, std)
            with torch.no_grad():
                logits = self.cnn(torch.from_numpy(x))
                proba = torch.softmax(logits, dim=1).numpy()
            labels: list[str] = self.cnn_meta.get("labels", self.class_names + [BACKGROUND_LABEL])
            n = len(self.class_names)
            out_idx, out_conf = [], []
            for row in proba:
                best_i, best_p, bg_p = -1, 0.0, 0.0
                for col, lab in enumerate(labels):
                    p = float(row[col])
                    if lab == BACKGROUND_LABEL:
                        bg_p = p
                    elif lab in self.class_names:
                        li = self.class_names.index(lab)
                        if p > best_p:
                            best_p, best_i = p, li
                if best_i >= 0 and best_p > bg_p:
                    out_idx.append(best_i); out_conf.append(best_p)
                else:
                    out_idx.append(-1); out_conf.append(float(max(best_p, bg_p)))
            return out_idx, out_conf
        except Exception as exc:
            print(f"[litter_ml] CNN inference failed: {exc}")
            return None

    def _classify_rf(self, crops) -> tuple[list[int], list[float]] | None:
        if self.pipeline is None or not self.class_names:
            return None
        try:
            X = np.stack([extract_features(c) for c in crops]).astype(np.float32)
            proba = self.pipeline.predict_proba(X)
            labels = list(self.pipeline.classes_)
            out_idx, out_conf = [], []
            for row in proba:
                best_i, best_p, bg_p = -1, 0.0, 0.0
                for col, lab in enumerate(labels):
                    p = float(row[col])
                    if lab == BACKGROUND_LABEL:
                        bg_p = p
                    elif str(lab) in self.class_names:
                        li = self.class_names.index(str(lab))
                        if p > best_p:
                            best_p, best_i = p, li
                if best_i >= 0 and best_p > bg_p:
                    out_idx.append(best_i); out_conf.append(best_p)
                else:
                    out_idx.append(-1); out_conf.append(float(max(best_p, bg_p)))
            return out_idx, out_conf
        except Exception as exc:
            print(f"[litter_ml] RF inference failed: {exc}")
            return None

    def _classify_heuristic(self, crops) -> tuple[list[int], list[float]]:
        idx, conf = [], []
        for c in crops:
            base = c if c is not None and c.size else np.zeros((8, 8, 3), np.uint8)
            g = cv2.cvtColor(cv2.resize(base, FIXED_CROP), cv2.COLOR_BGR2GRAY)
            e = (cv2.Canny(g, 60, 150) > 0).mean()
            hsv = cv2.cvtColor(cv2.resize(base, FIXED_CROP), cv2.COLOR_BGR2HSV)
            s = float(hsv[:, :, 1].mean() / 255.0)
            score = float(np.clip(0.25 + 1.4 * e + 0.5 * abs(s - 0.25), 0.05, 0.85))
            idx.append(-2); conf.append(score)
        return idx, conf

    def detect(self, img_bgr: np.ndarray, *,
               max_proposals: int = 350, raw: bool = False,
               working_max_side: int | None = None) -> list[dict]:
        """Detect litter. raw=True returns ungated per-proposal votes
        (no thresholds/floor/area-gating, no final NMS) for ensemble fusion."""
        if img_bgr is None or img_bgr.size == 0:
            return []
        H, W = img_bgr.shape[:2]
        cap = working_max_side or WORKING_MAX_SIDE
        scale = min(1.0, cap / max(1.0, float(max(H, W))))
        if scale < 1.0:
            work = cv2.resize(img_bgr, (int(W * scale), int(H * scale)),
                              interpolation=cv2.INTER_LINEAR)
        else:
            work, scale = img_bgr, 1.0

        proposals = propose_boxes(work, max_proposals=max_proposals)
        if not proposals:
            return []

        crops: list[np.ndarray] = []
        for x1, y1, x2, y2 in proposals:
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(work.shape[1], int(x2) + 1), min(work.shape[0], int(y2) + 1)
            if ix2 - ix1 < 4 or iy2 - iy1 < 4:
                crops.append(np.zeros((8, 8, 3), np.uint8))
            else:
                crops.append(work[iy1:iy2, ix1:ix2])

        result = self._classify_cnn(crops)
        if result is None:
            result = self._classify_rf(crops)
        if result is None:
            result = self._classify_heuristic(crops)
        pred_idx, pred_conf = result

        cfg = self.cfg
        names = self.class_names or ["other_litter"]
        floor = float(cfg.get("global_floor", 0.0))
        min_area = float(cfg.get("min_box_area", 0.0))
        dets: list[dict] = []
        for (x1, y1, x2, y2), pi, pc in zip(proposals, pred_idx, pred_conf):
            if pi == -1:
                continue
            if pi == -2:
                cls_name = "other_litter"
                conf = float(pc)
                if not raw and conf < max(float(cfg.get("confidence_threshold", 0.30)), floor):
                    continue
            else:
                cls_name = names[pi] if 0 <= pi < len(names) else f"class_{pi}"
                conf = float(pc)
                if not raw and conf < max(threshold_for(cfg, cls_name), floor):
                    continue
            X1, Y1, X2, Y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
            area = float((X2 - X1) * (Y2 - Y1))
            if not raw and area < min_area:
                continue
            dets.append({
                "class": cls_name,
                "confidence": round(conf, 4),
                "priority": priority_of(cfg, cls_name),
                "bin": bin_for(cfg, cls_name),
                "box": [round(float(X1), 2), round(float(Y1), 2),
                        round(float(X2), 2), round(float(Y2), 2)],
                "area": round(area, 2),
                "_score": conf,
            })

        if not dets:
            return []
        if raw:
            # ungated votes for WBF fusion; strip helper key only
            for d in dets:
                d.pop("_score", None)
            return dets
        boxes = np.array([d["box"] for d in dets], dtype=np.float32)
        scores = np.array([d["_score"] for d in dets], dtype=np.float32)
        keep = nms(boxes, scores, float(cfg.get("iou_threshold", 0.5)))
        max_det = int(cfg.get("max_detections", 100))
        out = []
        for i in keep[:max_det]:
            d = dict(dets[i])
            d.pop("_score", None)
            out.append(d)
        out.sort(key=lambda d: d["confidence"], reverse=True)
        return out


_detector: LitterDetector | None = None


def load_detector(force: bool = False) -> LitterDetector:
    global _detector
    if _detector is None or force:
        _detector = LitterDetector()
    return _detector


def reload_detector() -> LitterDetector:
    return load_detector(force=True)
