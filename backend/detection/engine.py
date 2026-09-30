"""YOLO detection engine (Ultralytics) with optional tiled inference.

Thread-safe: one engine instance is shared by all cameras/jobs; predictions are
serialized with a lock (a single model on one device cannot run concurrently
anyway). The model can be hot-swapped (model activation) without restarting.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from backend.core.settings import DetectionSettings
from backend.detection.tiling import merge_detections, tile_grid
from backend.detection.types import Detection, InferenceInfo

log = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested and requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            log.warning("Device %s requested but CUDA is not available; using CPU", requested)
            return "cpu"
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hardware_profile() -> dict[str, Any]:
    """Describe the hardware and recommend a model size / resolution for it."""
    import os

    import torch

    info: dict[str, Any] = {"cuda": torch.cuda.is_available(), "cpu_count": os.cpu_count()}
    if info["cuda"]:
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1e9
        info.update(gpu=props.name, vram_gb=round(vram, 1))
        info["recommended"] = ({"model": "yolo11l", "train_imgsz": 1280, "infer_imgsz": 1280} if vram >= 16 else
                               {"model": "yolo11m", "train_imgsz": 1024, "infer_imgsz": 1024} if vram >= 8 else
                               {"model": "yolo11s", "train_imgsz": 960, "infer_imgsz": 960})
    else:
        info["recommended"] = {"model": "yolo11s", "train_imgsz": 640, "infer_imgsz": 640,
                               "note": "CPU only: train on a GPU (e.g. Colab) for larger models/resolutions"}
    return info


class YoloEngine:
    def __init__(self, weights: str | Path, settings: DetectionSettings, version: str = "unknown") -> None:
        self._lock = threading.Lock()
        self.settings = settings
        self.version = version
        self.weights = str(weights)
        self._load(self.weights)

    def _load(self, weights: str) -> None:
        from ultralytics import YOLO

        if not Path(weights).is_file():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        model = YOLO(weights)
        if getattr(model, "task", "detect") not in ("detect", "segment"):
            raise ValueError(f"{weights} is a '{model.task}' model; a detection model is required")
        self.model = model
        raw = {int(k): v for k, v in model.names.items()}
        # Open-vocabulary models (YOLOE) ship a vocab.json next to the weights: many text prompts map onto
        # the canonical litter classes, and "distractor" prompts (car, shoe, poster...) are detected only so
        # they can be discarded - they compete with litter prompts and remove false positives.
        vocab_file = Path(weights).with_name("vocab.json")
        self.vocab: Optional[dict[str, Any]] = None
        self._raw_to_cls: Optional[dict[int, Optional[int]]] = None
        if vocab_file.is_file():
            import json
            self.vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
            canon = self.vocab.get("classes") or list(dict.fromkeys(self.vocab["prompt_to_class"].values()))
            self._raw_to_cls = {i: (canon.index(self.vocab["prompt_to_class"][n]) if n in self.vocab["prompt_to_class"] else None)
                                for i, n in raw.items()}
            self.names = dict(enumerate(canon))
        else:
            self.names = raw
        self.device = resolve_device(self.settings.inference.device)
        self.train_imgsz = int((getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 640))
        log.info("Loaded YOLO %s (%d classes) on %s", weights, len(self.names), self.device)

    def swap(self, weights: str | Path, version: str) -> None:
        """Hot-swap the model; in-flight predictions finish on the old model first."""
        with self._lock:
            old = (self.model, self.names, self.weights, self.version)
            try:
                self._load(str(weights))
                self.weights, self.version = str(weights), version
            except Exception:
                self.model, self.names, self.weights, self.version = old
                raise

    def update_settings(self, settings: DetectionSettings) -> None:
        with self._lock:
            self.settings = settings
            self.device = resolve_device(settings.inference.device)

    @property
    def class_names(self) -> list[str]:
        return [self.names[i] for i in sorted(self.names)]

    # ------------------------------------------------------------------ inference
    def _predict(self, image: np.ndarray, imgsz: int, s: DetectionSettings, source: str,
                 offset: tuple[int, int] = (0, 0)) -> list[Detection]:
        inf = s.inference
        res = self._run(image, imgsz, s)[0]
        return self._to_dets(res, source, offset)

    def _run(self, images: Any, imgsz: int, s: DetectionSettings) -> list[Any]:
        inf = s.inference
        kw: dict[str, Any] = {"half": True} if inf.half and self.device.startswith("cuda") else {}
        return self.model.predict(images, imgsz=imgsz, conf=s.confidence.weak_threshold, iou=inf.iou_threshold,
                                  max_det=inf.max_detections, device=self.device, agnostic_nms=inf.agnostic_nms,
                                  verbose=False, **kw)

    def _to_dets(self, res: Any, source: str, offset: tuple[int, int] = (0, 0)) -> list[Detection]:
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        ox, oy = offset
        if self._raw_to_cls is not None:
            # suppress litter boxes that are mostly inside a distractor (e.g. "wrapper" on a poster/car)
            from backend.detection.tiling import ios_matrix
            dis = np.array([self._raw_to_cls.get(int(c)) is None for c in cls])
            keep = ~dis
            if dis.any() and keep.any():
                ios = ios_matrix(xyxy[keep], xyxy[dis])
                stronger = conf[dis][None, :] > conf[keep][:, None]
                keep_idx = np.nonzero(keep)[0]
                keep[keep_idx[((ios > 0.8) & stronger).any(axis=1)]] = False
            xyxy, conf = xyxy[keep], conf[keep]
            cls = np.array([self._raw_to_cls[int(c)] for c in cls[keep]], dtype=int)
        return [Detection([float(b[0] + ox), float(b[1] + oy), float(b[2] + ox), float(b[3] + oy)], int(c),
                          self.names.get(int(c), str(c)), float(p), source=source)
                for b, p, c in zip(xyxy, conf, cls)]

    def detect_many(self, images: list[np.ndarray], settings: Optional[DetectionSettings] = None,
                    force_tiling: Optional[bool] = None) -> list[tuple[list[Detection], InferenceInfo]]:
        """Detect on several frames. Full-frame inference is sent to the model as ONE batch (big win on
        GPU); tiled inference falls back to per-frame detect()."""
        if not images:
            return []
        s = settings or self.settings
        if len(images) == 1 or force_tiling or (force_tiling is None and any(self.should_tile(i.shape, s) for i in images)):
            return [self.detect(i, s, force_tiling) for i in images]
        with self._lock:
            t0 = time.perf_counter()
            results = self._run(list(images), s.inference.image_size, s)
            per = (time.perf_counter() - t0) * 1000 / len(images)
            out = []
            for r in results:
                dets = self._to_dets(r, "full")[: s.inference.max_detections]
                out.append((dets, InferenceInfo(inference_ms=per, raw_before_merge=len(dets))))
            return out

    def should_tile(self, shape: tuple[int, ...], s: DetectionSettings | None = None) -> bool:
        s = s or self.settings
        t = s.tiling
        if not t.enabled:
            return False
        if t.mode == "always":
            return True
        return max(shape[:2]) > t.tile_size * t.auto_min_ratio

    def detect(self, image: np.ndarray, settings: Optional[DetectionSettings] = None,
               force_tiling: Optional[bool] = None) -> tuple[list[Detection], InferenceInfo]:
        """Detect objects in a BGR image. Returns detections >= weak threshold and timing info."""
        if image is None or not isinstance(image, np.ndarray) or image.ndim != 3 or image.size == 0:
            raise ValueError("detect() expects a non-empty BGR image (H, W, 3)")
        with self._lock:
            s = settings or self.settings
            t0 = time.perf_counter()
            info = InferenceInfo()
            tiled = self.should_tile(image.shape, s) if force_tiling is None else force_tiling
            dets: list[Detection] = []
            if not tiled or s.tiling.include_full_frame:
                dets += self._predict(image, s.inference.image_size, s, "full")
            if tiled:
                h, w = image.shape[:2]
                tiles = tile_grid(w, h, s.tiling.tile_size, s.tiling.overlap)
                for i, (x1, y1, x2, y2) in enumerate(tiles):
                    # tile inferred near native resolution -> small litter keeps its pixels
                    side = -(-max(x2 - x1, y2 - y1) // 32) * 32  # YOLO stride multiple
                    dets += self._predict(image[y1:y2, x1:x2], side, s, f"tile:{i}", (x1, y1))
                info.tiles, info.tiled = len(tiles), True
                info.raw_before_merge = len(dets)
                dets = merge_detections(dets, s.tiling.merge_iou, s.tiling.cross_tile_ios, s.inference.agnostic_nms)
            else:
                info.raw_before_merge = len(dets)
            dets = dets[: s.inference.max_detections]
            info.inference_ms = (time.perf_counter() - t0) * 1000
            return dets, info
