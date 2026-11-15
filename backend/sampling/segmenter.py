"""Scene segmentation for the outdoor-ROI proposal.

Model: UperNet with a ConvNeXt-tiny backbone trained on ADE20K (150 scene classes), Hugging Face
``openmmlab/upernet-convnext-tiny``, MIT licence, stored locally in ``models/roi_segmenter/upernet-convnext-tiny``
(no network access at run time). It is used only to locate ground surfaces and shop-interior structures; litter
itself is detected by the existing LDM YOLO model.

ADE20K classes mapped to (chosen after measuring on real Yatrikart CCTV, see docs/video_analysis/EVALUATION.md):
  walkable ground   road, sidewalk, path, earth, grass, sand, field, land, dirt track, floor
  clearly outdoor   the same minus "floor"
  shop interior     shelf, cabinet, case, refrigerator, counter, countertop, box, arcade machine, bookcase, buffet,
                    kitchen island, wardrobe, chest of drawers, ceiling
On these cameras ADE20K labels outdoor paver pavement mostly "floor" and product shelving as box / case /
"arcade machine", so a label alone cannot separate pavement from a shop floor. The proposal therefore takes the
walkable ground, carves the interior structures out of it, and lowers its confidence by the share that is only
"floor" - ambiguous scenes always go to the operator for confirmation.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from backend.sampling.roi_proposal import Masks

if TYPE_CHECKING:
    from backend.core.config import AppConfig

log = logging.getLogger(__name__)

OUTDOOR = ("road", "sidewalk", "path", "earth", "grass", "sand", "field", "land", "dirt track")
GROUND = (*OUTDOOR, "floor")
INDOOR = ("shelf", "cabinet", "case", "refrigerator", "counter", "countertop", "box", "arcade machine", "bookcase",
          "buffet", "kitchen island", "wardrobe", "chest of drawers", "ceiling")
SHORT_SIDE, MAX_LONG = 384, 768     # pavement/interior are large regions; keeps CPU time low


class UperNetSegmenter:
    name = "upernet-convnext-tiny (ADE20K)"

    def __init__(self, path: Path) -> None:
        import torch
        from transformers import UperNetForSemanticSegmentation

        self._torch = torch
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.model = UperNetForSemanticSegmentation.from_pretrained(str(path), local_files_only=True).eval()
        labels = {int(k): v.strip() for k, v in self.model.config.id2label.items()}
        self.labels = labels
        by_name = {v: k for k, v in labels.items()}
        missing = [n for n in (*GROUND, *INDOOR) if n not in by_name]
        if missing:
            raise RuntimeError(f"segmentation model lacks expected scene classes: {missing}")
        self.ground_ids = [by_name[n] for n in GROUND]
        self.indoor_ids = [by_name[n] for n in INDOOR]
        self.outdoor_ids = [by_name[n] for n in OUTDOOR]
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)
        self._lock = threading.Lock()

    def _tensor(self, image: np.ndarray):
        h, w = image.shape[:2]
        s = SHORT_SIDE / min(h, w)
        if max(h, w) * s > MAX_LONG:
            s = MAX_LONG / max(h, w)
        nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
        rgb = cv2.cvtColor(cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        return self._torch.from_numpy(x.transpose(2, 0, 1)[None].copy())

    def masks(self, image: np.ndarray) -> Masks:
        torch = self._torch
        with self._lock, torch.inference_mode():
            logits = self.model(pixel_values=self._tensor(image)).logits[0]     # (150, h', w')
            probs = torch.softmax(logits, dim=0)
            ground = probs[self.ground_ids].sum(0).numpy()
            indoor = probs[self.indoor_ids].sum(0).numpy()
            outdoor = probs[self.outdoor_ids].sum(0).numpy()
            top_p, top = probs.max(0)
            top_p, top = top_p.numpy(), top.numpy()
        relevant = np.isin(top, self.ground_ids + self.indoor_ids)
        certainty = float(top_p[relevant].mean()) if relevant.any() else float(top_p.mean())
        ids, counts = np.unique(top, return_counts=True)
        shares = {self.labels[int(i)]: round(float(c) / top.size, 3) for i, c in zip(ids, counts) if c / top.size >= 0.01}
        h, w = image.shape[:2]
        up = lambda m: cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)  # noqa: E731
        return Masks(up(ground), up(indoor), certainty, dict(sorted(shares.items(), key=lambda kv: -kv[1])), up(outdoor))


_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


def load_segmenter(cfg: "AppConfig") -> Optional[UperNetSegmenter]:
    """The shared segmenter, loaded on first use; None (manual ROI only) when disabled or unavailable."""
    if cfg.roi_segmenter == "off":
        return None
    key = str(cfg.roi_segmenter_dir)
    with _cache_lock:
        if key in _cache:
            return _cache[key] if isinstance(_cache[key], UperNetSegmenter) else None   # type: ignore[return-value]
        try:
            if not (Path(key) / "config.json").is_file():
                raise FileNotFoundError(f"segmentation weights not found in {key}")
            _cache[key] = UperNetSegmenter(Path(key))
            log.info("ROI segmenter loaded from %s", key)
        except Exception as exc:  # noqa: BLE001 - missing optional dependency/weights: manual ROI still works
            log.warning("ROI segmenter unavailable (%s); the operator will draw the ROI", exc)
            _cache[key] = exc
        return _cache[key] if isinstance(_cache[key], UperNetSegmenter) else None   # type: ignore[return-value]


def segmenter_status(cfg: "AppConfig") -> dict:
    s = _cache.get(str(cfg.roi_segmenter_dir))
    return {"enabled": cfg.roi_segmenter != "off", "loaded": isinstance(s, UperNetSegmenter),
            "error": None if s is None or isinstance(s, UperNetSegmenter) else str(s),
            "weights_present": (Path(cfg.roi_segmenter_dir) / "config.json").is_file()}
