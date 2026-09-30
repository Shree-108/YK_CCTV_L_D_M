"""Box proposals for manual labelling: union of the fine-tuned and the open-vocabulary models at LOW
confidence (tiled), merged with class-agnostic NMS. Proposals are a labelling aid, never labels.

    python -m training.propose --src <image folder> --out <proposals.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from backend.core.config import PROJECT_ROOT
from backend.core.settings import merged_settings
from backend.detection.engine import YoloEngine

MODELS = PROJECT_ROOT / "models" / "litter_yolo"


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.08)
    ap.add_argument("--max", type=int, default=40)
    a = ap.parse_args(argv)
    low = {"weak_threshold": a.conf, "candidate_threshold": a.conf, "confirmed_threshold": a.conf}
    engines = []
    # the synthetic-trained v1 fires on pavement texture when tiled, so only the open-vocab model proposes
    for version, extra in (("v2_yoloe_openvocab", {"inference": {"image_size": 960, "agnostic_nms": True}}),):
        s = merged_settings({**extra, "confidence": low, "tiling": {"enabled": True, "mode": "always", "tile_size": 640, "overlap": 0.25}})
        engines.append((version, s, YoloEngine(MODELS / version / "best.pt", s, version)))
    out: dict[str, list[dict]] = {}
    paths = sorted(p for p in Path(a.src).iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    for n, p in enumerate(paths, 1):
        img = cv2.imread(str(p))
        props = []
        for version, s, eng in engines:
            dets, _ = eng.detect(img, s, force_tiling=True)
            props += [{"bbox": [round(v, 1) for v in d.bbox], "cls": d.cls_name, "conf": round(d.confidence, 3), "src": version[:2]}
                      for d in dets]
        props.sort(key=lambda d: -d["conf"])
        kept: list[dict] = []
        for d in props:   # class-agnostic NMS across both models
            if all(iou(d["bbox"], k["bbox"]) < 0.5 for k in kept):
                kept.append(d)
        out[p.name] = kept[:a.max]
        print(f"[{n}/{len(paths)}] {p.name}: {len(kept)} proposals", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
