"""Tiled inference helpers: overlapping tiles, coordinate remapping, merging.

Original frame -> overlapping tiles -> YOLO per tile -> map boxes back to the
frame -> class-aware global NMS -> suppress cut-off partial boxes that lie
mostly inside a box from ANOTHER tile/pass. Boxes from the same pass are never
suppressed by containment, so two packets lying side by side stay two boxes.
"""
from __future__ import annotations

import numpy as np

from backend.detection.types import Detection


def tile_grid(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) tiles covering the frame with the given fractional overlap."""
    if tile <= 0:
        raise ValueError("tile size must be positive")
    stride = max(1, int(round(tile * (1.0 - overlap))))

    def starts(length: int) -> list[int]:
        if length <= tile:
            return [0]
        s = list(range(0, length - tile, stride))
        s.append(length - tile)          # last tile flush with the edge
        return sorted(set(s))

    return [(x, y, min(x + tile, width), min(y + tile, height)) for y in starts(height) for x in starts(width)]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def ios_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the SMALLER box area."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(np.minimum(area_a[:, None], area_b[None, :]), 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return keep


def merge_detections(dets: list[Detection], iou_thr: float, cross_source_ios: float,
                     agnostic: bool = False) -> list[Detection]:
    """Global class-aware NMS, then cross-source containment suppression."""
    if len(dets) <= 1:
        return list(dets)
    boxes = np.array([d.bbox for d in dets], dtype=np.float32)
    scores = np.array([d.confidence for d in dets], dtype=np.float32)
    keep: list[int] = []
    groups = [range(len(dets))] if agnostic else [
        [i for i, d in enumerate(dets) if d.cls_id == c] for c in sorted({d.cls_id for d in dets})]
    for g in groups:
        idx = np.array(list(g))
        keep.extend(int(idx[k]) for k in nms(boxes[idx], scores[idx], iou_thr))
    kept = [dets[i] for i in sorted(keep, key=lambda i: -dets[i].confidence)]

    # Cross-source containment: a partial box cut by a tile border is mostly inside the
    # full box found by a neighbouring tile/pass. Same-source neighbours are untouched.
    out: list[Detection] = []
    kb = np.array([d.bbox for d in kept], dtype=np.float32)
    ios = ios_matrix(kb, kb)
    suppressed = set()
    for i, d in enumerate(kept):
        if i in suppressed:
            continue
        out.append(d)
        for j in range(i + 1, len(kept)):
            o = kept[j]
            if j in suppressed or o.source == d.source:
                continue
            if (agnostic or o.cls_id == d.cls_id) and ios[i, j] >= cross_source_ios:
                suppressed.add(j)
    return out
