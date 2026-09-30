"""Model evaluation on a labelled dataset split, with the SAME engine used in production.

Produces: overall P/R/F1/mAP50/mAP50-95, per-class P/R/F1/AP50/AP50-95, recall by object
size (small <32^2 px, medium <96^2, large), report-group recall (chips / wrapper / plastic
packaging), a confidence-threshold sweep (P/R/F1/FP/FN), false positives on negative
images, latency, an optional tiled-vs-full comparison, per-image results (for the
ground-truth review page) and hard examples (FN, FP, tiny, low-confidence, blurred).

Matching: class-aware, greedy by descending confidence, IoU >= threshold (COCO-style).
AP: 101-point interpolated precision envelope. Detections are the raw model output
(no ROI/size filters) so the numbers describe the model, not a camera configuration.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable, Optional

import cv2
import numpy as np

from backend.core.settings import DetectionSettings, merged_settings
from backend.database.db import utcnow
from backend.detection.engine import YoloEngine
from backend.detection.tiling import iou_matrix
from backend.services.datasets import TINY_PX, DatasetService, size_bucket

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)
SWEEP = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
EVAL_FLOOR = 0.01


def match_image(gt: list[dict[str, Any]], preds: list[dict[str, Any]], iou_thr: float) -> tuple[list[int], list[int]]:
    """Greedy class-aware matching. Returns (pred_match -> gt index or -1, gt_match -> pred index or -1)."""
    pm, gm = [-1] * len(preds), [-1] * len(gt)
    if not preds or not gt:
        return pm, gm
    ious = iou_matrix(np.array([p["bbox"] for p in preds], np.float32), np.array([g["bbox"] for g in gt], np.float32))
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i]["confidence"]):
        best, best_iou = -1, iou_thr
        for gi in range(len(gt)):
            if gm[gi] == -1 and gt[gi]["cls_id"] == preds[pi]["cls_id"] and ious[pi, gi] >= best_iou:
                best, best_iou = gi, ious[pi, gi]
        if best >= 0:
            pm[pi], gm[best] = best, pi
    return pm, gm


def average_precision(confs: list[float], tp: list[bool], n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if not confs:
        return 0.0
    order = np.argsort(-np.asarray(confs))
    tps = np.asarray(tp, dtype=float)[order]
    ctp, cfp = np.cumsum(tps), np.cumsum(1 - tps)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(x, mrec, mpre)))


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def _nan(v: float) -> Optional[float]:
    return None if v != v else round(v, 4)


def _predict_all(engine: YoloEngine, ds: DatasetService, name: str, refs: list, s: DetectionSettings,
                 tiling: bool, progress: Optional[Callable[[float], None]], p0: float, p1: float) -> list[dict[str, Any]]:
    images = []
    for i, ref in enumerate(refs):
        img = cv2.imread(str(ref.path))
        if img is None:
            continue
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        dets, _ = engine.detect(img, s, force_tiling=tiling)
        ms = (time.perf_counter() - t0) * 1000
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append({
            "id": ref.path.relative_to(ds.path(name)).as_posix(), "split": ref.split, "negative": ref.negative,
            "width": w, "height": h, "latency_ms": ms, "gray": gray,
            "gt": ds.gt_boxes_px(name, ref, w, h),
            "preds": [{"cls_id": d.cls_id, "class": d.cls_name, "confidence": d.confidence,
                       "bbox": [float(v) for v in d.bbox]} for d in dets]})
        if progress:
            progress(p0 + (p1 - p0) * (i + 1) / len(refs))
    return images


def _metrics(images: list[dict[str, Any]], names: list[str], op_thr: float,
             report_groups: dict[str, list[str]]) -> dict[str, Any]:
    n_gt = defaultdict(int)
    for im in images:
        for g in im["gt"]:
            n_gt[g["cls_id"]] += 1
    # AP per class per IoU threshold
    ap = np.full((len(names), len(IOU_THRS)), np.nan)
    for ti, thr in enumerate(IOU_THRS):
        confs, tps = defaultdict(list), defaultdict(list)
        for im in images:
            pm, _ = match_image(im["gt"], im["preds"], float(thr))
            for p, m in zip(im["preds"], pm):
                confs[p["cls_id"]].append(p["confidence"])
                tps[p["cls_id"]].append(m >= 0)
        for c in range(len(names)):
            ap[c, ti] = average_precision(confs[c], tps[c], n_gt[c])
    # matches at IoU .5 for threshold-dependent metrics
    for im in images:
        im["pm"], im["gm"] = match_image(im["gt"], im["preds"], 0.5)

    def counts_at(t: float, cls: Optional[int] = None) -> tuple[int, int, int]:
        tp = fp = fn = 0
        for im in images:
            for p, m in zip(im["preds"], im["pm"]):
                if p["confidence"] >= t and (cls is None or p["cls_id"] == cls):
                    tp, fp = tp + (m >= 0), fp + (m < 0)
            for g, m in zip(im["gt"], im["gm"]):
                if cls is None or g["cls_id"] == cls:
                    fn += not (m >= 0 and im["preds"][m]["confidence"] >= t)
        return tp, fp, fn

    sweep = [{"threshold": t, **prf(*counts_at(t))} for t in SWEEP]
    per_class = []
    for c, n in enumerate(names):
        per_class.append({"class": n, "instances": n_gt[c], **prf(*counts_at(op_thr, c)),
                          "ap50": _nan(ap[c, 0]), "ap50_95": _nan(float(np.nanmean(ap[c])) if n_gt[c] else float("nan"))})

    def recall_where(pred: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        tot = hit = 0
        for im in images:
            for g, m in zip(im["gt"], im["gm"]):
                if pred(g):
                    tot += 1
                    hit += m >= 0 and im["preds"][m]["confidence"] >= op_thr
        return {"instances": tot, "detected": hit, "recall": round(hit / tot, 4) if tot else None}

    neg = [im for im in images if im["negative"]]
    neg_fp = sum(1 for im in neg for p in im["preds"] if p["confidence"] >= op_thr)
    lat = [im["latency_ms"] for im in images]
    valid_ap = ap[[c for c in range(len(names)) if n_gt[c] > 0]]
    return {
        "operating_threshold": op_thr,
        "overall": {**prf(*counts_at(op_thr)),
                    "mAP50": _nan(float(np.nanmean(valid_ap[:, 0]))) if len(valid_ap) else None,
                    "mAP50_95": _nan(float(np.nanmean(valid_ap))) if len(valid_ap) else None},
        "per_class": per_class,
        "by_size": {s: recall_where(lambda g, s=s: g["size"] == s) for s in ("small", "medium", "large")},
        "tiny_recall": recall_where(lambda g: g["area_px"] < TINY_PX),
        "report_groups": {k: recall_where(lambda g, v=v: g["class"] in v) for k, v in report_groups.items()},
        "threshold_sweep": sweep,
        "best_f1_threshold": max(sweep, key=lambda r: (r["f1"], -r["threshold"]))["threshold"],
        "negatives": {"images": len(neg), "false_positives": neg_fp,
                      "fp_per_image": round(neg_fp / len(neg), 3) if neg else None},
        "latency_ms": {"mean": round(float(np.mean(lat)), 1) if lat else None,
                       "p95": round(float(np.percentile(lat, 95)), 1) if lat else None},
        "images": len(images), "gt_objects": int(sum(n_gt.values())),
    }


def _hard_examples(images: list[dict[str, Any]], op_thr: float, weak: float, dataset: str, version: str,
                   source: str) -> list[dict[str, Any]]:
    out, now = [], utcnow()

    def row(kind: str, im: dict[str, Any], cls: str, conf: Optional[float], b: list[float]) -> dict[str, Any]:
        return {"created_at": now, "kind": kind, "source": source, "dataset": dataset, "split": im["split"],
                "image": im["id"], "class": cls, "confidence": conf, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                "model_version": version}

    for im in images:
        for g, m in zip(im["gt"], im["gm"]):
            matched_conf = im["preds"][m]["confidence"] if m >= 0 else None
            if matched_conf is None or matched_conf < op_thr:
                out.append(row("false_negative", im, g["class"], matched_conf, g["bbox"]))
                if g["area_px"] < TINY_PX * 4:
                    out.append(row("tiny", im, g["class"], matched_conf, g["bbox"]))
                x1, y1, x2, y2 = (int(max(0, v)) for v in g["bbox"])
                crop = im["gray"][y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
                if crop.size >= 16 and cv2.Laplacian(crop, cv2.CV_64F).var() < 30:
                    out.append(row("blurred", im, g["class"], matched_conf, g["bbox"]))
            if matched_conf is not None and weak <= matched_conf < op_thr:
                out.append(row("low_confidence", im, g["class"], matched_conf, g["bbox"]))
        for p, m in zip(im["preds"], im["pm"]):
            if m < 0 and p["confidence"] >= op_thr:
                out.append(row("false_positive", im, p["class"], p["confidence"], p["bbox"]))
    return out


def run_evaluation(engine: YoloEngine, ds: DatasetService, dataset: str, split: str, base: DetectionSettings,
                   report_groups: dict[str, list[str]], compare_tiling: bool = False,
                   include_negatives: bool = True, operating_threshold: Optional[float] = None,
                   progress: Optional[Callable[[float], None]] = None) -> dict[str, Any]:
    names = ds.classes(dataset)
    if names != engine.class_names:
        raise ValueError(f"dataset classes {names} do not match model classes {engine.class_names}")
    refs = [r for r in ds._images(dataset) if r.split == split and (include_negatives or not r.negative)]
    if not refs:
        raise ValueError(f"split '{split}' of dataset '{dataset}' has no images")
    s = merged_settings({**base.model_dump(), "confidence": {"weak_threshold": EVAL_FLOOR,
                                                              "candidate_threshold": EVAL_FLOOR,
                                                              "confirmed_threshold": EVAL_FLOOR}})
    op = operating_threshold if operating_threshold is not None else base.confidence.confirmed_threshold
    modes = [False, True] if compare_tiling else [engine.should_tile((1080, 1920), base)]
    results: dict[str, Any] = {}
    images_main = None
    for k, tiled in enumerate(modes):
        images = _predict_all(engine, ds, dataset, refs, s, tiled, progress, k / len(modes), (k + 1) / len(modes))
        results["tiled" if tiled else "full_frame"] = _metrics(images, names, op, report_groups)
        if images_main is None or not compare_tiling:
            images_main = images
    main_key = "full_frame" if compare_tiling else ("tiled" if modes[0] else "full_frame")
    main = results[main_key]
    tiling_verdict = None
    if compare_tiling:
        a, b = results["full_frame"], results["tiled"]
        d_small = ((b["by_size"]["small"]["recall"] or 0) - (a["by_size"]["small"]["recall"] or 0))
        d_map = (b["overall"]["mAP50"] or 0) - (a["overall"]["mAP50"] or 0)
        tiling_verdict = {
            "delta_mAP50": round(d_map, 4), "delta_small_recall": round(d_small, 4),
            "delta_precision": round(b["overall"]["precision"] - a["overall"]["precision"], 4),
            "latency_factor": round((b["latency_ms"]["mean"] or 0) / max(a["latency_ms"]["mean"] or 1, 1e-6), 2),
            "recommend_tiling": bool(d_small > 0.02 or d_map > 0.01),
            "note": "Measured on this split only; re-run on real CCTV data before changing production settings.",
        }
    per_image = [{"id": im["id"], "split": im["split"], "negative": im["negative"], "width": im["width"],
                  "height": im["height"],
                  "tp": sum(1 for p, m in zip(im["preds"], im["pm"]) if m >= 0 and p["confidence"] >= op),
                  "fp": sum(1 for p, m in zip(im["preds"], im["pm"]) if m < 0 and p["confidence"] >= op),
                  "fn": sum(1 for g, m in zip(im["gt"], im["gm"]) if not (m >= 0 and im["preds"][m]["confidence"] >= op)),
                  "gt": len(im["gt"])} for im in images_main]
    hard = _hard_examples(images_main, op, base.confidence.weak_threshold, dataset, engine.version, "evaluation")
    return {"model_version": engine.version, "dataset": dataset, "split": split, "created_at": utcnow(),
            "classes": names, "mode": main_key, **main, "modes": results if compare_tiling else None,
            "tiling_comparison": tiling_verdict, "per_image": per_image, "hard_examples_found": len(hard),
            "_hard": hard}
