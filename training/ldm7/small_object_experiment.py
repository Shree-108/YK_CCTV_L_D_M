"""Controlled small-object experiment: does inference size / tiling recover small litter for a given model?

    py -m training.ldm7.small_object_experiment --weights models/litter_yolo/v3_store_cctv/best.pt \
        --images D:/LDM7/ldm7_v1/images/test --labels D:/LDM7/ldm7_v1/labels/test --out D:/LDM7/exp/v3_small \
        [--imgsz 640 960 1280] [--tiling off on] [--tile-size 640] [--overlap 0.2] \
        [--thresholds 0.05 0.1 0.2 0.3] [--conf-floor 0.01] [--match-iou 0.3] [--device cpu]

RAM: every image is loaded once and every setting is run on it in turn; use a small labelled set on this PC and the
big runs on Colab. The labels must follow the ldm7 schema. A v3 model is scored through the COMPARISON mapping in
schema.py; a model with another schema is scored class-agnostically only.

For every ground-truth box and every setting it records the best IoU with any prediction above --conf-floor and the
best confidence of a prediction overlapping it with IoU >= --match-iou (any class and same class). From those:
recall at each threshold (class-agnostic and class-aware), per size bucket (small < 32x32 px, medium <= 96x96,
large), per class, and false positives per image (one-to-one greedy matching, class-agnostic, every prediction of
the model incl. unmapped v3 classes). Outputs: per_gt.csv, summary.csv, summary.json in --out.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from training.ldm7.common import SIZE_BUCKETS
from training.ldm7.evalcore import (TILER_SOURCE, UltralyticsPredictor, gt_bucket, greedy_match, iou_matrix,
                                    iter_labelled_images, map_predictions, predict_setting, safe_div)
from training.ldm7.schema import CLASSES, class_mapper_for


def make_settings(imgsz: Sequence[int], tiling: Sequence[str], tile_size: int, overlap: float,
                  merge_iou: float = 0.5) -> list[dict[str, Any]]:
    out = []
    for sz, t in itertools.product(imgsz, tiling):
        on = t == "on"
        out.append({"name": f"imgsz{sz}" + (f"_tile{tile_size}o{int(overlap * 100)}" if on else ""), "imgsz": int(sz),
                    "tiling": on, "tile_size": tile_size, "overlap": overlap, "include_full_frame": True,
                    "merge_iou": merge_iou})
    return out


def run_experiment(predictor: Any, images: Iterable[tuple[str, np.ndarray, np.ndarray]], settings: list[dict[str, Any]],
                   thresholds: Sequence[float] = (0.05, 0.1, 0.2, 0.3), conf_floor: float = 0.01,
                   match_iou: float = 0.3) -> dict[str, Any]:
    """images: iterable (streamed, one image in RAM at a time) of (image id, BGR array, GT (N,5) cls x1 y1 x2 y2 px).

    Returns {'per_gt': rows, 'summary': {...}}."""
    try:
        kind, mapper = class_mapper_for(predictor.names)
    except ValueError:
        kind, mapper = "unknown", None
    thresholds = sorted(float(t) for t in thresholds)
    per_gt: list[dict[str, Any]] = []
    fp_counts = {s["name"]: {t: 0 for t in thresholds} for s in settings}
    fp_bg = {s["name"]: {t: 0 for t in thresholds} for s in settings}
    n_img = n_bg = n_gt = 0
    for img_id, img, gts in images:
        n_img += 1
        n_bg += int(len(gts) == 0)
        n_gt += len(gts)
        for s in settings:
            raw = predict_setting(predictor, img, s, conf_floor)
            raw = raw[raw[:, 4] >= conf_floor]
            mapped = map_predictions(raw, mapper)[0] if mapper is not None else raw[:0]
            ious_all = iou_matrix(gts[:, 1:5], raw[:, :4]) if len(gts) else np.zeros((0, len(raw)))
            ious_map = iou_matrix(gts[:, 1:5], mapped[:, :4]) if len(gts) else np.zeros((0, len(mapped)))
            for gi, g in enumerate(gts):
                ok_any = ious_all[gi] >= match_iou
                same = (ious_map[gi] >= match_iou) & (mapped[:, 5] == g[0]) if len(mapped) else np.zeros(0, bool)
                per_gt.append({
                    "setting": s["name"], "image": img_id, "gt_index": gi, "class": CLASSES[int(g[0])],
                    "w_px": round(float(g[3] - g[1]), 1), "h_px": round(float(g[4] - g[2]), 1),
                    "area_px": round(float((g[3] - g[1]) * (g[4] - g[2])), 1), "bucket": gt_bucket(g),
                    "best_iou": round(float(ious_all[gi].max()), 4) if ious_all.shape[1] else 0.0,
                    "matched": bool(ok_any.any()),
                    "best_conf_any": round(float(raw[ok_any, 4].max()), 4) if ok_any.any() else 0.0,
                    "best_conf_same_class": round(float(mapped[same, 4].max()), 4) if same.any() else 0.0,
                })
            for t in thresholds:
                p = raw[raw[:, 4] >= t]
                tp, _ = greedy_match(p, gts, match_iou, class_aware=False)
                fp = int((~tp).sum())
                fp_counts[s["name"]][t] += fp
                if len(gts) == 0:
                    fp_bg[s["name"]][t] += fp
    summary: dict[str, Any] = {"model_schema": kind, "model_classes": list(predictor.names), "n_images": n_img,
                               "n_background_images": n_bg, "n_gt": n_gt,
                               "match_iou": match_iou, "conf_floor": conf_floor, "thresholds": thresholds,
                               "tiler": TILER_SOURCE, "settings": {}}
    if kind == "v3":
        summary["note"] = "v3 scored through the comparison-only mapping in training/ldm7/schema.py"
    elif kind == "unknown":
        summary["note"] = "model schema unknown: class-aware recall not computed"
    for s in settings:
        rows = [r for r in per_gt if r["setting"] == s["name"]]
        res: dict[str, Any] = {"config": s, "by_threshold": {}}
        for t in thresholds:
            def rec(sub: list[dict[str, Any]], key: str) -> Optional[float]:
                return safe_div(sum(1 for r in sub if r[key] >= t), len(sub))
            entry = {"recall_any": rec(rows, "best_conf_any"),
                     "recall_class": rec(rows, "best_conf_same_class") if mapper is not None else None,
                     "fp_per_image": safe_div(fp_counts[s["name"]][t], n_img),
                     "fp_per_background_image": safe_div(fp_bg[s["name"]][t], n_bg),
                     "by_bucket": {}, "by_class": {}}
            for b in SIZE_BUCKETS:
                sub = [r for r in rows if r["bucket"] == b]
                entry["by_bucket"][b] = {"n_gt": len(sub), "recall_any": rec(sub, "best_conf_any"),
                                         "recall_class": rec(sub, "best_conf_same_class") if mapper is not None else None}
            for c in CLASSES:
                sub = [r for r in rows if r["class"] == c]
                entry["by_class"][c] = {"n_gt": len(sub), "recall_any": rec(sub, "best_conf_any"),
                                        "recall_class": rec(sub, "best_conf_same_class") if mapper is not None else None}
            res["by_threshold"][str(t)] = entry
        summary["settings"][s["name"]] = res
    return {"per_gt": per_gt, "summary": summary}


def write_outputs(result: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = result["per_gt"]
    fields = ["setting", "image", "gt_index", "class", "w_px", "h_px", "area_px", "bucket", "best_iou", "matched",
              "best_conf_any", "best_conf_same_class"]
    with open(out / "per_gt.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setting", "threshold", "scope", "n_gt", "recall_any", "recall_class", "fp_per_image"])
        for name, res in result["summary"]["settings"].items():
            for t, e in res["by_threshold"].items():
                w.writerow([name, t, "all", "", e["recall_any"], e["recall_class"], e["fp_per_image"]])
                for b, v in e["by_bucket"].items():
                    w.writerow([name, t, f"bucket:{b}", v["n_gt"], v["recall_any"], v["recall_class"], ""])
                for c, v in e["by_class"].items():
                    w.writerow([name, t, f"class:{c}", v["n_gt"], v["recall_any"], v["recall_class"], ""])
    (out / "summary.json").write_text(json.dumps(result["summary"], indent=1), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280])
    ap.add_argument("--tiling", nargs="+", choices=["off", "on"], default=["off", "on"])
    ap.add_argument("--tile-size", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--conf-floor", type=float, default=0.01)
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if not any(a.images.glob("*")):
        print(f"ERROR: no images in {a.images}", file=sys.stderr)
        return 1
    imgs = ((p.name, img, gt) for p, img, gt in iter_labelled_images(a.images, a.labels))
    pred = UltralyticsPredictor(a.weights, a.device)
    res = run_experiment(pred, imgs, make_settings(a.imgsz, a.tiling, a.tile_size, a.overlap), a.thresholds,
                         a.conf_floor, a.match_iou)
    res["summary"]["weights"] = str(a.weights)
    write_outputs(res, a.out)
    print(f"{res['summary']['n_gt']} GT boxes x {len(res['summary']['settings'])} settings -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
