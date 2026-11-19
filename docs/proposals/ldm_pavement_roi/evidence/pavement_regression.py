"""Before/after regression evidence for the (revised) pavement-only patch on the two real cameras.

Run with the PATCHED backend on PYTHONPATH (the patch is not applied to the active application):
    PYTHONPATH=<patched copy> python pavement_regression.py --repo <project root> \
        --out-images <git-ignored dir> --out-json <results.json>

Read-only use of the project: the screenshots of the two existing automatic jobs and the production v3 weights.
No database write, no alert, no setting, no camera record: detect_image() is called with db=None and alerts=None,
in-memory settings, and "registered" is only simulated (camera_registered=True) - nothing is registered.
Each camera uses ONLY its own proposed polygon (CAM_A: docs/proposals/ldm_phaseABC/roi_proposal_CAM_A.json,
EZVIZ: docs/proposals/ldm_pavement_roi/roi_draft_EZVIZ_for_review.json). Both are proposals, NOT approved.

Regression evidence on 8 screenshots of 2 recordings - not an accuracy measurement.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sqlite3
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from backend.core.settings import merged_settings
from backend.detection import pavement as pv
from backend.detection.engine import YoloEngine
from backend.detection.filters import ROISet, apply_filters
from backend.detection.types import Detection
from backend.services.pipeline import detect_image
from backend.services.visualization import draw_pavement_roi

CAMERAS = {
    "CAM_A_indoor_counter": {"job": "sv_20261115_232628_e603d5",
                             "roi_file": "docs/proposals/ldm_phaseABC/roi_proposal_CAM_A.json",
                             "roi_key": "api_payload_roi_field_only", "ref": (2560, 1440)},
    "EZVIZ_outdoor": {"job": "sv_20261116_103744_eb0a51",
                      "roi_file": "docs/proposals/ldm_pavement_roi/roi_draft_EZVIZ_for_review.json",
                      "roi_key": "zones", "ref": (1920, 1080)},
}

# name -> (settings patch, camera id kind, use ROI). Full-frame inference everywhere except X (experimental crop).
PAV = {"enabled": True}
CONFIGS = {
    "R0_production_today": ({}, "upload", False),
    "G_general_image_mode": ({"pavement": PAV}, "upload", False),
    "C1_camera_roi_mode_no_size_rule": ({"pavement": {**PAV, "size": {"min_side_frac": 0, "min_area_frac": 0}}}, "cam", True),
    "C2_camera_roi_mode_proposed": ({"pavement": PAV}, "cam", True),
    "N_registered_camera_no_roi": ({"pavement": PAV}, "cam", False),
    "X_experimental_roi_crop": ({"pavement": PAV, "filters": {"roi_crop_inference": True}}, "cam", True),
}


def zones_for(repo: Path, cam: dict) -> list[dict]:
    raw = json.loads((repo / cam["roi_file"]).read_text(encoding="utf-8"))[cam["roi_key"]]
    return [{k: z[k] for k in ("name", "type", "mode", "polygon")} for z in raw]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--out-images", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    a = ap.parse_args()
    repo = a.repo.resolve()
    db = sqlite3.connect(f"file:{repo / 'storage' / 'litter_monitor.db'}?mode=ro", uri=True)
    engine = YoloEngine(repo / "models" / "litter_yolo" / "v3_store_cctv" / "best.pt", merged_settings({}), "v3_store_cctv")
    out: dict = {"model": "v3_store_cctv", "device": engine.device, "configs": {k: v[0] for k, v in CONFIGS.items()},
                 "cameras": {}}

    for cam_name, cam in CAMERAS.items():
        roi = ROISet.from_json(zones_for(repo, cam))           # this camera's own polygon only
        frames = db.execute("SELECT idx, frame_idx, original_path FROM sampled_frames WHERE job_id=? ORDER BY idx",
                            (cam["job"],)).fetchall()
        cdir = a.out_images / cam_name
        cdir.mkdir(parents=True, exist_ok=True)
        cres: dict = {"job": cam["job"], "roi_source": cam["roi_file"], "roi_status": "PROPOSED_NOT_APPROVED",
                      "screenshots": []}
        totals = {c: Counter() for c in CONFIGS}
        timings = {c: [] for c in CONFIGS}
        gate_ms: list[float] = []
        images = {}
        for idx, fidx, path in frames:
            img = cv2.imread(path)
            images[idx] = img
            h, w = img.shape[:2]
            shot: dict = {"screenshot": idx, "frame_index": fidx, "size": [w, h], "configs": {}}
            for cname, (patch, kind, use_roi) in CONFIGS.items():
                p = copy.deepcopy(patch)
                if "pavement" in p:
                    p["pavement"]["roi_reference_size"] = list(cam["ref"])
                s = merged_settings(p)
                cam_id = "upload" if kind == "upload" else f"{cam_name}"
                t0 = time.perf_counter()
                res = detect_image(engine, img, s, roi if use_roi else None, None, None, None, cam_id,
                                   want_diagnostic=True, camera_registered=kind == "cam")
                wall = (time.perf_counter() - t0) * 1000
                ann, diag = res.pop("annotated"), res.pop("diagnostic", None)
                res.pop("alerts")
                timings[cname].append({"inference_ms": res["inference_ms"], "wall_ms": round(wall, 1), "tiles": res["tiles"]})
                c = res["pavement"]["counts"] if "pavement" in res else dict(res["counts"])
                totals[cname].update(c)
                shot["configs"][cname] = {"counts": c, "mode": (res.get("pavement") or {}).get("status", "PAVEMENT_OFF"),
                                          "inference_ms": res["inference_ms"], "tiles": res["tiles"],
                                          "detections": res["detections"]}
                cv2.imwrite(str(cdir / f"s{idx}_{cname}.jpg"), ann, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if diag is not None:
                    cv2.imwrite(str(cdir / f"s{idx}_{cname}_diagnostic.jpg"), diag, [cv2.IMWRITE_JPEG_QUALITY, 88])
            prev = img.copy()
            draw_pavement_roi(prev, roi, float(np.clip(w / 900, 0.6, 1.4)))
            cv2.imwrite(str(cdir / f"s{idx}_original_with_proposed_roi.jpg"), prev, [cv2.IMWRITE_JPEG_QUALITY, 88])
            raw = [Detection(list(d["bbox"]), 0, d["class"], d["confidence"])
                   for d in shot["configs"]["R0_production_today"]["detections"]]
            s2 = merged_settings({"pavement": PAV})
            t0 = time.perf_counter()
            for _ in range(200):
                ds = [Detection(list(d.bbox), 0, d.cls_name, d.confidence) for d in raw]
                apply_filters(ds, img.shape, s2, None)
                pv.gate(ds, img.shape, roi, pv.CAMERA_ROI_MODE, s2.pavement)
            gate_ms.append((time.perf_counter() - t0) * 1000 / 200)
            # post-filter property: same boxes before and after (full-frame inference both times)
            b0 = sorted(tuple(d["bbox"]) for d in shot["configs"]["R0_production_today"]["detections"])
            b2 = sorted(tuple(d["bbox"]) for d in shot["configs"]["C2_camera_roi_mode_proposed"]["detections"])
            shot["raw_boxes_identical_R0_vs_C2"] = b0 == b2
            cres["screenshots"].append(shot)

        cres["totals"] = {c: dict(v) for c, v in totals.items()}
        cres["timing"] = {c: {"median_inference_ms": round(statistics.median(t["inference_ms"] for t in v), 1),
                              "median_wall_ms": round(statistics.median(t["wall_ms"] for t in v), 1),
                              "tiles": v[0]["tiles"]} for c, v in timings.items()}
        cres["timing"]["gate_only_ms_per_frame_median"] = round(statistics.median(gate_ms), 3)

        # what production confirms today and what happens to it under C2
        trans = []
        for sh in cres["screenshots"]:
            c2 = {tuple(d["bbox"]): d for d in sh["configs"]["C2_camera_roi_mode_proposed"]["detections"]}
            for d in sh["configs"]["R0_production_today"]["detections"]:
                if d["status"] in ("confirmed", "candidate"):
                    e = c2.get(tuple(d["bbox"]), {})
                    trans.append({"screenshot": sh["screenshot"], "class": d["class"], "confidence": d["confidence"],
                                  "bbox": d["bbox"], "before": d["status"], "after": e.get("outcome"),
                                  "reason": e.get("outcome_reason"),
                                  "membership": (e.get("geometry") or {}).get("membership")})
        cres["production_detections_before_vs_after"] = trans

        # 4. remaining confirmed detections: hard-negative CANDIDATES (pending human review), with recurrence
        hn = []
        hdir = a.out_images / "hard_negative_candidates"
        hdir.mkdir(parents=True, exist_ok=True)
        for sh in cres["screenshots"]:
            img = images[sh["screenshot"]]
            H, W = img.shape[:2]
            diag_len = math.hypot(W, H)
            for d in sh["configs"]["C2_camera_roi_mode_proposed"]["detections"]:
                if d.get("outcome") != pv.CONFIRMED_LITTER:
                    continue
                x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
                pad = 60
                crop = img[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
                name = f"{cam_name}_s{sh['screenshot']}_{x1}_{y1}_{x2}_{y2}"
                cv2.imwrite(str(hdir / f"{name}_original_crop.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cx, cy = (d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2
                rec = []
                for o in cres["screenshots"]:
                    if o["screenshot"] == sh["screenshot"]:
                        continue
                    for e in o["configs"]["C2_camera_roi_mode_proposed"]["detections"]:
                        ex, ey = (e["bbox"][0] + e["bbox"][2]) / 2, (e["bbox"][1] + e["bbox"][3]) / 2
                        if iou(d["bbox"], e["bbox"]) >= 0.3 or math.hypot(cx - ex, cy - ey) / diag_len <= 0.01:
                            rec.append({"screenshot": o["screenshot"], "confidence": e["confidence"], "class": e["class"],
                                        "outcome": e.get("outcome"), "bbox": e["bbox"]})
                g = d["geometry"]
                hn.append({"camera": cam_name, "screenshot": sh["screenshot"], "frame_index": sh["frame_index"],
                           "crop_file": f"storage/proposals/pavement_roi/hard_negative_candidates/{name}_original_crop.jpg",
                           "class": d["class"], "confidence": d["confidence"], "bbox": d["bbox"],
                           "box_w_px": g["box_w_px"], "box_h_px": g["box_h_px"], "min_side_frac": g["min_side_frac"],
                           "area_frac": g["area_frac"], "roi_membership": g["membership"], "zone": g.get("zone"),
                           "foot_point": g.get("foot_point"), "accepted_because": d.get("outcome_reason"),
                           "touches_frame_edge": y2 >= H - 1 or x2 >= W - 1 or x1 <= 0 or y1 <= 0,
                           "recurs_in_other_screenshots": rec,
                           "review_status": "HARD_NEGATIVE_CANDIDATE - pending human review; not added to any dataset"})
        cres["remaining_confirmed_hard_negative_candidates"] = hn
        out["cameras"][cam_name] = cres
        print(cam_name, json.dumps(cres["totals"]))
        print("  timing", json.dumps(cres["timing"]))
        print("  identical boxes R0 vs C2:", [s["raw_boxes_identical_R0_vs_C2"] for s in cres["screenshots"]])

    a.out_json.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
