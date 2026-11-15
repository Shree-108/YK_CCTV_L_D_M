"""Transfer reference annotations to every variant of the same scene, then write YOLO labels for the chosen ROI.

Per variant (homography reference -> variant, validated as crop + uniform zoom):
  * accepted boxes are mapped and clipped to the frame
      visible >= 50 % of the mapped area and both sides >= 4 px  -> label candidate ("transferred")
      otherwise                                                   -> IGNORE region ("clipped")
  * ambiguous regions and out-of-ROI litter are mapped as IGNORE regions (never labels)
  * the variant area NOT covered by the reference is recorded as "uncovered" (needs manual inspection)
  * a per-box alignment score: NCC between the reference crop warped into the variant and the variant crop
Writes annotations/per_image/<stem>.json (source of truth) and labels/train/<stem>.txt (in-ROI labels only).
Labels are CANDIDATES (label_status=transferred_candidate) until verify.py records the visual check.

    python -m experiments.exp_aug100.transfer_boxes --roi pav_kerb_gutter
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from experiments.exp_aug100.common import (ANN, IMG_DIR, LBL_DIR, SCENES, check_v3, features, fname, homography,
                                           load_json, read_manifest, save_json, write_manifest)

ROI_OPTIONS = {
    "pavement": {"zones": ["pavement"], "edge": "pav|kerb",
                 "desc": "pavement only; kerb, gutter and road litter are ignore regions"},
    "pav_kerb_gutter": {"zones": ["pavement", "kerb", "gutter"], "edge": "gutter|road",
                        "desc": "pavement + kerb + gutter/drain channel; carriageway excluded"},
}


def clip(b, w, h):
    return [max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])]


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def ncc_score(ref_g, var_g, H, box, pad=6):
    """NCC of the variant box region against the reference warped into variant coordinates."""
    h, w = var_g.shape
    x1, y1, x2, y2 = [int(round(v)) for v in (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    T = np.array([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]], float) @ H
    warped = cv2.warpPerspective(ref_g, T, (x2 - x1, y2 - y1), flags=cv2.INTER_LINEAR)
    a = warped.astype(np.float32).ravel(); b = var_g[y1:y2, x1:x2].astype(np.float32).ravel()
    if a.std() < 1 or b.std() < 1:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def roi_polygon(boundary, H, w, h):
    """In-ROI region = everything left of the boundary polyline (extended to the frame top/bottom)."""
    line = np.float32(boundary)
    top = line[0] + (line[0] - line[1]) * 50; bot = line[-1] + (line[-1] - line[-2]) * 50
    poly_ref = np.vstack([[-5000, top[1]], top, line, bot, [-5000, bot[1]]]).astype(np.float32)
    poly = cv2.perspectiveTransform(poly_ref.reshape(-1, 1, 2), H).reshape(-1, 2)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in poly]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--roi", choices=list(ROI_OPTIONS), required=True)
    a = ap.parse_args(argv)
    check_v3()
    roi = ROI_OPTIONS[a.roi]
    save_json(ANN / "roi_choice.json", {"roi": a.roi, **roi, "status": "PROVISIONAL - awaiting user approval",
                                        "options": ROI_OPTIONS})
    rows = {r["file"]: r for r in read_manifest()}
    summary = {}
    for scene, cfg in SCENES.items():
        ref_file = fname(cfg["reference"]); ref = load_json(ANN / f"reference_{scene}.json", None)
        ref_feat = features(IMG_DIR / ref_file); ref_g = ref_feat[0]
        s = dict(images=0, labels=0, clipped_ignored=0, out_of_roi_ignored=0, ambiguous_ignored=0, uncovered_images=0)
        for n in cfg["ids"]:
            f = fname(n); stem = f[:-4]
            if f == ref_file:
                H = np.eye(3); var_g = ref_g; inl, rms = 0, 0.0
            else:
                var_feat = features(IMG_DIR / f); var_g = var_feat[0]
                H, inl, rms = homography(ref_feat, var_feat)
            h, w = var_g.shape
            labels, ignore = [], []
            for b in ref["boxes"]:
                m = [float(v) for v in cv2.perspectiveTransform(np.float32([[b["xyxy"][:2]], [b["xyxy"][2:]]]), H).reshape(-1)]
                c = clip(m, w, h)
                vis = area(c) / max(area(m), 1e-6)
                rec = {"ref_id": b["id"], "xyxy": [round(v, 1) for v in c], "zone": b["zone"], "visible": round(vis, 3),
                       "ncc": None if f == ref_file else ncc_score(ref_g, var_g, H, c)}
                if vis < 0.5 or c[2] - c[0] < 4 or c[3] - c[1] < 4:
                    if area(c) > 0:
                        ignore.append({**rec, "why": "clipped"}); s["clipped_ignored"] += 1
                    continue
                if b["zone"] not in roi["zones"]:
                    ignore.append({**rec, "why": f"out_of_roi:{b['zone']}"}); s["out_of_roi_ignored"] += 1
                    continue
                labels.append({**rec, "origin": "reference" if f == ref_file else "transferred"})
            for b in ref["ambiguous"]:
                m = [float(v) for v in cv2.perspectiveTransform(np.float32([[b["xyxy"][:2]], [b["xyxy"][2:]]]), H).reshape(-1)]
                c = clip(m, w, h)
                if area(c) > 0:
                    ignore.append({"ref_id": b["id"], "xyxy": [round(v, 1) for v in c], "why": "ambiguous"}); s["ambiguous_ignored"] += 1
            # variant pixels not covered by the reference view
            Hi = np.linalg.inv(H); rh, rw = ref_g.shape
            cov = cv2.warpPerspective(np.full((rh, rw), 255, np.uint8), H, (w, h)) > 250
            unc = ~cov; frac = float(unc.mean())
            uncovered = []
            if frac > 0.0005:
                ys, xs = np.where(unc); uncovered = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                s["uncovered_images"] += 1
            rec = {"image": f, "scene": scene, "reference": ref_file, "H_ref_to_image": np.round(H, 6).tolist(),
                   "inliers": inl, "rms_px": round(rms, 3), "roi": a.roi,
                   "roi_polygon": roi_polygon(ref["boundaries"][roi["edge"]], H, w, h),
                   "labels": labels, "ignore": ignore, "uncovered_frac": round(frac, 5), "uncovered_bbox": uncovered,
                   "manual_added": [], "manual_removed": [], "status": "reference_reviewed" if f == ref_file else "transferred_candidate"}
            old = load_json(ANN / "per_image" / f"{stem}.json", None)
            if old:  # keep manual corrections made during verification
                rec["manual_added"] = old.get("manual_added", []); rec["manual_removed"] = old.get("manual_removed", [])
                if old.get("status") == "verified": rec["status"] = "verified"; rec["verified"] = old.get("verified")
            save_json(ANN / "per_image" / f"{stem}.json", rec)
            write_yolo(rec, w, h)
            r = rows[f]
            r["label_status"] = rec["status"] if rec["status"] != "reference_reviewed" else "manual_verified"
            r["n_transferred"] = str(sum(1 for b in labels if b["origin"] == "transferred"))
            r["n_manual"] = str(len(rec["manual_added"]) + (len(labels) if f == ref_file else 0))
            r["n_boxes"] = str(len(final_boxes(rec)))
            if f == ref_file:
                r["verified_by"] = "claude (visual review of every proposal)"; r["verified_at"] = "2026-11-13"
            s["images"] += 1; s["labels"] += len(final_boxes(rec))
        summary[scene] = s
        print(scene, s)
    write_manifest(sorted(rows.values(), key=lambda r: r["file"]))
    save_json(ANN / "transfer_summary.json", summary)
    check_v3()
    return 0


def final_boxes(rec):
    removed = set(rec.get("manual_removed", []))
    return [b["xyxy"] for b in rec["labels"] if b["ref_id"] not in removed] + [b["xyxy"] for b in rec.get("manual_added", [])]


def write_yolo(rec, w, h):
    lines = []
    for x1, y1, x2, y2 in final_boxes(rec):
        lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    (LBL_DIR / (rec["image"][:-4] + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
