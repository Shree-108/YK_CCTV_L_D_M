"""List CANDIDATE litter-free frames from dataset/ldm_v4_workspace (read-only) for a human to confirm.

Ranking aid only: real-CCTV, non-overlay frames sorted by how little v3 sees at low confidence. Nothing is copied
into the experiment until the user confirms a frame as genuinely litter-free (annotations/clean_confirmed.json).

    python -m experiments.exp_aug100.clean_candidates --top 24
"""
from __future__ import annotations

import argparse
import csv

import cv2
import numpy as np
from ultralytics import YOLO

from experiments.exp_aug100.common import ANN, DS, ROOT, V3, check_v3, save_json

WS = ROOT / "dataset" / "ldm_v4_workspace"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--top", type=int, default=24); a = ap.parse_args(argv)
    check_v3()
    with open(WS / "manifest.csv", newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["source_type"] == "real_cctv" and r["overlay_suspect"] != "yes"]
    m = YOLO(str(V3)); scored = []
    for r in rows:
        p = WS / "images" / "train" / r["image"]
        if not p.exists(): continue
        res = m.predict(str(p), imgsz=640, conf=0.10, verbose=False, device="cpu", agnostic_nms=True)[0]
        confs = [float(c) for c in res.boxes.conf]
        scored.append({"image": r["image"], "session": r["session"], "camera": r["camera"], "source_path": r["source_path"],
                       "frame_index": r["frame_index"], "timestamp_sec": r["timestamp_sec"], "conditions": r["conditions"],
                       "used_in_previous_training": r["used_in_previous_training"], "review_status": r["review_status"],
                       "v3_dets_conf010": len(confs), "v3_max_conf": round(max(confs), 3) if confs else 0.0})
    scored.sort(key=lambda s: (s["v3_dets_conf010"], s["v3_max_conf"]))
    top = scored[:a.top]
    save_json(ANN / "clean_candidates.json", {"note": "CANDIDATES ONLY - not confirmed litter-free", "n_real_frames_scanned": len(scored),
                                             "candidates": top})
    tiles = []
    for i, s in enumerate(top):
        im = cv2.imread(str(WS / "images" / "train" / s["image"])); im = cv2.resize(im, (400, int(im.shape[0] * 400 / im.shape[1])))
        lab = np.zeros((30, 400, 3), np.uint8)
        cv2.putText(lab, f"C{i:02d} {s['session'][:22]} f{s['frame_index']} t{s['timestamp_sec']}s", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        cv2.putText(lab, f"v3 dets@0.10={s['v3_dets_conf010']} max={s['v3_max_conf']}", (3, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        tiles.append(np.vstack([lab, im]))
    while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
    rowsimg = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    for k in range(0, len(rowsimg), 3):
        cv2.imwrite(str(DS / "review" / f"clean_candidates_{k // 3}.jpg"), np.vstack(rowsimg[k:k + 3]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"scanned {len(scored)} real frames")
    for i, s in enumerate(top):
        print(f"C{i:02d} {s['image']} session={s['session']} frame={s['frame_index']} t={s['timestamp_sec']}s dets={s['v3_dets_conf010']} max={s['v3_max_conf']}")
    check_v3()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
