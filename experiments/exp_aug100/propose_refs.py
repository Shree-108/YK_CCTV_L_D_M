"""Labelling AID for the two reference frames: low-confidence v3 proposals (full 1280 + tiled 640), merged with
class-agnostic NMS, rendered as zoomed review quadrants with a pixel grid. Proposals are never labels:
every box is accepted, corrected or rejected by a reviewer in annotations/reference_<scene>.json.

    python -m experiments.exp_aug100.propose_refs
"""
from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from experiments.exp_aug100.common import ANN, DS, IMG_DIR, SCENES, V3, check_v3, fname, save_json

REVIEW = DS / "review"


def nms(boxes, iou=0.5):
    boxes = sorted(boxes, key=lambda b: -b[4]); keep = []
    for b in boxes:
        ok = True
        for k in keep:
            x1, y1, x2, y2 = max(b[0], k[0]), max(b[1], k[1]), min(b[2], k[2]), min(b[3], k[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            u = (b[2]-b[0])*(b[3]-b[1]) + (k[2]-k[0])*(k[3]-k[1]) - inter
            if inter / (u + 1e-9) > iou or inter / ((b[2]-b[0])*(b[3]-b[1]) + 1e-9) > 0.8:
                ok = False; break
        if ok: keep.append(b)
    return keep


def grid_quadrants(im, boxes, out_prefix, zoom=2):
    h, w = im.shape[:2]
    for qi, (x0, y0) in enumerate(((0, 0), (w // 2, 0), (0, h // 2), (w // 2, h // 2))):
        q = im[y0:y0 + h // 2, x0:x0 + w // 2].copy()
        q = cv2.resize(q, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        for gx in range(0, w // 2, 40):  # grid every 40 source px, labelled in SOURCE coordinates
            cv2.line(q, (gx * zoom, 0), (gx * zoom, q.shape[0]), (90, 90, 90), 1)
            cv2.putText(q, str(x0 + gx), (gx * zoom + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for gy in range(0, h // 2, 40):
            cv2.line(q, (0, gy * zoom), (q.shape[1], gy * zoom), (90, 90, 90), 1)
            cv2.putText(q, str(y0 + gy), (2, gy * zoom + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for i, b in enumerate(boxes):
            p = [int((b[0]-x0)*zoom), int((b[1]-y0)*zoom), int((b[2]-x0)*zoom), int((b[3]-y0)*zoom)]
            if p[2] < 0 or p[3] < 0 or p[0] > q.shape[1] or p[1] > q.shape[0]: continue
            cv2.rectangle(q, p[:2], p[2:], (0, 0, 255), 1)
            cv2.putText(q, f"{i}:{b[4]:.2f}", (p[0], max(p[1]-3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imwrite(str(REVIEW / f"{out_prefix}_q{qi}.jpg"), q, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> int:
    check_v3(); REVIEW.mkdir(parents=True, exist_ok=True)
    m = YOLO(str(V3))
    for scene, cfg in SCENES.items():
        f = fname(cfg["reference"]); im = cv2.imread(str(IMG_DIR / f)); h, w = im.shape[:2]
        boxes = []
        r = m.predict(im, imgsz=1280, conf=0.05, verbose=False, device="cpu", agnostic_nms=True)[0]
        boxes += [b.xyxy[0].tolist() + [float(b.conf), "full1280"] for b in r.boxes]
        t, step = 640, 480
        for y in sorted({*range(0, h - t + 1, step), h - t}):
            for x in sorted({*range(0, w - t + 1, step), w - t}):
                r = m.predict(im[y:y+t, x:x+t], imgsz=640, conf=0.05, verbose=False, device="cpu", agnostic_nms=True)[0]
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    boxes.append([x1 + x, y1 + y, x2 + x, y2 + y, float(b.conf), "tile640"])
        boxes = nms(boxes)
        boxes.sort(key=lambda b: (b[1] // 60, b[0]))
        save_json(ANN / f"proposals_{scene}.json", {"image": f, "source": "v3_store_cctv (aid only)",
                  "boxes": [dict(id=i, xyxy=[round(v, 1) for v in b[:4]], conf=round(b[4], 3), via=b[5]) for i, b in enumerate(boxes)]})
        grid_quadrants(im, boxes, f"ref_{scene}")
        print(scene, f, len(boxes), "proposals")
    check_v3()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
