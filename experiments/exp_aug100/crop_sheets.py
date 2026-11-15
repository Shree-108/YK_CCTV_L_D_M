"""Zoomed crop sheets for review: one tile per box (proposal or label) with context, id and coordinates.

    python -m experiments.exp_aug100.crop_sheets --scene A_shopfront --source proposals|reference
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from experiments.exp_aug100.common import ANN, DS, IMG_DIR, SCENES, fname, load_json

REVIEW = DS / "review"


def tiles_for(im, boxes, tile=200, pad=24, label=lambda b: ""):
    h, w = im.shape[:2]; out = []
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        side = max(x2 - x1, y2 - y1) + 2 * pad
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        X1, Y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        X2, Y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        crop = im[Y1:Y2, X1:X2].copy()
        s = tile / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        t = np.zeros((tile + 34, tile, 3), np.uint8); t[:crop.shape[0], :crop.shape[1]] = crop
        cv2.rectangle(t, (int((x1 - X1) * s), int((y1 - Y1) * s)), (int((x2 - X1) * s), int((y2 - Y1) * s)), (0, 0, 255), 1)
        cv2.putText(t, label(b), (2, tile + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(t, f"{int(x1)},{int(y1)} {int(x2-x1)}x{int(y2-y1)}", (2, tile + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        out.append(t)
    return out


def write_sheets(tiles, prefix, cols=6, rows=4):
    REVIEW.mkdir(parents=True, exist_ok=True); paths = []
    per = cols * rows
    for k in range(0, len(tiles), per):
        chunk = tiles[k:k + per]
        while len(chunk) % cols: chunk.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(chunk[i:i + cols]) for i in range(0, len(chunk), cols)])
        p = REVIEW / f"{prefix}_{k // per}.jpg"; cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]); paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--source", default="proposals")
    a = ap.parse_args(argv)
    im = cv2.imread(str(IMG_DIR / fname(SCENES[a.scene]["reference"])))
    data = load_json(ANN / f"{'proposals' if a.source == 'proposals' else 'reference'}_{a.scene}.json", None)
    boxes = data["boxes"]
    lab = (lambda b: f"#{b['id']} c{b['conf']:.2f}") if a.source == "proposals" else (lambda b: f"#{b['id']} {b.get('zone','')}")
    for p in write_sheets(tiles_for(im, boxes, label=lab), f"crops_{a.source}_{a.scene}"):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
