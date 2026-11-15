"""Visual verification sheets.

  --what uncovered   border strips of each variant that the reference does not show (+40 px context)
  --what boxes       every transferred label as a zoomed tile (ordered by image), red = label, yellow = ignore

    python -m experiments.exp_aug100.review_sheets --what uncovered|boxes
"""
from __future__ import annotations

import argparse
import glob
import json

import cv2
import numpy as np

from experiments.exp_aug100.common import ANN, DS, IMG_DIR

REVIEW = DS / "review"


def recs():
    return [json.load(open(p)) for p in sorted(glob.glob(str(ANN / "per_image" / "*.json")))]


def uncovered():
    strips = []
    for r in recs():
        if r["uncovered_frac"] <= 0.0005: continue
        im = cv2.imread(str(IMG_DIR / r["image"])); h, w = im.shape[:2]
        H = np.array(r["H_ref_to_image"]); cov = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (w, h)) > 250
        unc = ~cov
        for name, sl, horiz in (("top", np.s_[0:48, :], True), ("bottom", np.s_[h - 48:h, :], True),
                                ("left", np.s_[:, 0:48], False), ("right", np.s_[:, w - 48:w], False)):
            if not unc[sl].any(): continue
            crop = im[sl].copy(); m = unc[sl]
            crop[m] = (0.5 * crop[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)  # magenta = not in reference
            if not horiz: crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            crop = cv2.resize(crop, (1280, int(crop.shape[0] * 1280 / crop.shape[1])))
            lab = np.zeros((16, 1280, 3), np.uint8)
            cv2.putText(lab, f"{r['image']} {name} (rotated)" if not horiz else f"{r['image']} {name}", (4, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            strips.append(np.vstack([lab, crop]))
    for k in range(0, len(strips), 12):
        cv2.imwrite(str(REVIEW / f"uncovered_{k // 12}.jpg"), np.vstack(strips[k:k + 12]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(len(strips), "strips")


def boxes(tile=96, cols=12, rows=9):
    tiles = []
    for r in recs():
        if r["status"] == "reference_reviewed": continue
        im = cv2.imread(str(IMG_DIR / r["image"])); h, w = im.shape[:2]
        n = r["image"][12:15]
        for b in r["labels"]:
            x1, y1, x2, y2 = b["xyxy"]; side = max(x2 - x1, y2 - y1) + 20
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            X1, Y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2)); X2, Y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
            c = im[Y1:Y2, X1:X2].copy(); s = tile / max(c.shape[:2])
            c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            t = np.zeros((tile + 12, tile, 3), np.uint8); t[:c.shape[0], :c.shape[1]] = c
            cv2.rectangle(t, (int((x1 - X1) * s), int((y1 - Y1) * s)), (int((x2 - X1) * s), int((y2 - Y1) * s)), (0, 0, 255), 1)
            cv2.putText(t, f"{n}#{b['ref_id']}", (1, tile + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
            tiles.append(t)
    per = cols * rows
    for k in range(0, len(tiles), per):
        ch = tiles[k:k + per]
        while len(ch) % cols: ch.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(REVIEW / f"boxes_{k // per:02d}.jpg"), np.vstack([np.hstack(ch[i:i + cols]) for i in range(0, len(ch), cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(len(tiles), "tiles in", (len(tiles) + per - 1) // per, "sheets")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--what", choices=["uncovered", "boxes"], required=True)
    {"uncovered": uncovered, "boxes": boxes}[ap.parse_args().what]()
