"""Reference-frame annotation decisions (visual review of every v3 proposal on 2026-11-13, by Claude; zoomed crop
sheets in dataset/exp_aug100/review/). Writes dataset/exp_aug100/annotations/reference_<scene>.json.

Decision vocabulary
  accept     identifiable litter on the ground -> box kept (proposal box, tight on inspection)
  ambiguous  >= 6 px but NOT identifiable as litter (speck, leaf-like, debris smear) -> never labelled;
             an IGNORE region at evaluation (a detection there is neither TP nor FP)
  reject     not litter (overlay text, foliage, vehicles, signage, bin contents, duplicates, texture)

Zones come from boundary polylines drawn on the reference frame (pavement | kerb | gutter | road). Whether
kerb/gutter boxes are labels depends on the ROI option chosen (roi_options.json); out-of-ROI litter becomes an
IGNORE region, so a change of ROI never requires re-annotation.

    python -m experiments.exp_aug100.reference_review
"""
from __future__ import annotations

import cv2
import numpy as np

from experiments.exp_aug100.common import ANN, DS, IMG_DIR, SCENES, fname, load_json, save_json

# boundary polylines (x, y) in reference-frame pixels, far -> near
BOUNDARIES = {
    # A: pavement | light kerb stones | dark drain channel | outer concrete strip + carriageway (= "road")
    "A_shopfront": {"pav|kerb": [(760, 90), (778, 380), (800, 720)],
                    "kerb|gutter": [(790, 90), (805, 260), (830, 720)],
                    "gutter|road": [(800, 90), (806, 135), (837, 260), (882, 720)]},
    "B_roadside_bin": {"pav|kerb": [(600, 100), (625, 300), (655, 380), (805, 720)],
                       "kerb|gutter": [(615, 100), (645, 300), (690, 380), (830, 720)],
                       "gutter|road": [(630, 100), (665, 300), (745, 380), (905, 720)]},
}

DECISIONS = {
    "A_shopfront": {
        "accept": {5: "plastic bag at kerb (far)", 8: "white paper", 9: "small paper piece", 10: "red wrapper",
                   11: "wrapper piece", 12: "plastic bag", 13: "wrapper", 14: "paper piece", 19: "chips packet",
                   20: "paper", 22: "red/white wrapper", 24: "wrapper", 27: "purple wrapper", 28: "red wrapper",
                   29: "white tube / paper roll", 32: "crumpled wrapper", 34: "tissue / paper", 36: "orange carton piece",
                   39: "yellow wrapper"},
        "ambiguous": {6: "white speck", 7: "ring / lid?", 17: "thin sliver", 23: "grey speck", 30: "white blob on road edge",
                      33: "leaf or paper", 37: "thin sliver", 41: "faint piece", 42: "faint piece", 44: "cut by frame edge"},
        "reject": {0: "CAM overlay text", 1: "foliage", 2: "foliage", 3: "car reflection", 4: "car reflection",
                   15: "road smudge", 16: "faint smudge", 18: "road smudge", 21: "kerb texture", 25: "speck", 26: "speck",
                   31: "leaf", 35: "merged box (34+36)", 38: "speck", 40: "dark smudge", 43: "faint smudge"},
    },
    "B_roadside_bin": {
        "accept": {25: "white piece at stand base", 27: "white pieces at tree base", 28: "round lid / plate",
                   34: "brown paper bag beside bin", 36: "white plastic", 38: "newspaper", 41: "white paper",
                   43: "white piece at bin base", 44: "red chips packet", 45: "white/blue plastic", 47: "red packet piece",
                   52: "food container", 55: "white piece", 59: "white lid", 60: "white plate / container",
                   65: "plastic bottle", 75: "white paper piece", 76: "white piece", 77: "white/blue plastic"},
        "ambiguous": {30: "speck", 31: "speck", 32: "white speck on road", 33: "speck", 35: "speck at bin edge",
                      37: "white item on road", 40: "speck", 42: "speck", 46: "faint piece", 48: "white item on drain stone",
                      50: "road speck", 53: "debris smear with mixed pieces", 54: "bluish fragment", 56: "road speck",
                      57: "road speck", 62: "small brown item", 64: "leaf-like", 66: "gutter fragment",
                      68: "gutter fragment", 69: "white piece on road", 70: "yellow leaf or wrapper", 72: "speck",
                      73: "paper inside planter", 78: "gutter fragment", 79: "gutter fragment"},
        "reject": {**{i: "foliage / vehicles / signage / far background" for i in range(0, 24)},
                   24: "road marking", 26: "scooter shadow", 29: "bin contents (inside bin)", 39: "duplicate of 38",
                   49: "duplicate of 48", 51: "speck", 58: "plant", 61: "faint", 63: "faint", 67: "duplicate of 66",
                   71: "faint", 74: "dark spot", 80: "road speck"},
    },
}


# manual zone decisions where the lines converge in the far field (visual check on the zoomed reference)
ZONE_OVERRIDE = {"A_shopfront": {5: "gutter", 36: "gutter"}}


def x_at(line, y):
    ys = [p[1] for p in line]; xs = [p[0] for p in line]
    return float(np.interp(y, ys, xs))


def zone_of(scene, box):
    cx, by = (box[0] + box[2]) / 2, box[3]  # ground contact point = bottom centre
    b = BOUNDARIES[scene]
    if cx < x_at(b["pav|kerb"], by): return "pavement"
    if cx < x_at(b["kerb|gutter"], by): return "kerb"
    if cx < x_at(b["gutter|road"], by): return "gutter"
    return "road"


def main() -> int:
    for scene, dec in DECISIONS.items():
        prop = load_json(ANN / f"proposals_{scene}.json", None)
        ids = {b["id"] for b in prop["boxes"]}
        decided = set(dec["accept"]) | set(dec["ambiguous"]) | set(dec["reject"])
        if decided != ids or len(decided) != sum(len(v) for v in dec.values()):
            raise SystemExit(f"{scene}: every proposal needs exactly one decision; missing {sorted(ids - decided)}, "
                             f"extra {sorted(decided - ids)}")
        by_id = {b["id"]: b for b in prop["boxes"]}
        out = {"image": prop["image"], "reviewed_by": "claude (visual, zoomed crops)", "reviewed_at": "2026-11-13",
               "boundaries": BOUNDARIES[scene], "boxes": [], "ambiguous": [], "rejected": []}
        for kind, key in (("accept", "boxes"), ("ambiguous", "ambiguous"), ("reject", "rejected")):
            for i, note in sorted(dec[kind].items()):
                b = by_id[i]
                zone = ZONE_OVERRIDE.get(scene, {}).get(i) or zone_of(scene, b["xyxy"])
                out[key].append({"id": i, "xyxy": b["xyxy"], "zone": zone, "note": note,
                                 "origin": "v3_proposal_accepted" if kind == "accept" else "v3_proposal"})
        save_json(ANN / f"reference_{scene}.json", out)
        zc = {}
        for b in out["boxes"]: zc[b["zone"]] = zc.get(b["zone"], 0) + 1
        print(scene, "accept", len(out["boxes"]), zc, "ambiguous", len(out["ambiguous"]), "reject", len(out["rejected"]))
        render(scene, out)
    return 0


def render(scene, ann):
    im = cv2.imread(str(IMG_DIR / ann["image"]))
    for name, col in (("pav|kerb", (255, 200, 0)), ("kerb|gutter", (255, 0, 255)), ("gutter|road", (0, 128, 255))):
        cv2.polylines(im, [np.int32(ann["boundaries"][name])], False, col, 1)
    for b in ann["ambiguous"]:
        p = [int(v) for v in b["xyxy"]]; cv2.rectangle(im, p[:2], p[2:], (0, 255, 255), 1)
    for b in ann["boxes"]:
        p = [int(v) for v in b["xyxy"]]; cv2.rectangle(im, p[:2], p[2:], (0, 255, 0), 1)
        cv2.putText(im, f"{b['id']}{b['zone'][0]}", (p[0], p[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.imwrite(str(DS / "review" / f"reference_{scene}.png"), im)


if __name__ == "__main__":
    raise SystemExit(main())
