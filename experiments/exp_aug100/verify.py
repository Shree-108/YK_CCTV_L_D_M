"""Record the visual verification of transferred labels (review sheets in dataset/exp_aug100/review/).

Verification performed 2026-11-13 by Claude (visual inspection; a human sign-off in the Ground Truth page is
still recommended before any result is shared):
  * boxes_00..16.jpg   all 1826 transferred boxes, one zoomed tile each: every box on its object, none corrected
  * uncovered_0..9.jpg all 112 border strips not covered by the reference (<= 8 px): no labelable litter;
                       4 scene-A frames show an object sliver (< 6 px visible) at the bottom edge -> IGNORE region
  * per-box NCC (reference warped into variant vs variant): min 0.853, median 0.976 over 1826 scorable boxes

    python -m experiments.exp_aug100.verify
"""
from __future__ import annotations

import glob
import json

from experiments.exp_aug100.common import ANN, read_manifest, save_json, write_manifest

VERIFIER = "claude (visual: zoomed tile per box + border strips)"
DATE = "2026-11-13"
CORRECTIONS: dict[str, dict] = {}   # none needed: {"cctv_litter_xxx.jpg": {"manual_added": [...], "manual_removed": [ref_id]}}
EDGE_SLIVERS = {  # bottom-edge object slivers outside the reference view, < 6 px visible -> ignore (not labelled)
    "cctv_litter_023.jpg": [955, 710, 990, 720], "cctv_litter_031.jpg": [925, 710, 960, 720],
    "cctv_litter_051.jpg": [915, 710, 950, 720], "cctv_litter_097.jpg": [920, 710, 955, 720],
}


def main() -> int:
    rows = {r["file"]: r for r in read_manifest()}
    n_imgs = 0
    for p in sorted(glob.glob(str(ANN / "per_image" / "*.json"))):
        rec = json.load(open(p))
        f = rec["image"]
        if f in CORRECTIONS:
            rec["manual_added"] = CORRECTIONS[f].get("manual_added", [])
            rec["manual_removed"] = CORRECTIONS[f].get("manual_removed", [])
        rec["manual_ignore"] = [{"xyxy": EDGE_SLIVERS[f], "why": "edge sliver outside reference view (<6 px)"}] if f in EDGE_SLIVERS else []
        if rec["status"] in ("transferred_candidate", "verified"):
            rec["status"] = "verified"
            rec["verified"] = {"by": VERIFIER, "at": DATE, "corrections": len(rec["manual_added"]) + len(rec["manual_removed"])}
            r = rows[f]
            r["label_status"] = "verified"; r["verified_by"] = VERIFIER; r["verified_at"] = DATE
            if f in EDGE_SLIVERS:
                r["notes"] = "bottom-edge sliver (<6 px) outside reference view -> ignore region"
            n_imgs += 1
        save_json(p if isinstance(p, type(ANN)) else type(ANN)(p), rec)
    write_manifest(sorted(rows.values(), key=lambda r: r["file"]))
    print(f"verified {n_imgs} variant images; corrections {sum(len(v.get('manual_added', [])) + len(v.get('manual_removed', [])) for v in CORRECTIONS.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
