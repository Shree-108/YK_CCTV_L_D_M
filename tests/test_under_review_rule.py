"""Owner rule 2026-11-21: "under review" counts ``candidate`` detections only - in the import CSV remark, in the
annotated image header and on the dashboard Results view. ``weak`` detections are counted in manifest.json only."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from backend.detection.types import CANDIDATE, CONFIRMED, WEAK, Detection
from backend.services import garbage_export as ge
from backend.services import results_view as rv
from backend.services import visualization as viz
from tests import test_result_bundle as trb

# one screenshot: 2 confirmed, 3 candidate, 5 weak
DETS = ([{"class": "food_wrapper", "confidence": 0.80 - i / 100, "bbox": [10 + 40 * i, 10, 30 + 40 * i, 30],
          "status": "confirmed"} for i in range(2)]
        + [{"class": "paper", "confidence": 0.50 - i / 100, "bbox": [10 + 40 * i, 60, 30 + 40 * i, 80],
            "status": "candidate"} for i in range(3)]
        + [{"class": "paper", "confidence": 0.30 - i / 100, "bbox": [10 + 40 * i, 110, 30 + 40 * i, 130],
            "status": "weak"} for i in range(5)])


def remark_count(remark: str) -> int:
    m = re.search(r"(\d+) possible objects? under review", remark)
    return int(m.group(1)) if m else 0


def test_csv_image_header_and_dashboard_agree_and_weak_is_manifest_only(tmp_path, monkeypatch):
    monkeypatch.setattr(trb, "LITTER", DETS)
    cfg = trb.write_cfg(tmp_path)
    job = trb.Job(tmp_path, ["LITTER_DETECTED"] + ["NO_LITTER"] * 3)

    # 1) the annotated image header (the counts annotate_litter writes come from hud_counts)
    dets = [Detection(d["bbox"], 0, d["class"], d["confidence"], d["status"]) for d in DETS]
    header = viz.hud_counts(dets)
    assert header == {"litter": 2, "under_review": 3}
    img = viz.annotate_litter(np.zeros((240, 320, 3), np.uint8), dets)          # renders with the same rule
    assert img.shape == (240, 320, 3)

    # 2) the import CSV remark (bundle CSV)
    m = job.build(cfg)
    rows = trb.rows_of(Path(m["folder"]))
    assert remark_count(rows[0]["remark"]) == 3

    # 3) the dashboard Results view (row field and remark)
    view = rv.view(job.query, ge.Settings(cfg, tmp_path / "x", tmp_path / "l.db"), rv.Filters())
    r1 = next(r for r in view["rows"] if r["idx"] == 1)
    assert r1["under_review"] == 3 and remark_count(r1["remark"]) == 3 and r1["remark"] == rows[0]["remark"]

    assert header["under_review"] == remark_count(rows[0]["remark"]) == r1["under_review"]

    # weak: recorded in manifest.json only - not in the CSV remark, not in the header, not on the dashboard
    man = json.loads((Path(m["folder"]) / "manifest.json").read_text(encoding="utf-8"))
    assert man["screenshots"][0]["weak_not_counted"] == 5 and man["counts"]["weak_not_counted"] == 5
    assert man["counts"]["under_review_not_counted"] == 3
    assert "weak" not in rows[0]["remark"] and "0.2" not in rows[0]["remark"] and "weak" not in r1
    assert "29.00" not in rows[0]["remark"]                                       # a weak score is never listed


def test_header_never_counts_weak():
    weak_only = [Detection([0, 0, 5, 5], 0, "paper", 0.3, WEAK)] * 4
    assert viz.hud_counts(weak_only) == {"litter": 0, "under_review": 0}
    mixed = [Detection([0, 0, 5, 5], 0, "paper", 0.9, CONFIRMED), Detection([0, 0, 5, 5], 0, "paper", 0.5, CANDIDATE)]
    assert viz.hud_counts(mixed) == {"litter": 1, "under_review": 1}
