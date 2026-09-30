"""Pre-label a folder of real CCTV images with the open-vocabulary teacher model.

Creates a YOLO dataset (images + labels + data.yaml) that you then REVIEW AND CORRECT in the
UI (Ground Truth page) before training. Auto-labels are a starting point, never ground truth.

    python -m training.autolabel --src ../CCTV_Litter_100_Images --name cctv_real_v1 \
        --model models/litter_yolo/v2_yoloe_openvocab/best.pt --val 0.15 --test 0.15

Split: frames are grouped by near-duplicate similarity (dHash) and whole groups go to one split,
so almost-identical frames never leak between train and test.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from backend.core.config import PROJECT_ROOT
from backend.core.settings import merged_settings
from backend.detection.engine import YoloEngine

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash(img: np.ndarray) -> np.ndarray:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (17, 16), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def group_frames(paths: list[Path], max_bits: int = 40) -> list[list[Path]]:
    """Greedy grouping of visually similar frames (same scene/moment)."""
    hashes = [(p, dhash(cv2.imread(str(p)))) for p in paths]
    groups: list[tuple[np.ndarray, list[Path]]] = []
    for p, h in hashes:
        for gh, members in groups:
            if int(np.count_nonzero(gh != h)) <= max_bits:
                members.append(p)
                break
        else:
            groups.append((h, [p]))
    return [m for _, m in groups]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True, help="new dataset folder name under dataset/")
    ap.add_argument("--model", default="models/litter_yolo/v2_yoloe_openvocab/best.pt")
    ap.add_argument("--conf", type=float, default=0.12, help="teacher confidence for proposed labels")
    ap.add_argument("--tiling", action="store_true", help="tiled teacher inference (slower, finds smaller objects)")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    src = Path(a.src)
    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    if not paths:
        print(f"error: no images in {src}", file=sys.stderr)
        return 2
    out = PROJECT_ROOT / "dataset" / a.name
    if out.exists():
        print(f"error: {out} exists (datasets are never overwritten)", file=sys.stderr)
        return 2
    model = Path(a.model)
    meta_f = model.with_name("model_meta.json")
    rec = json.loads(meta_f.read_text()).get("recommended_settings", {}) if meta_f.is_file() else {}
    s = merged_settings({**rec, "confidence": {"weak_threshold": a.conf, "candidate_threshold": a.conf,
                                               "confirmed_threshold": a.conf}})
    eng = YoloEngine(model, s, model.parent.name)

    groups = group_frames(paths)
    random.Random(a.seed).shuffle(groups)
    n = len(paths)
    split_of: dict[Path, str] = {}
    counts = {"test": 0, "val": 0}
    for g in groups:
        sp = ("test" if counts["test"] < a.test * n else "val" if counts["val"] < a.val * n else "train")
        if sp in counts:
            counts[sp] += len(g)
        for p in g:
            split_of[p] = sp

    total = 0
    for i, p in enumerate(paths, 1):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        dets, _ = eng.detect(img, s, force_tiling=a.tiling)
        sp = split_of[p]
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out / "images" / sp / p.name)
        lines = [f"{d.cls_id} {((d.bbox[0] + d.bbox[2]) / 2) / w:.6f} {((d.bbox[1] + d.bbox[3]) / 2) / h:.6f} "
                 f"{(d.bbox[2] - d.bbox[0]) / w:.6f} {(d.bbox[3] - d.bbox[1]) / h:.6f}" for d in dets]
        (out / "labels" / sp / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        total += len(lines)
        print(f"[{i}/{n}] {p.name} -> {sp}: {len(lines)} proposed boxes", flush=True)

    (out / "data.yaml").write_text(yaml.safe_dump({"path": str(out), "train": "images/train", "val": "images/val",
                                                   "test": "images/test", "names": dict(enumerate(eng.class_names))},
                                                  sort_keys=False))
    (out / "README.txt").write_text(
        f"AUTO-LABELLED by {model.parent.name} (conf>={a.conf}, tiling={a.tiling}) from {src.name}.\n"
        "These labels are UNREVIEWED proposals. Review and correct every image in the Ground Truth page\n"
        "before training or evaluating. Missing small litter must be added by hand.\n")
    print(f"done: {n} images, {total} proposed boxes, groups={len(groups)}, splits={dict(counts, train=n - sum(counts.values()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
