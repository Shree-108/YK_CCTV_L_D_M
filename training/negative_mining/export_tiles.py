"""Export reviewed mining folders as a TILED YOLO dataset, using the inference crop + tiling code (phase 2 item 6).

    python -m training.negative_mining.export_tiles storage/review --out dataset/negmine_tiles_v1 \
        [--test-videos ID ...] [--val-videos ID ...] [--background-ratio 0.3] [--overwrite]

Same geometry as inference (backend/services/roi_service.roi_detect + backend/detection/engine.YoloEngine.detect):
monitored-region mask from the review folder's manifest -> ``roi_crop`` (outside pixels filled with the region's
median colour, cropped to the region's bounding box + 1 % pad) -> ``tile_grid(tile_size, overlap)`` of the app's
tiling settings. Every tile is written at native resolution, exactly the pixels the detector sees.

Labels per tile
  * TP (class = ``true_class`` or the detected class) and drawn "missed" boxes -> litter class ids 0..9, UNCHANGED
    (the order of the model's own class list, stored in the manifests);
  * FP_shadow / FP_carton / FP_stock / FP_puddle -> negative classes appended AFTER them (10 fp_shadow, 11 fp_carton,
    12 fp_stock, 13 fp_puddle); FP_other -> no box (background), as in build_dataset.py;
  * a box goes into a tile when >= 50 % of its area is inside (clipped to the tile);
  * a tile that touches ANY unlabelled or capped detection is skipped (an unlabelled real litter in a training tile
    would teach "litter = background"); tiles without boxes are background, capped at ``background_ratio``.

SPLIT BY VIDEO (never by frame): ``--test-videos``, else the videos recorded on 2026-11-10 (file name starts with
20261110): the first one is held out for test. ``--val-videos``, else build_dataset.assign_splits picks one by hash.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from backend.detection.tiling import tile_grid
from backend.services import roi_service as rs
from backend.services.negative_mining import NEGATIVE_LABELS, normalise_label, read_csv, review_dirs
from training.negative_mining.build_dataset import NEGATIVE_CLASSES, DatasetError, assign_splits

HOLDOUT_PREFIX = "20261110"


def tiles_for(frame: np.ndarray, polygons: Optional[list[dict[str, Any]]], tile: int, overlap: float
              ) -> tuple[np.ndarray, int, int, list[tuple[int, int, int, int]]]:
    """(crop, x0, y0, tiles in crop coordinates) - the inference geometry."""
    if polygons:
        mask = rs.polygons_mask(polygons, frame.shape)
        crop, x0, y0 = rs.roi_crop(frame, mask)
    else:
        crop, x0, y0 = frame, 0, 0
    h, w = crop.shape[:2]
    return crop, x0, y0, tile_grid(w, h, tile, overlap)


def _inside(box: Sequence[float], t: tuple[int, int, int, int]) -> tuple[float, list[float]]:
    x1, y1, x2, y2 = box
    a = max(1e-6, (x2 - x1) * (y2 - y1))
    cx1, cy1, cx2, cy2 = max(x1, t[0]), max(y1, t[1]), min(x2, t[2]), min(y2, t[3])
    if cx2 <= cx1 or cy2 <= cy1:
        return 0.0, []
    return (cx2 - cx1) * (cy2 - cy1) / a, [cx1, cy1, cx2, cy2]


def holdout_videos(manifests: dict[str, dict[str, Any]]) -> list[str]:
    cands = sorted(v for v, m in manifests.items() if str(m.get("video_file", "")).startswith(HOLDOUT_PREFIX))
    return cands[:1]


def export(review_roots: Sequence[Path], out_dir: Path, *, val_videos: Sequence[str] = (),
           test_videos: Optional[Sequence[str]] = None, background_ratio: float = 0.3, min_inside: float = 0.5,
           overwrite: bool = False) -> dict[str, Any]:
    dirs = review_dirs(review_roots)
    if not dirs:
        raise DatasetError("no review folders found")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise DatasetError(f"{out_dir} is not empty (use --overwrite)")
        shutil.rmtree(out_dir)
    vids: dict[str, dict[str, Any]] = {}
    classes: Optional[list[str]] = None
    for d in dirs:
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if classes is None:
            classes = list(m.get("class_names") or [])
        elif m.get("class_names") and list(m["class_names"]) != classes:
            raise DatasetError(f"class list of {m['video_id']} differs from the other videos")
        vids[m["video_id"]] = {"dir": d, "manifest": m, "dets": read_csv(d / "detections.csv"),
                               "missed": read_csv(d / "missed.csv") if (d / "missed.csv").is_file() else []}
    if not classes:
        raise DatasetError("class list unknown (manifests have no class_names)")
    names = classes + NEGATIVE_CLASSES
    cid = {n: i for i, n in enumerate(names)}
    tests = list(test_videos) if test_videos is not None else holdout_videos({v: x["manifest"] for v, x in vids.items()})
    splits = assign_splits(list(vids), val_videos, tests)
    stats = {s: {"videos": [], "tiles": 0, "labelled_tiles": 0, "background_tiles": 0, "boxes": Counter()}
             for s in ("train", "val", "test")}
    skipped: Counter = Counter()
    for vid in sorted(vids):
        v, split = vids[vid], splits[vid]
        m, st = v["manifest"], stats[split]
        st["videos"].append(vid)
        tiling = m.get("tiling") or {}
        tile, overlap = int(tiling.get("tile_size", 1024)), float(tiling.get("overlap", 0.2))
        polys = (m.get("roi") or {}).get("polygons") if (m.get("roi") or {}).get("applied") else None
        by_frame: dict[str, dict[str, list]] = {}
        for r in v["dets"]:
            by_frame.setdefault(r["frame_file"], {"dets": [], "missed": []})["dets"].append(r)
        for r in v["missed"]:
            by_frame.setdefault(r["frame_file"], {"dets": [], "missed": []})["missed"].append(r)
        img_dir, lab_dir = out_dir / "images" / split, out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        background: list[tuple[str, np.ndarray]] = []
        labelled_here = 0
        for frame_file, group in sorted(by_frame.items()):
            src = v["dir"] / frame_file
            frame = cv2.imread(str(src)) if src.is_file() else None
            if frame is None:
                skipped["frame_image_missing"] += 1
                continue
            crop, x0, y0, tiles = tiles_for(frame, polys, tile, overlap)
            boxes = []                                   # (class name | None = FP_other | "?" = unlabelled, box)
            for r in group["dets"]:
                lab = normalise_label(r.get("label")) if r.get("capped", "0") != "1" else None
                b = [float(r["x1"]) - x0, float(r["y1"]) - y0, float(r["x2"]) - x0, float(r["y2"]) - y0]
                if lab is None:
                    boxes.append(("?", b))
                elif lab == "TP":
                    boxes.append(((r.get("true_class") or "").strip() or r["class"], b))
                elif lab in NEGATIVE_LABELS:
                    boxes.append((NEGATIVE_LABELS[lab], b))
                else:
                    boxes.append((None, b))
            for r in group["missed"]:
                boxes.append((r["class"], [float(r["x1"]) - x0, float(r["y1"]) - y0, float(r["x2"]) - x0,
                                           float(r["y2"]) - y0]))
            for ti, t in enumerate(tiles):
                lines, unsafe = [], False
                for cls, b in boxes:
                    frac, clipped = _inside(b, t)
                    if frac <= 0:
                        continue
                    if cls == "?":
                        unsafe = True
                        break
                    if cls is None or frac < min_inside:
                        continue
                    if cls not in cid:
                        raise DatasetError(f"{vid} {frame_file}: class {cls!r} not in {names}")
                    tw, th = t[2] - t[0], t[3] - t[1]
                    bx1, by1, bx2, by2 = clipped[0] - t[0], clipped[1] - t[1], clipped[2] - t[0], clipped[3] - t[1]
                    lines.append(f"{cid[cls]} {(bx1 + bx2) / 2 / tw:.6f} {(by1 + by2) / 2 / th:.6f} "
                                 f"{(bx2 - bx1) / tw:.6f} {(by2 - by1) / th:.6f}")
                    st["boxes"][cls] += 1
                if unsafe:
                    skipped["tile_with_unlabelled_detection"] += 1
                    continue
                stem = f"{vid}__{Path(frame_file).stem}_t{ti:02d}"
                img = crop[t[1]:t[3], t[0]:t[2]]
                if lines:
                    cv2.imwrite(str(img_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                    labelled_here += 1
                elif len(background) < 5000:
                    background.append((stem, img))
        cap = int(round(background_ratio * labelled_here))
        step = max(1, len(background) // cap) if cap else 0
        for stem, img in (background[::step][:cap] if cap else []):
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (lab_dir / f"{stem}.txt").write_text("", encoding="utf-8")
            st["background_tiles"] += 1
        st["labelled_tiles"] += labelled_here
        st["tiles"] = st["labelled_tiles"] + st["background_tiles"]
    seen: dict[str, str] = {}
    for s in ("train", "val", "test"):
        d = out_dir / "images" / s
        for p in d.glob("*.jpg") if d.is_dir() else []:
            vid = p.name.split("__", 1)[0]
            if seen.setdefault(vid, s) != s:
                raise DatasetError(f"video {vid} leaked into {seen[vid]} and {s}")
    yaml = [f"path: {out_dir.resolve().as_posix()}", "train: images/train", "val: images/val"]
    if stats["test"]["videos"]:
        yaml.append("test: images/test")
    yaml += [f"nc: {len(names)}", "names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    (out_dir / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    manifest = {"classes": names, "litter_classes": classes, "negative_classes": NEGATIVE_CLASSES,
                "split_unit": "video", "holdout_test": tests, "geometry": "roi_crop + tile_grid (inference)",
                "splits": {s: {**x, "boxes": dict(x["boxes"])} for s, x in stats.items()}, "skipped": dict(skipped),
                "background_ratio": background_ratio, "min_inside": min_inside,
                "sources": {vid: str(x["dir"]) for vid, x in vids.items()}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-videos", nargs="*", default=[])
    ap.add_argument("--test-videos", nargs="*", default=None, help="default: the first 2026-11-10 video")
    ap.add_argument("--background-ratio", type=float, default=0.3)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    try:
        m = export(a.review, a.out, val_videos=a.val_videos, test_videos=a.test_videos,
                   background_ratio=a.background_ratio, overwrite=a.overwrite)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({s: {k: v for k, v in x.items()} for s, x in m["splits"].items()}, indent=2))
    print("held out for test:", m["holdout_test"], "skipped:", m["skipped"] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
