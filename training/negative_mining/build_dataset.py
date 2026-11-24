"""Build a YOLO dataset from labelled negative-mining review folders (backend/services/negative_mining.py).

    python -m training.negative_mining.build_dataset storage/review --out dataset/negmine_v1 \
        [--val-videos ID ...] [--test-videos ID ...] [--background-ratio 0.3]

Rules
  * TP                         -> a box of its class (``true_class`` if the reviewer set one, else the detected class)
  * FP_shadow / FP_carton / FP_stock / FP_puddle
                               -> a box of an explicit negative class fp_shadow / fp_carton / fp_stock / fp_puddle
  * FP_other                   -> no box (the region stays unlabelled = background for the model)
  * a frame is used only when EVERY detection in it is labelled: an unlabelled TP left in a training image would
    teach the model that litter is background. Such frames are skipped and counted.
  * clean frames (no detection) -> background images (empty label file), capped at ``background_ratio`` x the number
    of labelled images.
  * SPLIT BY VIDEO: all frames of one video go to one split (no near-duplicate frames across train/val). Needs at
    least two videos; explicit --val-videos / --test-videos, otherwise a stable hash order.

The negative classes come AFTER the litter classes (litter class ids are unchanged). The detection engine drops
every class whose name starts with ``fp_`` (backend/detection/engine.py), so they never reach alerts or reports.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from backend.services.negative_mining import NEGATIVE_LABELS, load_review, normalise_label, review_dirs

NEGATIVE_CLASSES = list(dict.fromkeys(NEGATIVE_LABELS.values()))


class DatasetError(ValueError):
    pass


def _yolo_line(cls_id: int, r: dict[str, str]) -> str:
    w, h = float(r["frame_w"]), float(r["frame_h"])
    x1, y1 = max(0.0, float(r["x1"])), max(0.0, float(r["y1"]))
    x2, y2 = min(w, float(r["x2"])), min(h, float(r["y2"]))
    return f"{cls_id} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def assign_splits(video_ids: Sequence[str], val_videos: Sequence[str] = (), test_videos: Sequence[str] = (),
                  val_ratio: float = 0.2, test_ratio: float = 0.0) -> dict[str, str]:
    vids = sorted(set(video_ids))
    if len(vids) < 2:
        raise DatasetError(f"split by video needs at least 2 labelled videos, found {len(vids)} ({', '.join(vids) or '-'}). "
                           "Mine and label another video; splitting frames of one video would leak near-duplicate "
                           "frames into validation.")
    unknown = (set(val_videos) | set(test_videos)) - set(vids)
    if unknown:
        raise DatasetError(f"unknown video id(s): {', '.join(sorted(unknown))}")
    if set(val_videos) & set(test_videos):
        raise DatasetError("a video cannot be in both val and test")
    out = {v: "val" for v in val_videos} | {v: "test" for v in test_videos}
    rest = sorted((v for v in vids if v not in out), key=lambda v: zlib.crc32(v.encode()))
    if not val_videos:
        n_val = max(1, round(len(vids) * val_ratio))
        for v in rest[:n_val]:
            out[v] = "val"
        rest = rest[n_val:]
    if not test_videos and test_ratio > 0:
        n_test = min(len(rest) - 1, max(1, round(len(vids) * test_ratio)))
        for v in rest[:max(0, n_test)]:
            out[v] = "test"
        rest = rest[max(0, n_test):]
    for v in rest:
        out[v] = "train"
    if "train" not in out.values():
        raise DatasetError("no video left for train")
    if "val" not in out.values():
        raise DatasetError("no video assigned to val")
    return out


def build_dataset(review_roots: Sequence[Path], out_dir: Path, *, val_videos: Sequence[str] = (),
                  test_videos: Sequence[str] = (), val_ratio: float = 0.2, test_ratio: float = 0.0,
                  background_ratio: float = 0.3, base_classes: Optional[Sequence[str]] = None,
                  overwrite: bool = False) -> dict[str, Any]:
    dirs = review_dirs(review_roots)
    if not dirs:
        raise DatasetError("no review folders (manifest.json + detections.csv) found")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise DatasetError(f"{out_dir} is not empty (use --overwrite)")
        shutil.rmtree(out_dir)
    videos: dict[str, dict[str, Any]] = {}
    classes: Optional[list[str]] = list(base_classes) if base_classes else None
    for d in dirs:
        manifest, dets, frames = load_review(d)
        vid = manifest["video_id"]
        if vid in videos:
            raise DatasetError(f"video {vid} appears twice ({videos[vid]['dir']} and {d})")
        mc = manifest.get("class_names")
        if classes is None and mc:
            classes = list(mc)
        elif mc and classes != list(mc) and not base_classes:
            raise DatasetError(f"class list of {vid} differs from the other videos; pass --classes explicitly")
        videos[vid] = {"dir": d, "manifest": manifest, "dets": dets, "frames": frames}
    if not classes:
        raise DatasetError("class list unknown: manifests have no class_names; pass --classes")
    if set(classes) & set(NEGATIVE_CLASSES):
        raise DatasetError("base classes must not already contain the negative classes")
    names = list(classes) + NEGATIVE_CLASSES
    cid = {n: i for i, n in enumerate(names)}

    splits = assign_splits(list(videos), val_videos, test_videos, val_ratio, test_ratio)
    stats: dict[str, Any] = {s: {"videos": [], "images": 0, "labelled_images": 0, "background_images": 0,
                                 "boxes": Counter()} for s in ("train", "val", "test")}
    skipped = Counter()
    label_counts = Counter()
    for vid in sorted(videos):
        v, split = videos[vid], splits[vid]
        st = stats[split]
        st["videos"].append(vid)
        by_frame: dict[str, list[dict[str, str]]] = {}
        for r in v["dets"]:
            by_frame.setdefault(r["frame_file"], []).append(r)
        img_dir, lab_dir = out_dir / "images" / split, out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        labelled_here = 0
        for frame_file, rows in sorted(by_frame.items()):
            try:
                labs = [normalise_label(r.get("label")) for r in rows]
            except ValueError as exc:
                raise DatasetError(f"{vid} {frame_file}: {exc}") from exc
            if any(lab is None for lab in labs):
                skipped["frame_with_unlabelled_detection"] += 1
                continue
            src = v["dir"] / frame_file
            if not src.is_file():
                skipped["frame_image_missing"] += 1
                continue
            lines = []
            for r, lab in zip(rows, labs):
                label_counts[lab] += 1
                if lab == "TP":
                    cls = (r.get("true_class") or "").strip() or r["class"]
                    if cls not in cid or cls in NEGATIVE_CLASSES:
                        raise DatasetError(f"{r['det_id']}: class {cls!r} is not one of {classes}")
                elif lab in NEGATIVE_LABELS:
                    cls = NEGATIVE_LABELS[lab]
                else:
                    continue                                    # FP_other: no box
                lines.append(_yolo_line(cid[cls], r))
                st["boxes"][cls] += 1
            stem = f"{vid}__{Path(frame_file).stem}"
            shutil.copy2(src, img_dir / f"{stem}.jpg")
            (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            labelled_here += 1
        st["labelled_images"] += labelled_here
        clean = [f for f in v["frames"] if f.get("clean") == "1" and f.get("saved") == "1" and f.get("frame_file")]
        cap = int(round(background_ratio * labelled_here)) if labelled_here else 0
        step = max(1, len(clean) // cap) if cap else 0
        chosen = clean[::step][:cap] if cap else []
        for f in chosen:
            src = v["dir"] / f["frame_file"]
            if not src.is_file():
                skipped["background_image_missing"] += 1
                continue
            stem = f"{vid}__{Path(f['frame_file']).stem}"
            shutil.copy2(src, img_dir / f"{stem}.jpg")
            (lab_dir / f"{stem}.txt").write_text("", encoding="utf-8")
            st["background_images"] += 1
        st["images"] = st["labelled_images"] + st["background_images"]
    # leakage guard: no video in two splits, no image name in two splits
    seen: dict[str, str] = {}
    for s in ("train", "val", "test"):
        for p in (out_dir / "images" / s).glob("*.jpg") if (out_dir / "images" / s).is_dir() else []:
            vid = p.name.split("__", 1)[0]
            if seen.setdefault(vid, s) != s:
                raise DatasetError(f"video {vid} leaked into {seen[vid]} and {s}")
    if stats["train"]["labelled_images"] == 0:
        raise DatasetError("no fully labelled frame in the train videos: label detections.csv first")
    yaml = [f"path: {out_dir.resolve().as_posix()}", "train: images/train", "val: images/val"]
    if stats["test"]["videos"]:
        yaml.append("test: images/test")
    yaml.append(f"nc: {len(names)}")
    yaml.append("names:")
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (out_dir / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    manifest = {"classes": names, "litter_classes": list(classes), "negative_classes": NEGATIVE_CLASSES,
                "split_unit": "video", "splits": {s: {**st, "boxes": dict(st["boxes"])} for s, st in stats.items()},
                "label_counts": dict(label_counts), "skipped": dict(skipped), "background_ratio": background_ratio,
                "sources": {vid: str(v["dir"]) for vid, v in videos.items()}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review", nargs="+", type=Path, help="storage/review or individual <video_id> folders")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-videos", nargs="*", default=[])
    ap.add_argument("--test-videos", nargs="*", default=[])
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.0)
    ap.add_argument("--background-ratio", type=float, default=0.3)
    ap.add_argument("--classes", nargs="*", default=None, help="litter class names (default: from the manifests)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    try:
        m = build_dataset(a.review, a.out, val_videos=a.val_videos, test_videos=a.test_videos, val_ratio=a.val_ratio,
                          test_ratio=a.test_ratio, background_ratio=a.background_ratio, base_classes=a.classes,
                          overwrite=a.overwrite)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({s: {k: v for k, v in st.items()} for s, st in m["splits"].items()}, indent=2))
    print("skipped:", m["skipped"] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
