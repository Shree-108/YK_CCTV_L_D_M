"""Build a versioned YOLO dataset (train/val/test) from the REVIEWED images of the annotation workspace.

    python -m training.build_ldm_dataset --name cctv_ldm_v4
    python -m training.build_ldm_dataset --name cctv_ldm_v4 --include-dataset dataset/cctv_store_v2   # once restored

What goes in: only images whose review_status is 'annotated' or 'reviewed_empty', that are not overlay_suspect, whose
label file agrees with the status and whose manifest values are valid. Everything else is listed in the build report
with the reason. Nothing is ever labelled here: reviewed_empty images get the empty label file that a human approved.

Leakage control (an honest test set matters more than a big one):
  * all images of one recording session stay in one split
  * near-identical images (dHash <= --near-dup-bits) are tied together even across sessions
  * synthetic images and footage the current model was trained on (used_in_previous_training yes/likely) are
    locked to train, so val/test measure real, unseen CCTV
  * a split_lock in manifest.csv pins a session's images to a split (e.g. a whole new camera kept for test)
  * --include-dataset keeps that dataset's own split assignment (so its test images stay test)
Groups are assigned greedily (largest first) towards the --ratios while balancing images with and without litter.

The output folder is new and versioned: an existing dataset is never overwritten. data.yaml has no absolute 'path'
(Ultralytics then uses the yaml's own folder), so the dataset can be copied to a GPU machine / Colab as-is.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from backend.core.config import PROJECT_ROOT
from training.ldm_common import (BASE_MODEL_VERSION, DEFAULT_WORKSPACE, IMAGE_EXTS, DatasetPrepError, dhash,
                                 label_path, model_classes, now_iso, read_image, read_label_lines, read_manifest,
                                 validate_row_vocab, yaml_names)

SPLITS = ("train", "val", "test")


def check_label(lines: list[list[str]], n_classes: int) -> list[str]:
    probs = []
    for i, p in enumerate(lines, 1):
        if len(p) != 5:
            probs.append(f"line {i}: expected 5 values")
            continue
        try:
            c = int(p[0])
            cx, cy, w, h = map(float, p[1:])
        except ValueError:
            probs.append(f"line {i}: not numeric")
            continue
        if not 0 <= c < n_classes:
            probs.append(f"line {i}: class id {c} outside 0..{n_classes - 1}")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            probs.append(f"line {i}: coordinates not normalized / zero size")
    return probs


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _bits(hexes: list[str]) -> np.ndarray:
    return np.unpackbits(np.array([np.frombuffer(bytes.fromhex(h), np.uint8) for h in hexes]), axis=1)


def collect_workspace(ws: Path, classes: list[str], allow_synthetic_eval: bool) -> tuple[list[dict], list[dict]]:
    items, excluded = [], []
    for r in read_manifest(ws):
        img = ws / "images" / "train" / r["image"]
        lp = label_path(ws, r["image"])
        st = r["review_status"] or "unreviewed"

        def skip(reason: str) -> None:
            excluded.append({"image": r["image"], "reason": reason})

        if st not in ("annotated", "reviewed_empty"):
            skip(f"review_status {st}")
            continue
        vocab = validate_row_vocab(r)
        if vocab:
            skip("manifest values: " + "; ".join(vocab))
            continue
        if r["overlay_suspect"] == "yes":
            skip("overlay_suspect=yes (burned-in boxes); set to 'no' after checking if wrongly flagged")
            continue
        if not img.is_file():
            skip("image file missing")
            continue
        if not lp.is_file():
            skip("no label file (annotate it, or run ldm_review mark-empty for a reviewed empty image)")
            continue
        lines = read_label_lines(lp)
        probs = check_label(lines, len(classes))
        if probs:
            skip("invalid label: " + "; ".join(probs[:3]))
            continue
        if st == "annotated" and not lines:
            skip("status annotated but label is empty")
            continue
        if st == "reviewed_empty" and lines:
            skip("status reviewed_empty but label has boxes")
            continue
        lock = r["split_lock"]
        synthetic = r["source_type"] == "synthetic"
        if synthetic and lock in ("val", "test") and not allow_synthetic_eval:
            skip(f"synthetic image locked to {lock}; val/test must be real CCTV (or pass --allow-synthetic-eval)")
            continue
        if not lock and synthetic and not allow_synthetic_eval:
            lock = "train"
        if not lock and r["used_in_previous_training"] in ("yes", "likely"):
            lock = "train"
        items.append({"name": r["image"], "img": img, "label": lp, "lines": lines, "session": r["session"] or r["image"],
                      "dhash": r["dhash"] or dhash(read_image(img)), "lock": lock, "source_type": r["source_type"],
                      "category": r["category"], "subcategory": r["subcategory"], "conditions": r["conditions"],
                      "prev": r["used_in_previous_training"], "source_path": r["source_path"], "origin": "workspace"})
    return items, excluded


def collect_included(ds: Path, classes: list[str]) -> list[dict]:
    yml = ds / "data.yaml"
    if not yml.is_file():
        raise DatasetPrepError(f"{ds}: no data.yaml")
    if yaml_names(yml) != classes:
        raise DatasetPrepError(f"{ds}: class names {yaml_names(yml)} differ from model {BASE_MODEL_VERSION} {classes}")
    items = []
    for split in SPLITS:
        for img_dir, lbl_dir, neg in ((ds / "images" / split, ds / "labels" / split, False),
                                      (ds / "images" / "negative" / split, None, True)):
            if not img_dir.is_dir():
                continue
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() not in IMAGE_EXTS:
                    continue
                im = read_image(p)
                if im is None:
                    raise DatasetPrepError(f"{p}: unreadable image in included dataset")
                lines = [] if neg else read_label_lines(lbl_dir / f"{p.stem}.txt")
                probs = check_label(lines, len(classes))
                if probs:
                    raise DatasetPrepError(f"{p}: invalid label in included dataset: {probs[:2]}")
                items.append({"name": f"{ds.name}__{p.name}", "img": p, "label": None, "lines": lines,
                              "session": f"included:{ds.name}:{split}", "dhash": dhash(im), "lock": split,
                              "source_type": "previous_dataset", "category": "litter" if lines else "negative",
                              "subcategory": "", "conditions": "",
                              "prev": "yes", "source_path": str(p), "origin": ds.name})
    if not items:
        raise DatasetPrepError(f"{ds}: no images found (images/<split>/ missing - is the dataset restored?)")
    return items


def assign(items: list[dict], ratios: tuple[float, float, float], near_bits: int, seed: int) -> tuple[list[int], list[str]]:
    n = len(items)
    uf = UnionFind(n)
    first: dict[str, int] = {}
    for i, it in enumerate(items):
        if it["session"] in first:
            uf.union(i, first[it["session"]])
        else:
            first[it["session"]] = i
    bits = _bits([it["dhash"] for it in items])
    for i in range(n - 1):
        d = np.count_nonzero(bits[i + 1:] != bits[i], axis=1)
        for j in np.nonzero(d <= near_bits)[0]:
            uf.union(i, i + 1 + int(j))
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    warnings = []
    split_of = [""] * n
    cur = {s: Counter() for s in SPLITS}
    tot = Counter()
    for it in items:
        tot["img"] += 1
        tot["pos" if it["lines"] else "neg"] += 1
    unlocked = []
    for g, members in groups.items():
        locks = {items[i]["lock"] for i in members if items[i]["lock"]}
        if len(locks) > 1:
            sess = sorted({items[i]["session"] for i in members})
            raise DatasetPrepError(f"conflicting split locks {sorted(locks)} inside one group (same session or "
                                   f"near-identical images): sessions {sess[:5]}")
        if locks:
            s = locks.pop()
            for i in members:
                split_of[i] = s
                cur[s]["img"] += 1
                cur[s]["pos" if items[i]["lines"] else "neg"] += 1
        else:
            unlocked.append(members)
    rnd = random.Random(seed)
    rnd.shuffle(unlocked)
    unlocked.sort(key=len, reverse=True)
    ratio = dict(zip(SPLITS, ratios))
    for members in unlocked:
        g = Counter()
        for i in members:
            g["img"] += 1
            g["pos" if items[i]["lines"] else "neg"] += 1

        def deficit(s: str) -> float:
            return sum((ratio[s] * tot[k] - cur[s][k]) / max(tot[k], 1) for k in ("img", "pos", "neg"))

        s = max(SPLITS, key=lambda s: (deficit(s), -SPLITS.index(s)))
        for i in members:
            split_of[i] = s
        cur[s].update(g)
    if len(unlocked) < 3:
        warnings.append(f"only {len(unlocked)} independent group(s) free to place in val/test: record more separate "
                        f"sessions (other days / cameras) for an honest evaluation")
    for s in ("val", "test"):
        if not cur[s]["img"]:
            warnings.append(f"{s} split is EMPTY")
        elif not cur[s]["pos"]:
            warnings.append(f"{s} split has no litter images")
        elif not cur[s]["neg"]:
            warnings.append(f"{s} split has no images without litter (hard negatives / clean scenes)")
    return split_of, warnings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="new dataset version folder under dataset/, e.g. cctv_ldm_v4")
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    ap.add_argument("--include-dataset", type=Path, nargs="*", default=[],
                    help="existing YOLO datasets with the SAME class list to merge (their splits are kept)")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--near-dup-bits", type=int, default=10, help="dHash distance that ties images together (default 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-synthetic-eval", action="store_true", help="allow synthetic images in val/test (not advised)")
    ap.add_argument("--dry-run", action="store_true", help="report the split without writing the dataset")
    a = ap.parse_args(argv)

    out = PROJECT_ROOT / "dataset" / a.name
    try:
        if abs(sum(a.ratios) - 1) > 1e-6 or min(a.ratios) < 0:
            raise DatasetPrepError("--ratios must be non-negative and sum to 1")
        if out.exists():
            raise DatasetPrepError(f"{out} already exists - datasets are versioned, choose a new --name")
        classes = model_classes()
        if not (a.workspace / "data.yaml").is_file():
            raise DatasetPrepError(f"no workspace at {a.workspace}")
        if yaml_names(a.workspace / "data.yaml") != classes:
            raise DatasetPrepError("workspace class names differ from the model's class list")
        items, excluded = collect_workspace(a.workspace, classes, a.allow_synthetic_eval)
        for ds in a.include_dataset:
            items += collect_included(ds if ds.is_absolute() else PROJECT_ROOT / ds, classes)
        if not items:
            raise DatasetPrepError(f"no usable images: {len(excluded)} excluded (run 'python -m training.ldm_review "
                                   f"report' to see what still needs review)")
        split_of, warnings = assign(items, tuple(a.ratios), a.near_dup_bits, a.seed)
    except DatasetPrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = {s: Counter() for s in SPLITS}
    obj = {s: Counter() for s in SPLITS}
    sessions = {s: set() for s in SPLITS}
    for it, s in zip(items, split_of):
        summary[s]["images"] += 1
        summary[s]["with_litter" if it["lines"] else "without_litter"] += 1
        summary[s][it["source_type"]] += 1
        sessions[s].add(it["session"])
        for p in it["lines"]:
            obj[s][classes[int(p[0])]] += 1
    report = {
        "name": a.name, "created": now_iso(), "class_source": f"models/litter_yolo/{BASE_MODEL_VERSION}",
        "classes": classes, "workspace": str(a.workspace), "included": [str(d) for d in a.include_dataset],
        "ratios": list(a.ratios), "near_dup_bits": a.near_dup_bits, "seed": a.seed,
        "splits": {s: {**summary[s], "sessions": sorted(sessions[s]), "objects": dict(obj[s])} for s in SPLITS},
        "excluded": excluded, "warnings": warnings,
        "command": "python -m training.build_ldm_dataset " + " ".join(argv if argv is not None else sys.argv[1:]),
    }
    for s in SPLITS:
        print(f"{s:5s}: {summary[s]['images']:5d} images ({summary[s]['with_litter']} with litter, "
              f"{summary[s]['without_litter']} without), {sum(obj[s].values())} boxes, {len(sessions[s])} session group(s)")
    print(f"excluded: {len(excluded)} (see build_report.json)")
    for w in warnings:
        print(f"WARNING: {w}")
    if a.dry_run:
        print("dry run: nothing written")
        return 0

    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    with open(out / "split_manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "split", "session", "source_type", "category", "subcategory", "conditions", "boxes",
                    "used_in_previous_training", "origin", "source_path"])
        for it, s in zip(items, split_of):
            shutil.copy2(it["img"], out / "images" / s / it["name"])
            (out / "labels" / s / f"{Path(it['name']).stem}.txt").write_text(
                "".join(" ".join(p) + "\n" for p in it["lines"]), encoding="utf-8")
            w.writerow([it["name"], s, it["session"], it["source_type"], it["category"], it["subcategory"],
                        it["conditions"], len(it["lines"]), it["prev"], it["origin"], it["source_path"]])
    (out / "data.yaml").write_text(
        f"# {a.name}: generated {now_iso()} by training/build_ldm_dataset.py from {a.workspace.name}\n"
        f"# Class mapping = models/litter_yolo/{BASE_MODEL_VERSION} (verified). No 'path' key: the folder of this file\n"
        "# is the dataset root, so the folder can be moved or zipped for a GPU machine.\n"
        + yaml.safe_dump({"train": "images/train", "val": "images/val", "test": "images/test",
                          "names": dict(enumerate(classes))}, sort_keys=False), encoding="utf-8")
    (out / "build_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    lines = [f"# {a.name}", "", f"Built {report['created']} from `{a.workspace}`"
             + (f" + {', '.join(report['included'])}" if report["included"] else "") + ".", "",
             f"Classes (from models/litter_yolo/{BASE_MODEL_VERSION}): " + ", ".join(f"{i}={c}" for i, c in enumerate(classes)),
             "", "| split | images | with litter | without litter | boxes | real CCTV | synthetic | sessions |",
             "|---|---|---|---|---|---|---|---|"]
    for s in SPLITS:
        c = summary[s]
        lines.append(f"| {s} | {c['images']} | {c['with_litter']} | {c['without_litter']} | {sum(obj[s].values())} | "
                     f"{c['real_cctv'] + c['other_real']} | {c['synthetic']} | {len(sessions[s])} |")
    lines += ["", "Rules: one session = one split; near-identical frames tied; synthetic and previously-trained footage "
              "locked to train.", "", f"Reproduce: `{report['command']}` (same workspace + manifest.csv)", ""]
    if warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]
    lines += [f"{len(excluded)} workspace images excluded - see build_report.json.",
              f"Validate before training: `python -m training.validate_ldm_dataset dataset/{a.name}`", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"written: {out}")
    print(f"next: python -m training.validate_ldm_dataset dataset/{a.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
