"""Verify a seven-class YOLO dataset before ANY ldm7 training or evaluation.

    py -m training.ldm7.verify_dataset --root D:/LDM7/ldm7_v1 [--out D:/LDM7/ldm7_v1/verify_report]
        [--min-instances 50] [--near-bits 10] [--group-by session_id,camera_day]

Layout: <root>/data.yaml, <root>/images/{train,val,test}/*, <root>/labels/{train,val,test}/<stem>.txt,
optional <root>/manifest.csv (columns at least: image, camera_id, session_id, date; image = images/<split>/<file>
or the bare file name).

ERROR (exit code 1):
  data.yaml missing / class names not exactly the ldm7 schema / nc wrong
  train or val split missing or empty
  image unreadable; label file MISSING (a background image needs an EXISTING EMPTY .txt - absence is never
  silently treated as "no litter"); two images with the same stem in one split
  label line: not 5 values, non-numeric, class id not an integer in 0..6, coordinates outside 0..1, size <= 0,
  box extending outside the image
  exact duplicate (sha256) or near-duplicate (dHash) images in DIFFERENT splits (leakage)
  manifest present: image without a manifest row, row without camera_id/session_id/date, and any group
  (session / camera-day by default) that appears in more than one split (group leakage)
WARNING: orphan label files, duplicates inside one split, test split missing/empty, a class with fewer than
  --min-instances train instances, a class with no val/test instance, no manifest (group leakage unchecked),
  val/test without background images.
Reports: report.json + report.md (default <root>/verify_report/). Nothing in the dataset is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from training.ldm7.common import (DEFAULT_GROUP_BY, GROUP_KEYS, SIZE_BUCKETS, SPLITS, dhash, find_manifest_row,
                                  group_values, index_manifest, list_images, manifest_missing_columns, read_image,
                                  read_manifest, sha256_file, size_bucket, tree_hash)
from training.ldm7.schema import CLASSES, NUM_CLASSES, SCHEMA_VERSION, names_from_yaml

REQUIRED_SPLITS = ("train", "val")
EDGE_TOL = 0.002


class Issues:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def add(self, severity: str, typ: str, image: str, detail: str) -> None:
        self.items.append({"severity": severity, "type": typ, "image": image, "detail": detail})

    def count(self, severity: str) -> int:
        return sum(1 for i in self.items if i["severity"] == severity)

    def types(self, severity: Optional[str] = None) -> set[str]:
        return {i["type"] for i in self.items if severity is None or i["severity"] == severity}


def parse_label_file(lp: Path, img_w: int, img_h: int, image_id: str,
                     issues: Issues) -> list[tuple[int, float, float, float, float]]:
    """Strictly validate one YOLO label file. Invalid lines are reported as ERROR and skipped."""
    boxes = []
    for i, ln in enumerate(lp.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        p = ln.split()
        if not p:
            continue
        if len(p) != 5:
            issues.add("ERROR", "label_syntax", image_id, f"line {i}: expected 5 values, got {len(p)}")
            continue
        try:
            cf = float(p[0])
            cx, cy, w, h = (float(v) for v in p[1:])
        except ValueError:
            issues.add("ERROR", "label_syntax", image_id, f"line {i}: non-numeric value")
            continue
        if not all(np.isfinite([cf, cx, cy, w, h])):
            issues.add("ERROR", "label_syntax", image_id, f"line {i}: non-finite value")
            continue
        if cf != int(cf) or not 0 <= int(cf) < NUM_CLASSES:
            issues.add("ERROR", "class_id", image_id, f"line {i}: class id {p[0]} not an integer in 0..{NUM_CLASSES - 1}")
            continue
        if w <= 0 or h <= 0:
            issues.add("ERROR", "box_size", image_id, f"line {i}: width/height must be > 0")
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and w <= 1 and h <= 1):
            issues.add("ERROR", "not_normalized", image_id, f"line {i}: coordinates must be normalised to 0..1")
            continue
        if cx - w / 2 < -EDGE_TOL or cy - h / 2 < -EDGE_TOL or cx + w / 2 > 1 + EDGE_TOL or cy + h / 2 > 1 + EDGE_TOL:
            issues.add("ERROR", "box_outside_image", image_id, f"line {i}: box extends beyond the image")
            continue
        if w * img_w < 4 or h * img_h < 4:
            issues.add("WARNING", "tiny_box", image_id, f"line {i}: {w * img_w:.0f}x{h * img_h:.0f}px - real or mis-click?")
        boxes.append((int(cf), cx, cy, w, h))
    return boxes


def _near_pairs(hashes: list[str], bits: int) -> list[tuple[int, int, int]]:
    if len(hashes) < 2:
        return []
    m = np.unpackbits(np.array([np.frombuffer(bytes.fromhex(h), np.uint8) for h in hashes]), axis=1)
    out = []
    for i in range(len(hashes) - 1):
        d = np.count_nonzero(m[i + 1:] != m[i], axis=1)
        out += [(i, i + 1 + int(j), int(d[j])) for j in np.nonzero(d <= bits)[0]]
    return out


def verify(root: Path, min_instances: int = 50, near_bits: int = 10,
           group_by: Sequence[str] = DEFAULT_GROUP_BY) -> dict[str, Any]:
    root = Path(root)
    iss = Issues()
    bad_keys = [k for k in group_by if k not in GROUP_KEYS]
    if bad_keys:
        raise ValueError(f"unknown group key(s) {bad_keys}; allowed {GROUP_KEYS}")

    # ---- data.yaml
    yml = root / "data.yaml"
    names: list[str] = []
    if not yml.is_file():
        iss.add("ERROR", "data_yaml_missing", "data.yaml", f"{yml} not found")
    else:
        try:
            import yaml
            doc = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
            names = names_from_yaml(yml)
            if names != CLASSES:
                iss.add("ERROR", "class_names", "data.yaml", f"names {names} != ldm7 schema {CLASSES}")
            if "nc" in doc and doc["nc"] != NUM_CLASSES:
                iss.add("ERROR", "class_names", "data.yaml", f"nc {doc['nc']} != {NUM_CLASSES}")
        except Exception as exc:  # malformed YAML
            iss.add("ERROR", "data_yaml_invalid", "data.yaml", str(exc))

    # ---- manifest
    manifest_idx: Optional[dict[str, dict[str, str]]] = None
    mp = root / "manifest.csv"
    if mp.is_file():
        missing_cols = manifest_missing_columns(mp)
        if missing_cols:
            iss.add("ERROR", "manifest_columns", "manifest.csv", f"missing columns {missing_cols}")
        else:
            manifest_idx = index_manifest(read_manifest(mp))
    else:
        iss.add("WARNING", "no_manifest", "manifest.csv",
                "no manifest.csv: camera/session/day leakage between splits cannot be checked")

    # ---- per image
    records: list[dict[str, Any]] = []
    per_split: dict[str, dict[str, Any]] = {}
    for s in SPLITS:
        img_dir, lab_dir = root / "images" / s, root / "labels" / s
        imgs = list_images(img_dir)
        st = {"images": 0, "background_images": 0, "instances": Counter(), "images_with_class": Counter()}
        per_split[s] = st
        if not imgs:
            if s in REQUIRED_SPLITS:
                iss.add("ERROR", "empty_split", s, f"{img_dir} missing or has no images")
            else:
                iss.add("WARNING", "no_test_split", s, "no test images: nothing independent to evaluate on")
            continue
        stems = Counter(p.stem for p in imgs)
        for stem, n in stems.items():
            if n > 1:
                iss.add("ERROR", "duplicate_stem", f"images/{s}/{stem}.*", f"{n} images share one label file name")
        for p in imgs:
            iid = f"images/{s}/{p.name}"
            img = read_image(p)
            if img is None:
                iss.add("ERROR", "corrupt_image", iid, "cannot be decoded")
                continue
            h, w = img.shape[:2]
            st["images"] += 1
            lp = lab_dir / f"{p.stem}.txt"
            boxes: list[tuple[int, float, float, float, float]] = []
            if not lp.is_file():
                iss.add("ERROR", "missing_label", iid,
                        f"labels/{s}/{p.stem}.txt missing (background images need an EMPTY label file)")
            else:
                boxes = parse_label_file(lp, w, h, iid, iss)
                if not boxes and lp.read_text(encoding="utf-8", errors="replace").strip() == "":
                    st["background_images"] += 1
            for c in {b[0] for b in boxes}:
                st["images_with_class"][CLASSES[c]] += 1
            rec = {"id": iid, "split": s, "w": w, "h": h, "boxes": boxes, "sha": sha256_file(p), "dhash": dhash(img),
                   "path": p, "groups": []}
            if manifest_idx is not None:
                row = find_manifest_row(manifest_idx, s, p.name)
                if row is None:
                    iss.add("ERROR", "not_in_manifest", iid, "no manifest.csv row: its camera/session/day is unknown")
                else:
                    blanks = [k for k in ("camera_id", "session_id", "date") if not row.get(k)]
                    if blanks:
                        iss.add("ERROR", "manifest_blank", iid, f"manifest row has empty {blanks}")
                    rec["groups"] = group_values(row, group_by)
            records.append(rec)
        img_stems = {p.stem for p in imgs}
        if lab_dir.is_dir():
            for lp in sorted(lab_dir.glob("*.txt")):
                if lp.stem not in img_stems:
                    iss.add("WARNING", "orphan_label", f"labels/{s}/{lp.name}", "label file without image")

    # ---- duplicates / leakage
    by_sha: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_sha[r["sha"]].append(r)
    exact_pairs = set()
    for group in by_sha.values():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                ra, rb = group[a], group[b]
                exact_pairs.add((ra["id"], rb["id"]))
                if ra["split"] != rb["split"]:
                    iss.add("ERROR", "cross_split_duplicate", ra["id"], f"identical file in {rb['id']} (leakage)")
                else:
                    iss.add("WARNING", "duplicate_image", ra["id"], f"identical file {rb['id']} in the same split")
    for i, j, d in _near_pairs([r["dhash"] for r in records], near_bits):
        ra, rb = records[i], records[j]
        if (ra["id"], rb["id"]) in exact_pairs:
            continue
        if ra["split"] != rb["split"]:
            iss.add("ERROR", "cross_split_near_duplicate", ra["id"], f"near-identical to {rb['id']} ({d}/256 bits)")
        else:
            iss.add("INFO", "near_duplicate", ra["id"], f"near-identical to {rb['id']} ({d}/256 bits), same split")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for r in records:
        for g in r["groups"]:
            group_splits[g].add(r["split"])
    for g, sp in sorted(group_splits.items()):
        if len(sp) > 1:
            iss.add("ERROR", "group_leakage", g, f"group appears in {sorted(sp)}; one group must stay in one split")

    # ---- statistics
    size_stats = {s: {c: Counter() for c in CLASSES} for s in SPLITS}
    for r in records:
        for c, _cx, _cy, bw, bh in r["boxes"]:
            per_split[r["split"]]["instances"][CLASSES[c]] += 1
            size_stats[r["split"]][CLASSES[c]][size_bucket(bw * r["w"] * bh * r["h"])] += 1
    for c in CLASSES:
        n_train = per_split["train"]["instances"][c]
        if n_train < min_instances:
            iss.add("WARNING", "few_instances", c, f"only {n_train} train instances (< {min_instances})")
        for s in ("val", "test"):
            if per_split[s]["images"] and per_split[s]["instances"][c] == 0:
                iss.add("WARNING", "class_absent", c, f"no {c} instance in {s}: that class cannot be evaluated there")
    for s in ("val", "test"):
        if per_split[s]["images"] and not per_split[s]["background_images"]:
            iss.add("WARNING", "no_background", s, f"{s} has no background (empty-label) images: false positives untested")

    split_summary = {
        s: {"images": st["images"], "background_images": st["background_images"],
            "instances": {c: st["instances"][c] for c in CLASSES},
            "images_with_class": {c: st["images_with_class"][c] for c in CLASSES},
            "size_buckets": {c: {b: size_stats[s][c][b] for b in SIZE_BUCKETS} for c in CLASSES}}
        for s, st in per_split.items()}
    files = [(r["id"], r["path"]) for r in records]
    for s in SPLITS:
        files += [(f"labels/{s}/{lp.name}", lp) for lp in sorted((root / "labels" / s).glob("*.txt"))] \
            if (root / "labels" / s).is_dir() else []
    if yml.is_file():
        files.append(("data.yaml", yml))
    errors = iss.count("ERROR")
    return {
        "root": str(root), "schema": SCHEMA_VERSION, "classes": CLASSES, "data_yaml_names": names,
        "ok": errors == 0, "errors": errors, "warnings": iss.count("WARNING"), "infos": iss.count("INFO"),
        "issue_counts": dict(Counter(f"{i['severity']}:{i['type']}" for i in iss.items)),
        "group_by": list(group_by), "manifest": manifest_idx is not None, "groups": len(group_splits),
        "dataset_hash": tree_hash(files), "splits": split_summary, "issues": iss.items,
        "settings": {"min_instances": min_instances, "near_bits": near_bits},
    }


def write_reports(report: dict[str, Any], out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    jp, mp = out / "report.json", out / "report.md"
    jp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    L = [f"# ldm7 dataset verification - {'PASS' if report['ok'] else 'FAIL'}", "",
         f"Root: `{report['root']}`  ", f"Schema: {report['schema']}  ", f"Dataset hash: `{report['dataset_hash']}`  ",
         f"Errors: **{report['errors']}**, warnings: {report['warnings']}, infos: {report['infos']}  ",
         f"Group leakage check: {'by ' + ', '.join(report['group_by']) if report['manifest'] else 'NOT RUN (no manifest.csv)'}",
         "", "## Class distribution (instances / images containing the class)", "",
         "| class | " + " | ".join(SPLITS) + " |", "|---|" + "---|" * len(SPLITS)]
    for c in CLASSES:
        L.append(f"| {c} | " + " | ".join(f"{report['splits'][s]['instances'][c]} / {report['splits'][s]['images_with_class'][c]}"
                                          for s in SPLITS) + " |")
    L.append("| images (background) | " + " | ".join(f"{report['splits'][s]['images']} ({report['splits'][s]['background_images']})"
                                                     for s in SPLITS) + " |")
    L += ["", "## Box sizes per class, all splits (small < 32x32 px, medium <= 96x96 px, large)", "",
          "| class | small | medium | large |", "|---|---|---|---|"]
    for c in CLASSES:
        tot = Counter()
        for s in SPLITS:
            tot.update(report["splits"][s]["size_buckets"][c])
        L.append(f"| {c} | {tot['small']} | {tot['medium']} | {tot['large']} |")
    L += ["", "## Issues", ""]
    if not report["issues"]:
        L.append("None.")
    for sev in ("ERROR", "WARNING", "INFO"):
        rows = [i for i in report["issues"] if i["severity"] == sev]
        if rows:
            L += [f"### {sev} ({len(rows)})", "", "| type | item | detail |", "|---|---|---|"]
            L += [f"| {i['type']} | {i['image']} | {i['detail']} |" for i in rows[:500]]
            if len(rows) > 500:
                L.append(f"| ... | {len(rows) - 500} more in report.json | |")
            L.append("")
    mp.write_text("\n".join(L) + "\n", encoding="utf-8")
    return jp, mp


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="report folder (default <root>/verify_report)")
    ap.add_argument("--min-instances", type=int, default=50)
    ap.add_argument("--near-bits", type=int, default=10, help="dHash distance (of 256) counted as near-duplicate")
    ap.add_argument("--group-by", default=",".join(DEFAULT_GROUP_BY),
                    help=f"comma list from {GROUP_KEYS}; e.g. camera_id for an unseen-camera test split")
    a = ap.parse_args(argv)
    rep = verify(a.root, a.min_instances, a.near_bits, [k.strip() for k in a.group_by.split(",") if k.strip()])
    jp, mp = write_reports(rep, a.out or a.root / "verify_report")
    print(f"{'PASS' if rep['ok'] else 'FAIL'}: {rep['errors']} error(s), {rep['warnings']} warning(s) -> {mp}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
