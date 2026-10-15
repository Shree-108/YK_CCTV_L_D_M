"""Validate annotations: the annotation WORKSPACE (before splitting) or a BUILT dataset (before training).

    python -m training.validate_ldm_dataset dataset/ldm_v4_workspace     # workspace mode (has manifest.csv)
    python -m training.validate_ldm_dataset dataset/cctv_ldm_v4          # built-dataset mode

Exit code 1 if any ERROR is found. Reports (nothing else is touched):
    workspace: reports/validation_report.md|json + reports/files_to_fix.csv
    dataset:   validation_report.md|json + files_to_fix.csv in the dataset folder

Both modes check every label line: 5 values, integer class id inside the model's class list, positive size, box
inside the image (ERROR); boxes smaller than 4 px, covering > 60 % of the frame, duplicated or overlapping > 80 %
with a same-class box (WARNING). Every image is opened; exact and near-duplicate images are listed.

Workspace mode adds the review bookkeeping:
  ERROR    class list in data.yaml differs from the model; manifest values not in the allowed vocabulary
  ERROR    image missing / unreadable; label file without an image; reviewed image (annotated / reviewed_empty)
           without a label file; 'annotated' with an empty label; 'reviewed_empty' or category hard_negative/clean
           with litter boxes; category litter without boxes
  WARNING  label saved but status still unreviewed (run ldm_review sync); reviewed_empty without a category;
           labelled frame flagged for burned-in boxes (it will be excluded); image on disk not in manifest.csv
  INFO     empty-label images are tracked (they mean "a person reviewed this and found no litter")
  plus a split PREVIEW of the verified images (sessions per split) so leakage is visible before building.

Built-dataset mode reuses the project's checker (backend/services/datasets.py) and adds leakage checks:
near-identical images in different splits, one session in several splits, synthetic images in val/test, empty
val/test or val/test without litter (ERROR); val/test without litter-free images, val/test frames the current
model was trained on (WARNING).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from backend.core.config import PROJECT_ROOT
from backend.services.datasets import IMAGE_EXTS, DatasetService
from training.ldm_common import (BASE_MODEL_VERSION, DatasetPrepError, dhash, label_path, model_classes, read_image,
                                 read_manifest, validate_row_vocab, yaml_names)

SPLITS = ("train", "val", "test")
ERROR_TYPES = {"corrupt_image", "label_error", "orphan_label", "missing_label"}
ACTIONS = {
    "class_mapping": "regenerate data.yaml from the model class list; never re-order classes",
    "manifest_value": "fix the value in manifest.csv (allowed values: docs/LDM_ANNOTATION_GUIDE.md)",
    "image_missing": "restore the image or delete its manifest row",
    "image_unreadable": "re-export the frame; mark the row review_status=rejected",
    "label_syntax": "re-save the image in the Ground Truth page (or fix the line by hand)",
    "class_id": "the class id is outside the model's 10 classes: re-label the box",
    "box_size": "re-draw the box",
    "box_outside_image": "re-draw the box inside the image",
    "tiny_box": "check: real distant litter, or a mis-click?",
    "very_large_box": "check: a box over most of the frame is usually a mistake",
    "duplicate_box": "delete the duplicate box",
    "overlapping_boxes": "check: two boxes on one item?",
    "orphan_label": "label without image: delete it or restore the image",
    "reviewed_without_label": "annotate it, or run 'ldm_review mark-empty' if it was reviewed as no-litter",
    "annotated_but_empty": "add the boxes, or change review_status to reviewed_empty",
    "empty_status_with_boxes": "remove the boxes, or change review_status to annotated",
    "category_conflict": "fix category: litter = has boxes, hard_negative/clean = no boxes",
    "needs_sync": "run: python -m training.ldm_review sync",
    "needs_category": "set category hard_negative or clean in manifest.csv",
    "overlay_with_labels": "confirm the frame has no drawn boxes, then set overlay_suspect=no; else it stays excluded",
    "image_not_in_manifest": "import it with import_images.py or remove it",
    "exact_duplicate": "keep one copy (set the other to review_status=rejected)",
    "near_duplicate": "fine within one session; across sessions the build ties them to one split",
    "cross_split_near_duplicate": "rebuild the dataset (the build ties near-duplicates together)",
    "session_in_several_splits": "rebuild the dataset; never move files between splits by hand",
    "synthetic_in_eval": "rebuild without --allow-synthetic-eval",
    "seen_by_current_model": "prefer unseen footage for val/test (split_lock train for this session)",
    "empty_split": "add more independent sessions",
    "no_litter_in_split": "add sessions with litter",
    "no_negatives_in_split": "add litter-free sessions (hard negatives / clean scenes)",
}


class Report:
    def __init__(self) -> None:
        self.issues: list[dict[str, str]] = []

    def add(self, sev: str, typ: str, image: str, detail: str) -> None:
        self.issues.append({"severity": sev, "type": typ, "image": image, "detail": detail,
                            "action": ACTIONS.get(typ, "")})

    def count(self, sev: str) -> int:
        return sum(1 for i in self.issues if i["severity"] == sev)


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def check_label_file(lp: Path, n_classes: int, img_w: int, img_h: int, image_id: str,
                     rep: Report) -> list[tuple[int, float, float, float, float]]:
    """Validate one YOLO label file; returns the valid boxes. Issues go into rep."""
    boxes = []
    seen = set()
    for i, ln in enumerate(lp.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        p = ln.split()
        if not p:
            continue
        if len(p) != 5:
            rep.add("ERROR", "label_syntax", image_id, f"line {i}: expected 5 values, got {len(p)}")
            continue
        try:
            cf = float(p[0])
            cx, cy, w, h = map(float, p[1:])
        except ValueError:
            rep.add("ERROR", "label_syntax", image_id, f"line {i}: non-numeric value")
            continue
        if cf != int(cf) or not 0 <= int(cf) < n_classes:
            rep.add("ERROR", "class_id", image_id, f"line {i}: class id {p[0]} not in 0..{n_classes - 1}")
            continue
        c = int(cf)
        if w <= 0 or h <= 0:
            rep.add("ERROR", "box_size", image_id, f"line {i}: zero or negative width/height")
            continue
        if cx - w / 2 < -0.002 or cy - h / 2 < -0.002 or cx + w / 2 > 1.002 or cy + h / 2 > 1.002:
            rep.add("ERROR", "box_outside_image", image_id, f"line {i}: class {c} box extends beyond the image")
            continue
        if img_w and (w * img_w < 4 or h * img_h < 4):
            rep.add("WARNING", "tiny_box", image_id, f"line {i}: class {c} box {w * img_w:.0f}x{h * img_h:.0f}px")
        if w * h > 0.6:
            rep.add("WARNING", "very_large_box", image_id, f"line {i}: class {c} box covers {w * h:.0%} of the frame")
        key = (c, round(cx, 4), round(cy, 4), round(w, 4), round(h, 4))
        if key in seen:
            rep.add("WARNING", "duplicate_box", image_id, f"line {i}: identical class-{c} box")
        seen.add(key)
        boxes.append((c, cx, cy, w, h))
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            if boxes[a][0] == boxes[b][0] and boxes[a][1:] != boxes[b][1:] and _iou(boxes[a][1:], boxes[b][1:]) > 0.8:
                rep.add("WARNING", "overlapping_boxes", image_id, f"two class-{boxes[a][0]} boxes overlap > 80 %")
    return boxes


def near_duplicates(items: list[tuple[str, str, str]], bits: int) -> list[tuple[int, int, int]]:
    """items: (id, group, dhash hex). Returns (i, j, distance) pairs with distance <= bits."""
    if len(items) < 2:
        return []
    m = np.unpackbits(np.array([np.frombuffer(bytes.fromhex(h), np.uint8) for _, _, h in items]), axis=1)
    out = []
    for i in range(len(items) - 1):
        d = np.count_nonzero(m[i + 1:] != m[i], axis=1)
        out += [(i, i + 1 + int(j), int(d[j])) for j in np.nonzero(d <= bits)[0]]
    return out


# ---------------------------------------------------------------- workspace mode
def validate_workspace(ws: Path, dup_bits: int) -> tuple[dict, Report]:
    rep = Report()
    classes = model_classes()
    names = yaml_names(ws / "data.yaml")
    if names != classes:
        rep.add("ERROR", "class_mapping", "data.yaml", f"names {names} != {BASE_MODEL_VERSION} {classes}")
    rows = read_manifest(ws)
    in_manifest = {r["image"] for r in rows}
    img_dir = ws / "images" / "train"
    for p in sorted(img_dir.iterdir()) if img_dir.is_dir() else []:
        if p.suffix.lower() in IMAGE_EXTS and p.name not in in_manifest:
            rep.add("WARNING", "image_not_in_manifest", p.name, "image file has no manifest.csv row")
    stems = {Path(n).stem for n in in_manifest}
    for lf in sorted((ws / "labels" / "train").glob("*.txt")):
        if lf.stem not in stems:
            rep.add("ERROR", "orphan_label", lf.name, "label file without a manifest image")

    status, cats, empty_labels = Counter(), Counter(), []
    obj, sha_seen, hashes = Counter(), {}, []
    verified, unverified = [], []
    for r in rows:
        iid = r["image"]
        st = r["review_status"] or "unreviewed"
        status[st] += 1
        for prob in validate_row_vocab(r):
            rep.add("ERROR", "manifest_value", iid, prob)
        p = img_dir / iid
        if not p.is_file():
            rep.add("ERROR", "image_missing", iid, "listed in manifest.csv but not in images/train")
            continue
        raw = p.read_bytes()
        img = read_image(p)
        if img is None:
            rep.add("ERROR", "image_unreadable", iid, "cannot be decoded")
            continue
        h, w = img.shape[:2]
        digest = hashlib.sha256(raw).hexdigest()
        if digest in sha_seen:
            rep.add("WARNING", "exact_duplicate", iid, f"byte-identical to {sha_seen[digest]}")
        sha_seen.setdefault(digest, iid)
        hashes.append((iid, r["session"], r["dhash"] or dhash(img)))
        lp = label_path(ws, iid)
        has_label = lp.is_file()
        boxes = check_label_file(lp, len(classes), w, h, iid, rep) if has_label else []
        for b in boxes:
            obj[classes[b[0]]] += 1
        if has_label and not boxes and not lp.read_text(encoding="utf-8", errors="replace").strip():
            empty_labels.append(iid)
        if st in ("annotated", "reviewed_empty") and not has_label:
            rep.add("ERROR", "reviewed_without_label", iid, f"review_status {st} but no label file")
        if st == "annotated" and has_label and not boxes:
            rep.add("ERROR", "annotated_but_empty", iid, "status annotated but the label has no valid box")
        if st == "reviewed_empty" and boxes:
            rep.add("ERROR", "empty_status_with_boxes", iid, f"status reviewed_empty but {len(boxes)} box(es)")
        if r["category"] in ("hard_negative", "clean") and boxes:
            rep.add("ERROR", "category_conflict", iid, f"category {r['category']} but {len(boxes)} litter box(es)")
        if r["category"] == "litter" and st in ("annotated", "reviewed_empty") and has_label and not boxes:
            rep.add("ERROR", "category_conflict", iid, "category litter but no boxes")
        if has_label and st == "unreviewed":
            rep.add("WARNING", "needs_sync", iid, "label saved but review_status still unreviewed")
        if st == "reviewed_empty" and r["category"] in ("", "unknown"):
            rep.add("WARNING", "needs_category", iid, "no-litter image without category hard_negative / clean")
        if has_label and r["overlay_suspect"] == "yes":
            rep.add("WARNING", "overlay_with_labels", iid, "labelled, but flagged for burned-in boxes: will be excluded")
        if st in ("annotated", "reviewed_empty") and has_label and r["overlay_suspect"] != "yes":
            verified.append(iid)
            cats[r["category"] or "unknown"] += 1
        else:
            unverified.append(iid)
    for i, j, d in near_duplicates(hashes, dup_bits):
        same = hashes[i][1] == hashes[j][1]
        rep.add("WARNING" if not same else "INFO", "near_duplicate", hashes[i][0],
                f"{'same' if same else 'DIFFERENT'} session as {hashes[j][0]} (distance {d}/256)")

    # split preview of what the build would do with the verified images
    preview: dict = {"note": "no verified images yet - nothing to split"}
    if verified:
        from training.build_ldm_dataset import assign, collect_workspace
        items, _ = collect_workspace(ws, classes, False)
        if items:
            try:
                split_of, warns = assign(items, (0.7, 0.15, 0.15), 10, 0)
                per = {s: {"images": 0, "sessions": set()} for s in SPLITS}
                for it, s in zip(items, split_of):
                    per[s]["images"] += 1
                    per[s]["sessions"].add(it["session"])
                preview = {s: {"images": v["images"], "sessions": sorted(v["sessions"])} for s, v in per.items()}
                preview["warnings"] = warns
            except DatasetPrepError as exc:
                rep.add("ERROR", "session_in_several_splits", "", str(exc))
    summary = {
        "mode": "workspace", "path": str(ws), "classes": names, "images_listed": len(rows),
        "review_status": dict(status), "verified_images": len(verified), "unverified_images": len(unverified),
        "verified_by_category": dict(cats), "empty_label_images": len(empty_labels), "boxes_per_class": dict(obj),
        "split_preview": preview,
    }
    return summary, rep


# ---------------------------------------------------------------- built-dataset mode
def validate_dataset(ds: Path, leak_bits: int) -> tuple[dict, Report]:
    rep = Report()
    classes = model_classes()
    names = yaml_names(ds / "data.yaml")
    if names != classes:
        rep.add("ERROR", "class_mapping", "data.yaml", f"names {names} != {BASE_MODEL_VERSION} {classes}")
    stats = DatasetService(ds.parent).stats(ds.name)
    for i in stats["issues"]:
        if i["type"] in ("near_duplicate", "tiny_object"):
            continue            # reported below with split context / pixel threshold
        rep.add("ERROR" if i["type"] in ERROR_TYPES else "WARNING", i["type"], i["image"], i["detail"])
    flat: list[tuple[str, str, str]] = []
    counts = {s: Counter() for s in SPLITS}
    for s in SPLITS:
        img_dir = ds / "images" / s
        if not img_dir.is_dir():
            rep.add("ERROR", "empty_split", s, f"{img_dir} does not exist")
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            iid = f"images/{s}/{p.name}"
            img = read_image(p)
            if img is None:
                continue                        # reported by the project checker
            flat.append((iid, s, dhash(img)))
            counts[s]["images"] += 1
            lp = ds / "labels" / s / f"{p.stem}.txt"
            boxes = []
            if lp.is_file():
                # syntax / class / duplicate-box problems come from the project checker above; keep only the extra checks
                tmp = Report()
                boxes = check_label_file(lp, len(classes), img.shape[1], img.shape[0], iid, tmp)
                rep.issues += [x for x in tmp.issues if x["type"] in ("box_outside_image", "very_large_box",
                                                                     "overlapping_boxes", "tiny_box")]
            counts[s]["with_litter" if boxes else "without_litter"] += 1
    leaks = 0
    for i, j, d in near_duplicates(flat, leak_bits):
        if flat[i][1] != flat[j][1]:
            leaks += 1
            if leaks <= 200:
                rep.add("ERROR", "cross_split_near_duplicate", flat[i][0], f"near-identical to {flat[j][0]} ({d}/256)")
        else:
            rep.add("INFO", "near_duplicate", flat[i][0], f"same split as {flat[j][0]} ({d}/256)")
    sm = ds / "split_manifest.csv"
    if sm.is_file():
        with open(sm, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        sess_splits: dict[str, set] = defaultdict(set)
        for r in rows:
            sess_splits[r["session"]].add(r["split"])
            if r["split"] in ("val", "test"):
                if r["source_type"] == "synthetic":
                    rep.add("ERROR", "synthetic_in_eval", r["image"], f"synthetic image in {r['split']}")
                if r["used_in_previous_training"] in ("yes", "likely") and r["origin"] == "workspace":
                    rep.add("WARNING", "seen_by_current_model", r["image"],
                            f"{r['split']} image was (likely) in v3 training data - inflates the v3 baseline")
        for sname, sp in sess_splits.items():
            if len(sp) > 1:
                rep.add("ERROR", "session_in_several_splits", sname, f"session appears in {sorted(sp)}")
    else:
        rep.add("WARNING", "no_split_manifest", "", "split_manifest.csv missing: provenance / session checks skipped")
    for s in ("val", "test"):
        if not counts[s]["images"]:
            rep.add("ERROR", "empty_split", s, f"{s} has no images")
        elif not counts[s]["with_litter"]:
            rep.add("ERROR", "no_litter_in_split", s, f"{s} has no images with litter")
        elif not counts[s]["without_litter"]:
            rep.add("WARNING", "no_negatives_in_split", s, f"{s} has no litter-free images (false positives untested)")
    summary = {"mode": "dataset", "path": str(ds), "classes": names, "counts": {s: dict(c) for s, c in counts.items()},
               "objects_per_class": stats["objects_per_class"], "size_distribution": stats["size_distribution"],
               "empty_label_images": sum(c["without_litter"] for c in counts.values())}
    return summary, rep


# ---------------------------------------------------------------- output
def write_reports(out_dir: Path, summary: dict, rep: Report) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {**summary, "errors": rep.count("ERROR"), "warnings": rep.count("WARNING"), "infos": rep.count("INFO"),
               "issue_counts": dict(Counter(f"{i['severity']}:{i['type']}" for i in rep.issues))}
    (out_dir / "validation_report.json").write_text(json.dumps({**summary, "issues": rep.issues[:5000]}, indent=1,
                                                               default=sorted), encoding="utf-8")
    with open(out_dir / "files_to_fix.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["severity", "type", "image", "detail", "action"])
        w.writeheader()
        w.writerows(i for i in rep.issues if i["severity"] in ("ERROR", "WARNING"))
    md = [f"# Validation report: {Path(summary['path']).name} ({summary['mode']})", "",
          f"**{summary['errors']} error(s), {summary['warnings']} warning(s), {summary['infos']} info**", ""]
    if summary["mode"] == "workspace":
        md += [f"- Images listed: {summary['images_listed']}",
               f"- Verified (reviewed by a person, label file present, not overlay-flagged): "
               f"**{summary['verified_images']}**",
               f"- Not verified yet: **{summary['unverified_images']}**",
               f"- Empty-label images (reviewed: no litter): {summary['empty_label_images']}", "",
               "| review_status | images |", "|---|---|"]
        md += [f"| {k} | {v} |" for k, v in sorted(summary["review_status"].items())]
        md += ["", "| verified category | images |", "|---|---|"]
        md += [f"| {k} | {v} |" for k, v in sorted(summary["verified_by_category"].items())] or ["| (none) | 0 |"]
        md += ["", "Split preview of verified images:", "", "```",
               json.dumps(summary["split_preview"], indent=1, default=sorted), "```"]
    else:
        md += ["| split | images | with litter | without litter |", "|---|---|---|---|"]
        md += [f"| {s} | {c.get('images', 0)} | {c.get('with_litter', 0)} | {c.get('without_litter', 0)} |"
               for s, c in summary["counts"].items()]
    md += ["", "| issue | count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(summary["issue_counts"].items())]
    md += ["", "Files to fix: `files_to_fix.csv` (one row per problem, with the action to take).", ""]
    md += [f"- {i['severity']} {i['type']} `{i['image']}`: {i['detail']}"
           for i in rep.issues if i["severity"] != "INFO"][:80]
    p = out_dir / "validation_report.md"
    p.write_text("\n".join(md) + "\n", encoding="utf-8")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="dataset/ldm_v4_workspace or a built dataset folder")
    ap.add_argument("--dup-bits", type=int, default=6, help="dHash distance (of 256) reported as near-duplicate")
    a = ap.parse_args(argv)
    path = (a.path if a.path.is_absolute() else PROJECT_ROOT / a.path).resolve()
    try:
        if not (path / "data.yaml").is_file():
            raise DatasetPrepError(f"{path}: no data.yaml")
        if (path / "manifest.csv").is_file():
            summary, rep = validate_workspace(path, a.dup_bits)
            out_dir = path / "reports"
        else:
            summary, rep = validate_dataset(path, a.dup_bits)
            out_dir = path
    except DatasetPrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    md = write_reports(out_dir, summary, rep)
    text = md.read_text(encoding="utf-8").splitlines()
    print("\n".join(text[:45]))
    print(f"\nreport: {md}")
    return 1 if rep.count("ERROR") else 0


if __name__ == "__main__":
    sys.exit(main())
