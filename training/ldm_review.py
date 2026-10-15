"""Review bookkeeping and the dataset gap report for the annotation workspace.

    python -m training.ldm_review sync          # record what was annotated in the Ground Truth page
    python -m training.ldm_review mark-empty    # write EMPTY label files for images a human marked reviewed_empty
    python -m training.ldm_review mark-empty --images a.jpg b.jpg --category clean   # ...or name them directly
    python -m training.ldm_review report        # gap report + lists of images that still need a human

sync (never invents labels, only reads what a human saved):
  label file with boxes          + status unreviewed -> annotated   (category unknown -> litter)
  label file saved with 0 boxes  + status unreviewed -> reviewed_empty (category stays for you to choose:
                                                        hard_negative or clean)
  anything contradictory (e.g. reviewed_empty but the label has boxes) is REPORTED, not changed.

mark-empty only writes a label for an image that a person has explicitly reviewed as containing no litter.
An unreviewed image never gets an empty label: "no label" and "no litter" are different statements.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from training.ldm_common import (CATEGORIES, DEFAULT_WORKSPACE, SUBCATEGORIES, DatasetPrepError,
                                 label_path, model_classes, read_label_lines, read_manifest, split_tags,
                                 validate_row_vocab, write_manifest)

# Minimums for a first CCTV fine-tune of v3 (guidance, not magic numbers: more real, varied data is always better)
TARGETS = {
    "litter_images": 300,
    "hard_negative_images": 200,
    "clean_images": 100,
    "boxes_per_class": 50,
    "small_boxes": 150,                  # < 32x32 px in the original frame: distant litter
    "images_per_hard_negative_type": 20,
    "images_per_condition": 30,
    "sessions": 8,                       # independent recordings: needed for honest val/test splits
}
KEY_CONDITIONS = {"day": ("day",), "dusk": ("dusk",), "night / IR": ("night", "ir"), "rain / wet ground": ("rain_wet",),
                  "glare / harsh shadow": ("glare", "harsh_shadow")}


def _boxes(lines: list[list[str]]) -> list[tuple[int, float, float]]:
    out = []
    for p in lines:
        try:
            out.append((int(float(p[0])), float(p[3]), float(p[4])))
        except (ValueError, IndexError):
            out.append((-1, 0.0, 0.0))
    return out


def sync(ws: Path) -> int:
    rows = read_manifest(ws)
    changed, problems = Counter(), []
    names = {r["image"] for r in rows}
    for r in rows:
        problems += [f"{r['image']}: {p}" for p in validate_row_vocab(r)]
        if not (ws / "images" / "train" / r["image"]).is_file():
            problems.append(f"{r['image']}: image file missing from images/train")
            continue
        lp = label_path(ws, r["image"])
        if not lp.is_file():
            if r["review_status"] in ("annotated", "reviewed_empty"):
                problems.append(f"{r['image']}: status '{r['review_status']}' but no label file "
                                f"(run 'mark-empty' or annotate it)")
            continue
        n = len(read_label_lines(lp))
        st = r["review_status"] or "unreviewed"
        if n and st == "unreviewed":
            r["review_status"] = "annotated"
            changed["unreviewed -> annotated"] += 1
            if r["category"] in ("", "unknown"):
                r["category"] = "litter"
                changed["category unknown -> litter"] += 1
        elif not n and st == "unreviewed":
            r["review_status"] = "reviewed_empty"
            changed["unreviewed -> reviewed_empty"] += 1
        elif n and st == "reviewed_empty":
            problems.append(f"{r['image']}: marked reviewed_empty but its label has {n} box(es) - fix one of them")
        elif not n and st == "annotated":
            problems.append(f"{r['image']}: marked annotated but its label file is empty")
        if n and r["category"] in ("hard_negative", "clean"):
            problems.append(f"{r['image']}: category '{r['category']}' but it has litter boxes")
    for lf in sorted((ws / "labels" / "train").glob("*.txt")):
        if not any(Path(n).stem == lf.stem for n in names):
            problems.append(f"{lf.name}: label file without a manifest row")
    write_manifest(ws, rows)
    for k, v in changed.items():
        print(f"{k}: {v}")
    if not changed:
        print("no status changes")
    for p in problems[:200]:
        print(f"PROBLEM {p}")
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


def mark_empty(ws: Path, images: list[str] | None, category: str | None) -> int:
    rows = read_manifest(ws)
    by_name = {r["image"]: r for r in rows}
    if images:
        missing = [i for i in images if i not in by_name]
        if missing:
            print(f"ERROR: not in manifest: {missing}", file=sys.stderr)
            return 2
        targets = [by_name[i] for i in images]
        for r in targets:
            r["review_status"] = "reviewed_empty"
            if category:
                r["category"] = category
    else:
        targets = [r for r in rows if r["review_status"] == "reviewed_empty"]
    written, refused = 0, []
    for r in targets:
        lp = label_path(ws, r["image"])
        if lp.is_file():
            if read_label_lines(lp):
                refused.append(f"{r['image']}: label already has boxes - not emptied (fix the review status instead)")
                r["review_status"] = "annotated"
            continue
        lp.write_text("", encoding="utf-8")
        written += 1
    write_manifest(ws, rows)
    print(f"empty label files written: {written}")
    for x in refused:
        print(f"REFUSED {x}")
    need_cat = [r["image"] for r in targets if r["category"] in ("", "unknown")]
    if need_cat:
        print(f"{len(need_cat)} of these still need category hard_negative or clean in manifest.csv")
    return 0


def report(ws: Path, classes: list[str]) -> dict:
    rows = read_manifest(ws)
    status, cats, srcs, sess, cams = Counter(), Counter(), Counter(), Counter(), Counter()
    subs, conds = Counter(), Counter()
    obj, small = Counter(), 0
    usable, needs_annot, ambiguous, needs_cat, overlay, vocab = [], [], [], [], [], []
    usable_by = defaultdict(Counter)
    for r in rows:
        st = r["review_status"] or "unreviewed"
        status[st] += 1
        srcs[r["source_type"]] += 1
        vocab += [f"{r['image']}: {p}" for p in validate_row_vocab(r)]
        lines = read_label_lines(label_path(ws, r["image"]))
        is_usable = st in ("annotated", "reviewed_empty") and r["overlay_suspect"] != "yes"
        if r["overlay_suspect"] == "yes":
            overlay.append(r["image"])
        if st == "unreviewed":
            needs_annot.append(r["image"])
        elif st == "ambiguous":
            ambiguous.append(f"{r['image']}\t{r['notes']}")
        if st == "reviewed_empty" and r["category"] in ("", "unknown"):
            needs_cat.append(r["image"])
        if not is_usable:
            continue
        usable.append(r["image"])
        cats[r["category"] or "unknown"] += 1
        sess[r["session"]] += 1
        cams[r["camera"] or "(none)"] += 1
        usable_by[r["source_type"]][r["category"] or "unknown"] += 1
        for t in split_tags(r["subcategory"]):
            subs[t] += 1
        for t in split_tags(r["conditions"]):
            conds[t] += 1
        w, h = float(r["width"] or 0), float(r["height"] or 0)
        for c, bw, bh in _boxes(lines):
            if 0 <= c < len(classes):
                obj[classes[c]] += 1
                if w and bw * w * bh * h < 32 * 32:
                    small += 1
    real_usable = sum(usable_by["real_cctv"].values()) + sum(usable_by["other_real"].values())
    gaps = []

    def need(label: str, have: int, target: int) -> None:
        if have < target:
            gaps.append({"item": label, "have": have, "target": target, "missing": target - have})

    need("genuine litter images (real)", usable_by["real_cctv"]["litter"] + usable_by["other_real"]["litter"],
         TARGETS["litter_images"])
    need("hard-negative images (real)", usable_by["real_cctv"]["hard_negative"] + usable_by["other_real"]["hard_negative"],
         TARGETS["hard_negative_images"])
    need("clean-scene images (real)", usable_by["real_cctv"]["clean"] + usable_by["other_real"]["clean"],
         TARGETS["clean_images"])
    for c in classes:
        need(f"boxes of class '{c}'", obj[c], TARGETS["boxes_per_class"])
    need("small / distant litter boxes (< 32x32 px)", small, TARGETS["small_boxes"])
    for t in SUBCATEGORIES["hard_negative"]:
        need(f"hard negative: {t}", subs[t], TARGETS["images_per_hard_negative_type"])
    for label, tags in KEY_CONDITIONS.items():
        need(f"condition: {label}", sum(conds[t] for t in tags), TARGETS["images_per_condition"])
    need("independent recording sessions", len(sess), TARGETS["sessions"])

    out = {
        "workspace": str(ws), "images_listed": len(rows), "usable_images": len(usable), "usable_real": real_usable,
        "review_status": dict(status), "source_type_all": dict(srcs),
        "usable_by_source_and_category": {k: dict(v) for k, v in usable_by.items()},
        "usable_by_category": dict(cats), "sessions": dict(sess), "cameras": dict(cams),
        "hard_negative_and_litter_tags": dict(subs), "conditions": dict(conds),
        "boxes_per_class": {c: obj[c] for c in classes}, "small_boxes": small,
        "overlay_suspect": len(overlay), "needs_annotation": len(needs_annot), "ambiguous": len(ambiguous),
        "needs_category": len(needs_cat), "manifest_value_problems": vocab[:200], "gaps": gaps, "targets": TARGETS,
    }
    rep = ws / "reports"
    rep.mkdir(exist_ok=True)
    (rep / "gap_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (rep / "needs_annotation.txt").write_text("\n".join(needs_annot) + ("\n" if needs_annot else ""), encoding="utf-8")
    (rep / "ambiguous.txt").write_text("\n".join(ambiguous) + ("\n" if ambiguous else ""), encoding="utf-8")
    (rep / "needs_category.txt").write_text("\n".join(needs_cat) + ("\n" if needs_cat else ""), encoding="utf-8")
    (rep / "overlay_suspect.txt").write_text("\n".join(overlay) + ("\n" if overlay else ""), encoding="utf-8")
    (rep / "gap_report.md").write_text(_markdown(out), encoding="utf-8")
    write_queue(rows, rep / "annotation_queue.csv")
    return out


def write_queue(rows: list[dict[str, str]], path: Path) -> None:
    """Work list for annotators: what to open next, in order, and why."""
    q = []
    for r in rows:
        st = r["review_status"] or "unreviewed"
        if st == "ambiguous":
            q.append((3, r, "ambiguous: needs a decision (box it, or reviewed_empty, or rejected)"))
        elif st != "unreviewed":
            continue
        elif r["overlay_suspect"] == "yes":
            q.append((9, r, "flagged for burned-in boxes: skip, or check and set overlay_suspect=no"))
        elif r["source_type"] == "synthetic":
            q.append((2, r, "synthetic: train only, annotate after the real frames"))
        else:
            q.append((1, r, "real CCTV: annotate first"))
    q.sort(key=lambda x: (x[0], x[1]["session"], x[1]["image"]))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["priority", "image", "session", "source_type", "overlay_suspect", "reason"])
        for pr, r, why in q:
            w.writerow([pr, r["image"], r["session"], r["source_type"], r["overlay_suspect"], why])


def _table(d: dict, k1: str = "value", k2: str = "images") -> str:
    if not d:
        return "_none_\n"
    return f"| {k1} | {k2} |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in sorted(d.items(), key=lambda x: -x[1]))


def _markdown(o: dict) -> str:
    s = ["# Dataset gap report\n", f"Workspace: `{o['workspace']}`\n",
         f"- Images listed: **{o['images_listed']}**",
         f"- Usable now (reviewed, no overlay): **{o['usable_images']}** (real CCTV: {o['usable_real']})",
         f"- Still need annotation / review: **{o['needs_annotation']}** (see `needs_annotation.txt`)",
         f"- Ambiguous, waiting for a decision: **{o['ambiguous']}** (see `ambiguous.txt`)",
         f"- Reviewed empty but no category yet: **{o['needs_category']}** (see `needs_category.txt`)",
         f"- Flagged for burned-in boxes: **{o['overlay_suspect']}** (see `overlay_suspect.txt`)\n",
         "## Review status (all images)\n", _table(o["review_status"], "status"),
         "## Source type (all images)\n", _table(o["source_type_all"], "source"),
         "## Usable images by category\n", _table(o["usable_by_category"], "category"),
         "## Boxes per class (usable images)\n", _table(o["boxes_per_class"], "class", "boxes"),
         f"Small boxes (< 32x32 px): **{o['small_boxes']}**\n",
         "## Sessions (usable images)\n", _table(o["sessions"], "session"),
         "## Conditions tagged (usable images)\n", _table(o["conditions"], "condition"),
         "## Gaps against the first fine-tune targets\n"]
    if o["gaps"]:
        s.append("| needed | have | target | missing |\n|---|---|---|---|\n"
                 + "".join(f"| {g['item']} | {g['have']} | {g['target']} | {g['missing']} |\n" for g in o["gaps"]))
    else:
        s.append("All targets met.\n")
    if o["manifest_value_problems"]:
        s.append("\n## manifest.csv values to fix\n" + "".join(f"- {p}\n" for p in o["manifest_value_problems"]))
    return "\n".join(s) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("sync", "mark-empty", "report"))
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    ap.add_argument("--images", nargs="*", help="mark-empty: image names reviewed as containing no litter")
    ap.add_argument("--category", choices=[c for c in CATEGORIES if c in ("hard_negative", "clean")],
                    help="mark-empty --images: also set their category")
    a = ap.parse_args(argv)
    if not (a.workspace / "manifest.csv").is_file():
        print(f"ERROR: no workspace at {a.workspace} (run extract_frames or import_images first)", file=sys.stderr)
        return 2
    try:
        if a.command == "sync":
            return sync(a.workspace)
        if a.command == "mark-empty":
            return mark_empty(a.workspace, a.images, a.category)
        o = report(a.workspace, model_classes())
    except DatasetPrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print((a.workspace / "reports" / "gap_report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
