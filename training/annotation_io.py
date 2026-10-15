"""Hand images to an external annotation tool (CVAT, Label Studio, LabelImg) and bring YOLO labels back safely.

    python -m training.annotation_io export --batch real_priority1          # images still to annotate, real CCTV first
    python -m training.annotation_io import <cvat_export.zip or folder>      # non-empty labels only
    python -m training.annotation_io import <export> --empty-means-reviewed  # ...and empty ones (read the warning!)

export writes dataset/ldm_v4_workspace/annotation_batches/<batch>/ with
    images/            copies of the images to label (git-ignored)
    obj.names          class names in model order (LabelImg / Label Studio "classes.txt" style)
    cvat_labels.json   paste into CVAT -> Project/Task -> Labels -> Raw: creates the 10 labels in the right ORDER
    batch.csv          which workspace images are in the batch and why
By default it takes unreviewed images, skips frames flagged for burned-in boxes, and puts real CCTV before synthetic.

import reads a YOLO 1.1 export (CVAT: obj.names + obj_train_data/*.txt; Label Studio / LabelImg: classes.txt +
*.txt). It refuses an export whose class list differs from the model's, checks every line, and copies valid label
files into labels/train/ (an existing different label is kept as .txt.bak, and only replaced with --overwrite).

!! CVAT and Label Studio write an EMPTY .txt for every image in the task, including images nobody opened. An empty
file therefore does NOT prove "reviewed, no litter". Empty files are skipped unless you pass --empty-means-reviewed,
which you should only do after every image of that task was looked at (CVAT job state "completed").
After importing, run `python -m training.ldm_review sync` to record the review state.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from training.ldm_common import (DEFAULT_WORKSPACE, DatasetPrepError, label_path, model_classes, read_manifest,
                                 safe_name, write_manifest)

NAMES_FILES = ("obj.names", "classes.txt")
NOT_LABELS = {"train.txt", "valid.txt", "test.txt", "obj.data", "classes.txt", "obj.names", "notes.txt", "readme.txt"}


def parse_label_text(text: str, n_classes: int) -> tuple[list[str], list[str]]:
    """(normalized lines, problems) for one YOLO label file. Boxes must lie inside the image."""
    lines, probs = [], []
    for i, ln in enumerate(text.splitlines(), 1):
        p = ln.split()
        if not p:
            continue
        if len(p) != 5:
            probs.append(f"line {i}: expected 5 values (class cx cy w h), got {len(p)}")
            continue
        try:
            c_f = float(p[0])
            cx, cy, w, h = map(float, p[1:])
        except ValueError:
            probs.append(f"line {i}: non-numeric value")
            continue
        if c_f != int(c_f) or not 0 <= int(c_f) < n_classes:
            probs.append(f"line {i}: class id {p[0]} not in 0..{n_classes - 1}")
            continue
        if w <= 0 or h <= 0:
            probs.append(f"line {i}: zero or negative box size")
            continue
        if cx - w / 2 < -0.002 or cy - h / 2 < -0.002 or cx + w / 2 > 1.002 or cy + h / 2 > 1.002:
            probs.append(f"line {i}: box extends outside the image")
            continue
        lines.append(f"{int(c_f)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines, probs


# ---------------------------------------------------------------- export
def export(ws: Path, batch: str, include_synthetic: bool, include_overlay: bool, images: list[str] | None,
           limit: int | None) -> int:
    classes = model_classes()
    rows = read_manifest(ws)
    if images:
        wanted = set(images)
        sel = [r for r in rows if r["image"] in wanted]
        missing = wanted - {r["image"] for r in sel}
        if missing:
            raise DatasetPrepError(f"not in manifest: {sorted(missing)[:10]}")
    else:
        sel = [r for r in rows if (r["review_status"] or "unreviewed") == "unreviewed"
               and (include_overlay or r["overlay_suspect"] != "yes")
               and (include_synthetic or r["source_type"] != "synthetic")]
        sel.sort(key=lambda r: (r["source_type"] == "synthetic", r["session"], r["image"]))
    if limit:
        sel = sel[:limit]
    if not sel:
        print("nothing to export (no matching unreviewed images)")
        return 0
    out = ws / "annotation_batches" / safe_name(batch)
    if out.exists():
        raise DatasetPrepError(f"{out} already exists - choose another --batch name")
    (out / "images").mkdir(parents=True)
    for r in sel:
        shutil.copy2(ws / "images" / "train" / r["image"], out / "images" / r["image"])
    (out / "obj.names").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (out / "cvat_labels.json").write_text(json.dumps([{"name": c, "attributes": []} for c in classes], indent=1),
                                          encoding="utf-8")
    with open(out / "batch.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "session", "source_type", "overlay_suspect", "notes"])
        for r in sel:
            w.writerow([r["image"], r["session"], r["source_type"], r["overlay_suspect"], r["notes"]])
    (out / "README.txt").write_text(
        f"Annotation batch '{batch}' - {len(sel)} images, exported {dt.datetime.now():%Y-%m-%d %H:%M}.\n"
        "CVAT: create a task, Labels -> Raw -> paste cvat_labels.json (keeps the class ORDER), upload images/*.\n"
        "Annotate following docs/LDM_ANNOTATION_GUIDE.md. Mark the job 'completed' only when every image was checked.\n"
        "Export: 'YOLO 1.1', then: python -m training.annotation_io import <export.zip>\n", encoding="utf-8")
    print(f"exported {len(sel)} images to {out}")
    return 0


# ---------------------------------------------------------------- import
def _open_export(src: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if src.is_dir():
        return src, None
    if src.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(src) as z:
            for m in z.namelist():
                if Path(m).is_absolute() or ".." in Path(m).parts:
                    raise DatasetPrepError(f"unsafe path in zip: {m}")
            z.extractall(tmp.name)
        return Path(tmp.name), tmp
    raise DatasetPrepError(f"{src}: expected a folder or a .zip export")


def import_labels(ws: Path, src: Path, empty_means_reviewed: bool, overwrite: bool, dry_run: bool) -> int:
    classes = model_classes()
    root, tmp = _open_export(src)
    try:
        names_file = next((p for n in NAMES_FILES for p in root.rglob(n)), None)
        if names_file is None:
            raise DatasetPrepError("no obj.names / classes.txt in the export: cannot verify the class order")
        names = [ln.strip() for ln in names_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if names != classes:
            raise DatasetPrepError(f"class list in export {names} differs from the model {classes}. "
                                   "Recreate the labels from cvat_labels.json (same names, same order).")
        rows = read_manifest(ws)
        by_stem = {Path(r["image"]).stem: r for r in rows}
        res = {k: [] for k in ("imported", "unchanged", "empty_skipped", "empty_imported", "not_in_workspace",
                               "invalid", "kept_existing")}
        stamp = f"labels imported from {src.name} {dt.date.today().isoformat()}"
        for lf in sorted(root.rglob("*.txt")):
            if lf.name.lower() in NOT_LABELS:
                continue
            r = by_stem.get(lf.stem)
            if r is None:
                res["not_in_workspace"].append(lf.name)
                continue
            lines, probs = parse_label_text(lf.read_text(encoding="utf-8", errors="replace"), len(classes))
            if probs:
                res["invalid"].append(f"{lf.name}: {'; '.join(probs[:3])}")
                continue
            if not lines and not empty_means_reviewed:
                res["empty_skipped"].append(r["image"])
                continue
            new = "".join(x + "\n" for x in lines)
            dst = label_path(ws, r["image"])
            if dst.is_file():
                old = dst.read_text(encoding="utf-8", errors="replace")
                if old.split() == new.split():
                    res["unchanged"].append(r["image"])
                    continue
                if not overwrite:
                    res["kept_existing"].append(f"{r['image']}: differs from the existing label (use --overwrite)")
                    continue
                if not dry_run:
                    shutil.copy2(dst, dst.with_suffix(".txt.bak"))
            if not dry_run:
                dst.write_text(new, encoding="utf-8")
                r["notes"] = "; ".join(x for x in (r["notes"], stamp) if x)
            res["imported" if lines else "empty_imported"].append(r["image"])
        if not dry_run:
            write_manifest(ws, rows)
    finally:
        if tmp is not None:
            tmp.cleanup()
    for k, v in res.items():
        print(f"{k:18s} {len(v)}")
    for k in ("invalid", "kept_existing", "not_in_workspace"):
        for v in res[k][:30]:
            print(f"  {k}: {v}")
    if res["empty_skipped"]:
        print(f"NOTE: {len(res['empty_skipped'])} empty label files were NOT treated as 'reviewed, no litter'. If every "
              "image in that task was checked, re-run with --empty-means-reviewed (or save them in the Ground Truth page).")
    print("dry run: nothing written" if dry_run else "next: python -m training.ldm_review sync")
    return 1 if res["invalid"] else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="copy images to annotate into a batch folder")
    e.add_argument("--batch", required=True, help="batch name, e.g. real_priority1")
    e.add_argument("--images", nargs="*", help="specific workspace image names (default: unreviewed images)")
    e.add_argument("--include-synthetic", action="store_true")
    e.add_argument("--include-overlay", action="store_true", help="also frames flagged for burned-in boxes")
    e.add_argument("--limit", type=int)
    i = sub.add_parser("import", help="bring YOLO labels from an annotation tool export back into the workspace")
    i.add_argument("export", type=Path, help="folder or .zip (CVAT 'YOLO 1.1', Label Studio 'YOLO', LabelImg)")
    i.add_argument("--empty-means-reviewed", action="store_true",
                   help="treat empty label files as reviewed 'no litter' (only if every image was checked)")
    i.add_argument("--overwrite", action="store_true", help="replace differing existing labels (kept as .txt.bak)")
    i.add_argument("--dry-run", action="store_true")
    for p in (e, i):
        p.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    a = ap.parse_args(argv)
    try:
        if not (a.workspace / "manifest.csv").is_file():
            raise DatasetPrepError(f"no workspace at {a.workspace}")
        if a.cmd == "export":
            return export(a.workspace, a.batch, a.include_synthetic, a.include_overlay, a.images, a.limit)
        return import_labels(a.workspace, a.export, a.empty_means_reviewed, a.overwrite, a.dry_run)
    except DatasetPrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
