"""Copy existing images (screenshots, NVR exports, scheduled captures) into the annotation workspace.

    python -m training.import_images "C:/SHREE/DATASET/CCTV_Litter_100_Images (2)/CCTV_Litter_100_Images" \
        --source-type synthetic --session street_aug100 --used-in-previous-training likely
    python -m training.import_images storage/captures --source-type real_cctv

Sources are COPIED, never moved or modified. For each file:
  * unreadable / corrupt          -> reported, not copied
  * exact duplicate (same bytes)  -> skipped (also against images already in the workspace)
  * near-duplicate (dHash)        -> skipped unless --keep-near-duplicates (reported either way)
  * very small (< 320 px side)    -> copied, noted in the manifest
  * burned-in analytics boxes     -> copied, flagged overlay_suspect=yes (excluded from the build until cleared)
  * the app's annotated outputs (storage/alerts, storage/videos, annotated*.jpg, *_crop.jpg) -> refused

Scheduled captures (storage/captures/<camera>/<YYYYmmdd_HHMMSS_id>/original.jpg) are recognised automatically:
camera, session (<camera>_<date>) and capture time are filled in; only original.jpg is taken.
--source-type is required on purpose: real CCTV and synthetic images must never be mixed up.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

from training.ldm_common import (CATEGORIES, CONDITIONS, DEFAULT_WORKSPACE, IMAGE_EXTS, SOURCE_TYPES, YES_NO,
                                 DatasetPrepError, dhash, ensure_workspace, hamming, is_project_overlay_output,
                                 model_classes, now_iso, overlay_suspect, read_image, read_manifest, safe_name,
                                 sha256_file, split_tags, write_manifest)


def capture_info(p: Path) -> dict[str, str] | None:
    """camera/session/time for storage/captures/<camera>/<YYYYmmdd_HHMMSS_xxxxxx>/original.jpg."""
    if p.name.lower() != "original.jpg" or p.parent.parent.parent.name.lower() != "captures":
        return None
    cam, cap_id = p.parent.parent.name, p.parent.name
    try:
        t = dt.datetime.strptime(cap_id[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return {"camera": cam, "session": f"{cam}_{cap_id[:8]}", "recorded_at": "", "stem": cap_id}
    return {"camera": cam, "session": f"{cam}_{t:%Y%m%d}", "recorded_at": t.isoformat(timespec="seconds"), "stem": cap_id}


def _files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            out.append(p)
        else:
            raise DatasetPrepError(f"not found: {p}")
    # inside a capture folder only the original frame is training material
    return [q for q in out if not (q.parent.parent.parent.name.lower() == "captures" and q.name.lower() != "original.jpg")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="image files or folders (searched recursively)")
    ap.add_argument("--source-type", required=True, choices=SOURCE_TYPES)
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    ap.add_argument("--session", help="session name for all files (default: the file's parent folder name)")
    ap.add_argument("--camera")
    ap.add_argument("--category", choices=CATEGORIES, default="unknown",
                    help="HINT only (e.g. a folder of shelf photos = hard_negative); images still need review")
    ap.add_argument("--conditions", help=f"';'-separated tags: {', '.join(CONDITIONS)}")
    ap.add_argument("--used-in-previous-training", choices=YES_NO, default="unknown")
    ap.add_argument("--split-lock", choices=("train", "val", "test"))
    ap.add_argument("--near-dup-bits", type=int, default=4, help="dHash distance treated as near-identical (default 4)")
    ap.add_argument("--keep-near-duplicates", action="store_true")
    ap.add_argument("--notes")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    report: dict[str, list] = {k: [] for k in ("imported", "corrupt", "exact_duplicate", "near_duplicate",
                                               "refused_overlay_output", "overlay_suspect", "small")}
    try:
        bad = [t for t in split_tags(a.conditions or "") if t not in CONDITIONS]
        if bad:
            raise DatasetPrepError(f"unknown condition tag(s) {bad}; allowed: {', '.join(CONDITIONS)}")
        ws = ensure_workspace(a.workspace, model_classes())
        rows = read_manifest(ws)
        by_sha = {r["sha256"]: r["image"] for r in rows if r["sha256"] and not r["frame_index"]}
        hashes = [(r["image"], r["dhash"]) for r in rows if r["dhash"]]
        names = {r["image"] for r in rows}
        for src in _files(a.sources):
            if is_project_overlay_output(src):
                report["refused_overlay_output"].append(str(src))
                continue
            img = read_image(src)
            if img is None:
                report["corrupt"].append(str(src))
                continue
            digest = sha256_file(src)
            if digest in by_sha:
                report["exact_duplicate"].append(f"{src} == {by_sha[digest]}")
                continue
            h = dhash(img)
            near = next((n for n, hh in hashes if hamming(h, hh) <= a.near_dup_bits), None)
            if near:
                report["near_duplicate"].append(f"{src} ~ {near}")
                if not a.keep_near_duplicates:
                    continue
            cap = capture_info(src)
            session = safe_name(a.session or (cap["session"] if cap else src.parent.name))
            stem = safe_name(cap["stem"] if cap else src.stem)
            name = f"{session}__{stem}{src.suffix.lower()}"
            i = 1
            while name in names:
                name = f"{session}__{stem}_{i}{src.suffix.lower()}"
                i += 1
            hh, ww = img.shape[:2]
            notes = [a.notes] if a.notes else []
            if min(hh, ww) < 320:
                notes.append(f"small image {ww}x{hh}")
                report["small"].append(name)
            if near:
                notes.append(f"near-duplicate of {near}")
            suspect = overlay_suspect(img)
            if suspect:
                report["overlay_suspect"].append(name)
            if not a.dry_run:
                shutil.copy2(src, ws / "images" / "train" / name)
            rows.append({
                "image": name, "source_type": a.source_type, "source_path": str(src.resolve()), "session": session,
                "camera": a.camera or (cap["camera"] if cap else ""), "frame_index": "", "timestamp_sec": "",
                "recorded_at": cap["recorded_at"] if cap else "", "width": str(ww), "height": str(hh),
                "sha256": digest, "dhash": h, "overlay_suspect": "yes" if suspect else "no",
                "used_in_previous_training": a.used_in_previous_training, "category": a.category, "subcategory": "",
                "conditions": ";".join(split_tags(a.conditions or "")), "review_status": "unreviewed",
                "split_lock": a.split_lock or "", "notes": "; ".join(notes), "added_at": now_iso(),
            })
            names.add(name)
            by_sha[digest] = name
            hashes.append((name, h))
            report["imported"].append(name)
        if not a.dry_run:
            write_manifest(ws, rows)
            out = ws / "reports" / f"import_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
            out.write_text(json.dumps({"sources": a.sources, "source_type": a.source_type, **report}, indent=1),
                           encoding="utf-8")
    except DatasetPrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for k, v in report.items():
        print(f"{k:24s} {len(v)}")
    for k in ("corrupt", "refused_overlay_output"):
        for v in report[k][:20]:
            print(f"  {k}: {v}")
    print("dry run: nothing written" if a.dry_run else f"workspace: {a.workspace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
