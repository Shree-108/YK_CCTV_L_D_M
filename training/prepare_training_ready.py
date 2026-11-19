"""Build LDM_TRAINING_READY: a scene-split, audited YOLO dataset from the AUG100 images (CCTV_Litter_100_Images).

    python -m training.prepare_training_ready \
        --source "C:\\SHREE\\DATASET\\CCTV_Litter_100_Images (2)\\CCTV_Litter_100_Images" \
        --out "C:\\SHREE\\DATASET\\LDM_TRAINING_READY"

Facts this script is built on (checked, not assumed - see dataset_summary.csv of every build):
  * The source folder holds 100 JPEGs and NO annotation files. Its README says they are synthetic/augmented variants
    of TWO source scenes. ``archive/spotgarbage-GINI-master`` is a different dataset (web photos, image-level labels,
    one coarse box per image at most): not CCTV and not compatible, so it is audited and excluded.
  * The only box labels for these 100 images are the class-0 ``litter`` labels in ``dataset/exp_aug100``
    (labels/train/*.txt + annotations/per_image/*.json + manifest.csv). They are used only when the SHA-256 of every
    source image matches the image those labels were made for (dataset/exp_aug100/source_sha256.txt).
  * One class (``litter``) cannot be mapped onto the ten v3_store_cctv classes without inventing labels, so this
    dataset is for a SEPARATE one-class model. v3_store_cctv is never modified.

Split (scene-based, so no image of the test scene - or a transform of it - is ever seen in training):
  test  = every image of ``--test-scene`` (default B_roadside_bin, 50 images): an independent scene.
  train = the other scene minus ``--val-count`` images; val = those images (same scene as train: used only for
          checkpoint selection / early stopping; its metrics are optimistic by construction).
  70/20/10 is NOT used: with two scenes, any random 70/20/10 split puts transforms of the same scene in train and
  test (leakage). The limitation is written to split_summary.csv and README.md.

Safety: the source folder and dataset/exp_aug100 are only read (source hashes are checked before and after);
the output folder must not exist yet (nothing is ever overwritten); v3_store_cctv/best.pt is hash-checked.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(r"C:\SHREE\DATASET\CCTV_Litter_100_Images (2)\CCTV_Litter_100_Images")
DEFAULT_OUT = Path(r"C:\SHREE\DATASET\LDM_TRAINING_READY")
DEFAULT_LABELS = PROJECT_ROOT / "dataset" / "exp_aug100"
V3_WEIGHTS = PROJECT_ROOT / "models" / "litter_yolo" / "v3_store_cctv" / "best.pt"
CLASS_NAMES = ["litter"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
TINY_PX = 16          # boxes whose shorter side is below this (original pixels) are reported as tiny
NEAR_DUP_BITS = 6     # dHash Hamming distance at or below which two images are near-duplicates


class PrepError(Exception):
    pass


# ------------------------------------------------------------------------------------------------ helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(img: np.ndarray, size: int = 8) -> int:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    s = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Optional[list[str]] = None) -> None:
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def folder_hashes(folder: Path) -> dict[str, str]:
    return {p.name: sha256(p) for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def parse_yolo(path: Path, w: int, h: int) -> tuple[list[list[float]], list[str]]:
    """Boxes [cls, cx, cy, bw, bh] and the problems found (never corrected silently)."""
    boxes, issues = [], []
    if not path.is_file():
        return boxes, ["label_file_missing"]
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(f"line{n}:not_5_fields")
            continue
        try:
            c, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:])
        except ValueError:
            issues.append(f"line{n}:not_numeric")
            continue
        if c < 0 or c >= len(CLASS_NAMES):
            issues.append(f"line{n}:class_id_{c}_out_of_range")
        if bw <= 0 or bh <= 0:
            issues.append(f"line{n}:zero_or_negative_size")
        if not (0 <= cx - bw / 2 + 1e-6 and cx + bw / 2 <= 1 + 1e-6 and 0 <= cy - bh / 2 + 1e-6
                and cy + bh / 2 <= 1 + 1e-6):
            issues.append(f"line{n}:outside_image")
        boxes.append([c, cx, cy, bw, bh])
    seen = Counter(tuple(round(v, 5) for v in b) for b in boxes)
    issues += [f"duplicate_box_x{k}" for b, k in seen.items() if k > 1]
    return boxes, issues


# ------------------------------------------------------------------------------------------------ build
def audit_archive(source: Path) -> dict[str, Any]:
    arch = source / "archive"
    if not arch.is_dir():
        return {"present": False}
    files = [p for p in arch.rglob("*") if p.is_file()]
    imgs = [p for p in files if p.suffix.lower() in IMAGE_EXTS | {".gif", ".tif", ".jpe"}]
    return {"present": True, "files": len(files), "image_like_files": len(imgs),
            "csv_files": sum(1 for p in files if p.suffix.lower() == ".csv"),
            "decision": "EXCLUDED: SpotGarbage/GINI web photos with image-level labels (at most one coarse box per "
                        "image); not CCTV, different label semantics, contains nested duplicate copy of itself"}


def build(source: Path, labels_root: Path, out: Path, test_scene: str = "B_roadside_bin", val_count: int = 10,
          seed: int = 0, make_zip: bool = True) -> dict[str, Any]:
    source, labels_root, out = Path(source), Path(labels_root), Path(out)
    if not source.is_dir():
        raise PrepError(f"source folder not found: {source}")
    if out.exists() and any(out.iterdir()):
        raise PrepError(f"{out} already exists and is not empty: refusing to overwrite. Choose a new --out.")
    manifest_path = labels_root / "manifest.csv"
    if not manifest_path.is_file():
        raise PrepError(f"no label manifest at {manifest_path}: the source images have no annotations of their own")
    v3_before = sha256(V3_WEIGHTS) if V3_WEIGHTS.is_file() else None
    src_before = folder_hashes(source)
    if not src_before:
        raise PrepError(f"no images in {source}")

    expected: dict[str, str] = {}
    sha_file = labels_root / "source_sha256.txt"
    if sha_file.is_file():
        for line in sha_file.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                digest, name = line.split(None, 1)
                expected[name.strip()] = digest
    man = {r["file"]: r for r in read_csv(manifest_path)}
    scenes = sorted({r["scene"] for r in man.values()})
    if test_scene not in scenes:
        raise PrepError(f"test scene {test_scene!r} not in the label manifest scenes {scenes}")

    # ---- per-image audit
    records: list[dict[str, Any]] = []
    for name, digest in src_before.items():
        p = source / name
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        m = man.get(name)
        rec: dict[str, Any] = {"file": name, "sha256": digest, "format": p.suffix.lower().lstrip("."),
                               "bytes": p.stat().st_size, "decodable": img is not None}
        issues: list[str] = []
        if img is None:
            issues.append("corrupt_or_undecodable")
            h = w = 0
        else:
            h, w = img.shape[:2]
            rec["dhash"] = f"{dhash(img):016x}"
        rec.update(width=w, height=h)
        if m is None:
            issues.append("no_label_record")
        if expected and expected.get(name) != digest:
            issues.append("source_hash_differs_from_labelled_image")
        if m and m.get("sha256") and m["sha256"] != digest:
            issues.append("manifest_hash_mismatch")
        lbl = labels_root / "labels" / "train" / (Path(name).stem + ".txt")
        boxes, lissues = parse_yolo(lbl, w or 1, h or 1)
        issues += lissues
        per = labels_root / "annotations" / "per_image" / (Path(name).stem + ".json")
        ignore = []
        if per.is_file():
            pj = json.loads(per.read_text(encoding="utf-8"))
            ignore = [[round(float(v), 1) for v in r["xyxy"]] for r in pj.get("ignore", []) + pj.get("manual_ignore", [])
                      if r.get("xyxy")]
        sides = [min(b[3] * w, b[4] * h) for b in boxes] if w else []
        rec.update(
            scene=(m or {}).get("scene", ""), role=(m or {}).get("role", ""),
            reference_id=(m or {}).get("reference_id", ""), label_status=(m or {}).get("label_status", ""),
            verified_by=(m or {}).get("verified_by", ""), source_type=(m or {}).get("source_type", ""),
            n_boxes=len(boxes), n_ignore_regions=len(ignore),
            boxes_lt16px=sum(1 for s in sides if s < TINY_PX),
            min_box_side_px=round(min(sides), 1) if sides else "", max_box_side_px=round(max(sides), 1) if sides else "",
            label_file=str(lbl), _boxes=boxes, _ignore=ignore, _issues=issues)
        if (m or {}).get("label_status") not in ("verified", "manual_verified"):
            issues.append("labels_not_verified")
        records.append(rec)

    # ---- duplicates / near-duplicates
    by_sha = defaultdict(list)
    for r in records:
        by_sha[r["sha256"]].append(r["file"])
    hashes = [(r["file"], int(r["dhash"], 16)) for r in records if r.get("dhash")]
    parent = {f: f for f, _ in hashes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming(hashes[i][1], hashes[j][1]) <= NEAR_DUP_BITS:
                parent[find(hashes[i][0])] = find(hashes[j][0])
    groups = defaultdict(list)
    for f, _ in hashes:
        groups[find(f)].append(f)
    gid = {f: (f"nd{k:03d}" if len(members) > 1 else "") for k, members in enumerate(sorted(groups.values()))
           for f in members}
    for r in records:
        r["exact_duplicate_of"] = ";".join(x for x in by_sha[r["sha256"]] if x != r["file"])
        r["near_duplicate_group"] = gid.get(r["file"], "")

    blocking = [r for r in records if any(i for i in r["_issues"] if i not in ("labels_not_verified",)
                                           and not i.startswith("duplicate_box"))]
    unverified = [r for r in records if "labels_not_verified" in r["_issues"]]
    if blocking or unverified:
        bad = "; ".join(f"{r['file']}: {','.join(r['_issues'])}" for r in (blocking + unverified)[:10])
        raise PrepError(f"{len(blocking)} image(s) with blocking problems, {len(unverified)} unverified: {bad}")

    # ---- split
    rng = random.Random(seed)
    train_pool = sorted(r["file"] for r in records if r["scene"] != test_scene and r["role"] != "reference")
    val = set(rng.sample(train_pool, min(val_count, len(train_pool))))
    for r in records:
        r["split"] = "test" if r["scene"] == test_scene else ("val" if r["file"] in val else "train")

    # ---- write
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    ignore_out: dict[str, list[list[float]]] = {}
    for r in records:
        shutil.copy2(source / r["file"], out / "images" / r["split"] / r["file"])
        lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in r["_boxes"]]
        (out / "labels" / r["split"] / (Path(r["file"]).stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if r["_ignore"]:
            ignore_out[r["file"]] = r["_ignore"]
        if sha256(out / "images" / r["split"] / r["file"]) != r["sha256"]:
            raise PrepError(f"copy of {r['file']} does not match its source")
    (out / "ignore_regions.json").write_text(json.dumps(
        {"_note": "Pixel xyxy regions per image that are neither litter nor background (ambiguous objects, litter "
                  "outside the labelling ROI, <50% visible). A prediction centred in one counts as neither TP nor FP.",
         "regions": ignore_out}, indent=1), encoding="utf-8")
    (out / "data.yaml").write_text(
        "# LDM_TRAINING_READY - one-class litter dataset (AUG100). Scene split: see split_summary.csv.\n"
        "# 'path' is set by the training notebook to wherever this folder is mounted.\n"
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames:\n  0: litter\n", encoding="utf-8")

    manifest_fields = ["file", "split", "scene", "role", "reference_id", "source_type", "label_status", "verified_by",
                       "sha256", "dhash", "format", "bytes", "width", "height", "n_boxes", "n_ignore_regions",
                       "boxes_lt16px", "exact_duplicate_of", "near_duplicate_group"]
    write_csv(out / "dataset_manifest.csv", [{k: r.get(k, "") for k in manifest_fields} for r in records],
              manifest_fields)
    write_csv(out / "annotation_audit.csv", [{
        "file": r["file"], "split": r["split"], "label_file_present": "label_file_missing" not in r["_issues"],
        "decodable": r["decodable"], "n_boxes": r["n_boxes"], "n_ignore_regions": r["n_ignore_regions"],
        "invalid_boxes": sum(1 for i in r["_issues"] if i.startswith("line")),
        "duplicate_boxes": sum(1 for i in r["_issues"] if i.startswith("duplicate_box")),
        "boxes_lt16px": r["boxes_lt16px"], "min_box_side_px": r["min_box_side_px"],
        "max_box_side_px": r["max_box_side_px"], "empty_label": r["n_boxes"] == 0,
        "label_status": r["label_status"], "issues": ";".join(r["_issues"]) or "none"} for r in records])

    split_counts = {s: [r for r in records if r["split"] == s] for s in SPLITS}
    write_csv(out / "class_distribution.csv", [
        {"class_id": 0, "class_name": "litter", "split": s, "images_with_class": sum(1 for r in rs if r["n_boxes"]),
         "boxes": sum(r["n_boxes"] for r in rs), "boxes_lt16px": sum(r["boxes_lt16px"] for r in rs)}
        for s, rs in split_counts.items()])
    rationale = {
        "train": "scene(s) other than the test scene, minus the validation images; includes the scene reference image",
        "val": f"{val_count} images of the TRAINING scene (seed {seed}): checkpoint selection only, optimistic metrics",
        "test": f"every image of {test_scene}: an independent scene never seen in training or validation"}
    write_csv(out / "split_summary.csv", [
        {"split": s, "images": len(rs), "scenes": ";".join(sorted({r['scene'] for r in rs})),
         "boxes": sum(r["n_boxes"] for r in rs), "empty_images": sum(1 for r in rs if not r["n_boxes"]),
         "share": round(len(rs) / len(records), 3), "rationale": rationale[s]} for s, rs in split_counts.items()])

    arch = audit_archive(source)
    sides_all = [r for r in records]
    summary = [
        ("built_at", dt.datetime.now().astimezone().isoformat(timespec="seconds")),
        ("source_folder", str(source)), ("label_source", str(labels_root)),
        ("images_total", len(records)), ("images_decodable", sum(r["decodable"] for r in records)),
        ("corrupt_images", sum(not r["decodable"] for r in records)),
        ("formats", ";".join(f"{k}:{v}" for k, v in Counter(r["format"] for r in records).items())),
        ("dimensions", ";".join(f"{k}:{v}" for k, v in Counter(f"{r['width']}x{r['height']}" for r in records).items())),
        ("annotation_files_in_source_folder", 0),
        ("annotation_format_used", "YOLO txt (class cx cy w h, normalised) from dataset/exp_aug100"),
        ("classes", "0:litter"), ("class_mapping_vs_v3_store_cctv",
                                  "INCOMPATIBLE: 1 generic class vs 10 specific v3 classes; no mapping invented -> "
                                  "separate one-class model"),
        ("scenes", ";".join(f"{k}:{v}" for k, v in Counter(r["scene"] for r in records).items())),
        ("images_are", "synthetic/augmented variants of 2 reference frames (source README)"),
        ("boxes_total", sum(r["n_boxes"] for r in records)),
        ("boxes_lt16px", sum(r["boxes_lt16px"] for r in sides_all)),
        ("images_without_litter", sum(1 for r in records if not r["n_boxes"])),
        ("images_with_multiple_litter", sum(1 for r in records if r["n_boxes"] > 1)),
        ("ignore_regions_total", sum(r["n_ignore_regions"] for r in records)),
        ("exact_duplicate_images", sum(1 for r in records if r["exact_duplicate_of"])),
        ("near_duplicate_groups_dhash", len({r["near_duplicate_group"] for r in records if r["near_duplicate_group"]})),
        ("label_verification", ";".join(f"{k}:{v}" for k, v in Counter(r["verified_by"] for r in records).items())),
        ("label_verification_caveat", "labels were transferred from 2 reference frames by homography and checked by an "
                                      "AI review (claude), not by a human annotator"),
        ("archive_folder", json.dumps(arch)),
        ("split_method", f"scene-based: test={test_scene}, val={val_count} from training scene, seed={seed}"),
        ("split_70_20_10", "not used: only 2 scenes; a 70/20/10 split would leak transforms of the test scene"),
        ("v3_store_cctv_sha256", v3_before or "not found"),
    ]
    write_csv(out / "dataset_summary.csv", [{"metric": k, "value": v} for k, v in summary], ["metric", "value"])
    (out / "README.md").write_text(_readme(records, test_scene, val_count, seed), encoding="utf-8")

    if folder_hashes(source) != src_before:
        raise PrepError("the source folder changed while building: output is not trustworthy")
    if v3_before and sha256(V3_WEIGHTS) != v3_before:
        raise PrepError("v3_store_cctv/best.pt changed during the build")
    zip_path = None
    if make_zip:
        zip_path = Path(shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name))
    return {"out": str(out), "zip": str(zip_path) if zip_path else None,
            "splits": {s: len(rs) for s, rs in split_counts.items()},
            "boxes": {s: sum(r["n_boxes"] for r in rs) for s, rs in split_counts.items()}}


def _readme(records: list[dict[str, Any]], test_scene: str, val_count: int, seed: int) -> str:
    n = Counter(r["split"] for r in records)
    return f"""# LDM_TRAINING_READY (one-class `litter`, AUG100)

Built by `python -m training.prepare_training_ready` (LDM repository). Do not edit by hand: rebuild into a new folder.

| split | images | contents |
|---|---|---|
| train | {n['train']} | training scene (incl. its reference frame) |
| val | {n['val']} | {val_count} images of the training scene (seed {seed}) - checkpoint selection only |
| test | {n['test']} | all of `{test_scene}` - independent scene |

Limitations (read before trusting any metric):
* All 100 images are synthetic transforms of **two** reference frames. Test metrics measure transfer between two
  synthetic scenes, not real CCTV accuracy across stores.
* Labels were transferred by homography and reviewed by an AI, not by a human annotator.
* ~60% of boxes are smaller than 16 px (1280x720 originals). At 640 px training size they become ~8 px.
* `ignore_regions.json` lists ambiguous / outside-ROI objects; the notebook does not count predictions there as FP.
* One class only: this model cannot replace the 10-class v3_store_cctv model.
"""


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--test-scene", default="B_roadside_bin")
    ap.add_argument("--val-count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args(argv)
    try:
        res = build(a.source, a.labels, a.out, a.test_scene, a.val_count, a.seed, not a.no_zip)
    except PrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
