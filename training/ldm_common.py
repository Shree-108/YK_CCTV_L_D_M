"""Shared helpers for preparing the CCTV litter fine-tuning dataset (workspace -> versioned dataset).

Workflow (see docs/LDM_DATASET_GUIDE.md):
  1. extract_frames.py / import_images.py   copy source material into the WORKSPACE + manifest.csv
  2. annotate in the Ground Truth page (dataset "ldm_v4_workspace"), fill category/conditions in manifest.csv
  3. ldm_review.py sync | mark-empty | report  record review state, write empty labels, gap report
  4. build_ldm_dataset.py                      group-aware train/val/test split -> dataset/<version>/
  5. validate_ldm_dataset.py                   checks before any training

The workspace is NOT a training dataset: its data.yaml has no val split, so Ultralytics refuses to train on it.
Sources are only ever read and copied; nothing here moves, edits or deletes original recordings or images.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

from backend.core.config import PROJECT_ROOT

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".ts", ".h264", ".264", ".dav"}

BASE_MODEL_VERSION = "v3_store_cctv"
DEFAULT_WORKSPACE = PROJECT_ROOT / "dataset" / "ldm_v4_workspace"

# ---------------------------------------------------------------- manifest vocabulary
SOURCE_TYPES = ("real_cctv", "synthetic", "other_real")
CATEGORIES = ("unknown", "litter", "hard_negative", "clean")
REVIEW_STATUSES = ("unreviewed", "annotated", "reviewed_empty", "ambiguous", "rejected")
YES_NO = ("yes", "no", "likely", "unknown")
SUBCATEGORIES = {
    "litter": ("snack_packet", "wrapper", "paper", "newspaper", "cup", "bottle", "plastic_piece", "cardboard",
               "small_distant", "occluded"),
    "hard_negative": ("product_display", "shop_inventory", "car", "motorcycle", "tyre", "entrance", "shadow",
                      "reflection", "pavement_marking", "person_carrying"),
    "clean": ("empty_pavement", "shop_surroundings"),
}
CONDITIONS = ("day", "dusk", "night", "ir", "rain_wet", "glare", "harsh_shadow", "low_light", "crowded", "blur",
              "compression")

FIELDS = ["image", "source_type", "source_path", "session", "camera", "frame_index", "timestamp_sec", "recorded_at",
          "width", "height", "sha256", "dhash", "overlay_suspect", "used_in_previous_training", "category",
          "subcategory", "conditions", "review_status", "split_lock", "notes", "added_at"]


class DatasetPrepError(Exception):
    pass


# ---------------------------------------------------------------- class mapping (never invented)
def model_classes(version: str = BASE_MODEL_VERSION) -> list[str]:
    """Class list of a registered model, cross-checked across its class_names.json, data.yaml and model_meta.json."""
    d = PROJECT_ROOT / "models" / "litter_yolo" / version
    found: dict[str, list[str]] = {}
    if (d / "class_names.json").is_file():
        found["class_names.json"] = json.loads((d / "class_names.json").read_text(encoding="utf-8"))
    if (d / "model_meta.json").is_file():
        found["model_meta.json"] = json.loads((d / "model_meta.json").read_text(encoding="utf-8")).get("classes", [])
    if (d / "data.yaml").is_file():
        names = (yaml.safe_load((d / "data.yaml").read_text(encoding="utf-8")) or {}).get("names", [])
        found["data.yaml"] = [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else list(names)
    lists = [v for v in found.values() if v]
    if not lists:
        raise DatasetPrepError(f"no class list found for model {version} in {d}")
    if any(v != lists[0] for v in lists):
        raise DatasetPrepError(f"class lists disagree for model {version}: {found}")
    return list(lists[0])


def yaml_names(path: Path) -> list[str]:
    names = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("names", [])
    return [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else list(names)


# ---------------------------------------------------------------- hashing
def dhash(img: np.ndarray) -> str:
    """256-bit difference hash (same scheme as backend/services/datasets.py), as 64 hex chars."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (17, 16), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


def hamming(a: str, b: str) -> int:
    return int(np.unpackbits(np.frombuffer(bytes.fromhex(a), np.uint8) ^ np.frombuffer(bytes.fromhex(b), np.uint8)).sum())


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_image(p: Path) -> np.ndarray | None:
    """cv2.imread that also works with non-ASCII Windows paths."""
    try:
        return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


_HUE_FAMILIES = ((0, 10), (170, 180), (20, 35), (40, 85), (85, 100), (100, 130), (130, 170))


def _thin_segments(mask: np.ndarray, horizontal: bool, min_len: int) -> list[tuple[int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for x, y, w, h, _a in stats[1:]:
        if (horizontal and w >= min_len and h <= 4) or (not horizontal and h >= min_len and w <= 4):
            out.append((int(x), int(y), int(x + w - 1), int(y + h - 1)))
    return out


def overlay_suspect(img: np.ndarray, min_len: int = 30, gap: int = 9, tol: int = 6) -> bool:
    """Heuristic for burned-in analytics boxes (person / footfall / previous-model overlays).

    Looks for a box CORNER: a thin, dead-straight horizontal line and a vertical line of the same saturated colour
    whose ends meet. Small gaps from video compression are bridged first. Tuned for recall - a missed overlay
    pollutes training, a false flag only parks a frame: measured on the store footage, it flagged 70 of 90 sampled
    frames of the overlay video (the unflagged ones had no visible boxes) and 13 of 90 clean frames, where the blue
    fridge-door frame forms corners. A human has the final word: set overlay_suspect=no in manifest.csv to keep a
    frame that was flagged wrongly.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 110)
    k = cv2.getStructuringElement
    for lo, hi in _HUE_FAMILIES:
        mask = (sat & (hsv[:, :, 0] >= lo) & (hsv[:, :, 0] < hi)).astype(np.uint8) * 255
        if cv2.countNonZero(mask) < min_len:
            continue
        mh = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k(cv2.MORPH_RECT, (gap, 1)))
        mv = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k(cv2.MORPH_RECT, (1, gap)))
        # thin straight runs: survive a long 1-px kernel, vanish under a 5-px-thick one
        th = cv2.subtract(cv2.morphologyEx(mh, cv2.MORPH_OPEN, k(cv2.MORPH_RECT, (min_len, 1))),
                          cv2.morphologyEx(mh, cv2.MORPH_OPEN, k(cv2.MORPH_RECT, (min_len, 5))))
        tv = cv2.subtract(cv2.morphologyEx(mv, cv2.MORPH_OPEN, k(cv2.MORPH_RECT, (1, min_len))),
                          cv2.morphologyEx(mv, cv2.MORPH_OPEN, k(cv2.MORPH_RECT, (5, min_len))))
        for hx1, hy1, hx2, hy2 in _thin_segments(th, True, min_len):
            for vx1, vy1, vx2, vy2 in _thin_segments(tv, False, min_len):
                if (min(abs(hx1 - vx1), abs(hx2 - vx2), abs(hx1 - vx2), abs(hx2 - vx1)) <= tol
                        and min(abs(hy1 - vy1), abs(hy2 - vy2), abs(hy1 - vy2), abs(hy2 - vy1)) <= tol):
                    return True
    return False


def is_project_overlay_output(p: Path) -> bool:
    """Files the app itself wrote WITH boxes drawn on them - never training material."""
    parts = [x.lower() for x in Path(p).resolve().parts]
    name = Path(p).name.lower()
    if "storage" in parts and ("alerts" in parts or "videos" in parts or "hard_examples" in parts):
        return True
    return name.startswith("annotated") or name.endswith("_crop.jpg")


# ---------------------------------------------------------------- workspace + manifest
def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_workspace(ws: Path, classes: list[str]) -> Path:
    ws = Path(ws)
    (ws / "images" / "train").mkdir(parents=True, exist_ok=True)
    (ws / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (ws / "reports").mkdir(exist_ok=True)
    yml = ws / "data.yaml"
    if yml.is_file():
        if yaml_names(yml) != classes:
            raise DatasetPrepError(f"{yml} class names differ from model {BASE_MODEL_VERSION}: refusing to mix mappings")
    else:
        yml.write_text(
            "# ANNOTATION WORKSPACE - not a training dataset (no val split on purpose).\n"
            "# images/train holds EVERY candidate frame; the real split happens in training/build_ldm_dataset.py.\n"
            f"# Class mapping copied from models/litter_yolo/{BASE_MODEL_VERSION} - do not reorder.\n"
            + yaml.safe_dump({"train": "images/train", "names": dict(enumerate(classes))}, sort_keys=False),
            encoding="utf-8")
    if not (ws / "manifest.csv").is_file():
        write_manifest(ws, [])
    return ws


def manifest_path(ws: Path) -> Path:
    return Path(ws) / "manifest.csv"


def read_manifest(ws: Path) -> list[dict[str, str]]:
    p = manifest_path(ws)
    if not p.is_file():
        return []
    with open(p, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FIELDS:
            r.setdefault(k, "")
            r[k] = (r[k] or "").strip()
    return rows


def write_manifest(ws: Path, rows: Iterable[dict[str, str]]) -> None:
    p = manifest_path(ws)
    tmp = p.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:   # BOM: opens cleanly in Excel
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    try:
        os.replace(tmp, p)
    except PermissionError as exc:
        tmp.unlink(missing_ok=True)
        raise DatasetPrepError(f"cannot write {p}: close it in Excel (or any editor) and run again") from exc


def label_path(ws: Path, image: str) -> Path:
    return Path(ws) / "labels" / "train" / f"{Path(image).stem}.txt"


def read_label_lines(p: Path) -> list[list[str]]:
    if not p.is_file():
        return []
    return [ln.split() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]


def split_tags(value: str) -> list[str]:
    return [t.strip() for t in value.replace(",", ";").split(";") if t.strip()]


def safe_name(s: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_." else "_" for c in s.strip())
    return out.strip("._") or "x"


def validate_row_vocab(r: dict[str, str]) -> list[str]:
    """Problems with hand-edited manifest values (typos would silently skew the split and the gap report)."""
    probs = []
    if r["source_type"] not in SOURCE_TYPES:
        probs.append(f"source_type '{r['source_type']}' not in {SOURCE_TYPES}")
    if r["category"] and r["category"] not in CATEGORIES:
        probs.append(f"category '{r['category']}' not in {CATEGORIES}")
    if r["review_status"] and r["review_status"] not in REVIEW_STATUSES:
        probs.append(f"review_status '{r['review_status']}' not in {REVIEW_STATUSES}")
    allowed_sub = {s for v in SUBCATEGORIES.values() for s in v}
    probs += [f"unknown subcategory '{t}'" for t in split_tags(r["subcategory"]) if t not in allowed_sub]
    probs += [f"unknown condition '{t}'" for t in split_tags(r["conditions"]) if t not in CONDITIONS]
    if r["split_lock"] and r["split_lock"] not in ("train", "val", "test"):
        probs.append(f"split_lock '{r['split_lock']}' must be train, val, test or empty")
    for k in ("overlay_suspect", "used_in_previous_training"):
        if r[k] and r[k] not in YES_NO:
            probs.append(f"{k} '{r[k]}' not in {YES_NO}")
    return probs
