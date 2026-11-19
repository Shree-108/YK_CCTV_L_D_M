"""Small shared helpers for the ldm7 tools (hashing and image reading are reused from training/ldm_common.py)."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable, Optional

try:  # reuse the project's helpers (same 256-bit dHash scheme as backend/services/datasets.py)
    from training.ldm_common import dhash, hamming, read_image, sha256_file  # noqa: F401  (re-exported)
except ImportError:  # e.g. Colab with only training/ uploaded: ldm_common needs backend.core.config
    import cv2
    import numpy as np

    def dhash(img: "np.ndarray") -> str:
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (17, 16), interpolation=cv2.INTER_AREA)
        return np.packbits((g[:, 1:] > g[:, :-1]).flatten()).tobytes().hex()

    def hamming(a: str, b: str) -> int:
        return int(np.unpackbits(np.frombuffer(bytes.fromhex(a), np.uint8)
                                 ^ np.frombuffer(bytes.fromhex(b), np.uint8)).sum())

    def sha256_file(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def read_image(p: Path) -> "Optional[np.ndarray]":
        try:
            return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        except (OSError, ValueError):
            return None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")

# COCO-style size buckets on the box area in ORIGINAL image pixels
SMALL_MAX = 32 * 32
MEDIUM_MAX = 96 * 96
SIZE_BUCKETS = ("small", "medium", "large")

MANIFEST_REQUIRED = ("image", "camera_id", "session_id", "date")
GROUP_KEYS = ("camera_id", "session_id", "date", "camera_day")
DEFAULT_GROUP_BY = ("session_id", "camera_day")


def size_bucket(area_px: float) -> str:
    if area_px < SMALL_MAX:
        return "small"
    if area_px <= MEDIUM_MAX:
        return "medium"
    return "large"


def list_images(folder: Path) -> list[Path]:
    if not Path(folder).is_dir():
        return []
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in rows]


def manifest_missing_columns(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        header = [h.strip() for h in (next(csv.reader(f), []) or [])]
    return [c for c in MANIFEST_REQUIRED if c not in header]


def group_values(row: dict[str, str], group_by: Iterable[str]) -> list[str]:
    """Group identifiers of one manifest row, e.g. ['session_id=S1', 'camera_day=CAM_A@2026-11-19']."""
    out = []
    for k in group_by:
        if k == "camera_day":
            v = f"{row.get('camera_id', '')}@{row.get('date', '')}" if row.get("camera_id") and row.get("date") else ""
        else:
            v = row.get(k, "")
        if v:
            out.append(f"{k}={v}")
    return out


def tree_hash(files: Iterable[tuple[str, Path]]) -> str:
    """Deterministic content hash of (relative name, file) pairs (dataset version fingerprint)."""
    h = hashlib.sha256()
    for rel, p in sorted(files, key=lambda x: x[0]):
        h.update(rel.replace("\\", "/").encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(p).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def find_manifest_row(index: dict[str, dict[str, str]], split: str, name: str) -> Optional[dict[str, str]]:
    for key in (f"images/{split}/{name}", f"{split}/{name}", name):
        if key in index:
            return index[key]
    return None


def index_manifest(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    idx = {}
    for r in rows:
        key = r.get("image", "").replace("\\", "/")
        while key.startswith("./"):
            key = key[2:]
        if key:
            idx[key] = r
    return idx
