"""Shared paths, guards and geometry for the AUG100 experiment.

Hard rules enforced here:
  * the original dataset is only read; copies are hash-checked
  * v3_store_cctv weights must keep their recorded SHA-256 (checked before and after every step)
  * nothing in models/litter_yolo/ or dataset/ldm_v4_workspace/ is written
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Original AUG100 source folder (read only, outside the repository). Only prepare.py needs it:
#   set AUG100_SOURCE_DIR=<folder containing cctv_litter_001.jpg ... cctv_litter_100.jpg>
SRC_DIR = Path(os.environ["AUG100_SOURCE_DIR"]) if os.environ.get("AUG100_SOURCE_DIR") else None
DS = ROOT / "dataset" / "exp_aug100"
IMG_DIR = DS / "images" / "train"
LBL_DIR = DS / "labels" / "train"
MANIFEST = DS / "manifest.csv"
ANN = DS / "annotations"               # reference boxes, ROI, review decisions (json, the source of truth)
EXP = Path(__file__).resolve().parent
RUNS = ROOT / "storage" / "runs"
V3 = ROOT / "models" / "litter_yolo" / "v3_store_cctv" / "best.pt"
V3_SUMS = ROOT / "models" / "backups" / "v3_store_cctv_20261012" / "SHA256SUMS"
DATASET_VERSION = "exp_aug100_v1"

SCENES = {
    "A_shopfront": {"ids": list(range(1, 101, 2)), "reference": 83},
    "B_roadside_bin": {"ids": list(range(2, 101, 2)), "reference": 6},
}

MANIFEST_FIELDS = ["image_id", "file", "source_path", "image_path", "label_path", "sha256", "scene", "role",
                   "reference_id", "aug_scale", "aug_tx", "aug_ty", "aug_rot_deg", "aug_rms_px", "aug_inliers",
                   "label_status", "verified_by", "verified_at", "n_boxes", "n_transferred", "n_manual",
                   "dataset_version", "source_type", "notes"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_v3() -> str:
    want = V3_SUMS.read_text().split()[0].strip().lower()
    got = sha256(V3)
    if got != want:
        raise SystemExit(f"STOP: v3_store_cctv/best.pt hash changed ({got} != {want})")
    return got


def image_id(scene: str, n: int) -> str:
    return f"aug100_{scene[0]}_{n:03d}"


def fname(n: int) -> str:
    return f"cctv_litter_{n:03d}.jpg"


def scene_of(n: int) -> str:
    return "A_shopfront" if n % 2 else "B_roadside_bin"


def read_manifest() -> list[dict]:
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict]) -> None:
    tmp = MANIFEST.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(MANIFEST)


def load_json(p: Path, default):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def save_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ geometry (validated in docs/.../geometry.py)
_ORB = cv2.ORB_create(6000, scaleFactor=1.2, nlevels=10)
_BF = cv2.BFMatcher(cv2.NORM_HAMMING)


def features(path: Path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    k, d = _ORB.detectAndCompute(cv2.createCLAHE(2.0, (8, 8)).apply(g), None)
    return g, k, d


def homography(a, b):
    """Homography mapping image a -> image b, plus inlier count and inlier RMS (px)."""
    (_, ka, da), (_, kb, db) = a, b
    m = _BF.knnMatch(da, db, k=2)
    good = [x for x, y in (p for p in m if len(p) == 2) if x.distance < 0.75 * y.distance]
    if len(good) < 12:
        return None, 0, float("inf")
    s = np.float32([ka[g.queryIdx].pt for g in good])
    d = np.float32([kb[g.trainIdx].pt for g in good])
    H, mask = cv2.findHomography(s, d, cv2.USAC_MAGSAC, 3.0, maxIters=10000, confidence=0.999)
    if H is None:
        return None, 0, float("inf")
    inl = mask.ravel().astype(bool)
    pr = cv2.perspectiveTransform(s[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    return H, int(inl.sum()), float(np.sqrt(((pr - d[inl]) ** 2).sum(1).mean()))


def similarity_params(H) -> dict:
    A = H[:2, :2] / H[2, 2]
    det = float(np.linalg.det(A))
    return dict(scale=float(np.sqrt(abs(det))), rot_deg=float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))),
                tx=float(H[0, 2] / H[2, 2]), ty=float(H[1, 2] / H[2, 2]), flip=det < 0,
                persp=float(np.hypot(H[2, 0], H[2, 1]) / H[2, 2]),
                aniso=float(np.hypot(A[0, 0], A[1, 0]) / (det / np.hypot(A[0, 0], A[1, 0]))))


def transfer_ok(p: dict, inliers: int, rms: float) -> tuple[bool, str]:
    if p["flip"]: return False, "flip"
    if inliers < 500: return False, f"inliers {inliers} < 500"
    if rms > 2.0: return False, f"rms {rms:.2f} > 2 px"
    if abs(p["rot_deg"]) > 0.5: return False, f"rotation {p['rot_deg']:.2f} deg"
    if not 0.9 <= p["scale"] <= 1.2: return False, f"scale {p['scale']:.3f}"
    if p["persp"] > 1e-5: return False, f"perspective {p['persp']:.2e}"
    if abs(p["aniso"] - 1) > 0.01: return False, f"anisotropy {p['aniso']:.3f}"
    return True, "ok"


def map_box(H, box):
    """Map an xyxy box through H (corners), return axis-aligned xyxy."""
    x1, y1, x2, y2 = box
    c = cv2.perspectiveTransform(np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2), H).reshape(-1, 2)
    return [float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max())]


def point_in_poly(pt, poly) -> bool:
    return cv2.pointPolygonTest(np.float32(poly).reshape(-1, 1, 2), (float(pt[0]), float(pt[1])), False) >= 0
