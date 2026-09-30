"""YOLO dataset discovery, statistics, validation, browsing and label editing.

Supported layouts (per dataset folder containing data.yaml):
  images/{train,val,test}/  + labels/{train,val,test}/        (Ultralytics default)
  {train,val,test}/images/  + {train,val,test}/labels/
  images/negative/{split}/  -> hard-negative images (no label file = no objects)

Validation checks: corrupt/unreadable images, images without labels, labels without
images, invalid class ids, malformed lines, coordinates outside [0,1], zero-area and
duplicate boxes, exact-duplicate images (md5), near-duplicate frames (dHash),
class imbalance, extremely small objects.
"""
from __future__ import annotations

import hashlib
import shutil
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
SMALL_PX, MEDIUM_PX = 32 * 32, 96 * 96            # COCO size buckets (pixel area in the original image)
TINY_PX = 16 * 16


def size_bucket(area_px: float) -> str:
    return "small" if area_px < SMALL_PX else "medium" if area_px < MEDIUM_PX else "large"


@dataclass
class ImageRef:
    split: str
    path: Path
    label: Optional[Path]      # None for negatives (no label file expected)
    negative: bool


class DatasetError(Exception):
    pass


class DatasetService:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ discovery
    def list(self) -> list[dict[str, Any]]:
        out = []
        for yml in sorted(self.root.glob("*/data.yaml")):
            try:
                out.append(self.summary(yml.parent.name))
            except DatasetError as exc:
                out.append({"name": yml.parent.name, "error": str(exc)})
        return out

    def path(self, name: str) -> Path:
        p = (self.root / name).resolve()
        if p.parent != self.root.resolve() or not (p / "data.yaml").is_file():
            raise DatasetError(f"dataset '{name}' not found")
        return p

    def classes(self, name: str) -> list[str]:
        data = yaml.safe_load((self.path(name) / "data.yaml").read_text(encoding="utf-8")) or {}
        names = data.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=int)]
        return list(names)

    def _images(self, name: str) -> list[ImageRef]:
        root = self.path(name)
        refs: list[ImageRef] = []
        for split in SPLITS:
            for img_dir, lbl_dir in ((root / "images" / split, root / "labels" / split),
                                     (root / split / "images", root / split / "labels")):
                if img_dir.is_dir():
                    for p in sorted(img_dir.iterdir()):
                        if p.suffix.lower() in IMAGE_EXTS:
                            refs.append(ImageRef(split, p, lbl_dir / f"{p.stem}.txt", False))
            neg = root / "images" / "negative" / split
            if neg.is_dir():
                refs += [ImageRef(split, p, None, True) for p in sorted(neg.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        return refs

    def resolve_image(self, name: str, image_id: str) -> ImageRef:
        root = self.path(name)
        p = (root / image_id).resolve()
        if root not in p.parents or p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
            raise DatasetError("image not found")
        for ref in self._images(name):
            if ref.path.resolve() == p:
                return ref
        raise DatasetError("image not part of the dataset")

    # ------------------------------------------------------------------ labels
    @staticmethod
    def read_labels(ref: ImageRef, n_classes: int) -> tuple[list[dict[str, Any]], list[str]]:
        """Return (boxes, problems). Boxes: {cls, cx, cy, w, h} normalized."""
        boxes: list[dict[str, Any]] = []
        problems: list[str] = []
        if ref.label is None or not ref.label.is_file():
            return boxes, problems
        for i, line in enumerate(ref.label.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                problems.append(f"line {i}: expected 5 values, got {len(parts)}")
                continue
            try:
                c = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                problems.append(f"line {i}: non-numeric value")
                continue
            if not 0 <= c < n_classes:
                problems.append(f"line {i}: invalid class id {c}")
            if not all(0 <= v <= 1 for v in (cx, cy, w, h)):
                problems.append(f"line {i}: coordinates outside [0,1]")
            if w <= 0 or h <= 0:
                problems.append(f"line {i}: zero-area box")
            boxes.append({"cls": c, "cx": cx, "cy": cy, "w": w, "h": h})
        return boxes, problems

    def gt_boxes_px(self, name: str, ref: ImageRef, width: int, height: int) -> list[dict[str, Any]]:
        names = self.classes(name)
        boxes, _ = self.read_labels(ref, len(names))
        out = []
        for i, b in enumerate(boxes):
            x1, y1 = (b["cx"] - b["w"] / 2) * width, (b["cy"] - b["h"] / 2) * height
            x2, y2 = (b["cx"] + b["w"] / 2) * width, (b["cy"] + b["h"] / 2) * height
            out.append({"index": i, "cls_id": b["cls"], "class": names[b["cls"]] if 0 <= b["cls"] < len(names) else
                        f"invalid:{b['cls']}", "bbox": [x1, y1, x2, y2], "area_px": (x2 - x1) * (y2 - y1),
                        "size": size_bucket((x2 - x1) * (y2 - y1))})
        return out

    def write_labels(self, name: str, image_id: str, boxes: list[dict[str, Any]]) -> dict[str, Any]:
        """Overwrite a label file from normalized boxes; keeps <file>.txt.bak of the previous version."""
        ref = self.resolve_image(name, image_id)
        n = len(self.classes(name))
        if ref.label is None:
            raise DatasetError("negative images have no label file; move the image to a labelled split first")
        lines = []
        for b in boxes:
            c, cx, cy, w, h = int(b["cls"]), float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["h"])
            if not 0 <= c < n:
                raise DatasetError(f"invalid class id {c}")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise DatasetError("box coordinates must be normalized and non-empty")
            lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        ref.label.parent.mkdir(parents=True, exist_ok=True)
        if ref.label.is_file():
            shutil.copy2(ref.label, ref.label.with_suffix(".txt.bak"))
        ref.label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        with self._lock:
            self._cache.pop(name, None)
        return {"image": image_id, "objects": len(lines)}

    # ------------------------------------------------------------------ stats + validation
    def _mtime(self, name: str) -> float:
        root = self.path(name)
        return max((p.stat().st_mtime for p in root.rglob("*") if p.is_file()), default=0.0)

    def summary(self, name: str) -> dict[str, Any]:
        s = self.stats(name)
        return {k: s[k] for k in ("name", "classes", "splits", "totals", "created", "readme", "issue_counts")}

    def stats(self, name: str) -> dict[str, Any]:
        mtime = self._mtime(name)
        with self._lock:
            hit = self._cache.get(name)
            if hit and hit[0] == mtime:
                return hit[1]
        result = self._compute_stats(name)
        with self._lock:
            self._cache[name] = (mtime, result)
        return result

    def _compute_stats(self, name: str) -> dict[str, Any]:
        root = self.path(name)
        names = self.classes(name)
        refs = self._images(name)
        splits: dict[str, Counter] = defaultdict(Counter)
        obj_per_class, img_per_class = Counter(), Counter()
        sizes = Counter()
        areas: list[float] = []
        issues: list[dict[str, str]] = []
        md5s: dict[str, str] = {}
        hashes: list[tuple[str, np.ndarray]] = []
        widths: Counter = Counter()
        seen_labels = set()
        for ref in refs:
            iid = ref.path.relative_to(root).as_posix()
            sp = splits[ref.split]
            sp["images"] += 1
            sp["negatives"] += int(ref.negative)
            raw = ref.path.read_bytes()
            digest = hashlib.md5(raw).hexdigest()  # noqa: S324 - duplicate detection, not security
            if digest in md5s:
                issues.append({"type": "duplicate_image", "image": iid, "detail": f"identical to {md5s[digest]}"})
            md5s.setdefault(digest, iid)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                issues.append({"type": "corrupt_image", "image": iid, "detail": "cannot be decoded"})
                continue
            h, w = img.shape[:2]
            widths[f"{w}x{h}"] += 1
            small = cv2.resize(img, (17, 16), interpolation=cv2.INTER_AREA)
            hashes.append((iid, (small[:, 1:] > small[:, :-1]).flatten()))
            if ref.label is not None:
                seen_labels.add(ref.label.resolve())
                if not ref.label.is_file():
                    issues.append({"type": "missing_label", "image": iid, "detail": "no label file"})
            boxes, problems = self.read_labels(ref, len(names))
            issues += [{"type": "label_error", "image": iid, "detail": p} for p in problems]
            keyset = set()
            classes_here = set()
            for b in boxes:
                key = (b["cls"], round(b["cx"], 4), round(b["cy"], 4), round(b["w"], 4), round(b["h"], 4))
                if key in keyset:
                    issues.append({"type": "duplicate_box", "image": iid, "detail": f"class {b['cls']}"})
                keyset.add(key)
                if not 0 <= b["cls"] < len(names) or b["w"] <= 0 or b["h"] <= 0:
                    continue
                area = b["w"] * w * b["h"] * h
                areas.append(area)
                sizes[size_bucket(area)] += 1
                if area < TINY_PX:
                    issues.append({"type": "tiny_object", "image": iid,
                                   "detail": f"{names[b['cls']]} {b['w'] * w:.0f}x{b['h'] * h:.0f}px"})
                obj_per_class[names[b["cls"]]] += 1
                classes_here.add(names[b["cls"]])
                sp["objects"] += 1
            if not boxes and not ref.negative:
                sp["empty_labels"] += 1
            for c in classes_here:
                img_per_class[c] += 1
        # labels without images
        for split in SPLITS:
            for lbl_dir in (root / "labels" / split, root / split / "labels"):
                if lbl_dir.is_dir():
                    for lf in lbl_dir.glob("*.txt"):
                        if lf.resolve() not in seen_labels:
                            issues.append({"type": "orphan_label", "image": lf.relative_to(root).as_posix(),
                                           "detail": "label file has no image"})
        # near-duplicate frames (dHash hamming distance <= 4 of 256 bits)
        if len(hashes) <= 5000:
            mat = np.array([h for _, h in hashes])
            for i in range(len(hashes)):
                d = np.count_nonzero(mat[i + 1:] != mat[i], axis=1)
                for j in np.nonzero(d <= 4)[0][:3]:
                    issues.append({"type": "near_duplicate", "image": hashes[i][0],
                                   "detail": f"near-identical to {hashes[i + 1 + j][0]}"})
        total_obj = sum(obj_per_class.values())
        counts = [obj_per_class.get(n, 0) for n in names]
        nonzero = [c for c in counts if c]
        imbalance = (max(nonzero) / min(nonzero)) if nonzero else None
        for n in names:
            if obj_per_class.get(n, 0) == 0:
                issues.append({"type": "class_without_labels", "image": "", "detail": n})
        readme = next((p for p in (root / "README.txt", root / "README.md") if p.is_file()), None)
        return {
            "name": name, "path": str(root), "classes": names,
            "created": root.stat().st_ctime,
            "readme": readme.read_text(encoding="utf-8", errors="replace")[:2000] if readme else None,
            "splits": {k: dict(v) for k, v in splits.items()},
            "totals": {"images": len(refs), "objects": total_obj,
                       "negatives": sum(v["negatives"] for v in splits.values())},
            "objects_per_class": {n: obj_per_class.get(n, 0) for n in names},
            "images_per_class": {n: img_per_class.get(n, 0) for n in names},
            "size_distribution": {k: sizes.get(k, 0) for k in ("small", "medium", "large")},
            "box_area_px": ({"mean": float(np.mean(areas)), "median": float(np.median(areas)),
                             "min": float(np.min(areas)), "max": float(np.max(areas)),
                             "histogram": np.histogram(np.sqrt(areas), bins=[0, 8, 16, 24, 32, 48, 64, 96, 128, 192,
                                                                             256, 10000])[0].tolist(),
                             "histogram_edges_side_px": [0, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]}
                            if areas else None),
            "resolutions": dict(widths.most_common(10)),
            "class_imbalance_ratio": round(imbalance, 2) if imbalance else None,
            "issue_counts": dict(Counter(i["type"] for i in issues)),
            "issues": issues[:2000],
        }

    # ------------------------------------------------------------------ browsing
    def list_images(self, name: str, split: Optional[str] = None, cls: Optional[str] = None,
                    size: Optional[str] = None, status: Optional[str] = None, offset: int = 0,
                    limit: int = 60) -> dict[str, Any]:
        root = self.path(name)
        names = self.classes(name)
        items = []
        for ref in self._images(name):
            if split and ref.split != split:
                continue
            boxes, problems = self.read_labels(ref, len(names))
            classes = sorted({names[b["cls"]] for b in boxes if 0 <= b["cls"] < len(names)})
            st = ("negative" if ref.negative else "issues" if problems else "annotated" if boxes else "empty")
            if cls and cls not in classes:
                continue
            if status and st != status:
                continue
            if size:
                img = cv2.imread(str(ref.path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
                if img is None:
                    continue
                w, h = img.shape[1] * 8, img.shape[0] * 8
                if not any(size_bucket(b["w"] * w * b["h"] * h) == size for b in boxes):
                    continue
            items.append({"id": ref.path.relative_to(root).as_posix(), "split": ref.split, "file": ref.path.name,
                          "objects": len(boxes), "classes": classes, "status": st})
        return {"total": len(items), "items": items[offset: offset + limit]}

    def image_detail(self, name: str, image_id: str) -> dict[str, Any]:
        ref = self.resolve_image(name, image_id)
        img = cv2.imread(str(ref.path))
        if img is None:
            raise DatasetError("image cannot be decoded")
        h, w = img.shape[:2]
        _, problems = self.read_labels(ref, len(self.classes(name)))
        return {"id": image_id, "split": ref.split, "negative": ref.negative, "width": w, "height": h,
                "ground_truth": self.gt_boxes_px(name, ref, w, h), "problems": problems,
                "editable": ref.label is not None}

    # ------------------------------------------------------------------ versions / export
    def create_version(self, source: str, new_name: str) -> dict[str, Any]:
        src = self.path(source)
        if not new_name.replace("_", "").replace("-", "").isalnum() or len(new_name) > 64:
            raise DatasetError("dataset name may only contain letters, digits, '-' and '_'")
        dest = self.root / new_name
        if dest.exists():
            raise DatasetError(f"dataset {new_name} already exists")
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("*.bak", "*.cache"))
        data = yaml.safe_load((dest / "data.yaml").read_text(encoding="utf-8")) or {}
        data["path"] = str(dest)
        (dest / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return self.summary(new_name)

    def export_zip(self, name: str, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / name
        return Path(shutil.make_archive(str(base), "zip", root_dir=self.path(name)))
