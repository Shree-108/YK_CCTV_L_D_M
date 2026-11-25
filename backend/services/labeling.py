"""Keyboard review (labelling) of mined detections - phase 2 item 6.

Works on the review folders written by training/negative_mining/mine_tiles.py (and mine_review.py):

    storage/review/<video_id>/
        detections.csv      one row per mined box; this module fills ``label`` (autosave, atomic rewrite)
        missed.csv          real litter the model missed, drawn by the reviewer ("draw box" mode)
        frames/ crops/      images (crops at native pixel scale = the scale of the inference tiles)
        manifest.json

Labels: TP / FP_shadow / FP_carton / FP_stock / FP_puddle / FP_other, or "" (unlabelled / undo). ``group_id`` (same
box in consecutive sampled frames, written by the miner) lets one keypress label the whole group. Rows whose crop was
not saved (``capped`` = 1, above the per-class cap) are not offered for review.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import os
import threading
from pathlib import Path
from typing import Any, Optional

from backend.services.negative_mining import LABELS, VIDEO_ID_RE, read_csv

MISSED_COLUMNS = ["missed_id", "video_id", "frame_file", "frame_idx", "class", "x1", "y1", "x2", "y2", "frame_w",
                  "frame_h", "created_at", "actor"]
_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


class LabelError(ValueError):
    pass


def _lock(path: Path) -> threading.Lock:
    with _GUARD:
        return _LOCKS.setdefault(str(path), threading.Lock())


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _write_atomic(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in columns})
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8", newline="")
    os.replace(tmp, path)


def _columns(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


class ReviewStore:
    """All review folders under ``root`` (one per mined video)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def video_dir(self, video_id: str) -> Path:
        if not VIDEO_ID_RE.match(video_id or ""):
            raise LabelError("invalid video id")
        d = self.root / video_id
        if not (d / "detections.csv").is_file():
            raise FileNotFoundError(video_id)
        return d

    def sessions(self) -> list[dict[str, Any]]:
        out = []
        if not self.root.is_dir():
            return out
        for d in sorted(p for p in self.root.iterdir() if (p / "detections.csv").is_file()):
            rows = read_csv(d / "detections.csv")
            offered = [r for r in rows if _offered(r)]
            out.append({"video_id": d.name, "detections": len(rows), "offered": len(offered),
                        "labelled": sum(1 for r in offered if (r.get("label") or "").strip()),
                        "missed": len(read_csv(d / "missed.csv")) if (d / "missed.csv").is_file() else 0,
                        "groups": len({r.get("group_id") or r["det_id"] for r in offered}),
                        "below_030": sum(1 for r in rows if r.get("below_030") == "1"),
                        "video_file": _manifest(d).get("video_file"), "frame_size": _manifest(d).get("frame_size")})
        return out

    def items(self, video_id: str) -> dict[str, Any]:
        """One item per group (representative = highest-confidence offered row), in mining order."""
        d = self.video_dir(video_id)
        rows = read_csv(d / "detections.csv")
        groups: dict[str, list[dict[str, str]]] = {}
        for r in rows:
            if _offered(r):
                groups.setdefault(r.get("group_id") or r["det_id"], []).append(r)
        items = []
        for gid, members in groups.items():
            rep = max(members, key=lambda r: float(r.get("conf") or 0))
            labels = {(m.get("label") or "").strip() for m in members}
            items.append({"group_id": gid, "det_id": rep["det_id"], "members": len(members),
                          "frames": sorted({int(m["frame_idx"]) for m in members}),
                          "class": rep["class"], "conf": float(rep["conf"]),
                          "min_conf": min(float(m["conf"]) for m in members),
                          "below_030": any(m.get("below_030") == "1" for m in members),
                          "bbox": [float(rep[k]) for k in ("x1", "y1", "x2", "y2")],
                          "crop_box": [int(float(rep[k])) for k in ("crop_x", "crop_y", "crop_w", "crop_h")]
                          if rep.get("crop_w") else None,
                          "frame_file": rep["frame_file"], "frame_idx": int(rep["frame_idx"]),
                          "timestamp_sec": float(rep.get("timestamp_sec") or 0),
                          "label": labels.pop() if len(labels) == 1 else "mixed",
                          "status": rep.get("status", ""), "in_roi": rep.get("in_roi", "")})
        items.sort(key=lambda it: (min(it["frames"]), it["det_id"]))
        return {"video_id": video_id, "manifest": _manifest(d), "labels": list(LABELS), "items": items,
                "missed": read_csv(d / "missed.csv") if (d / "missed.csv").is_file() else []}

    def set_label(self, video_id: str, target: str, label: str, *, whole_group: bool = True) -> dict[str, Any]:
        """``target`` = a det_id or a group_id; ``label`` "" clears (undo). Autosaved (atomic rewrite)."""
        if label not in ("", *LABELS):
            raise LabelError(f"label must be one of {', '.join(LABELS)} or empty")
        d = self.video_dir(video_id)
        path = d / "detections.csv"
        with _lock(path):
            cols = _columns(path)
            rows = read_csv(path)
            row = next((r for r in rows if r["det_id"] == target), None)
            gid = (row.get("group_id") if row else None) or (target if any(r.get("group_id") == target for r in rows)
                                                             else None)
            if row is None and gid is None:
                raise KeyError(target)
            changed = []
            for r in rows:
                # the whole chain (capped members included: same box in neighbouring frames) or just this row
                if (whole_group and gid and r.get("group_id") == gid) or r["det_id"] == target:
                    r["label"] = label
                    changed.append(r["det_id"])
            _write_atomic(path, cols, rows)
        return {"video_id": video_id, "label": label, "changed": changed}

    def set_true_class(self, video_id: str, det_ids: list[str], cls: str) -> str:
        """Correct the class of TP rows ("" = keep the detected class)."""
        d = self.video_dir(video_id)
        classes = _manifest(d).get("class_names") or []
        if cls and classes and cls not in classes:
            raise LabelError(f"class must be one of {', '.join(classes)}")
        path = d / "detections.csv"
        with _lock(path):
            cols = _columns(path)
            rows = read_csv(path)
            ids = set(det_ids)
            for r in rows:
                if r["det_id"] in ids:
                    r["true_class"] = cls
            _write_atomic(path, cols, rows)
        return cls

    def add_missed(self, video_id: str, frame_file: str, bbox: list[float], cls: str, actor: str = "") -> dict:
        d = self.video_dir(video_id)
        classes = _manifest(d).get("class_names") or []
        if classes and cls not in classes:
            raise LabelError(f"class must be one of {', '.join(classes)}")
        if not (d / frame_file).resolve().is_relative_to((d / "frames").resolve()) or not (d / frame_file).is_file():
            raise LabelError("unknown frame")
        x1, y1, x2, y2 = (float(v) for v in bbox)
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise LabelError("box too small")
        size = _manifest(d).get("frame_size") or [0, 0]
        path = d / "missed.csv"
        with _lock(path):
            rows = read_csv(path) if path.is_file() else []
            mid = f"{video_id}_m{len(rows) + 1:04d}_{dt.datetime.now().strftime('%H%M%S%f')[:9]}"
            row = {"missed_id": mid, "video_id": video_id, "frame_file": frame_file,
                   "frame_idx": int(Path(frame_file).stem.lstrip("f") or 0), "class": cls,
                   "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
                   "frame_w": size[0], "frame_h": size[1], "created_at": _now(), "actor": actor}
            _write_atomic(path, MISSED_COLUMNS, [*rows, row])
        return row

    def delete_missed(self, video_id: str, missed_id: str) -> bool:
        d = self.video_dir(video_id)
        path = d / "missed.csv"
        if not path.is_file():
            return False
        with _lock(path):
            rows = read_csv(path)
            keep = [r for r in rows if r["missed_id"] != missed_id]
            if len(keep) == len(rows):
                return False
            _write_atomic(path, MISSED_COLUMNS, keep)
        return True

    def image_path(self, video_id: str, rel: str) -> Optional[Path]:
        d = self.video_dir(video_id)
        p = (d / rel).resolve()
        if not (p.is_relative_to((d / "crops").resolve()) or p.is_relative_to((d / "frames").resolve())):
            return None
        return p if p.is_file() else None


def _offered(r: dict[str, str]) -> bool:
    return r.get("capped", "0") != "1" and bool(r.get("crop_file"))


def _manifest(d: Path) -> dict[str, Any]:
    import json
    try:
        return json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
