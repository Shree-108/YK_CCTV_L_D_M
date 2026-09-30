"""Experiment tracking: discovers Ultralytics training runs (results.csv + args.yaml) on disk."""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any, Optional

import yaml

ARTIFACTS = ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png",
             "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png", "labels.jpg", "val_batch0_pred.jpg",
             "val_batch0_labels.jpg", "train_batch0.jpg")
AUG_KEYS = ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr",
            "mosaic", "mixup", "copy_paste", "erasing", "close_mosaic", "multi_scale")
CFG_KEYS = ("model", "data", "epochs", "patience", "batch", "imgsz", "device", "optimizer", "lr0", "lrf",
            "momentum", "weight_decay", "seed", "cos_lr", "workers", "cache", "rect", "single_cls")


def _f(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class ExperimentService:
    def __init__(self, roots: list[Path], project_root: Path) -> None:
        self.roots, self.project_root = roots, project_root

    def _run_dirs(self) -> list[Path]:
        seen, out = set(), []
        for r in self.roots:
            if r.is_dir():
                for csvf in r.rglob("results.csv"):
                    d = csvf.parent.resolve()
                    if d not in seen and "litter_yolo" not in d.parts:
                        seen.add(d)
                        out.append(d)
        return sorted(out, key=lambda d: d.stat().st_mtime, reverse=True)

    def _id(self, d: Path) -> str:
        try:
            return d.relative_to(self.project_root).as_posix().replace("/", "__")
        except ValueError:
            return d.name

    def resolve(self, exp_id: str) -> Path:
        for d in self._run_dirs():
            if self._id(d) == exp_id:
                return d
        raise KeyError(exp_id)

    @staticmethod
    def _rows(d: Path) -> list[dict[str, Optional[float]]]:
        with (d / "results.csv").open(newline="") as fh:
            return [{k.strip(): _f(v) for k, v in r.items()} for r in csv.DictReader(fh)]

    def _summary(self, d: Path) -> dict[str, Any]:
        args = yaml.safe_load((d / "args.yaml").read_text()) if (d / "args.yaml").is_file() else {}
        rows = self._rows(d)
        best = max(rows, key=lambda r: 0.1 * (r.get("metrics/mAP50(B)") or 0) + 0.9 * (r.get("metrics/mAP50-95(B)") or 0),
                   default={})
        planned = args.get("epochs")
        status = ("complete" if planned and len(rows) >= planned
                  else "stopped early / incomplete" if (d / "weights" / "last.pt").is_file() else "no weights")
        return {
            "id": self._id(d), "name": d.name, "path": str(d),
            "date": dt.datetime.fromtimestamp(d.stat().st_mtime, dt.timezone.utc).isoformat(timespec="seconds"),
            "model": Path(str(args.get("model", "?"))).name, "dataset": Path(str(args.get("data", "?"))).parent.name,
            "epochs_planned": planned, "epochs_run": len(rows), "imgsz": args.get("imgsz"), "status": status,
            "best": {"epoch": int(best.get("epoch") or 0), "precision": best.get("metrics/precision(B)"),
                     "recall": best.get("metrics/recall(B)"), "mAP50": best.get("metrics/mAP50(B)"),
                     "mAP50_95": best.get("metrics/mAP50-95(B)")} if best else None,
            "has_best_weights": (d / "weights" / "best.pt").is_file(),
        }

    def list(self) -> list[dict[str, Any]]:
        out = []
        for d in self._run_dirs():
            try:
                out.append(self._summary(d))
            except Exception as exc:  # noqa: BLE001 - one broken run must not hide the others
                out.append({"id": self._id(d), "name": d.name, "error": str(exc)})
        return out

    def detail(self, exp_id: str) -> dict[str, Any]:
        d = self.resolve(exp_id)
        args = yaml.safe_load((d / "args.yaml").read_text()) if (d / "args.yaml").is_file() else {}
        return {**self._summary(d),
                "config": {k: args.get(k) for k in CFG_KEYS if k in args},
                "augmentation": {k: args.get(k) for k in AUG_KEYS if k in args},
                "epochs": self._rows(d),
                "artifacts": [a for a in ARTIFACTS if (d / a).is_file()]}

    def artifact(self, exp_id: str, name: str) -> Path:
        if name not in ARTIFACTS:
            raise KeyError(name)
        p = self.resolve(exp_id) / name
        if not p.is_file():
            raise KeyError(name)
        return p
