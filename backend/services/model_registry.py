"""YOLO model registry: models/litter_yolo/<version>/{best.pt,last.pt,data.yaml,class_names.json,model_meta.json}.

Models are never deleted automatically. Registering copies artefacts into a new
version folder; activation just flips the ``active`` flag and hot-swaps the engine.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from backend.database.db import Database, utcnow

VERSION_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


def metrics_from_results_csv(path: Path) -> dict[str, Any]:
    """Best-epoch metrics from an Ultralytics results.csv (fitness = 0.1*mAP50 + 0.9*mAP50-95)."""
    if not path.is_file():
        return {}
    with path.open(newline="") as fh:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]
    if not rows:
        return {}

    def f(r: dict[str, str], k: str) -> float:
        try:
            return float(r.get(k, "nan"))
        except ValueError:
            return float("nan")

    best = max(rows, key=lambda r: 0.1 * f(r, "metrics/mAP50(B)") + 0.9 * f(r, "metrics/mAP50-95(B)"))
    p, r_ = f(best, "metrics/precision(B)"), f(best, "metrics/recall(B)")
    return {"epoch": int(float(best["epoch"])), "epochs_run": len(rows),
            "precision": round(p, 4), "recall": round(r_, 4),
            "f1": round(2 * p * r_ / (p + r_), 4) if p + r_ > 0 else 0.0,
            "mAP50": round(f(best, "metrics/mAP50(B)"), 4), "mAP50_95": round(f(best, "metrics/mAP50-95(B)"), 4),
            "source": "ultralytics validation during training (val split)"}


class ModelRegistry:
    def __init__(self, db: Database, root: Path) -> None:
        self.db, self.root = db, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def register_run(self, run_dir: Path, version: str, name: str, notes: str = "",
                     dataset_yaml: Optional[Path] = None, extra_meta: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Copy an Ultralytics training run into a new version folder and register it."""
        from ultralytics import YOLO

        if not VERSION_RE.match(version):
            raise ValueError("version may only contain letters, digits, '.', '_' and '-'")
        run_dir = Path(run_dir)
        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(f"{best} not found")
        dest = self.root / version
        if dest.exists():
            raise FileExistsError(f"model version {version} already exists (versions are immutable)")
        dest.mkdir(parents=True)
        shutil.copy2(best, dest / "best.pt")
        if (run_dir / "weights" / "last.pt").is_file():
            shutil.copy2(run_dir / "weights" / "last.pt", dest / "last.pt")
        for extra in ("results.csv", "args.yaml", "results.png", "confusion_matrix.png", "BoxPR_curve.png"):
            if (run_dir / extra).is_file():
                shutil.copy2(run_dir / extra, dest / extra)
        if dataset_yaml and Path(dataset_yaml).is_file():
            shutil.copy2(dataset_yaml, dest / "data.yaml")
        m = YOLO(str(dest / "best.pt"))
        names = [m.names[i] for i in sorted(m.names)]
        ckpt = getattr(m, "ckpt", None) or {}
        (dest / "class_names.json").write_text(json.dumps(names, indent=2))
        meta = {"version": version, "name": name, "trained_at": ckpt.get("date"),
                "base_model": (ckpt.get("train_args") or {}).get("model"),
                "imgsz": (ckpt.get("train_args") or {}).get("imgsz"),
                "classes": names, "metrics": metrics_from_results_csv(run_dir / "results.csv"),
                "registered_at": utcnow(), "notes": notes, **(extra_meta or {})}
        (dest / "model_meta.json").write_text(json.dumps(meta, indent=2))
        self._upsert(dest, meta)
        return meta

    def _upsert(self, folder: Path, meta: dict[str, Any]) -> None:
        self.db.upsert_model({"version": meta["version"], "name": meta.get("name", meta["version"]),
                              "path": str(folder / "best.pt"), "trained_at": meta.get("trained_at"),
                              "classes": meta.get("classes", []),
                              "metrics": {**meta.get("metrics", {}), **({"evaluation": meta["evaluation"]}
                                                                        if meta.get("evaluation") else {})},
                              "notes": meta.get("notes")})

    def sync(self) -> list[dict[str, Any]]:
        """Register every version folder on disk that has model_meta.json (idempotent)."""
        for meta_file in sorted(self.root.glob("*/model_meta.json")):
            meta = json.loads(meta_file.read_text())
            if (meta_file.parent / "best.pt").is_file():
                self._upsert(meta_file.parent, meta)
        return self.db.list_models()

    def active(self) -> Optional[dict[str, Any]]:
        m = self.db.active_model()
        if m is None:
            models = [x for x in self.db.list_models() if Path(x["path"]).is_file()]
            if models:  # first boot: activate the newest registered model
                self.db.set_active_model(models[-1]["version"])
                m = self.db.active_model()
        return m
