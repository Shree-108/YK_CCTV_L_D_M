"""Register an Ultralytics training run as a new immutable model version.

    python -m training.register_model --run backend/runs/detect/runs/litter_s1 --version v1_litter_s1 \
        --name "yolo11s litter (synthetic v1)" --data dataset/cctv_litter_dataset_v1/data.yaml

Copies best.pt/last.pt/results into models/litter_yolo/<version>/ and records metadata.
Existing versions are never overwritten or deleted. Activation is a separate step
(POST /api/models/activate, or --activate here).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.core.config import AppConfig
from backend.database.db import Database
from backend.services.model_registry import ModelRegistry


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Ultralytics run dir containing weights/best.pt")
    p.add_argument("--version", required=True, help="new version id, e.g. v2_real_cctv")
    p.add_argument("--name", default=None)
    p.add_argument("--data", default=None, help="dataset data.yaml used for training")
    p.add_argument("--notes", default="")
    p.add_argument("--activate", action="store_true", help="make it the active model")
    a = p.parse_args(argv)
    cfg = AppConfig()
    cfg.ensure_dirs()
    db = Database(cfg.db_path)
    reg = ModelRegistry(db, cfg.yolo_models_dir)
    try:
        meta = reg.register_run(Path(a.run), a.version, a.name or a.version, a.notes, Path(a.data) if a.data else None)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if a.activate:
        db.set_active_model(a.version)
    print(f"registered {a.version}: {len(meta['classes'])} classes, metrics={meta['metrics']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
