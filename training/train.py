"""Train a litter detector on a reviewed YOLO dataset, then register it as a new model version.

Local (CPU, slow):   python -m training.train --data dataset/cctv_real_v1_autolabel/data.yaml --model yolo11s.pt
GPU / Colab:         python -m training.train --data <data.yaml> --model yolo11m.pt --imgsz 1280 --batch 16 --device 0

Augmentation is chosen for FIXED CCTV cameras:
  hsv_v / hsv_s   brightness & saturation shifts (day / dusk / overcast / IR-ish)       ON
  scale 0.5       objects at different distances from the camera                      ON
  degrees 5       small tilt only; the camera does not rotate                           LOW
  perspective     tiny; the viewpoint is fixed                                          LOW
  fliplr 0.5      litter has no left/right orientation                                  ON
  flipud 0        the ground is never above the camera                                  OFF
  mosaic 1.0      many small objects per image, varied context                           ON (off last 10 epochs)
  mixup 0.05      mild, avoids ghost-like litter                                        LOW
  copy_paste 0    needs segmentation labels                                             OFF
Blur / noise / JPEG artefacts come from Ultralytics' Albumentations hook when installed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.core.config import AppConfig
from backend.database.db import Database
from backend.services.model_registry import ModelRegistry

CCTV_AUG = dict(hsv_h=0.015, hsv_s=0.5, hsv_v=0.45, degrees=5.0, translate=0.1, scale=0.5, shear=1.0,
                perspective=0.0003, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.05, copy_paste=0.0, close_mosaic=10)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11s.pt", help="yolo11n/s/m/l/x.pt (pretrained COCO) or a previous best.pt")
    ap.add_argument("--imgsz", type=int, default=1280, help="high resolution helps small litter")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=25, help="early stopping")
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU (auto if omitted)")
    ap.add_argument("--name", default=None, help="run name / new model version")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze", type=int, default=None, help="freeze the first N layers (faster fine-tuning on CPU)")
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    cfg = AppConfig()
    project = cfg.storage_dir / "runs"
    name = a.name or f"{Path(a.model).stem}_{Path(a.data).parent.name}_{a.imgsz}"
    model = YOLO(a.model)
    model.train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, patience=a.patience, batch=a.batch, device=a.device,
                project=str(project), name=name, seed=a.seed, deterministic=True, multi_scale=0.0, cos_lr=True,
                plots=True, freeze=a.freeze, **CCTV_AUG)
    run = project / name
    print(f"training finished: {run}")
    if a.no_register:
        return 0
    reg = ModelRegistry(Database(cfg.db_path), cfg.yolo_models_dir)
    meta = reg.register_run(run, name, name, notes=f"Trained on {a.data} (imgsz {a.imgsz}).", dataset_yaml=Path(a.data))
    print(f"registered model version {name}: {meta['metrics']}  -> evaluate it, then deploy from the Model Lab")
    return 0


if __name__ == "__main__":
    sys.exit(main())
