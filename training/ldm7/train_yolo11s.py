"""Train the seven-class LDM detector from COCO-pretrained YOLO11s and write an INSTALLABLE package to a staging dir.

GPU (Colab):   python -m training.ldm7.train_yolo11s --data /content/ldm7_v1 --staging /content/drive/MyDrive/LDM7/staging \
                   --device 0 [--imgsz 1280] [--epochs 150] [--zip]
Smoke (CPU):   py -m training.ldm7.train_yolo11s --data <tiny dataset> --staging <tmp> --smoke     (proves the code path)

Rules enforced here:
  * starts from models/pretrained/yolo11s.pt (COCO) - never from v3_store_cctv or any models/litter_yolo weights:
    v3 has a different, incompatible 10-class schema; fine-tuning it would carry its label semantics over.
  * refuses to start unless training/ldm7/verify_dataset.py passes (zero ERRORs) on --data.
  * no downloads: the base weights must exist locally.
  * the package goes to --staging/<version>/ (never models/): best.pt, class_names.json, data.yaml (dataset copy),
    train_args.json, args.yaml, results.csv, dataset_info.json, verify_report.json, evaluation.json,
    model_meta.json, README.txt, SHA256SUMS - exactly what training/install_trained_model.py verifies
    (SHA256SUMS, model_meta.version == folder name, model_meta.classes == class_names.json == classes inside
    best.pt). That installer has no class-count assumption, so a 7-class package needs NO extra flag.
  * nothing is installed or activated here. Install with:
        py -m training.install_trained_model --package <staging>/v4_ldm7_yolo11s   (or the .zip)

imgsz default 1280: the target objects (e.g. a ~68x32 px wafer packet in a 2560x1440 frame) shrink to ~17x8 px at
640. 1280 is a HYPOTHESIS - it must be justified by small_object_experiment.py / compare_models.py on the
independent test set (train a 640 baseline with the same data and compare) before it is kept; it costs ~4x compute.

Augmentation (fixed CCTV cameras; adapted from CCTV_AUG in training/train.py):
  flipud 0 (ground never above camera), fliplr 0.5, small rotation/shear/perspective (fixed viewpoint),
  hsv brightness/saturation (day/dusk/IR-ish), scale 0.4 (not 0.5: strong zoom-out erases already tiny litter),
  MILD mosaic 0.5 (off for the last 15 epochs) and mixup 0.05 (no ghost litter), copy_paste 0 (box labels only).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Optional

from training.ldm7.schema import CLASSES, MODEL_VERSION, SCHEMA_VERSION, write_data_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = PROJECT_ROOT / "models" / "pretrained" / "yolo11s.pt"
DEFAULT_STAGING = PROJECT_ROOT / "runs" / "ldm7_staging"
FORBIDDEN_BASE_DIR = PROJECT_ROOT / "models" / "litter_yolo"
FORBIDDEN_STAGING_DIR = PROJECT_ROOT / "models"

LDM7_AUG = dict(hsv_h=0.015, hsv_s=0.5, hsv_v=0.45, degrees=3.0, translate=0.1, scale=0.4, shear=1.0,
                perspective=0.0003, flipud=0.0, fliplr=0.5, mosaic=0.5, mixup=0.05, copy_paste=0.0, close_mosaic=15)


class TrainError(Exception):
    pass


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _inside(p: Path, parent: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def check_base(base: Path) -> None:
    if _inside(base, FORBIDDEN_BASE_DIR):
        raise TrainError(f"{base} is a project litter model: ldm7 must start from COCO-pretrained YOLO11s "
                         f"({DEFAULT_BASE}), never from v3_store_cctv or another litter model")
    if not Path(base).is_file():
        raise TrainError(f"base weights {base} not found (no automatic download)")


def check_staging(staging: Path, version: str) -> Path:
    if _inside(staging, FORBIDDEN_STAGING_DIR):
        raise TrainError(f"staging {staging} is inside models/: the backend registry auto-lists that folder; "
                         f"install only through training/install_trained_model.py")
    pkg = Path(staging) / version
    if pkg.exists():
        raise TrainError(f"{pkg} already exists: packages are immutable, use a new --version or staging dir")
    return pkg


def write_sha256sums(pkg: Path) -> None:
    """'<sha256>  <relative path>' lines, the format training/install_trained_model.py parses."""
    lines = [f"{_sha(p)}  {p.relative_to(pkg).as_posix()}" for p in sorted(pkg.rglob("*"))
             if p.is_file() and p.name != "SHA256SUMS"]
    (pkg / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_package(run_dir: Path, staging: Path, data_root: Path, verify_report: dict[str, Any],
                  train_args: dict[str, Any], base_model: str, imgsz: int, version: str = MODEL_VERSION,
                  dataset_version: Optional[str] = None, evaluation: Optional[dict[str, Any]] = None,
                  smoke: bool = False, check_weights: bool = True, make_zip: bool = False) -> Path:
    """Assemble <staging>/<version>/ from an Ultralytics run dir (needs weights/best.pt)."""
    pkg = check_staging(staging, version)
    best = Path(run_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise TrainError(f"{best} missing: training did not produce a best checkpoint")
    if check_weights:
        from ultralytics import YOLO
        m = YOLO(str(best))
        inside = [m.names[i] for i in sorted(m.names)]
        if inside != CLASSES:
            raise TrainError(f"best.pt classes {inside} != ldm7 schema {CLASSES}")
    pkg.mkdir(parents=True)
    shutil.copy2(best, pkg / "best.pt")
    for name in ("results.csv", "args.yaml"):
        if (Path(run_dir) / name).is_file():
            shutil.copy2(Path(run_dir) / name, pkg / name)
    shutil.copy2(Path(data_root) / "data.yaml", pkg / "data.yaml")
    (pkg / "class_names.json").write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")
    (pkg / "train_args.json").write_text(json.dumps(train_args, indent=1, default=str), encoding="utf-8")
    ds_version = dataset_version or Path(data_root).name
    (pkg / "dataset_info.json").write_text(json.dumps({
        "dataset_version": ds_version, "dataset_hash": verify_report.get("dataset_hash"),
        "root": str(data_root), "schema": SCHEMA_VERSION, "splits": verify_report.get("splits"),
        "verify_errors": verify_report.get("errors"), "verify_warnings": verify_report.get("warnings")},
        indent=1), encoding="utf-8")
    (pkg / "verify_report.json").write_text(json.dumps(verify_report, indent=1), encoding="utf-8")
    (pkg / "evaluation.json").write_text(json.dumps(evaluation or {"note": "no evaluation recorded"}, indent=1),
                                         encoding="utf-8")
    meta = {
        "version": version, "name": version, "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "sha256": _sha(pkg / "best.pt"), "classes": CLASSES, "schema": SCHEMA_VERSION, "base_model": base_model,
        "imgsz": imgsz, "dataset_version": ds_version, "dataset_hash": verify_report.get("dataset_hash"),
        "smoke_test": bool(smoke),
        "metrics": (evaluation or {}).get("test") or (evaluation or {}).get("val") or {},
        "notes": "Seven-class LDM detector trained by training/ldm7/train_yolo11s.py. Install with "
                 "training/install_trained_model.py; never activated automatically.",
    }
    (pkg / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (pkg / "README.txt").write_text(
        f"{version} - seven-class LDM litter detector ({', '.join(CLASSES)}).\n"
        f"Base: {base_model}. imgsz {imgsz}. Dataset {ds_version} ({verify_report.get('dataset_hash')}).\n"
        f"{'SMOKE TEST PACKAGE - NOT A REAL MODEL. ' if smoke else ''}"
        "Install: py -m training.install_trained_model --package <this folder or zip>  (never activates).\n"
        "Rollback: delete models/litter_yolo/<version>/ while not active; if activated, re-activate v3_store_cctv.\n"
        "The app's class metadata (backend/core/class_metadata.json) has no entry for plastic_packet, food_waste "
        "or glass yet - add them (separate approved backend change) before using this model for alerts.\n",
        encoding="utf-8")
    write_sha256sums(pkg)
    if make_zip:
        zp = Path(staging) / f"ldm_model_{version}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(pkg.rglob("*")):
                if p.is_file():
                    z.write(p, Path(version) / p.relative_to(pkg))
    return pkg


def _metrics_dict(metrics: Any) -> dict[str, Any]:
    out: dict[str, Any] = {k: float(v) for k, v in getattr(metrics, "results_dict", {}).items()}
    per = {}
    try:
        for k, ci in enumerate(metrics.box.ap_class_index):
            p, r, ap50, ap = metrics.box.class_result(k)
            per[CLASSES[int(ci)]] = {"precision": float(p), "recall": float(r), "ap50": float(ap50), "ap50_95": float(ap)}
    except Exception as exc:  # metrics API drift - keep the run, record why
        per = {"error": str(exc)}
    out["per_class"] = per
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="verified ldm7 dataset root (has data.yaml)")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    ap.add_argument("--version", default=MODEL_VERSION)
    ap.add_argument("--dataset-version", default=None)
    ap.add_argument("--imgsz", type=int, default=1280, help="1280 is a hypothesis: justify it against a 640 run")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-instances", type=int, default=50)
    ap.add_argument("--smoke", action="store_true", help="1 epoch at 320 px, marks the package smoke_test")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args(argv)
    try:
        check_base(a.base)
        check_staging(a.staging, a.version)
        from training.ldm7.verify_dataset import verify, write_reports
        rep = verify(a.data, min_instances=a.min_instances)
        write_reports(rep, a.staging / f"{a.version}_verify")
        if not rep["ok"]:
            raise TrainError(f"verify_dataset found {rep['errors']} error(s) - see {a.staging / (a.version + '_verify')}"
                             "/report.md; training refused")
        has_test = rep["splits"]["test"]["images"] > 0
        runtime_dir = a.staging / f"{a.version}_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        data_yaml = write_data_yaml(runtime_dir, path_value=str(Path(a.data).resolve()),
                                    splits=("train", "val", "test") if has_test else ("train", "val"),
                                    filename="data_runtime.yaml")
        epochs, imgsz = (1, 320) if a.smoke else (a.epochs, a.imgsz)
        args = dict(data=str(data_yaml), imgsz=imgsz, epochs=epochs, patience=a.patience, batch=a.batch,
                    device=a.device, workers=a.workers, seed=a.seed, deterministic=True, cos_lr=True, plots=True,
                    project=str((a.staging / "runs").resolve()), name=a.version, exist_ok=False, **LDM7_AUG)
        if a.smoke:
            args.update(batch=2, workers=0, plots=False)
        from ultralytics import YOLO
        model = YOLO(str(a.base))
        model.train(**args)
        run_dir = Path(model.trainer.save_dir)
        best = YOLO(str(run_dir / "weights" / "best.pt"))
        evaluation = {"val": _metrics_dict(best.val(data=str(data_yaml), split="val", imgsz=imgsz, device=a.device,
                                                    plots=False, verbose=False))}
        if has_test:
            evaluation["test"] = _metrics_dict(best.val(data=str(data_yaml), split="test", imgsz=imgsz,
                                                        device=a.device, plots=False, verbose=False))
        else:
            evaluation["test_note"] = "NO TEST SPLIT - metrics are validation-only (optimistic, used for checkpoint choice)"
        pkg = build_package(run_dir, a.staging, Path(a.data), rep, {**args, "base": str(a.base)},
                            base_model=str(a.base.name), imgsz=imgsz, version=a.version,
                            dataset_version=a.dataset_version, evaluation=evaluation, smoke=a.smoke, make_zip=a.zip)
    except TrainError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"package ready (NOT installed, NOT active): {pkg}\n"
          f"install: py -m training.install_trained_model --package \"{pkg}\"" + (" --allow-smoke" if a.smoke else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
