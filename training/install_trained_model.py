"""Install a model package produced by training/colab/LDM_YOLO11s_Training.ipynb into models/litter_yolo/<version>/.

    python -m training.install_trained_model --package "C:\\Downloads\\ldm_model_v4_aug100_litter1c.zip"

Controlled model selection:
  * A NEW version folder only: an existing folder (any version, including v3_store_cctv) is never overwritten.
  * The package is checked before anything is copied: SHA256SUMS, model_meta.json, class_names.json, and the class
    list stored inside best.pt itself.
  * Nothing is activated and the database is not touched here. On its next start the backend's model registry
    (backend/services/model_registry.py ModelRegistry.sync) lists the folder as an available, NON-active model; it can
    then be chosen per analysis in Image Analysis -> Model version.
  * Smoke-test packages (a notebook test run, not a real model) are refused unless --allow-smoke.

Rollback: while the new version is not active, deleting its folder removes it. If someone activated it, re-activate
v3_store_cctv on the Models page (POST /api/models/activate) - v3 weights are never modified by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTECTED = {"v3_store_cctv"}
REQUIRED = ("best.pt", "model_meta.json", "class_names.json", "SHA256SUMS")


class InstallError(Exception):
    pass


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_package(folder: Path, allow_smoke: bool = False, load_model: bool = True) -> dict[str, Any]:
    missing = [f for f in REQUIRED if not (folder / f).is_file()]
    if missing:
        raise InstallError(f"package incomplete, missing {missing}")
    for line in (folder / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        p = folder / rel.strip()
        if ".." in Path(rel.strip()).parts or not p.is_file() or _sha(p) != digest:
            raise InstallError(f"checksum failed for {rel.strip()}")
    meta = json.loads((folder / "model_meta.json").read_text(encoding="utf-8"))
    names = json.loads((folder / "class_names.json").read_text(encoding="utf-8"))
    if meta.get("version") != folder.name:
        raise InstallError(f"model_meta version {meta.get('version')!r} != folder {folder.name!r}")
    if meta.get("classes") != names:
        raise InstallError("model_meta classes and class_names.json differ")
    if meta.get("smoke_test") and not allow_smoke:
        raise InstallError("this is a SMOKE-TEST package (not a trained model); use --allow-smoke only for testing")
    if load_model:
        from ultralytics import YOLO
        m = YOLO(str(folder / "best.pt"))
        inside = [m.names[i] for i in sorted(m.names)]
        if inside != names:
            raise InstallError(f"best.pt classes {inside} differ from class_names.json {names}")
    return meta


def install(package: Path, models_dir: Path, allow_smoke: bool = False, load_model: bool = True) -> dict[str, Any]:
    package = Path(package)
    with tempfile.TemporaryDirectory() as tmp:
        if package.is_file():
            with zipfile.ZipFile(package) as z:
                for n in z.namelist():
                    if n.startswith(("/", "\\")) or ".." in Path(n).parts:
                        raise InstallError(f"unsafe path in zip: {n}")
                z.extractall(tmp)
            tops = [p for p in Path(tmp).iterdir() if p.is_dir()]
            if len(tops) != 1:
                raise InstallError("the zip must contain exactly one version folder")
            src = tops[0]
        elif package.is_dir():
            src = package
        else:
            raise InstallError(f"{package} not found")
        version = src.name
        if version in PROTECTED:
            raise InstallError(f"{version} is the production model: refusing")
        dest = Path(models_dir) / version
        if dest.exists():
            raise InstallError(f"{dest} already exists: versions are immutable, choose a new MODEL_VERSION")
        meta = verify_package(src, allow_smoke, load_model)
        protected_before = {v: _sha(Path(models_dir) / v / "best.pt") for v in PROTECTED
                            if (Path(models_dir) / v / "best.pt").is_file()}
        shutil.copytree(src, dest)
        for v, h in protected_before.items():
            if _sha(Path(models_dir) / v / "best.pt") != h:
                raise InstallError(f"{v} changed during install")
    return {"installed": str(dest), "version": version, "classes": meta.get("classes"),
            "active": False, "note": "Restart the backend; then choose it in Image Analysis -> Model version."}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Install a trained LDM model package (never activates it)")
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models" / "litter_yolo")
    ap.add_argument("--allow-smoke", action="store_true")
    a = ap.parse_args(argv)
    try:
        print(json.dumps(install(a.package, a.models_dir, a.allow_smoke), indent=2))
    except (InstallError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
