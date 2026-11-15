"""Copy the 100 AUG100 images into dataset/exp_aug100 (hash-checked), measure each variant's transform from its
scene reference, and write manifest.csv + data.yaml. Idempotent; never overwrites a copy with different bytes.

    python -m experiments.exp_aug100.prepare
"""
from __future__ import annotations

import datetime as dt
import shutil

from experiments.exp_aug100.common import (DATASET_VERSION, DS, IMG_DIR, LBL_DIR, MANIFEST, SCENES, SRC_DIR, check_v3,
                                           features, fname, homography, image_id, read_manifest, sha256,
                                           similarity_params, transfer_ok, write_manifest)


def main() -> int:
    if SRC_DIR is None or not SRC_DIR.is_dir():
        raise SystemExit("STOP: set AUG100_SOURCE_DIR to the original AUG100 image folder (read only)")
    v3 = check_v3()
    IMG_DIR.mkdir(parents=True, exist_ok=True); LBL_DIR.mkdir(parents=True, exist_ok=True)
    before = {p.name: sha256(p) for p in sorted(SRC_DIR.iterdir()) if p.is_file()}
    old = {r["file"]: r for r in read_manifest()} if MANIFEST.exists() else {}
    rows = []
    for scene, cfg in SCENES.items():
        ref_n = cfg["reference"]
        ref_feat = features(SRC_DIR / fname(ref_n))
        for n in cfg["ids"]:
            f = fname(n); src = SRC_DIR / f; dst = IMG_DIR / f
            if dst.exists() and sha256(dst) != before[f]:
                raise SystemExit(f"STOP: {dst} exists with different bytes; refusing to overwrite")
            if not dst.exists():
                shutil.copy2(src, dst)
            if sha256(dst) != before[f]:
                raise SystemExit(f"STOP: copy of {f} does not match the original")
            row = dict(old.get(f, {}))
            row.update(image_id=image_id(scene, n), file=f, source_path=f,  # file name only (no machine path)
                       image_path=f"dataset/exp_aug100/images/train/{f}", label_path=f"dataset/exp_aug100/labels/train/{f[:-4]}.txt",
                       sha256=before[f], scene=scene, role="reference" if n == ref_n else "variant",
                       reference_id=image_id(scene, ref_n), dataset_version=DATASET_VERSION, source_type="synthetic")
            if n == ref_n:
                row.update(aug_scale="1.000", aug_tx="0.0", aug_ty="0.0", aug_rot_deg="0.000", aug_rms_px="0.00", aug_inliers="")
            else:
                H, inl, rms = homography(ref_feat, features(src))
                p = similarity_params(H)
                ok, why = transfer_ok(p, inl, rms)
                row.update(aug_scale=f"{p['scale']:.3f}", aug_tx=f"{p['tx']:.1f}", aug_ty=f"{p['ty']:.1f}",
                           aug_rot_deg=f"{p['rot_deg']:.3f}", aug_rms_px=f"{rms:.2f}", aug_inliers=str(inl))
                if not ok:
                    row["notes"] = f"transfer refused: {why} -> manual annotation"
            row.setdefault("label_status", "none")
            for k in ("verified_by", "verified_at", "n_boxes", "n_transferred", "n_manual", "notes"):
                row.setdefault(k, "")
            if not row["label_status"]:
                row["label_status"] = "none"
            rows.append(row)
            print(f"{f} {scene} {row['role']:9s} scale={row['aug_scale']} rms={row['aug_rms_px']} {row.get('notes','')}")
    rows.sort(key=lambda r: r["file"])
    write_manifest(rows)
    after = {p.name: sha256(p) for p in sorted(SRC_DIR.iterdir()) if p.is_file()}
    if after != before:
        raise SystemExit("STOP: the original dataset changed during preparation")
    (DS / "source_sha256.txt").write_text(
        f"# SHA-256 of the original AUG100 source folder ({SRC_DIR.name}, via AUG100_SOURCE_DIR) at {dt.datetime.now().isoformat(timespec='seconds')} (unchanged before/after copy)\n"
        + "".join(f"{h}  {n}\n" for n, h in before.items()), encoding="utf-8")
    (DS / "data.yaml").write_text(
        "# ANNOTATION VIEW for the Ground Truth page. NOT a training config:\n"
        "# training uses experiments/exp_aug100/data_*.yaml (scene-split path lists).\n"
        "path: .\ntrain: images/train\nval: images/train\nnames:\n  0: litter\n", encoding="utf-8")
    print(f"\n{len(rows)} images; v3 sha256 unchanged: {v3 == check_v3()}; original dataset unchanged: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
