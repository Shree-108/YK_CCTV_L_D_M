"""Build dataset/cctv_store_v2: hand-reviewed labels for the store camera + street CCTV frames, the
remapped synthetic starter set, and background-only negatives (fridges, shelves, glass, cars, road).

    python -m training.build_store_dataset --labels <specs.json> --street <CCTV_Litter_100_Images> --clips <clipframes>

specs.json maps image name -> [[class_id, x1, y1, x2, y2], ...] in pixels (class ids in CLASSES order).
Splits are by SOURCE (scene / video clip), never by frame, so the test clip is unseen footage.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import yaml

from backend.core.config import PROJECT_ROOT

CLASSES = ["plastic_bag", "plastic_packaging", "food_wrapper", "paper", "cardboard",
           "plastic_bottle", "cup", "newspaper", "plastic_piece", "other_litter"]
# synthetic v1 (13 classes) -> the 10 classes above
SYN_MAP = {0: 2, 1: 2, 2: 5, 3: 6, 4: 9, 5: 3, 6: 4, 7: 0, 8: 9, 9: 3, 10: 2, 11: 1, 12: 9}
# frames with encoder artefacts (green rectangle / corrupted strip) are excluded
BAD = {"real_111.jpg", "c3min_099.jpg", "realistic_099.jpg"}


def split_of(name: str, scenes: dict[str, int]) -> str | None:
    if name in BAD:
        return None
    if name.startswith("cctv_"):
        return "train" if scenes.get(name) in (0, 2) else None   # scene B (1) is kept unlabeled for visual tests
    clip = name.split("_")[0]
    return {"c3min": "train", "real": "train", "realistic": "val", "clear": "test"}.get(clip)


def write(out: Path, split: str, name: str, img, boxes) -> None:
    h, w = img.shape[:2]
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "images" / split / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    (out / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--street", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--scenes", required=True, help="scenes.json: street image -> scene id")
    ap.add_argument("--name", default="cctv_store_v2")
    a = ap.parse_args(argv)
    out = PROJECT_ROOT / "dataset" / a.name
    if out.exists():
        shutil.rmtree(out)
    specs = json.loads(Path(a.labels).read_text())
    scenes = json.loads(Path(a.scenes).read_text())
    counts: dict[str, list[int]] = {}
    for name, boxes in sorted(specs.items()):
        split = split_of(name, scenes)
        if split is None:
            continue
        img = cv2.imread(str(Path(a.street if name.startswith("cctv_") else a.clips) / name))
        write(out, split, name, img, boxes)
        c = counts.setdefault(split, [0, 0])
        c[0] += 1
        c[1] += len(boxes)

    # negatives: store interior (fridge, shelves, products, glass doors) and street clutter (cars, road, leaves)
    neg = 0
    for p in sorted(Path(a.clips).glob("c3min_*.jpg"))[::3] + sorted(Path(a.clips).glob("real_*.jpg"))[::3]:
        img = cv2.imread(str(p))
        write(out, "train", f"neg_store_{p.stem}.jpg", img[:, 330:], [])
        neg += 1
    for n, (name, sc) in enumerate(sorted(scenes.items())):
        if sc == 0 and n % 4 == 0:
            img = cv2.imread(str(Path(a.street) / name))
            write(out, "train", f"neg_road_{Path(name).stem}.jpg", img[120:720, 940:1280], [])
            write(out, "train", f"neg_shop_{Path(name).stem}.jpg", img[0:300, 0:420], [])
            neg += 2

    # synthetic starter set, classes remapped (train/val only: it is not CCTV footage, so never in test)
    syn = PROJECT_ROOT / "dataset" / "cctv_litter_dataset_v1"
    for split in ("train", "val"):
        for ip in sorted((syn / "images" / split).glob("*.jpg")):
            lp = syn / "labels" / split / (ip.stem + ".txt")
            img = cv2.imread(str(ip))
            h, w = img.shape[:2]
            boxes = []
            for line in (lp.read_text().split("\n") if lp.is_file() else []):
                if line.strip():
                    c, x, y, bw, bh = map(float, line.split())
                    boxes.append([SYN_MAP[int(c)], (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
            write(out, split, f"syn_{ip.name}", img, boxes)
        for ip in sorted((syn / "images" / "negative" / split).glob("*.jpg")):
            write(out, split, f"synneg_{ip.name}", cv2.imread(str(ip)), [])

    (out / "data.yaml").write_text(yaml.safe_dump({"path": str(out), "train": "images/train", "val": "images/val",
                                                   "test": "images/test", "names": dict(enumerate(CLASSES))}, sort_keys=False))
    shutil.copy(a.labels, out / "labels_source.json")
    (out / "README.txt").write_text(
        "CCTV store litter dataset v2 (10 classes). Real frames hand-reviewed; split by source:\n"
        "train = street scenes A+C, store clips '3min' + 'real_life', synthetic, negatives\n"
        "val   = store clip 'realistic' (same scene as 3min: used only to pick the epoch) + synthetic val\n"
        "test  = store clip 'clear_recognizable' (unseen litter layout)\n")
    for split in ("train", "val", "test"):
        imgs = len(list((out / "images" / split).glob("*.jpg")))
        objs = sum(len(p.read_text().split()) // 5 for p in (out / "labels" / split).glob("*.txt"))
        print(f"{split}: {imgs} images, {objs} boxes")
    print("real labelled:", counts, "negatives added:", neg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
