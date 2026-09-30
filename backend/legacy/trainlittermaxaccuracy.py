#!/usr/bin/env python3
"""
trainlittermaxaccuracy.py — ML training for the CCTV litter model (25 classes).

NO YOLO / NO torch-hub downloads. Everything trains on CPU from scratch.

Inputs : one or more YOLO-format datasets (--data a.yaml b.yaml ...).
         Label ids are remapped per dataset into the CANONICAL 25-class order
         (local 13 first, 12 TACO-sourced classes appended), so the original
         dataset.yaml (13 names) and D:/litter_extra/data.yaml (25 names)
         can be mixed freely.

Stages:
  1. PREFLIGHT — validate datasets, report combined class balance.
  2. DATASET   — GT positives (+augmented variants, oversampling rare classes)
                 + random negatives + hard negatives from the proposer.
  3. TRAIN RF  — StandardScaler + RandomForest on v4 handcrafted features.
  4. TRAIN CNN — tiny CNN from scratch on 48px crops, class-weighted CE.
  5. VALIDATE  — full propose->classify->NMS detection on val images,
                 IoU>=0.5 matching, per-class P/R/F1. Winner (RF vs CNN)
                 is exported.
  6. TUNE      — per-class confidence sweep maximising F1 (skippable).
  7. EXPORT    — litter_cnn.pt and/or litter_ml_model.pkl + litter_config.json
                 into backend/ (plus a copy under <project>/backend_bundle/).

Usage:
    python trainlittermaxaccuracy.py --data ../dataset/cctv_litter_dataset_v1/data.yaml D:/litter_extra/data.yaml --project D:/runs-ml
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import yaml

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("Install scikit-learn first:  pip install scikit-learn")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import litter_ml
from litter_ml import (BACKGROUND_LABEL, CNN_INPUT, extract_features,
                       propose_boxes, box_iou, nms,
                       preprocess_crops_for_cnn, get_cnn_module)

# ----------------------------------------------------------------------------------
# Canonical 25-class taxonomy: local 13 (fixed order) + 12 TACO-sourced classes.
# ----------------------------------------------------------------------------------
CANONICAL = [
    "chips_packet", "food_wrapper", "plastic_bottle", "plastic_cup", "metal_can",
    "paper", "cardboard", "plastic_bag", "food_container", "tissue",
    "cigarette_packet", "plastic_packaging", "other_litter",
    "cigarette_butt", "bottle_cap", "straw", "carton", "paper_cup",
    "glass_bottle", "lid", "styrofoam", "foil", "plastic_film", "pop_tab",
    "cutlery",
]

PRIORITY_MAP = {
    "chips_packet": "critical", "plastic_bag": "critical",
    "glass_bottle": "critical",
    "food_wrapper": "high", "plastic_packaging": "high",
    "food_container": "high", "cigarette_packet": "high",
    "plastic_bottle": "medium", "metal_can": "medium", "plastic_cup": "medium",
    "paper": "medium", "straw": "medium", "cutlery": "medium",
    "plastic_film": "medium",
    "cigarette_butt": "high", "styrofoam": "high",
    "cardboard": "low", "tissue": "low", "other_litter": "low",
    "bottle_cap": "low", "carton": "low", "paper_cup": "low", "lid": "low",
    "foil": "low", "pop_tab": "low",
}

BIN_MAP = {
    "chips_packet": "general", "food_wrapper": "general", "tissue": "general",
    "food_container": "general", "cigarette_packet": "general",
    "other_litter": "general", "cigarette_butt": "general",
    "plastic_bag": "general", "paper_cup": "general", "styrofoam": "general",
    "plastic_bottle": "plastic", "plastic_cup": "plastic",
    "plastic_packaging": "plastic", "straw": "plastic", "cutlery": "plastic",
    "plastic_film": "plastic", "lid": "plastic",
    "metal_can": "metal", "bottle_cap": "metal", "foil": "metal",
    "pop_tab": "metal",
    "paper": "paper", "cardboard": "paper", "carton": "paper",
    "glass_bottle": "glass",
}

PRIORITY_COLORS = {
    "critical": "#f87171", "high": "#fb923c",
    "medium": "#fbbf24", "low": "#34d399",
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="one or more data.yaml paths")
    p.add_argument("--project", default="runs", help="output dir for bundle (D:/runs-ml spares C:)")
    p.add_argument("--estimators", type=int, default=250)
    p.add_argument("--neg-ratio", type=float, default=2.0)
    p.add_argument("--cnn-epochs", type=int, default=25)
    p.add_argument("--cnn-batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--no-cnn", action="store_true", help="skip CNN, RF only")
    return p.parse_args()


# ----------------------------------------------------------------------------------
# Dataset loading (multi-yaml, canonical remap)
# ----------------------------------------------------------------------------------

def load_dataset(data_yaml: Path):
    cfg = yaml.safe_load(data_yaml.read_text())
    names = cfg.get("names")
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names or [])
    if not names:
        sys.exit(f"{data_yaml} has no 'names' entry.")
    remap = {}
    for old_idx, n in enumerate(names):
        if n not in CANONICAL:
            sys.exit(f"{data_yaml}: class '{n}' not in canonical taxonomy")
        remap[old_idx] = CANONICAL.index(n)
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        cand = (data_yaml.parent / root).resolve()
        root = cand if cand.exists() else data_yaml.parent.resolve()
    return {"yaml": data_yaml, "cfg": cfg, "names": names, "remap": remap, "root": root}


def split_images(ds, split: str) -> list[Path]:
    rel = ds["cfg"].get(split, f"images/{split}")
    d = ds["root"] / rel
    if not d.exists():
        d = ds["root"] / "images" / split
    if not d.exists():
        return []
    files = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXT and p.is_file()]
    # nested negative layout (local dataset)
    neg = ds["root"] / "images" / "negative" / split
    if neg.exists():
        files += [p for p in neg.iterdir() if p.suffix.lower() in IMG_EXT and p.is_file()]
    return sorted(files)


def read_gt(ds, img_p: Path, W: int, H: int) -> list[tuple[int, list[float]]]:
    lbl = Path(str(img_p.parent).replace("images", "labels")) / f"{img_p.stem}.txt"
    gt = []
    if lbl.exists():
        for ln in lbl.read_text().splitlines():
            parts = ln.split()
            if len(parts) != 5:
                continue
            try:
                old = int(float(parts[0]))
            except ValueError:
                continue
            if old not in ds["remap"]:
                continue
            cid = ds["remap"][old]
            cx, cy, w, h = (float(parts[1]) * W, float(parts[2]) * H,
                            float(parts[3]) * W, float(parts[4]) * H)
            gt.append((cid, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    return gt


def preflight(datasets) -> None:
    print("Datasets:")
    total_imgs = collections_count(datasets)
    print(f"  {total_imgs['train']} train / {total_imgs['val']} val images "
          f"across {len(datasets)} dataset(s)")
    print(f"  Canonical classes ({len(CANONICAL)}): {CANONICAL}")


def collections_count(datasets):
    out = {}
    for split in ("train", "val"):
        out[split] = sum(len(split_images(ds, split)) for ds in datasets)
    return out


# ----------------------------------------------------------------------------------
# Box helpers + augmentation
# ----------------------------------------------------------------------------------

def expand_box(box, W, H, frac=0.10):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return [max(0, x1 - frac * bw), max(0, y1 - frac * bh),
            min(W, x2 + frac * bw), min(H, y2 + frac * bh)]


def crop_or_none(img, box):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(max(0, box[0])), int(max(0, box[1])),
                      int(min(W, box[2] + 1)), int(min(H, box[3] + 1)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return img[y1:y2, x1:x2]


def max_iou(box, others) -> float:
    if not others:
        return 0.0
    a = np.array([box], np.float32)
    b = np.array(others, np.float32)
    return float(box_iou(a, b)[0].max())


def augment_crop(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric + mild geometric jitter for classifier robustness."""
    img = crop
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    ang = float(rng.uniform(-12, 12))
    if abs(ang) > 1.0:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.7:
        alpha = float(rng.uniform(0.75, 1.25))  # contrast
        beta = float(rng.uniform(-25, 25))      # brightness
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


# ----------------------------------------------------------------------------------
# Training-set build (shared positives/negatives for RF + CNN)
# ----------------------------------------------------------------------------------

def build_training_set(datasets, neg_ratio: float, rng: np.random.Generator):
    print("\n" + "=" * 74 + "\nBUILDING TRAINING SET\n" + "=" * 74)
    # gather GT + size prior
    all_items = []  # (ds, img_path)
    sizes = []
    gt_cache: dict[str, list] = {}
    for ds in datasets:
        for img_p in split_images(ds, "train"):
            im = cv2.imread(str(img_p))
            if im is None:
                continue
            H, W = im.shape[:2]
            gt = read_gt(ds, img_p, W, H)
            gt_cache[str(img_p)] = gt
            all_items.append((ds, img_p))
            for _, b in gt:
                sizes.append((b[2] - b[0], b[3] - b[1]))
    if not sizes:
        sys.exit("No GT boxes found — cannot train.")
    sizes_arr = np.array(sizes)

    # class frequency -> augmentation multiplicity (rare classes get more variants)
    freq = collections.Counter(cid for gt in gt_cache.values() for cid, _ in gt)
    print("  train boxes per class:")
    for ci, n in enumerate(CANONICAL):
        print(f"    {n:20s} {freq.get(ci, 0):5d}")
    maxf = max(freq.values())

    pos_crops: list[np.ndarray] = []   # BGR crops (for CNN)
    pos_labels: list[str] = []
    rf_X: list[np.ndarray] = []
    rf_y: list = []
    n_neg = 0

    def add_positive(crop, cname, multiplicity):
        variants = [crop] + [augment_crop(crop, rng) for _ in range(multiplicity - 1)]
        for v in variants:
            pos_crops.append(v)
            pos_labels.append(cname)
            rf_X.append(extract_features(v))
            rf_y.append(cname)

    for ds, img_p in all_items:
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        H, W = im.shape[:2]
        gt = gt_cache[str(img_p)]
        gt_boxes = [b for _, b in gt]
        for cid, b in gt:
            crop = crop_or_none(im, expand_box(b, W, H))
            if crop is None:
                continue
            # rare classes (<25% of max) get 4 variants, common get 2
            mult = 4 if freq[cid] < 0.25 * maxf else 2
            add_positive(crop, CANONICAL[cid], mult)
        # random negatives
        n_want = min(8, int(neg_ratio * max(1, len(gt))) if gt else 4)
        for _ in range(n_want):
            sw, sh = sizes_arr[rng.integers(0, len(sizes_arr))]
            sw *= float(rng.uniform(0.8, 1.4))
            sh *= float(rng.uniform(0.8, 1.4))
            bw = float(np.clip(sw, 10, W * 0.5))
            bh = float(np.clip(sh, 10, H * 0.5))
            b = [float(rng.uniform(0, max(1, W - bw))), float(rng.uniform(0, max(1, H - bh))),
                 0, 0]
            b[2], b[3] = b[0] + bw, b[1] + bh
            if max_iou(b, gt_boxes) > 0.15:
                continue
            crop = crop_or_none(im, b)
            if crop is None:
                continue
            pos_crops.append(crop); pos_labels.append(BACKGROUND_LABEL)
            rf_X.append(extract_features(crop)); rf_y.append(BACKGROUND_LABEL)
            n_neg += 1
        # hard negatives from proposer
        try:
            small = im if max(H, W) <= 960 else cv2.resize(
                im, (int(W * 960 / max(H, W)), int(H * 960 / max(H, W))))
            props = propose_boxes(small, max_proposals=60)
            sx = small.shape[1] / W
            hard = 0
            for pb in props:
                b = [pb[0] / sx, pb[1] / sx, pb[2] / sx, pb[3] / sx]
                if max_iou(b, gt_boxes) > 0.2 or hard >= 4:
                    continue
                crop = crop_or_none(im, b)
                if crop is None:
                    continue
                pos_crops.append(crop); pos_labels.append(BACKGROUND_LABEL)
                rf_X.append(extract_features(crop)); rf_y.append(BACKGROUND_LABEL)
                n_neg += 1; hard += 1
        except Exception:
            pass

    X = np.stack(rf_X).astype(np.float32)
    y = np.array(rf_y, dtype=object)
    n_pos = int((y != BACKGROUND_LABEL).sum())
    print(f"  positives (augmented): {n_pos}  negatives: {n_neg}  X: {X.shape}")
    neg_idx = np.where(y == BACKGROUND_LABEL)[0]
    max_neg = int(n_pos * max(1.0, neg_ratio) * 1.5) or 1
    keep = np.ones(len(y), bool)
    if len(neg_idx) > max_neg:
        drop = rng.choice(neg_idx, size=len(neg_idx) - max_neg, replace=False)
        keep[drop] = False
        X, y = X[keep], y[keep]
        print(f"  capped negatives -> {X.shape}")
    keep_pos = [pos_crops[i] for i in np.where(keep)[0]]
    keep_lab = [pos_labels[i] for i in np.where(keep)[0]]
    return X, y, keep_pos, keep_lab


# ----------------------------------------------------------------------------------
# RF training
# ----------------------------------------------------------------------------------

def train_rf(X, y, estimators: int):
    print("\n" + "=" * 74 + "\nTRAINING RandomForest (v4 features)\n" + "=" * 74)
    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=estimators, min_samples_leaf=2,
                               class_weight="balanced_subsample",
                               n_jobs=-1, random_state=42),
    )
    clf.fit(X, y)
    print(f"  train accuracy (crops): {clf.score(X, y):.4f}")
    return clf


# ----------------------------------------------------------------------------------
# CNN training (from scratch, CPU)
# ----------------------------------------------------------------------------------

class CropDataset:
    def __init__(self, crops, labels, label_list, mean, std, train: bool, rng):
        self.crops = crops
        self.y = np.array([label_list.index(l) for l in labels], dtype=np.int64)
        self.labels_list = label_list
        self.mean = mean
        self.std = std
        self.train = train
        self.rng = rng

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        import torch
        c = self.crops[i]
        if self.train:
            c = augment_crop(c, self.rng)
        x = preprocess_crops_for_cnn([c], self.mean, self.std)[0]
        return torch.from_numpy(x), int(self.y[i])


def build_val_crops(datasets):
    crops, labels = [], []
    for ds in datasets:
        for img_p in split_images(ds, "val"):
            im = cv2.imread(str(img_p))
            if im is None:
                continue
            H, W = im.shape[:2]
            gt = read_gt(ds, img_p, W, H)
            for cid, b in gt:
                c = crop_or_none(im, expand_box(b, W, H))
                if c is not None:
                    crops.append(c); labels.append(CANONICAL[cid])
            for _ in range(6):
                bw = float(np.random.uniform(15, W * 0.3))
                bh = float(np.random.uniform(15, H * 0.3))
                b = [float(np.random.uniform(0, max(1, W - bw))),
                     float(np.random.uniform(0, max(1, H - bh))), 0, 0]
                b[2], b[3] = b[0] + bw, b[1] + bh
                if max_iou(b, [g for _, g in gt]) > 0.15:
                    continue
                c = crop_or_none(im, b)
                if c is not None:
                    crops.append(c); labels.append(BACKGROUND_LABEL)
    return crops, labels


def train_cnn(train_crops, train_labels, val_crops, val_labels,
              epochs: int, batch: int, seed: int):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    print("\n" + "=" * 74 + "\nTRAINING tiny CNN from scratch (CPU)\n" + "=" * 74)

    label_list = CANONICAL + [BACKGROUND_LABEL]
    # dataset mean/std from a sample (RGB)
    sample = np.stack([
        cv2.resize(c, (CNN_INPUT, CNN_INPUT))[..., ::-1]
        for c in train_crops[::max(1, len(train_crops) // 2000)][:2000]
    ]).astype(np.float32) / 255.0
    mean = [float(sample[..., i].mean()) for i in range(3)]
    std = [float(sample[..., i].std()) + 1e-6 for i in range(3)]
    print(f"  norm mean={mean} std={std}")

    rng = np.random.default_rng(seed)
    trainds = CropDataset(train_crops, train_labels, label_list, mean, std, True, rng)
    valds = CropDataset(val_crops, val_labels, label_list, mean, std, False, rng)
    train_loader = torch.utils.data.DataLoader(trainds, batch_size=batch,
                                               shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(valds, batch_size=batch,
                                             shuffle=False, num_workers=0)

    counts = collections.Counter(train_labels)
    w = torch.tensor([max(1, len(train_labels)) / max(1, counts.get(l, 1))
                      for l in label_list], dtype=torch.float32)
    w = w / w.mean()

    TinyLitterCNN = get_cnn_module()
    net = TinyLitterCNN(n_classes=len(label_list))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        net.train()
        tot_loss, n = 0.0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()
            tot_loss += float(loss) * len(xb); n += len(xb)
        sched.step()
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = net(xb).argmax(1)
                correct += int((pred == yb).sum()); total += len(xb)
        acc = correct / max(1, total)
        print(f"  epoch {ep+1:2d}/{epochs} loss={tot_loss/max(1,n):.4f} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    print(f"  best val crop accuracy: {best_acc:.4f}")
    net.load_state_dict(best_state)
    meta = {"class_names": CANONICAL, "labels": label_list,
            "n_classes": len(label_list), "norm_mean": mean, "norm_std": std,
            "best_val_acc": round(best_acc, 4),
            "feature_version": litter_ml.FEATURE_VERSION, "engine": "tiny-cnn"}
    return net, meta


# ----------------------------------------------------------------------------------
# Detection-level validation (RF vs CNN shootout)
# ----------------------------------------------------------------------------------

def make_detector_with_rf(rf_clf):
    det = litter_ml.LitterDetector()
    det.pipeline = rf_clf
    det.cnn = None
    det.class_names = list(CANONICAL)
    det.cfg = litter_ml.load_config()
    det.cfg["class_names"] = list(CANONICAL)
    return det


def make_detector_with_cnn(net, meta):
    det = litter_ml.LitterDetector()
    det.cnn = net
    net.eval()
    det.cnn_meta = meta
    det.pipeline = None
    det.class_names = list(CANONICAL)
    det.cfg = litter_ml.load_config()
    det.cfg["class_names"] = list(CANONICAL)
    return det


def score_detector(det, datasets, conf=0.15):
    per = {n: {"tp": 0, "fp": 0, "fn": 0} for n in CANONICAL}
    n_imgs = 0
    for ds in datasets:
        for img_p in split_images(ds, "val"):
            im = cv2.imread(str(img_p))
            if im is None:
                continue
            n_imgs += 1
            H, W = im.shape[:2]
            gt = read_gt(ds, img_p, W, H)
            # raw detect at low threshold: temporarily patch thresholds
            cfg_save = det.cfg.get("class_thresholds", {})
            det.cfg["class_thresholds"] = {}
            det.cfg["confidence_threshold"] = conf
            dets = det.detect(im)
            det.cfg["class_thresholds"] = cfg_save
            matched = [False] * len(gt)
            for d in sorted(dets, key=lambda d: -d["confidence"]):
                try:
                    ci = CANONICAL.index(d["class"])
                except ValueError:
                    per_other = None
                    continue
                best, best_iou = -1, 0.5
                for gi, (cid, gb) in enumerate(gt):
                    if matched[gi] or cid != ci:
                        continue
                    iou = float(box_iou(np.array([d["box"]], np.float32),
                                        np.array([gb], np.float32))[0, 0])
                    if iou > best_iou:
                        best, best_iou = gi, iou
                if best >= 0:
                    matched[best] = True
                    per[d["class"]]["tp"] += 1
                else:
                    per[d["class"]]["fp"] += 1
            for gi, (cid, _) in enumerate(gt):
                if not matched[gi]:
                    per[CANONICAL[cid]]["fn"] += 1
    return per, n_imgs


def report(per, n_imgs, title):
    print(f"\n{title} ({n_imgs} val images, IoU>=0.5)")
    print(f"  {'class':20s} {'prio':9s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    tps = fps = fns = 0
    f1s = {}
    for n in CANONICAL:
        tp, fp, fn = per[n]["tp"], per[n]["fp"], per[n]["fn"]
        tps += tp; fps += fp; fns += fn
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        f1s[n] = f1
        print(f"  {n:20s} {PRIORITY_MAP.get(n, 'low'):9s} {p:7.3f} {r:7.3f} {f1:7.3f} "
              f"(tp={tp} fp={fp} fn={fn})")
    mp = tps / max(1, tps + fps); mr = tps / max(1, tps + fns)
    mf1 = 2 * mp * mr / max(1e-9, mp + mr)
    print(f"  MICRO-AVG: P {mp:.3f} R {mr:.3f} F1 {mf1:.3f}")
    return mf1, f1s


def tune_thresholds(det, datasets):
    print("\n" + "=" * 74 + "\nPER-CLASS CONFIDENCE TUNING (F1 sweep)\n" + "=" * 74)
    cache = []
    for ds in datasets:
        for img_p in split_images(ds, "val"):
            im = cv2.imread(str(img_p))
            if im is None:
                continue
            H, W = im.shape[:2]
            gt = read_gt(ds, img_p, W, H)
            cfg_save = det.cfg.get("class_thresholds", {})
            det.cfg["class_thresholds"] = {}
            det.cfg["confidence_threshold"] = 0.05
            dets = [{"ci": CANONICAL.index(d["class"]), "conf": d["confidence"],
                     "box": d["box"]} for d in det.detect(im) if d["class"] in CANONICAL]
            det.cfg["class_thresholds"] = cfg_save
            cache.append((gt, dets))
    grid = np.arange(0.15, 0.76, 0.05)
    best = {}
    print(f"  {'class':20s} {'conf':>6s} {'F1':>7s} {'P':>7s} {'R':>7s}")
    for ci, name in enumerate(CANONICAL):
        best_f1, best_t, best_pr = -1.0, 0.30, (0.0, 0.0)
        for t in grid:
            tp = fp = fn = 0
            for gt, dets in cache:
                mine = [d for d in dets if d["ci"] == ci and d["conf"] >= float(t)]
                gboxes = [b for cid, b in gt if cid == ci]
                matched = [False] * len(gboxes)
                for d in sorted(mine, key=lambda d: -d["conf"]):
                    hit = -1
                    for gi, gb in enumerate(gboxes):
                        if matched[gi]:
                            continue
                        iou = float(box_iou(np.array([d["box"]], np.float32),
                                            np.array([gb], np.float32))[0, 0])
                        if iou >= 0.5:
                            hit = gi
                            break
                    if hit >= 0:
                        matched[hit] = True
                        tp += 1
                    else:
                        fp += 1
                fn += sum(1 for m in matched if not m)
            p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
            f1 = 2 * p * r / max(1e-9, p + r)
            if f1 > best_f1:
                best_f1, best_t, best_pr = f1, float(t), (p, r)
        best[name] = round(best_t, 2)
        print(f"  {name:20s} {best_t:6.2f} {best_f1:7.3f} {best_pr[0]:7.3f} {best_pr[1]:7.3f}")
    return best


# ----------------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------------

def export_for_backend(winner: str, rf_clf, cnn_net, cnn_meta,
                       thresholds, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    global_conf = round(float(min(thresholds.values())) if thresholds else 0.30, 2)
    cfg = {
        "class_names": CANONICAL,
        "priority_map": {n: PRIORITY_MAP.get(n, "low") for n in CANONICAL},
        "bin_map": {n: BIN_MAP.get(n, "general") for n in CANONICAL},
        "priority_colors": PRIORITY_COLORS,
        "confidence_threshold": global_conf,
        "class_thresholds": thresholds,
        "iou_threshold": 0.5,
        "image_size": 960,
        "max_detections": 100,
        "engine": "tiny-cnn" if winner == "cnn" else "ml-classical-rf",
    }
    (out_dir / "litter_config.json").write_text(json.dumps(cfg, indent=2))
    shutil.copy2(out_dir / "litter_config.json", HERE / "litter_config.json")

    import torch
    if cnn_net is not None:
        bundle = dict(cnn_meta)
        bundle["state_dict"] = {k: v.cpu() for k, v in cnn_net.state_dict().items()}
        torch.save(bundle, out_dir / "litter_cnn.pt")
        if winner == "cnn":
            shutil.copy2(out_dir / "litter_cnn.pt", HERE / "litter_cnn.pt")
    if rf_clf is not None:
        joblib.dump({"pipeline": rf_clf, "class_names": CANONICAL,
                     "feature_version": litter_ml.FEATURE_VERSION,
                     "engine": "ml-classical-rf"},
                    out_dir / "litter_ml_model.pkl", compress=3)
        if winner == "rf":
            shutil.copy2(out_dir / "litter_ml_model.pkl", HERE / "litter_ml_model.pkl")
    # stale RF must not shadow a CNN winner (delete; C: is tight)
    if winner == "cnn" and (HERE / "litter_ml_model.pkl").exists():
        (HERE / "litter_ml_model.pkl").unlink()
        print("  (stale RF pickle deleted so CNN takes precedence)")
    print("\n" + "=" * 74 + f"\nBackend bundle written to {out_dir} + backend/\n" + "=" * 74)
    print(f"  winner: {winner}")
    print("Restart the API (or POST /reload-model) and verify: curl localhost:8000/health")


if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    datasets = [load_dataset(Path(a).resolve()) for a in args.data]
    preflight(datasets)

    X, y, cnn_crops, cnn_labels = build_training_set(
        datasets, neg_ratio=args.neg_ratio, rng=rng)

    rf_clf = train_rf(X, y, estimators=args.estimators)

    cnn_net = cnn_meta = None
    if not args.no_cnn:
        val_crops, val_labels = build_val_crops(datasets)
        print(f"\nCNN sets: train={len(cnn_crops)} crops, val={len(val_crops)} crops")
        cnn_net, cnn_meta = train_cnn(cnn_crops, cnn_labels, val_crops, val_labels,
                                      epochs=args.cnn_epochs, batch=args.cnn_batch,
                                      seed=args.seed)

    print("\n" + "=" * 74 + "\nSHOOTOUT — detection-level validation\n" + "=" * 74)
    det_rf = make_detector_with_rf(rf_clf)
    per_rf, n_imgs = score_detector(det_rf, datasets)
    f1_rf, _ = report(per_rf, n_imgs, "RANDOM FOREST")
    winner, det_win = "rf", det_rf
    if cnn_net is not None:
        det_cnn = make_detector_with_cnn(cnn_net, cnn_meta)
        per_cnn, _ = score_detector(det_cnn, datasets)
        f1_cnn, _ = report(per_cnn, n_imgs, "TINY CNN")
        print(f"\n  RF micro-F1 {f1_rf:.3f} vs CNN micro-F1 {f1_cnn:.3f}")
        if f1_cnn >= f1_rf:
            winner, det_win = "cnn", det_cnn
    print(f"  WINNER: {winner}")

    thresholds = {} if args.no_tune else tune_thresholds(det_win, datasets)
    try:
        import db
        db.init_db()
        db.log_analysis_run(winner, len(CANONICAL), len(X),
                            round(float(f1_rf if winner == "rf" else f1_cnn), 4),
                            str(Path(args.project) / "backend_bundle"),
                            str(Path(args.project) / "backend_bundle" / "litter_config.json"),
                            notes="25-class RF+CNN shootout")
    except Exception as e:
        print(f"  (analysis log skipped: {e})")
    export_for_backend(winner, rf_clf, cnn_net, cnn_meta, thresholds,
                       Path(args.project) / "backend_bundle")
