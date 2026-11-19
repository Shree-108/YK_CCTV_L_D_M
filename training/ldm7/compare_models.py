"""Compare v3_store_cctv with a v4 ldm7 candidate on the SAME independent, labelled seven-class test set.

    py -m training.ldm7.compare_models --root D:/LDM7/ldm7_v1 --split test \
        --candidate D:/LDM7/staging/v4_ldm7_yolo11s/best.pt --candidate-imgsz 1280 \
        [--v3 models/litter_yolo/v3_store_cctv/best.pt --v3-imgsz 640] [--op-conf 0.30] --out D:/LDM7/compare_v1

Both models see the same images (streamed one at a time, CPU by default). Per model:
  * per class (ldm7 schema): precision / recall at --op-conf (production floor 0.30) and AP50 (IoU 0.5,
    all-point interpolation over predictions >= --conf-floor). v3 is scored through the comparison-only mapping in
    training/ldm7/schema.py; predictions of UNMAPPED v3 classes (plastic_bag, cup, plastic_piece) are counted as
    false positives in a separate 'unmapped_v3' bucket and are excluded from per-class precision.
  * class-agnostic "any litter" precision / recall / AP50 (every prediction, mapped or not) - the fair localisation
    comparison, since v3 never learnt the ldm7 class definitions.
  * recall per size bucket (small < 32x32 px, medium <= 96x96, large), false positives per CLEAN (empty-label)
    image, and a confidence calibration table (precision per confidence bin).
The markdown report states explicitly when the test set is missing, too small, lacks clean images or classes, or
its independence from the training data is unverified - in those cases it draws NO conclusion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from training.ldm7.common import SIZE_BUCKETS, list_images
from training.ldm7.evalcore import (UltralyticsPredictor, average_precision, gt_bucket, greedy_match,
                                    iter_labelled_images, map_predictions, predict_setting, safe_div)
from training.ldm7.schema import (CLASSES, LDM7_WITHOUT_V3_SOURCE, V3_CLASSES, class_mapper_for, mapping_markdown)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V3 = PROJECT_ROOT / "models" / "litter_yolo" / "v3_store_cctv" / "best.pt"
CAL_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]

# sufficiency thresholds below which the report refuses to conclude
MIN_IMAGES = 100
MIN_CLEAN_IMAGES = 20
MIN_PER_CLASS = 30
MIN_SMALL = 30


def evaluate_model(predictor: Any, images: Iterable[tuple[str, np.ndarray, np.ndarray]], imgsz: int,
                   op_conf: float = 0.30, conf_floor: float = 0.001, iou: float = 0.5,
                   tiling: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    kind, mapper = class_mapper_for(predictor.names)        # unknown schema -> ValueError (never guessed)
    setting = {"imgsz": imgsz, "tiling": False, **(tiling or {})}
    cls_records: dict[int, list[tuple[float, bool]]] = {c: [] for c in range(len(CLASSES))}
    agn_records: list[tuple[float, bool]] = []
    n_gt = np.zeros(len(CLASSES), int)
    bucket_gt = {b: 0 for b in SIZE_BUCKETS}
    bucket_hit_cls = {b: 0 for b in SIZE_BUCKETS}
    bucket_hit_agn = {b: 0 for b in SIZE_BUCKETS}
    unmapped_fp = 0
    unmapped_by_class: dict[str, int] = {}
    n_img = n_clean = clean_fp = 0
    for _img_id, img, gts in images:
        n_img += 1
        raw = predict_setting(predictor, img, setting, conf_floor)
        raw = raw[raw[:, 4] >= conf_floor]
        mapped, unmapped = map_predictions(raw, mapper)
        for c in gts[:, 0].astype(int):
            n_gt[c] += 1
        tp, which = greedy_match(mapped, gts, iou, class_aware=True)
        for k in range(len(mapped)):
            cls_records[int(mapped[k, 5])].append((float(mapped[k, 4]), bool(tp[k])))
        atp, awhich = greedy_match(raw, gts, iou, class_aware=False)
        agn_records += [(float(raw[k, 4]), bool(atp[k])) for k in range(len(raw))]
        hit_c = {int(which[k]) for k in range(len(mapped)) if tp[k] and mapped[k, 4] >= op_conf}
        hit_a = {int(awhich[k]) for k in range(len(raw)) if atp[k] and raw[k, 4] >= op_conf}
        for gi, g in enumerate(gts):
            b = gt_bucket(g)
            bucket_gt[b] += 1
            bucket_hit_cls[b] += gi in hit_c
            bucket_hit_agn[b] += gi in hit_a
        u = unmapped[unmapped[:, 4] >= op_conf]
        unmapped_fp += len(u)
        for c in u[:, 5].astype(int):
            name = predictor.names[c]
            unmapped_by_class[name] = unmapped_by_class.get(name, 0) + 1
        if len(gts) == 0:
            n_clean += 1
            clean_fp += int((raw[:, 4] >= op_conf).sum())

    per_class = {}
    for c, name in enumerate(CLASSES):
        recs = cls_records[c]
        above = [t for s, t in recs if s >= op_conf]
        tp_n = sum(above)
        per_class[name] = {"n_gt": int(n_gt[c]), "tp": tp_n, "fp": len(above) - tp_n,
                           "precision": safe_div(tp_n, len(above)), "recall": safe_div(tp_n, int(n_gt[c])),
                           "ap50": average_precision([s for s, _ in recs], [t for _, t in recs], int(n_gt[c])),
                           "no_source_class": kind == "v3" and name in LDM7_WITHOUT_V3_SOURCE}
    total_gt = int(n_gt.sum())
    a_above = [t for s, t in agn_records if s >= op_conf]
    agnostic = {"n_gt": total_gt, "tp": sum(a_above), "fp": len(a_above) - sum(a_above),
                "precision": safe_div(sum(a_above), len(a_above)), "recall": safe_div(sum(a_above), total_gt),
                "ap50": average_precision([s for s, _ in agn_records], [t for _, t in agn_records], total_gt)}
    aps = [v["ap50"] for v in per_class.values() if v["ap50"] is not None]
    calib = []
    allrec = [r for recs in cls_records.values() for r in recs]
    for lo, hi in zip(CAL_BINS[:-1], CAL_BINS[1:]):
        sub = [t for s, t in allrec if lo <= s < hi]
        confs = [s for s, _ in allrec if lo <= s < hi]
        calib.append({"bin": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n_pred": len(sub),
                      "mean_conf": float(np.mean(confs)) if confs else None, "precision": safe_div(sum(sub), len(sub))})
    return {
        "schema": kind, "model_classes": list(predictor.names), "imgsz": imgsz, "tiling": bool(setting["tiling"]),
        "op_conf": op_conf, "conf_floor": conf_floor, "iou": iou, "n_images": n_img, "n_clean_images": n_clean,
        "per_class": per_class, "map50_over_classes_with_gt": float(np.mean(aps)) if aps else None,
        "agnostic": agnostic,
        "recall_by_bucket": {b: {"n_gt": bucket_gt[b], "recall_class": safe_div(bucket_hit_cls[b], bucket_gt[b]),
                                 "recall_any": safe_div(bucket_hit_agn[b], bucket_gt[b])} for b in SIZE_BUCKETS},
        "fp_per_clean_image": safe_div(clean_fp, n_clean),
        "unmapped_v3_fp": unmapped_fp, "unmapped_v3_fp_by_class": unmapped_by_class,
        "calibration": calib,
    }


def sufficiency(results: dict[str, dict[str, Any]], independence: Optional[str]) -> list[str]:
    """Reasons the test set cannot support a conclusion (empty list = sufficient)."""
    r = next(iter(results.values()), None)
    if r is None or r["n_images"] == 0:
        return ["NO TEST SET: no labelled seven-class test images were found - nothing can be concluded."]
    out = []
    if r["n_images"] < MIN_IMAGES:
        out.append(f"only {r['n_images']} test images (< {MIN_IMAGES})")
    if r["n_clean_images"] < MIN_CLEAN_IMAGES:
        out.append(f"only {r['n_clean_images']} clean (empty-label) images (< {MIN_CLEAN_IMAGES}): false-positive "
                   f"rates are not measurable")
    few = [f"{c} ({v['n_gt']})" for c, v in r["per_class"].items() if v["n_gt"] < MIN_PER_CLASS]
    if few:
        out.append(f"classes with < {MIN_PER_CLASS} test instances (per-class numbers not meaningful): " + ", ".join(few))
    if r["recall_by_bucket"]["small"]["n_gt"] < MIN_SMALL:
        out.append(f"only {r['recall_by_bucket']['small']['n_gt']} small (< 32x32 px) test boxes (< {MIN_SMALL})")
    if independence != "verified":
        out.append("independence of the test set from the training data is "
                   + ("NOT verified (no manifest.csv / verify not run)" if independence is None else independence))
    return out


def _f(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def render_markdown(results: dict[str, dict[str, Any]], reasons: list[str], test_desc: str) -> str:
    L = ["# v3_store_cctv vs ldm7 candidate - comparison on the seven-class test set", "", f"Test set: `{test_desc}`", ""]
    if reasons:
        L += ["## VERDICT: INSUFFICIENT EVIDENCE - DO NOT DRAW A CONCLUSION", "",
              "The test set cannot support a model decision:", ""] + [f"* {x}" for x in reasons] + [
              "", "The numbers below are shown for debugging only.", ""]
    else:
        L += ["## Test set sufficiency: OK (thresholds in compare_models.py)", ""]
    if not results or next(iter(results.values()))["n_images"] == 0:
        return "\n".join(L) + "\n"
    names = list(results)
    L += ["## Class-agnostic (any litter) at the operating threshold", "",
          "| model | imgsz | precision | recall | AP50 | FP / clean image | unmapped-v3 FP |", "|---|---|---|---|---|---|---|"]
    for n in names:
        r = results[n]
        L.append(f"| {n} | {r['imgsz']} | {_f(r['agnostic']['precision'])} | {_f(r['agnostic']['recall'])} | "
                 f"{_f(r['agnostic']['ap50'])} | {_f(r['fp_per_clean_image'])} | {r['unmapped_v3_fp']} |")
    L += ["", "## Per class (P / R at operating threshold, AP50)", "",
          "| class | n_gt | " + " | ".join(f"{n} P | {n} R | {n} AP50" for n in names) + " |",
          "|---|---|" + "---|---|---|" * len(names)]
    first = results[names[0]]
    for c in CLASSES:
        cells = []
        for n in names:
            v = results[n]["per_class"][c]
            flag = " (no v3 class)" if v["no_source_class"] else ""
            cells.append(f"{_f(v['precision'])} | {_f(v['recall'])}{flag} | {_f(v['ap50'])}")
        L.append(f"| {c} | {first['per_class'][c]['n_gt']} | " + " | ".join(cells) + " |")
    L += ["", "## Recall by object size (class-aware / any class)", "",
          "| bucket | n_gt | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for b in SIZE_BUCKETS:
        L.append(f"| {b} | {first['recall_by_bucket'][b]['n_gt']} | " + " | ".join(
            f"{_f(results[n]['recall_by_bucket'][b]['recall_class'])} / {_f(results[n]['recall_by_bucket'][b]['recall_any'])}"
            for n in names) + " |")
    L += ["", "## Confidence calibration (class-aware precision per confidence bin)", "",
          "| bin | " + " | ".join(f"{n} n | {n} mean conf | {n} precision" for n in names) + " |",
          "|---|" + "---|---|---|" * len(names)]
    for i in range(len(CAL_BINS) - 1):
        L.append(f"| {first['calibration'][i]['bin']} | " + " | ".join(
            f"{results[n]['calibration'][i]['n_pred']} | {_f(results[n]['calibration'][i]['mean_conf'])} | "
            f"{_f(results[n]['calibration'][i]['precision'])}" for n in names) + " |")
    if any(r["schema"] == "v3" for r in results.values()):
        L += ["", "## v3 -> ldm7 comparison mapping (comparison only, never applied to live output)", "",
              mapping_markdown()]
        for n in names:
            if results[n]["unmapped_v3_fp_by_class"]:
                L.append(f"\nUnmapped v3 predictions >= op-conf for {n}: {results[n]['unmapped_v3_fp_by_class']}")
    return "\n".join(L) + "\n"


def check_test_independence(root: Path) -> Optional[str]:
    if not (root / "manifest.csv").is_file():
        return None
    from training.ldm7.verify_dataset import verify
    rep = verify(root)
    leaks = [i for i in rep["issues"] if i["type"] in ("group_leakage", "cross_split_duplicate",
                                                        "cross_split_near_duplicate", "not_in_manifest")]
    return "verified" if not leaks else f"VIOLATED ({len(leaks)} leakage issue(s), see verify report)"


def compare(models: dict[str, tuple[Any, int]], image_source: Callable[[], Iterable], test_desc: str,
            out: Path, op_conf: float = 0.30, conf_floor: float = 0.001,
            independence: Optional[str] = None) -> dict[str, Any]:
    results = {name: evaluate_model(pred, image_source(), imgsz, op_conf, conf_floor) for name, (pred, imgsz) in models.items()}
    reasons = sufficiency(results, independence)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"test_set": test_desc, "independence": independence, "sufficient": not reasons,
           "insufficiency_reasons": reasons, "results": results}
    (out / "comparison.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    (out / "comparison_report.md").write_text(render_markdown(results, reasons, test_desc), encoding="utf-8")
    return doc


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="seven-class dataset root (verified)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--v3", type=Path, default=DEFAULT_V3)
    ap.add_argument("--v3-imgsz", type=int, default=640)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--candidate-imgsz", type=int, default=1280)
    ap.add_argument("--op-conf", type=float, default=0.30)
    ap.add_argument("--conf-floor", type=float, default=0.001)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    img_dir, lab_dir = a.root / "images" / a.split, a.root / "labels" / a.split
    desc = f"{a.root} [{a.split}]"
    if not list_images(img_dir):
        doc = compare({}, lambda: iter(()), desc, a.out, a.op_conf, a.conf_floor, None)
        print(f"NO TEST SET at {img_dir}: report written, no conclusion -> {a.out / 'comparison_report.md'}")
        return 1
    v3 = UltralyticsPredictor(a.v3, a.device)
    if v3.names != V3_CLASSES:
        print(f"ERROR: {a.v3} is not v3_store_cctv (classes {v3.names})", file=sys.stderr)
        return 1
    cand = UltralyticsPredictor(a.candidate, a.device)
    doc = compare({"v3_store_cctv": (v3, a.v3_imgsz), "candidate": (cand, a.candidate_imgsz)},
                  lambda: ((p.name, img, gt) for p, img, gt in iter_labelled_images(img_dir, lab_dir)),
                  desc, a.out, a.op_conf, a.conf_floor, check_test_independence(a.root))
    print(f"{'SUFFICIENT' if doc['sufficient'] else 'INSUFFICIENT'} test set -> {a.out / 'comparison_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
