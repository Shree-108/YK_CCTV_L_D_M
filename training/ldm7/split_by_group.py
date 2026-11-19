"""Build train/val/test from a LABELLED pool by group (camera / session / day) - never by random image.

    py -m training.ldm7.split_by_group --pool D:/LDM7/pool --out D:/LDM7/ldm7_v1 \
        --test-group camera_id=CAM_C --test-group session_id=2026-12-03_night [--val-frac 0.2] [--seed 0]

Pool layout: <pool>/images/<file>, <pool>/labels/<stem>.txt (an EMPTY file = reviewed background image),
<pool>/manifest.csv with at least image, camera_id, session_id, date (image = file name or images/<file>).

Grouping: images sharing ANY of the --group-by values (default session_id and camera_day = camera_id@date) or
near-identical images (dHash <= --near-bits) are tied into one connected group (union-find, the same idea as
training/build_ldm_dataset.py). A group is placed in exactly one split.

Test split: groups matching --test-group (key=value, or a bare value matched against any key) go to test and
nothing else does - the designated independent test set is kept untouched. Without --test-group, --test-frac of the
remaining groups is drawn as test and the report says the test set is NOT independent by design.
Deterministic: groups are ordered by their sorted ids and shuffled with random.Random(--seed); no hash() seeding.
Files are COPIED (sources never modified); the output folder must not exist. verify_dataset runs at the end.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from training.ldm7.common import (DEFAULT_GROUP_BY, GROUP_KEYS, MANIFEST_REQUIRED, SPLITS, dhash, group_values,
                                  index_manifest, list_images, manifest_missing_columns, read_image, read_manifest)
from training.ldm7.schema import write_data_yaml


class SplitError(Exception):
    pass


class _UF:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _item_matches(it: dict[str, Any], spec: str) -> bool:
    """spec 'key=value' matches a manifest column or a group id; a bare value matches any of them."""
    row, groups = it["row"], it["groups"]
    if "=" in spec:
        k, v = spec.split("=", 1)
        return spec in groups or row.get(k) == v
    return any(g.split("=", 1)[1] == spec for g in groups) or spec in (row.get(k) for k in MANIFEST_REQUIRED[1:])


def collect_pool(pool: Path, group_by: Sequence[str]) -> list[dict[str, Any]]:
    pool = Path(pool)
    mp = pool / "manifest.csv"
    if not mp.is_file():
        raise SplitError(f"{mp} missing: a group split needs camera_id/session_id/date for every image")
    missing = manifest_missing_columns(mp)
    if missing:
        raise SplitError(f"manifest.csv lacks columns {missing} (required {MANIFEST_REQUIRED})")
    idx = index_manifest(read_manifest(mp))
    items, problems = [], []
    for p in list_images(pool / "images"):
        row = idx.get(f"images/{p.name}") or idx.get(p.name)
        lp = pool / "labels" / f"{p.stem}.txt"
        if row is None:
            problems.append(f"{p.name}: no manifest row")
            continue
        if not lp.is_file():
            problems.append(f"{p.name}: label file missing (background images need an EMPTY .txt)")
            continue
        groups = group_values(row, group_by)
        if not groups:
            problems.append(f"{p.name}: empty {list(group_by)} in manifest")
            continue
        img = read_image(p)
        if img is None:
            problems.append(f"{p.name}: unreadable image")
            continue
        items.append({"name": p.name, "path": p, "label": lp, "row": row, "groups": groups, "dhash": dhash(img),
                      "n_boxes": sum(1 for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip())})
    if problems:
        raise SplitError(f"{len(problems)} pool problem(s), fix them first:\n  " + "\n  ".join(problems[:50]))
    if not items:
        raise SplitError(f"no images in {pool / 'images'}")
    return items


def assign_splits(items: list[dict[str, Any]], test_groups: Sequence[str], val_frac: float, test_frac: float,
                  seed: int, near_bits: int) -> tuple[list[str], list[str]]:
    """Return (split per item, warnings). Pure function of its inputs (deterministic)."""
    n = len(items)
    uf = _UF(n)
    first: dict[str, int] = {}
    for i, it in enumerate(items):
        for g in it["groups"]:
            if g in first:
                uf.union(i, first[g])
            else:
                first[g] = i
    if near_bits >= 0 and n > 1:
        bits = np.unpackbits(np.array([np.frombuffer(bytes.fromhex(it["dhash"]), np.uint8) for it in items]), axis=1)
        for i in range(n - 1):
            d = np.count_nonzero(bits[i + 1:] != bits[i], axis=1)
            for j in np.nonzero(d <= near_bits)[0]:
                uf.union(i, i + 1 + int(j))
    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[uf.find(i)].append(i)
    comp_list = sorted(comps.values(), key=lambda m: sorted(items[i]["name"] for i in m))
    warnings: list[str] = []
    split_of = [""] * n
    designated, free = [], []
    for members in comp_list:
        if any(_item_matches(items[i], spec) for i in members for spec in test_groups):
            designated.append(members)
        else:
            free.append(members)
    for spec in test_groups:
        if not any(_item_matches(items[i], spec) for m in designated for i in m):
            raise SplitError(f"--test-group {spec!r} matches no image in the pool")
    for members in designated:
        for i in members:
            split_of[i] = "test"
    rnd = random.Random(seed)
    rnd.shuffle(free)
    total = sum(len(m) for m in free)
    want = {"test": 0.0 if test_groups else test_frac * total, "val": val_frac * total}
    if not test_groups and test_frac > 0:
        warnings.append("no --test-group: test was drawn from the same cameras/days as train - it is NOT an "
                        "independent test set; designate unseen cameras/sessions for a real evaluation")
    have = {"test": 0, "val": 0}
    for members in free:
        target = "train"
        for s in ("test", "val"):
            if have[s] + len(members) / 2 < want[s] or (want[s] > 0 and have[s] == 0):
                target = s
                break
        have[target] = have.get(target, 0) + len(members)
        for i in members:
            split_of[i] = target
    if len(free) < 2:
        warnings.append(f"only {len(free)} free group(s): train/val cannot both be filled independently")
    for s in SPLITS:
        if s not in split_of:
            warnings.append(f"{s} split is EMPTY")
    return split_of, warnings


def build(pool: Path, out: Path, test_groups: Sequence[str] = (), val_frac: float = 0.2, test_frac: float = 0.0,
          seed: int = 0, near_bits: int = 10, group_by: Sequence[str] = DEFAULT_GROUP_BY,
          run_verify: bool = True) -> dict[str, Any]:
    out = Path(out)
    if out.exists():
        raise SplitError(f"{out} already exists: dataset versions are immutable, choose a new name")
    bad = [k for k in group_by if k not in GROUP_KEYS]
    if bad:
        raise SplitError(f"unknown group key(s) {bad}")
    items = collect_pool(pool, group_by)
    split_of, warnings = assign_splits(items, test_groups, val_frac, test_frac, seed, near_bits)
    rows = []
    for it, s in zip(items, split_of):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        shutil.copy2(it["path"], out / "images" / s / it["name"])
        shutil.copy2(it["label"], out / "labels" / s / it["label"].name)
        rows.append({**{k: it["row"].get(k, "") for k in MANIFEST_REQUIRED}, "image": f"images/{s}/{it['name']}",
                     "split": s, "groups": ";".join(it["groups"]), "boxes": it["n_boxes"]})
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    write_data_yaml(out)
    fields = list(MANIFEST_REQUIRED) + ["split", "groups", "boxes"]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["image"]))
    groups_per_split: dict[str, set[str]] = defaultdict(set)
    for it, s in zip(items, split_of):
        groups_per_split[s].update(it["groups"])
    report: dict[str, Any] = {
        "pool": str(pool), "out": str(out), "seed": seed, "val_frac": val_frac, "test_frac": test_frac,
        "test_groups": list(test_groups), "group_by": list(group_by), "near_bits": near_bits,
        "independent_test": bool(test_groups), "warnings": warnings,
        "splits": {s: {"images": split_of.count(s), "groups": sorted(groups_per_split[s])} for s in SPLITS}}
    if run_verify:
        from training.ldm7.verify_dataset import verify, write_reports
        rep = verify(out, group_by=group_by, near_bits=near_bits)
        write_reports(rep, out / "verify_report")
        report["verify"] = {"ok": rep["ok"], "errors": rep["errors"], "warnings": rep["warnings"],
                            "dataset_hash": rep["dataset_hash"]}
    (out / "split_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-group", action="append", default=[], help="key=value or bare value; repeatable")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.0, help="only used when no --test-group is given")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--near-bits", type=int, default=10, help="-1 disables near-duplicate tying")
    ap.add_argument("--group-by", default=",".join(DEFAULT_GROUP_BY), help=f"comma list from {GROUP_KEYS}")
    a = ap.parse_args(argv)
    try:
        rep = build(a.pool, a.out, a.test_group, a.val_frac, a.test_frac, a.seed, a.near_bits,
                    [k.strip() for k in a.group_by.split(",") if k.strip()])
    except SplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: rep[k] for k in ("out", "splits", "warnings", "verify") if k in rep}, indent=1))
    return 0 if rep.get("verify", {}).get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
