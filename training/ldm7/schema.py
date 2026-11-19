"""Fixed seven-class schema of the v4_ldm7 litter detector + the v3 -> ldm7 COMPARISON mapping.

Class ids and names below are a contract: data.yaml, label files, class_names.json and model_meta.json must use them
exactly, in this order. Never re-order, rename or insert classes; a different schema is a different model family.

    0 plastic_packet  wafer/chips/snack packets, plastic wrappers, sachets
    1 plastic_bottle  discarded plastic water/beverage bottles
    2 paper           loose paper, tissues, paper wrappers
    3 food_waste      food scraps, fruit peels, discarded food
    4 other_litter    clearly discarded litter not covered above
    5 glass           discarded glass bottles, visible glass fragments
    6 cardboard       discarded cardboard, cartons, packaging pieces

V3_TO_LDM7 is used ONLY to score the old v3_store_cctv model on a 7-class test set (compare_models.py,
small_object_experiment.py). It is NEVER used to relabel live detections or to convert training labels: v3 labels
were never annotated with the 7-class definitions, so converting them would invent labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import yaml

MODEL_VERSION = "v4_ldm7_yolo11s"
SCHEMA_VERSION = "ldm7-1"

CLASSES: list[str] = [
    "plastic_packet",
    "plastic_bottle",
    "paper",
    "food_waste",
    "other_litter",
    "glass",
    "cardboard",
]
NUM_CLASSES = len(CLASSES)
NAME_TO_ID: dict[str, int] = {n: i for i, n in enumerate(CLASSES)}
ID_TO_NAME: dict[int, str] = dict(enumerate(CLASSES))

CLASS_DEFINITIONS: dict[str, str] = {
    "plastic_packet": "wafer/chips/snack packets, plastic wrappers, sachets",
    "plastic_bottle": "discarded plastic water/beverage bottles",
    "paper": "loose paper, tissues, paper wrappers",
    "food_waste": "food scraps, fruit peels, discarded food",
    "other_litter": "clearly discarded litter not covered by the other six classes",
    "glass": "discarded glass bottles, visible glass fragments",
    "cardboard": "discarded cardboard, cartons, packaging pieces",
}

# ---------------------------------------------------------------- v3 comparison mapping
V3_VERSION = "v3_store_cctv"
V3_CLASSES: list[str] = [
    "plastic_bag", "plastic_packaging", "food_wrapper", "paper", "cardboard",
    "plastic_bottle", "cup", "newspaper", "plastic_piece", "other_litter",
]

# (target ldm7 class or None, quality, rationale). quality: exact | approximate | unmapped.
# Decided conservatively: a v3 class is mapped only when almost every object it was trained on falls into ONE ldm7
# class. Ambiguous classes are unmapped; their predictions are reported in a separate "unmapped_v3" false-positive
# bucket (per-class scoring) but still count in the class-AGNOSTIC "any litter" scoring, so v3 localisation is not
# penalised for a naming mismatch.
V3_MAPPING_TABLE: dict[str, tuple[Optional[str], str, str]] = {
    "plastic_packaging": ("plastic_packet", "approximate",
                          "flexible plastic packaging / wrappers; rigid trays or clamshells would be other_litter"),
    "food_wrapper": ("plastic_packet", "approximate",
                     "store-front food wrappers are mostly plastic/foil snack wrappers; PAPER wrappers belong to "
                     "'paper' and are mis-attributed by this mapping"),
    "paper": ("paper", "exact", "same definition"),
    "newspaper": ("paper", "exact", "newspaper is loose paper"),
    "cardboard": ("cardboard", "exact", "same definition"),
    "plastic_bottle": ("plastic_bottle", "exact", "same definition"),
    "other_litter": ("other_litter", "exact", "same catch-all definition"),
    "plastic_bag": (None, "unmapped",
                    "a carry bag is neither a snack packet (ldm7 plastic_packet) nor clearly other_litter in v3's "
                    "training labels; v3 is also known to confuse bags and packets"),
    "cup": (None, "unmapped", "paper, plastic or glass cup: material unknown, no single ldm7 class"),
    "plastic_piece": (None, "unmapped",
                      "fragment of a packet, a bottle or something else: no single ldm7 class"),
}
V3_TO_LDM7: dict[str, Optional[str]] = {k: v[0] for k, v in V3_MAPPING_TABLE.items()}
# ldm7 classes that v3 can never predict (per-class v3 recall is 0 by construction, not by measurement)
LDM7_WITHOUT_V3_SOURCE: list[str] = [c for c in CLASSES if c not in {t for t in V3_TO_LDM7.values() if t}]


def v3_id_to_ldm7_id(v3_id: int) -> Optional[int]:
    """Comparison-only mapping of a v3 class id to an ldm7 class id (None = unmapped)."""
    target = V3_TO_LDM7[V3_CLASSES[int(v3_id)]]
    return None if target is None else NAME_TO_ID[target]


def class_mapper_for(model_names: Sequence[str]) -> tuple[str, list[Optional[int]]]:
    """Return (kind, per-model-class ldm7 id) for a model's class list.

    kind is 'ldm7' (identity), 'v3' (comparison mapping) or raises ValueError for any other schema - an unknown
    model must never be scored through a guessed mapping.
    """
    names = list(model_names)
    if names == CLASSES:
        return "ldm7", list(range(NUM_CLASSES))
    if names == V3_CLASSES:
        return "v3", [v3_id_to_ldm7_id(i) for i in range(len(V3_CLASSES))]
    raise ValueError(f"model classes {names} are neither the ldm7 schema nor {V3_VERSION}: no mapping defined")


def names_from_yaml(path: Path) -> list[str]:
    names = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def write_data_yaml(root: Path, path_value: Optional[str] = None, splits: Sequence[str] = ("train", "val", "test"),
                    filename: str = "data.yaml") -> Path:
    """Write <root>/data.yaml for an Ultralytics YOLO dataset with the fixed ldm7 names.

    path_value None (default) omits 'path': Ultralytics 8.4 (data/utils.py check_det_dataset) then resolves the split
    folders against the folder holding the yaml, so the dataset stays relocatable (Colab). Do NOT write "." - a
    relative path is resolved against the current working directory / DATASETS_DIR instead. Pass an absolute path
    for a runtime copy that lives outside the dataset folder.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    doc: dict = {} if path_value is None else {"path": path_value}
    for s in splits:
        doc[s] = f"images/{s}"
    doc["nc"] = NUM_CLASSES
    doc["names"] = dict(enumerate(CLASSES))
    out = root / filename
    out.write_text(f"# LDM seven-class schema {SCHEMA_VERSION} - generated by training/ldm7/schema.py; do not re-order\n"
                   + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return out


def mapping_markdown() -> str:
    rows = ["| v3 class | ldm7 class | quality | rationale |", "|---|---|---|---|"]
    for k in V3_CLASSES:
        t, q, why = V3_MAPPING_TABLE[k]
        rows.append(f"| {k} | {t or '(unmapped)'} | {q} | {why} |")
    rows.append("")
    rows.append("ldm7 classes v3 can never predict: " + ", ".join(LDM7_WITHOUT_V3_SOURCE))
    return "\n".join(rows)
