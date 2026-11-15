"""Fingerprint the PRODUCTION data of LDM and compare two fingerprints, to prove a demo run did not touch it.

    python deploy/check_demo_isolation.py snapshot <out.json>
    python deploy/check_demo_isolation.py compare <before.json> <after.json>     # exit 1 if anything changed

What is fingerprinted (files are only read; the SQLite database is hashed as a file, never opened as a database,
because even a read-only SQLite open can touch the -shm file):
  storage/   every file: size, mtime and (for *.db / *.db-wal / *.db-shm) SHA-256
  dataset/   every file: size and mtime
  models/    every file: size and mtime; SHA-256 of every *.pt and *.safetensors
  .env       SHA-256 only (the content is never printed)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREES = {"storage": ROOT / "storage", "dataset": ROOT / "dataset", "models": ROOT / "models"}
HASH_SUFFIXES = {"storage": (".db", ".db-wal", ".db-shm"), "dataset": (), "models": (".pt", ".safetensors")}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot() -> dict:
    snap: dict = {"root": str(ROOT), "trees": {}}
    for name, base in TREES.items():
        files = {}
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file():
                    st = p.stat()
                    rec = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
                    if p.name.endswith(HASH_SUFFIXES[name]):
                        rec["sha256"] = sha256(p)
                    files[str(p.relative_to(base))] = rec
        snap["trees"][name] = files
    env = ROOT / ".env"
    snap["env_sha256"] = sha256(env) if env.exists() else None
    return snap


def compare(a: dict, b: dict) -> list[str]:
    diffs = []
    for name in TREES:
        fa, fb = a["trees"].get(name, {}), b["trees"].get(name, {})
        for k in sorted(set(fa) | set(fb)):
            if k not in fb: diffs.append(f"{name}/{k}: DELETED")
            elif k not in fa: diffs.append(f"{name}/{k}: ADDED")
            elif fa[k] != fb[k]: diffs.append(f"{name}/{k}: CHANGED")
    if a.get("env_sha256") != b.get("env_sha256"):
        diffs.append(".env: CHANGED")
    return diffs


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "snapshot":
        s = snapshot()
        Path(argv[1]).write_text(json.dumps(s), encoding="utf-8")
        print(f"snapshot: " + ", ".join(f"{n} {len(f)} files" for n, f in s["trees"].items()) + f" -> {argv[1]}")
        return 0
    if len(argv) == 3 and argv[0] == "compare":
        a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in argv[1:])
        d = compare(a, b)
        if d:
            print(f"PRODUCTION DATA CHANGED ({len(d)}):"); [print("  " + x) for x in d[:50]]
            return 1
        print("OK: production storage, dataset, models and .env unchanged")
        return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
