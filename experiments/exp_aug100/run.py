"""Train one AUG100 configuration on one scene fold. Experimental only: never registers a model.

    python -m experiments.exp_aug100.run --config pilot --fold AtoB
    python -m experiments.exp_aug100.run --config S1 --fold AtoB
    python -m experiments.exp_aug100.run --config S1 --fold BtoA --final --epochs 17     # fixed epochs, no early stop

Guards: v3 hash unchanged before/after; starting weights hash matches; dataset validation passed and every label
verified; run directory must not exist (no overwrite). Records RAM / CPU / time to <run>/run_meta.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import threading
import time

import psutil
import yaml

from experiments.exp_aug100.common import ANN, EXP, ROOT, RUNS, check_v3, read_manifest, sha256


class Monitor(threading.Thread):
    def __init__(self, every=2.0):
        super().__init__(daemon=True); self.every = every; self.stop = False
        self.proc = psutil.Process(os.getpid()); self.peak_rss = 0; self.min_avail = 1 << 62; self.cpu = []; self.swap_peak = 0
        self.proc.cpu_percent(None)

    def run(self):
        while not self.stop:
            try:
                rss = self.proc.memory_info().rss + sum(c.memory_info().rss for c in self.proc.children(recursive=True))
                self.peak_rss = max(self.peak_rss, rss)
                vm = psutil.virtual_memory(); self.min_avail = min(self.min_avail, vm.available)
                self.swap_peak = max(self.swap_peak, psutil.swap_memory().used)
                self.cpu.append(self.proc.cpu_percent(None))
            except psutil.Error:
                pass
            time.sleep(self.every)


def load_cfg(name: str) -> dict:
    cfg = yaml.safe_load((EXP / "configs" / "base.yaml").read_text())
    cfg.update(yaml.safe_load((EXP / "configs" / f"{name}.yaml").read_text()) or {})
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--fold", choices=["AtoB", "BtoA"], required=True)
    ap.add_argument("--final", action="store_true", help="fixed epochs, early stopping off")
    ap.add_argument("--epochs", type=int, help="required with --final")
    a = ap.parse_args(argv)

    v3_before = check_v3()
    val = json.loads((ANN / "validation.json").read_text())
    if val["errors"]: raise SystemExit("STOP: dataset validation has errors - run validate.py")
    bad = [r["file"] for r in read_manifest() if r["label_status"] not in ("verified", "manual_verified")]
    if bad: raise SystemExit(f"STOP: unverified labels: {bad[:5]}")
    cfg = load_cfg(a.config)
    wpath = ROOT / cfg.pop("model"); want = cfg.pop("model_sha256")
    if sha256(wpath) != want: raise SystemExit(f"STOP: {wpath} hash differs from the recorded official weights")
    if a.final:
        if not a.epochs: raise SystemExit("--final needs --epochs")
        cfg["epochs"] = a.epochs; cfg["patience"] = 0
    name = f"exp_aug100_{a.config}_{a.fold}" + ("_final" if a.final else "")
    run_dir = RUNS / name
    if run_dir.exists(): raise SystemExit(f"STOP: {run_dir} exists; refusing to overwrite")

    from ultralytics import YOLO
    mon = Monitor(); mon.start(); t0 = time.time(); status, err = "completed", None
    epoch_times = []
    try:
        model = YOLO(str(wpath))
        model.add_callback("on_train_epoch_end", lambda tr: epoch_times.append(round(time.time() - t0, 1)))
        model.train(data=str(EXP / f"data_{a.fold}.yaml"), project=str(RUNS), name=name, exist_ok=False, val=True, **cfg)
    except Exception as e:  # recorded, not hidden
        status, err = "failed", f"{type(e).__name__}: {e}"
    finally:
        mon.stop = True; mon.join(timeout=5)
    wall = time.time() - t0
    v3_after = check_v3()
    meta = {"run": name, "config": a.config, "fold": a.fold, "final": a.final, "status": status, "error": err,
            "started": dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"), "wall_s": round(wall, 1),
            "epoch_end_s": epoch_times, "hyperparameters": cfg, "start_weights": str(wpath.relative_to(ROOT)), "start_weights_sha256": want,
            "peak_rss_gb": round(mon.peak_rss / 2**30, 2), "min_system_available_gb": round(mon.min_avail / 2**30, 2),
            "peak_swap_used_gb": round(mon.swap_peak / 2**30, 2),
            "cpu_percent_mean": round(sum(mon.cpu) / max(len(mon.cpu), 1), 1), "cpu_threads": psutil.cpu_count(),
            "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1), "platform": platform.platform(),
            "v3_sha256_before": v3_before, "v3_sha256_after": v3_after, "registered": False,
            "checkpoints": sorted(p.name for p in (run_dir / "weights").glob("*.pt")) if (run_dir / "weights").exists() else []}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "hyperparameters"}, indent=1))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
