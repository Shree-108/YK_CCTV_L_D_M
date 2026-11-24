"""Precision, recall and false alerts per hour, before vs after, from LABELLED review folders.

    # before = the current model, after = the same detections restricted to the auto ROI (in_roi == 1)
    python -m training.negative_mining.report_metrics --before storage/review --after-roi

    # before = old model run, after = new model run (each mined + labelled separately)
    python -m training.negative_mining.report_metrics --before storage/review_v3 --after storage/review_v4 \
        [--ground-truth gt.json] [--alert-conf 0.6] [--out report.md]

Definitions (stated in the report, because each choice changes the numbers)
  * Only detections with conf >= ``alert_conf`` count (default: confirmed_threshold from the mining manifest).
    Detections whose class starts with ``fp_`` (negative classes of a retrained model) are never alerts.
  * Detection level: precision = TP / (TP + FP) over labelled sampled detections.
  * Event level: detections of the same video, label group (TP or FP) and class are chained across sampled frames
    when the gap is <= ``gap_sec`` and box IoU >= 0.3. One chain = one alert (the live app also de-duplicates).
    False alerts per hour = FP events / video hours.
  * Recall needs ground truth. With --ground-truth {"<video_id>": <number of litter objects>} it is
    TP events / GT objects. Without it the report gives RELATIVE recall: TP events of a run / TP events found by
    either run (pooled), which only compares the two runs and says nothing about litter neither run found.
  * Unlabelled detections above alert_conf make the numbers incomplete; they are counted and flagged.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from backend.services.negative_mining import load_review, normalise_label, review_dirs

IOU_LINK = 0.3


@dataclass
class Det:
    video_id: str
    t: float
    cls: str
    conf: float
    box: tuple[float, float, float, float]
    label: Optional[str]
    in_roi: Optional[bool]


@dataclass
class Event:
    video_id: str
    group: str            # TP | FP
    cls: str
    t0: float
    t1: float
    box: tuple[float, float, float, float]
    labels: Counter = field(default_factory=Counter)


@dataclass
class Run:
    name: str
    dets: list[Det]
    hours: float
    alert_conf: float
    videos: list[str]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def load_run(roots: Sequence[Path], name: str, alert_conf: Optional[float] = None) -> Run:
    dirs = review_dirs(roots)
    if not dirs:
        raise ValueError(f"{name}: no review folders under {', '.join(map(str, roots))}")
    dets: list[Det] = []
    secs, confs, vids = 0.0, [], []
    for d in dirs:
        m, rows, _ = load_review(d)
        secs += float(m.get("duration_sec") or 0)
        vids.append(m["video_id"])
        th = (m.get("thresholds") or {}).get("confirmed_threshold")
        if th is not None:
            confs.append(float(th))
        for r in rows:
            roi = r.get("in_roi", "")
            dets.append(Det(r["video_id"], float(r["timestamp_sec"]), r["class"], float(r["conf"]),
                            (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])),
                            normalise_label(r.get("label")), None if roi == "" else roi in ("1", "true", "True")))
    ac = alert_conf if alert_conf is not None else (min(confs) if confs else 0.6)
    return Run(name, dets, secs / 3600.0, ac, vids)


def restrict_to_roi(run: Run, name: str = "after (auto ROI)") -> Run:
    if any(d.in_roi is None for d in run.dets):
        raise ValueError("some detections have no in_roi value: mine with an ROI (--roi) to use --after-roi")
    return Run(name, [d for d in run.dets if d.in_roi], run.hours, run.alert_conf, run.videos)


def events(dets: Sequence[Det], gap_sec: float) -> list[Event]:
    out: list[Event] = []
    open_: dict[tuple[str, str, str], list[Event]] = {}
    for d in sorted(dets, key=lambda x: (x.video_id, x.t)):
        g = "TP" if d.label == "TP" else "FP"
        key = (d.video_id, g, d.cls)
        best = None
        for ev in open_.get(key, []):
            if d.t - ev.t1 <= gap_sec and iou(ev.box, d.box) >= IOU_LINK:
                best = ev if best is None or iou(ev.box, d.box) > iou(best.box, d.box) else best
        if best is None:
            best = Event(d.video_id, g, d.cls, d.t, d.t, d.box)
            out.append(best)
            open_.setdefault(key, []).append(best)
        best.t1, best.box = d.t, d.box
        best.labels[d.label] += 1
    return out


def _match(a: Event, b: Event, gap_sec: float) -> bool:
    return (a.video_id == b.video_id and a.t0 - gap_sec <= b.t1 and b.t0 - gap_sec <= a.t1
            and iou(a.box, b.box) >= IOU_LINK)


def pooled_tp(a: list[Event], b: list[Event], gap_sec: float) -> int:
    matched = sum(1 for e in b if any(_match(e, x, gap_sec) for x in a))
    return len(a) + len(b) - matched


def metrics(run: Run, gap_sec: float) -> dict[str, Any]:
    alerts = [d for d in run.dets if d.conf >= run.alert_conf and not d.cls.startswith("fp_")]
    labelled = [d for d in alerts if d.label is not None]
    unl = len(alerts) - len(labelled)
    tp = sum(d.label == "TP" for d in labelled)
    fp = len(labelled) - tp
    ev = events(labelled, gap_sec)
    tp_ev = [e for e in ev if e.group == "TP"]
    fp_ev = [e for e in ev if e.group == "FP"]
    fp_types = Counter(d.label for d in labelled if d.label != "TP")
    return {"run": run.name, "videos": run.videos, "video_hours": round(run.hours, 4), "alert_conf": run.alert_conf,
            "detections": len(alerts), "unlabelled": unl, "tp": tp, "fp": fp,
            "precision_det": round(tp / (tp + fp), 4) if tp + fp else None,
            "tp_events": len(tp_ev), "fp_events": len(fp_ev),
            "precision_event": round(len(tp_ev) / len(ev), 4) if ev else None,
            "false_alerts_per_hour": round(len(fp_ev) / run.hours, 2) if run.hours else None,
            "fp_detections_per_hour": round(fp / run.hours, 2) if run.hours else None,
            "fp_by_type": dict(fp_types), "_tp_events": tp_ev}


def compare(before: Run, after: Run, gap_sec: float, gt: Optional[dict[str, int]] = None) -> dict[str, Any]:
    mb, ma = metrics(before, gap_sec), metrics(after, gap_sec)
    if gt:
        total = sum(int(gt.get(v, 0)) for v in sorted(set(before.videos) | set(after.videos)))
        for m in (mb, ma):
            m["recall"] = round(min(1.0, m["tp_events"] / total), 4) if total else None
            m["recall_kind"] = "absolute (ground truth)"
    else:
        pool = pooled_tp(mb["_tp_events"], ma["_tp_events"], gap_sec)
        for m in (mb, ma):
            m["recall"] = round(m["tp_events"] / pool, 4) if pool else None
            m["recall_kind"] = "relative (pooled TP events of both runs; no ground truth)"
    for m in (mb, ma):
        m.pop("_tp_events")
    return {"before": mb, "after": ma, "gap_sec": gap_sec,
            "complete": mb["unlabelled"] == 0 and ma["unlabelled"] == 0}


def to_markdown(rep: dict[str, Any]) -> str:
    b, a = rep["before"], rep["after"]
    def f(v: Any, pct: bool = False) -> str:
        if v is None:
            return "n/a"
        return f"{v:.1%}" if pct else str(v)
    rows = [("Detections >= alert conf", "detections", False), ("Unlabelled (excluded)", "unlabelled", False),
            ("TP / FP detections", None, False), ("Precision (detections)", "precision_det", True),
            ("TP / FP events", None, False), ("Precision (events)", "precision_event", True),
            ("Recall", "recall", True), ("False alerts per hour", "false_alerts_per_hour", False),
            ("FP detections per hour", "fp_detections_per_hour", False)]
    out = [f"| Metric | {b['run']} | {a['run']} |", "|---|---|---|"]
    for title, k, pct in rows:
        if title.startswith("TP / FP det"):
            out.append(f"| {title} | {b['tp']} / {b['fp']} | {a['tp']} / {a['fp']} |")
        elif title.startswith("TP / FP ev"):
            out.append(f"| {title} | {b['tp_events']} / {b['fp_events']} | {a['tp_events']} / {a['fp_events']} |")
        else:
            out.append(f"| {title} | {f(b[k], pct)} | {f(a[k], pct)} |")
    out += ["", f"Video hours: {b['video_hours']} (before), {a['video_hours']} (after). Alert threshold: "
                f"{b['alert_conf']} / {a['alert_conf']}. Event linking: gap <= {rep['gap_sec']} s, IoU >= {IOU_LINK}.",
            f"Recall: {b['recall_kind']}.",
            f"FP by type (before): {b['fp_by_type'] or '-'}; (after): {a['fp_by_type'] or '-'}."]
    if not rep["complete"]:
        out.append("**INCOMPLETE: some detections above the alert threshold are not labelled; numbers will change.**")
    return "\n".join(out) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", nargs="+", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--after", nargs="+", type=Path)
    g.add_argument("--after-roi", action="store_true", help="after = before restricted to in_roi == 1")
    ap.add_argument("--alert-conf", type=float, default=None)
    ap.add_argument("--gap-sec", type=float, default=3.0)
    ap.add_argument("--ground-truth", type=Path, default=None, help='JSON {"<video_id>": n_litter_objects}')
    ap.add_argument("--out", type=Path, default=None, help="write the markdown report here (and .json next to it)")
    a = ap.parse_args(argv)
    try:
        before = load_run(a.before, "before", a.alert_conf)
        after = restrict_to_roi(before) if a.after_roi else load_run(a.after, "after", a.alert_conf)
    except (ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    gt = json.loads(a.ground_truth.read_text(encoding="utf-8")) if a.ground_truth else None
    rep = compare(before, after, a.gap_sec, gt)
    md = to_markdown(rep)
    print(md)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(md, encoding="utf-8")
        a.out.with_suffix(".json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return 0 if rep["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
