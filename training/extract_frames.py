"""Extract candidate frames from CCTV recordings into the annotation workspace (sources are only read).

    python -m training.extract_frames "C:/SHREE/Screen Captures/cctv_litter_poc_3min_h264.mp4" --every 2 \
        --session store_entrance_poc --camera store_entrance --conditions day --used-in-previous-training likely

* One frame every --every seconds, but a frame is kept only when it differs from the last kept frame of the same
  recording by at least --min-diff dHash bits (out of 256): a static scene does not produce 90 copies of itself.
* Every kept frame gets a manifest.csv row with the source video, frame index, video timestamp and, if
  --recorded-at is given, the wall-clock time of that frame. Re-running is safe: frames already listed are skipped.
* Frames that look like they carry burned-in analytics boxes are flagged overlay_suspect=yes (excluded from the
  dataset build until a human clears them). The app's own annotated outputs (storage/videos, storage/alerts) are
  refused outright: they have boxes drawn into every frame.
* --session: one recording session = one camera, one continuous time window. The dataset build never puts frames of
  one session into different splits, so give each recording (or each day) its own session name.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import cv2

from training.ldm_common import (CONDITIONS, DEFAULT_WORKSPACE, SOURCE_TYPES, VIDEO_EXTS, YES_NO, DatasetPrepError,
                                 dhash, ensure_workspace, hamming, is_project_overlay_output, model_classes, now_iso,
                                 overlay_suspect, read_manifest, safe_name, sha256_file, split_tags, write_manifest)


def _videos(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in VIDEO_EXTS)
        elif p.is_file():
            out.append(p)
        else:
            raise DatasetPrepError(f"not found: {p}")
    return out


def extract(video: Path, ws: Path, rows: list[dict[str, str]], a: argparse.Namespace) -> dict[str, int]:
    session = safe_name(a.session or video.stem)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise DatasetPrepError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0 or fps > 240:
        cap.release()
        raise DatasetPrepError(f"{video}: unknown frame rate; cannot sample by time")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(a.every * fps))
    start_f = int(a.start * fps)
    end_f = int(a.end * fps) if a.end else (n_frames or 10**12)
    t0 = dt.datetime.fromisoformat(a.recorded_at) if a.recorded_at else None
    known = {r["image"] for r in rows}
    src_hash = sha256_file(video)
    stats = {"sampled": 0, "kept": 0, "near_duplicate": 0, "already_listed": 0, "overlay_suspect": 0, "unreadable": 0}
    last_hash: str | None = None
    idx = 0
    if start_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        idx = start_f
    next_pick = start_f
    while idx < end_f and (a.max_frames is None or stats["kept"] < a.max_frames):
        if idx < next_pick:            # grab() decodes nothing visible: much faster than seeking in H.264
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        next_pick += step
        stats["sampled"] += 1
        if frame is None or frame.size == 0:
            stats["unreadable"] += 1
            continue
        frame_idx = idx - 1
        t_sec = frame_idx / fps
        name = f"{session}__f{frame_idx:07d}.jpg"
        if name in known:
            stats["already_listed"] += 1
            last_hash = dhash(frame)
            continue
        h = dhash(frame)
        if last_hash is not None and hamming(h, last_hash) < a.min_diff:
            stats["near_duplicate"] += 1
            continue
        last_hash = h
        suspect = a.source_has_overlay or overlay_suspect(frame)
        stats["overlay_suspect"] += int(suspect)
        if not a.dry_run:
            out = ws / "images" / "train" / name
            if not cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise DatasetPrepError(f"cannot write {out}")
        hh, ww = frame.shape[:2]
        rows.append({
            "image": name, "source_type": a.source_type, "source_path": str(video.resolve()),
            "session": session, "camera": a.camera or "", "frame_index": str(frame_idx),
            "timestamp_sec": f"{t_sec:.3f}",
            "recorded_at": (t0 + dt.timedelta(seconds=t_sec)).isoformat(timespec="seconds") if t0 else "",
            "width": str(ww), "height": str(hh), "sha256": src_hash, "dhash": h,
            "overlay_suspect": "yes" if suspect else "no", "used_in_previous_training": a.used_in_previous_training,
            "category": "unknown", "subcategory": "", "conditions": ";".join(split_tags(a.conditions or "")),
            "review_status": "unreviewed", "split_lock": a.split_lock or "", "notes": a.notes or "", "added_at": now_iso(),
        })
        known.add(name)
        stats["kept"] += 1
    cap.release()
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="video files or folders (searched recursively)")
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    ap.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames (default 2)")
    ap.add_argument("--min-diff", type=int, default=10,
                    help="minimum dHash difference (bits of 256) to the last kept frame; 0 keeps every sample")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--end", type=float, default=None, help="end time in seconds")
    ap.add_argument("--max-frames", type=int, default=None, help="cap per video")
    ap.add_argument("--session", help="recording session name (default: video file name); one video only")
    ap.add_argument("--camera", help="camera id / location, e.g. store_entrance")
    ap.add_argument("--recorded-at", help="wall-clock start of the video, ISO format, e.g. 2026-09-25T15:41:00+05:30")
    ap.add_argument("--source-type", choices=SOURCE_TYPES, default="real_cctv")
    ap.add_argument("--conditions", help=f"tags for the whole video, ';'-separated: {', '.join(CONDITIONS)}")
    ap.add_argument("--used-in-previous-training", choices=YES_NO, default="unknown",
                    help="was this footage part of the v3 training data? yes/likely -> kept in train only")
    ap.add_argument("--split-lock", choices=("train", "val", "test"), help="force every frame of this video into a split")
    ap.add_argument("--source-has-overlay", action="store_true",
                    help="the whole recording has burned-in analytics boxes: flag every frame")
    ap.add_argument("--notes")
    ap.add_argument("--dry-run", action="store_true", help="count only; write nothing")
    a = ap.parse_args(argv)

    try:
        bad = [t for t in split_tags(a.conditions or "") if t not in CONDITIONS]
        if bad:
            raise DatasetPrepError(f"unknown condition tag(s) {bad}; allowed: {', '.join(CONDITIONS)}")
        if a.recorded_at:
            dt.datetime.fromisoformat(a.recorded_at)
        videos = _videos(a.videos)
        if a.session and len(videos) != 1:
            raise DatasetPrepError("--session can only be used with a single video")
        refused = [v for v in videos if is_project_overlay_output(v)]
        for v in refused:
            print(f"REFUSED {v}: the app's annotated output (boxes drawn in) - use the original recording instead")
        videos = [v for v in videos if v not in refused]
        ws = ensure_workspace(a.workspace, model_classes())
        rows = read_manifest(ws)
        for v in videos:
            s = extract(v, ws, rows, a)
            print(f"{v.name}: sampled {s['sampled']}, kept {s['kept']}, near-duplicates skipped {s['near_duplicate']}, "
                  f"already listed {s['already_listed']}, overlay-suspect {s['overlay_suspect']}, unreadable {s['unreadable']}")
            if not a.dry_run:
                write_manifest(ws, rows)    # after every video: a crash never loses earlier work
    except (DatasetPrepError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if a.dry_run:
        print("dry run: nothing written")
    else:
        print(f"workspace: {a.workspace} ({len(rows)} images listed in manifest.csv)")
    return 1 if refused and not videos else 0


if __name__ == "__main__":
    sys.exit(main())
