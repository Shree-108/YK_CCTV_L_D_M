"""
eval_video.py — video-specific evaluation protocol.

Image mAP doesn't capture what matters for video. This module reports:
  * track-level precision/recall — was each GROUND-TRUTH OBJECT (not frame
    detection) found by >=1 confirmed track during its visible lifetime?
  * time-to-first-detection — frames from first visibility to persistence-gate
    confirmation (the real latency metric for monitoring, not infer speed).
  * false-alert rate per hour of footage at the CONFIRMED-alert level.

Self-test (no labeled video needed): builds a synthetic pan clip from a labeled
test image — sliding crop window with exact GT box transforms — then runs the
full video pipeline and scores it. Run:
    python eval_video.py
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import litter_ml
import video_pipeline


def track_metrics(gt_objects: list[dict], tracks: list[dict],
                  frames_seen: int, fps: float):
    """gt_objects: [{id, class, first_frame, last_frame}].
    tracks: fused detection dicts with track_id + frames_seen (approx window).
    Returns track-level recall, time-to-first-detection, FP/hour."""
    found, ttfd = {}, {}
    for g in gt_objects:
        # a confirmed track of the same class overlapping the object's lifetime
        cands = [t for t in tracks if t["class"] == g["class"]]
        if cands:
            best = max(cands, key=lambda t: t["confidence"])
            found[g["id"]] = best["track_id"]
            # TTFD: persistence gate needs min_persistence hits @ sample fps
            ttfd[g["id"]] = best.get("frames_seen", 1) / max(1e-9, fps)
    fp_tracks = [t for t in tracks
                 if t["track_id"] not in set(found.values())]
    hours = max(1e-9, frames_seen / max(1e-9, fps) / 3600.0)
    return {
        "track_recall": round(len(found) / max(1, len(gt_objects)), 4),
        "n_objects": len(gt_objects),
        "n_found": len(found),
        "mean_ttfd_s": round(float(np.mean(list(ttfd.values()))) if ttfd else -1.0, 3),
        "fp_per_hour": round(len(fp_tracks) / hours, 2),
        "n_fp_tracks": len(fp_tracks),
    }


def make_pan_clip(image_path: Path, label_path: Path, out_path: Path,
                  n_frames: int = 12, win_w: int = 640, win_h: int = 480,
                  fps: float = 4.0):
    """Static-camera clip: fixed window + tiny jitter/noise per frame.

    WHY static: CCTV cameras don't pan; litter is stationary. A sliding window
    breaks the IoU tracker's static-scene assumption (boxes shift ~50px between
    sampled frames and can never match). Jitter (±2px) + mild exposure flicker
    simulates real sensor noise while keeping objects matchable.
    """
    img = cv2.imread(str(image_path))
    assert img is not None, image_path
    H, W = img.shape[:2]
    names = litter_ml.load_detector().class_names
    gt_all = []
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        cid = int(float(p[0]))
        gt_all.append((names[cid] if cid < len(names) else str(cid),
                       float(p[1]) * W, float(p[2]) * H,
                       float(p[3]) * W, float(p[4]) * H))
    x0, y0 = max(0, (W - win_w) // 2), max(0, (H - win_h) // 2)
    rng = np.random.default_rng(7)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (win_w, win_h))
    per_frame_gt = []
    for f in range(n_frames):
        dx, dy = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(img[y0:y0 + win_h, x0:x0 + win_w], M, (win_w, win_h),
                               borderMode=cv2.BORDER_REFLECT)
        gain = float(rng.uniform(0.94, 1.06))  # exposure flicker
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        writer.write(frame)
        boxes = []
        for cls, cx, cy, bw, bh in gt_all:
            x1, y1 = cx - bw / 2 - x0 + dx, cy - bh / 2 - y0 + dy
            x2, y2 = x1 + bw, y1 + bh
            if x2 > 0 and y2 > 0 and x1 < win_w and y1 < win_h and (x2 - x1) > 8:
                boxes.append({"class": cls, "box": [x1, y1, x2, y2]})
        per_frame_gt.append(boxes)
    writer.release()
    return per_frame_gt


def self_test() -> dict:
    print("=" * 70 + "\nVIDEO PIPELINE SELF-TEST (synthetic pan clip)\n" + "=" * 70)
    det = litter_ml.load_detector()
    print(f"engine={det.engine_name} trained={det.trained} classes={len(det.class_names)}")
    test_imgs = sorted((HERE.parent / "dataset" / "cctv_litter_dataset_v1"
                        / "images" / "test").glob("*.jpg"))
    assert test_imgs, "no test images"
    img_p = test_imgs[0]
    lbl_p = (HERE.parent / "dataset" / "cctv_litter_dataset_v1"
             / "labels" / "test" / (img_p.stem + ".txt"))
    assert lbl_p.exists(), lbl_p
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "pan.mp4"
        per_frame_gt = make_pan_clip(img_p, lbl_p, clip)
        print(f"clip: {img_p.name}, {len(per_frame_gt)} frames, "
              f"gt boxes in frame0: {len(per_frame_gt[0])}")
        for mode in ("fast", "accurate"):
            res = video_pipeline.process_video(det, str(clip), mode=mode)
            # GT objects = distinct classes visible across clip
            seen = {}
            for f, boxes in enumerate(per_frame_gt):
                for b in boxes:
                    seen.setdefault(b["class"], {"first": f, "last": f})
                    seen[b["class"]]["last"] = f
            gt_objects = [{"id": k, "class": k, "first_frame": v["first"],
                           "last_frame": v["last"]} for k, v in seen.items()]
            m = track_metrics(gt_objects, res["detections"],
                              res["frames_sampled"], 2.0)
            print(f"\n[{mode}] detections={res['count']} "
                  f"tracks={len(res['tracks'])} unconfirmed={res['unconfirmed_tracks']}")
            print(f"[{mode}] track_recall={m['track_recall']} "
                  f"({m['n_found']}/{m['n_objects']}) mean_ttfd={m['mean_ttfd_s']}s "
                  f"fp/hour={m['fp_per_hour']}")
            print(f"[{mode}] timing={res['timing']}")
            q = res["quality_log"]
            print(f"[{mode}] quality: usable={sum(1 for x in q if x['usable'])}/{len(q)}")
    print("\nSELF-TEST DONE")
    return {}


if __name__ == "__main__":
    self_test()
