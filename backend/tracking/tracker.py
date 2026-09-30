"""ByteTrack-style tracker specialised for (mostly static) litter + temporal confirmation.

Association (per processed frame):
  1. candidate-tier detections are matched to live tracks by IoU (Hungarian);
  2. remaining WEAK detections may keep an unmatched track alive (ByteTrack's
     low-score stage) but never count as a confirmation hit and never start tracks;
  3. unmatched candidates start new tentative tracks.

Temporal confirmation: a track is CONFIRMED once, within the last
``temporal_window`` processed frames, it has >= ``min_confirmations`` candidate
hits whose mean confidence >= confirmed_threshold, and (optionally) it has not
moved more than ``max_center_shift_frac`` of the frame diagonal (carried items
are not litter). Once confirmed a track stays confirmed until it is lost.
The class label is a confidence-weighted vote over the track's history.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.core.settings import DetectionSettings
from backend.detection.tiling import iou_matrix
from backend.detection.types import CANDIDATE, CONFIRMED, WEAK, Detection


@dataclass
class Track:
    track_id: int
    bbox: list[float]
    first_frame: int
    last_frame: int
    window: int
    hits: deque = field(default_factory=deque)          # per processed frame: conf if candidate hit else None
    class_votes: Counter = field(default_factory=Counter)
    cls_ids: dict = field(default_factory=dict)
    origin: tuple[float, float] = (0.0, 0.0)
    max_shift_px: float = 0.0
    missed: int = 0
    total_hits: int = 0
    confirmed: bool = False
    confirmed_frame: Optional[int] = None
    best_conf: float = 0.0
    moving: bool = False

    def __post_init__(self) -> None:
        self.hits = deque(maxlen=self.window)
        x1, y1, x2, y2 = self.bbox
        self.origin = ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def cls_name(self) -> str:
        return self.class_votes.most_common(1)[0][0] if self.class_votes else "unknown"

    @property
    def cls_id(self) -> int:
        return self.cls_ids.get(self.cls_name, -1)

    @property
    def hit_count(self) -> int:
        return sum(1 for h in self.hits if h is not None)

    @property
    def mean_hit_conf(self) -> float:
        v = [h for h in self.hits if h is not None]
        return float(np.mean(v)) if v else 0.0

    def observe(self, det: Detection, is_hit: bool, frame_idx: int) -> None:
        # static objects: smooth the box so jitter doesn't move the alert location
        a = 0.6
        self.bbox = [a * n + (1 - a) * o for n, o in zip(det.bbox, self.bbox)]
        cx, cy = (det.bbox[0] + det.bbox[2]) / 2, (det.bbox[1] + det.bbox[3]) / 2
        self.max_shift_px = max(self.max_shift_px, math.hypot(cx - self.origin[0], cy - self.origin[1]))
        self.last_frame, self.missed = frame_idx, 0
        self.class_votes[det.cls_name] += det.confidence
        self.cls_ids[det.cls_name] = det.cls_id
        self.hits.append(det.confidence if is_hit else None)
        if is_hit:
            self.total_hits += 1
            self.best_conf = max(self.best_conf, det.confidence)

    def miss(self) -> None:
        self.missed += 1
        self.hits.append(None)


@dataclass
class TrackEvent:
    kind: str            # "confirmed" | "lost"
    track: Track


class LitterTracker:
    def __init__(self, settings: DetectionSettings) -> None:
        self.s = settings
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame_idx = -1

    def update_settings(self, settings: DetectionSettings) -> None:
        self.s = settings

    @property
    def active_tracks(self) -> list[Track]:
        return list(self.tracks)

    @property
    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed]

    def visible_tracks(self, grace: int = 2) -> list[Track]:
        """Tracks seen within the last ``grace`` processed frames (brief occlusions tolerated)."""
        return [t for t in self.tracks if t.missed <= grace]

    def _match(self, tracks: list[Track], dets: list[Detection], thr: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        ious = iou_matrix(np.array([t.bbox for t in tracks], np.float32), np.array([d.bbox for d in dets], np.float32))
        rows, cols = linear_sum_assignment(1.0 - ious)
        pairs = [(r, c) for r, c in zip(rows, cols) if ious[r, c] >= thr]
        mt, md = {r for r, _ in pairs}, {c for _, c in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(dets)) if j not in md]

    def update(self, dets: list[Detection], frame_idx: int, frame_shape: tuple[int, ...]) -> list[TrackEvent]:
        """Associate this frame's filtered detections. Sets det.track_id and promotes
        detections of confirmed tracks to status CONFIRMED. Returns track events."""
        ts = self.s.tracking
        self.frame_idx = frame_idx
        events: list[TrackEvent] = []
        high = [d for d in dets if d.status == CANDIDATE]
        low = [d for d in dets if d.status == WEAK]

        pairs, un_t, un_d = self._match(self.tracks, high, ts.iou_match)
        for ti, di in pairs:
            self.tracks[ti].observe(high[di], True, frame_idx)
            high[di].track_id = self.tracks[ti].track_id
        rem = [self.tracks[i] for i in un_t]
        pairs2, un_t2, _ = self._match(rem, low, ts.low_iou_match)
        for ti, di in pairs2:
            rem[ti].observe(low[di], False, frame_idx)
            low[di].track_id = rem[ti].track_id
        for i in un_t2:
            rem[i].miss()
        for di in un_d:
            d = high[di]
            t = Track(self._next_id, list(d.bbox), frame_idx, frame_idx, ts.temporal_window)
            self._next_id += 1
            t.observe(d, True, frame_idx)
            d.track_id = t.track_id
            self.tracks.append(t)

        diag = math.hypot(frame_shape[1], frame_shape[0])
        for t in self.tracks:
            t.moving = ts.stationarity_enabled and t.max_shift_px > ts.max_center_shift_frac * diag
            if (not t.confirmed and not t.moving and t.hit_count >= ts.min_confirmations
                    and t.mean_hit_conf >= self.s.confidence.confirmed_threshold):
                t.confirmed, t.confirmed_frame = True, frame_idx
                events.append(TrackEvent("confirmed", t))

        by_id = {t.track_id: t for t in self.tracks}
        for d in dets:
            t = by_id.get(d.track_id) if d.track_id is not None else None
            if t is None:
                continue
            if t.confirmed:
                d.status, d.cls_name, d.cls_id = CONFIRMED, t.cls_name, t.cls_id
            elif t.moving:
                d.reason = "moving"  # stays candidate/weak; cannot confirm while moving

        alive = []
        for t in self.tracks:
            if t.missed > ts.max_missed_frames:
                if t.confirmed:
                    events.append(TrackEvent("lost", t))
            else:
                alive.append(t)
        self.tracks = alive
        return events
