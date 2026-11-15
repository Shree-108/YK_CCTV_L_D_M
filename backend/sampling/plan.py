"""Which four frames to analyse. Pure functions (no I/O) so the sampling is testable and reproducible.

Rule: split the usable frames into 4 equal segments and draw one random frame per segment
(``random.Random(seed)``). Adjacent segments could still yield two frames a few frames apart (end of one,
start of the next), so a minimum gap between consecutive picks is enforced by redrawing within the segment;
the seed fully determines the result. A video too short for 4 well-separated frames is refused - the job
never silently analyses fewer or more than four screenshots.
"""
from __future__ import annotations

import datetime as dt
import random
import re
from dataclasses import asdict, dataclass
from typing import Optional

MIN_GAP_FRACTION = 0.05     # consecutive picks are at least 5 % of the usable length apart ...
MIN_GAP_FRAMES = 2          # ... and never closer than this many frames
MAX_REDRAWS = 200


class TooShort(ValueError):
    pass


@dataclass(frozen=True)
class Sample:
    idx: int                 # 1..count in video order
    frame: int               # 0-based frame number
    seconds: float           # planned position (frame / fps); the extractor records the decoded time
    segment: tuple[int, int]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["segment"] = list(self.segment)
        return d


def segments(first: int, last: int, count: int) -> list[tuple[int, int]]:
    n = last - first + 1
    return [(first + i * n // count, first + (i + 1) * n // count - 1) for i in range(count)]


def min_gap(usable: int) -> int:
    return max(MIN_GAP_FRAMES, int(usable * MIN_GAP_FRACTION))


def plan_samples(frames: int, fps: float, count: int, seed: int) -> list[Sample]:
    """Exactly ``count`` frames, one per equal segment, consecutive picks >= min_gap() apart.
    Frame 0 is skipped (many NVR exports start with an incomplete picture)."""
    if count < 1:
        raise ValueError("count must be >= 1")
    if not fps or fps <= 0 or not frames or frames <= 0:
        raise TooShort("The video's frame rate or length could not be determined, so four screenshots cannot be planned.")
    first, last = 1, frames - 1
    usable = last - first + 1
    gap = min_gap(usable)
    if usable < count * (gap + 1):
        raise TooShort(f"The video is too short for {count} well-separated screenshots "
                       f"({frames} frames at {fps:.1f} fps; at least {count * (gap + 1) + 1} frames are needed).")
    rng = random.Random(seed)
    segs = segments(first, last, count)
    picks: list[int] = []
    for lo, hi in segs:
        lo_ok = max(lo, picks[-1] + gap) if picks else lo
        if lo_ok > hi:                                   # cannot happen with the length check above
            raise TooShort("The video is too short for well-separated screenshots.")
        f = rng.randint(lo, hi)
        tries = 0
        while f < lo_ok and tries < MAX_REDRAWS:
            f, tries = rng.randint(lo, hi), tries + 1
        picks.append(max(f, lo_ok))
    return [Sample(i, f, round(f / fps, 6), seg) for i, (f, seg) in enumerate(zip(picks, segs), start=1)]


def new_seed() -> int:
    return random.SystemRandom().randrange(1, 2 ** 31)


# ---------------------------------------------------------------------- time helpers
def hhmmss(seconds: Optional[float], millis: bool = False) -> str:
    if seconds is None:
        return ""
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}" if millis else f"{h:02d}:{m:02d}:{s:02d}"


def parse_recording_start(value: Optional[str]) -> Optional[dt.datetime]:
    """ISO-8601 local recording start (as typed by the operator); None when blank. Raises ValueError if invalid."""
    if not value:
        return None
    return dt.datetime.fromisoformat(value.strip())


def clock_time(start: Optional[dt.datetime], seconds: Optional[float]) -> Optional[dt.datetime]:
    if start is None or seconds is None:
        return None
    return start + dt.timedelta(seconds=seconds)


# NVR export names such as 20261029_20261029183009_20261029183120_183009.mp4: <day>_<start>_<end>_...
_NVR_NAME = re.compile(r"(?:^|[^0-9])(\d{8})_(\d{14})_(\d{14})(?:[^0-9]|$)")
_COMPACT = re.compile(r"(?:^|[^0-9])(20\d{12})(?:[^0-9]|$)")     # ...20261029183009...


def recording_start_from_name(name: str) -> Optional[str]:
    """A recording start parsed from the file name, as an ISO string - a *suggestion* the operator confirms.
    None when the name carries no unambiguous timestamp (nothing is guessed)."""
    for rx, grp in ((_NVR_NAME, 2), (_COMPACT, 1)):
        m = rx.search(name)
        if m:
            try:
                return dt.datetime.strptime(m.group(grp), "%Y%m%d%H%M%S").isoformat()
            except ValueError:
                return None
    return None
