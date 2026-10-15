"""Rebuild text rows from OCR word boxes (with small-angle deskew)."""
from __future__ import annotations

import math
import statistics

from invoice_extractor.models import Line, Word


def deskew(words: list[Word]) -> tuple[list[Word], float]:
    """Rotate word coordinates by the median text angle so rows of a tilted photo stay level."""
    angles = [w.angle for w in words if abs(w.angle) < 15]
    if len(angles) < 5:
        return words, 0.0
    angle = statistics.median(angles)
    if abs(angle) < 0.3:
        return words, 0.0
    a = math.radians(-angle)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for w in words:
        pts = [(w.x0, w.y0), (w.x1, w.y0), (w.x1, w.y1), (w.x0, w.y1)]
        rot = [(x * ca - y * sa, x * sa + y * ca) for x, y in pts]
        xs, ys = [p[0] for p in rot], [p[1] for p in rot]
        # rotating an axis-aligned box inflates it; shrink back to the original height around the centre
        cy, hh = (min(ys) + max(ys)) / 2, (w.y1 - w.y0) / 2
        out.append(Word(w.text, w.conf, min(xs), cy - hh, max(xs), cy + hh, w.seg, 0.0))
    return out, angle


def group_lines(words: list[Word]) -> list[Line]:
    """Cluster words into rows by vertical overlap; rows sorted top-down, words left-right."""
    if not words:
        return []
    med_h = statistics.median(w.h for w in words)
    lines: list[Line] = []
    for w in sorted(words, key=lambda w: (w.cy, w.x0)):
        best, best_ov = None, 0.0
        for ln in lines[-8:]:
            ov = min(w.y1, ln.y1) - max(w.y0, ln.y0)
            if ov < 0.5 * min(w.h, ln.h) or abs(w.cy - ln.cy) > 0.6 * med_h:
                continue
            # a word that horizontally collides with a word already in the row belongs to another row
            if any(min(w.x1, o.x1) - max(w.x0, o.x0) > 0.3 * min(w.w, o.w) for o in ln.words):
                continue
            if ov > best_ov:
                best, best_ov = ln, ov
        if best is None:
            lines.append(Line([w]))
        else:
            best.words.append(w)
    for ln in lines:
        ln.words.sort(key=lambda w: w.x0)
    lines.sort(key=lambda ln: ln.cy)
    return lines


def median_line_height(lines: list[Line]) -> float:
    return statistics.median(ln.h for ln in lines) if lines else 10.0
