"""Outdoor-ROI proposal from the four sampled screenshots.

A *segmenter* turns each screenshot into two probability maps: outdoor ground (pavement, walkway, road edge,
forecourt) and shop interior (indoor floor, shelves, displays, counters). This module is segmenter-agnostic:

0. Ground     - walkable ground minus shop-interior structures (shelving, displays, counters, ceiling)
1. View check  - are the four frames the same camera view? (gradient-structure correlation)
2. Fusion      - static view: average the maps of all readable frames -> one camera ROI;
                 moved view: one ROI per frame, and the job needs per-frame confirmation.
3. Polygon     - largest remaining ground region -> one simplified include polygon. The interior is excluded
                 by construction (carved out of the ground before the polygon is drawn).
4. Confidence  - agreement across frames x plausible size x separation from the interior x segmenter
                 certainty x outdoor specificity (share of the polygon labelled clearly-outdoor rather than an
                 ambiguous "floor"). Below ``AUTO_ACCEPT`` the proposal is "uncertain": an operator must confirm.

Nothing here decides silently: the runner pauses for confirmation unless the job allowed auto-accept AND the
confidence reaches the threshold.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

AUTO_ACCEPT = 0.70          # minimum confidence for an automatic ROI to be used without confirmation
STATIC_VIEW = 0.55          # gradient correlation at or above which two frames show the same view
WORK_WIDTH = 320
MIN_REGION = 0.02           # ignore regions smaller than 2 % of the frame
MAX_POINTS = 16


@dataclass
class Masks:
    ground: np.ndarray          # float32 HxW in [0,1]: probability of outdoor ground
    indoor: np.ndarray          # float32 HxW in [0,1]: probability of shop interior
    certainty: float = 1.0      # segmenter's own mean confidence over the frame (0..1)
    labels: dict = field(default_factory=dict)   # optional: share of the frame per scene class (for the UI)
    outdoor: Optional[np.ndarray] = None          # probability of a *clearly outdoor* surface (road, sidewalk...)


class Segmenter(Protocol):
    name: str

    def masks(self, image: np.ndarray) -> Masks: ...


@dataclass
class FrameProposal:
    idx: int
    zones: list[dict]
    confidence: float
    ground_share: float
    indoor_share: float
    labels: dict


@dataclass
class Proposal:
    available: bool
    method: str
    zones: list[dict]                       # job-level ROI (static view) - include + exclude zones
    confidence: Optional[float]
    status: str                             # proposed | uncertain | unavailable
    static_view: Optional[bool]
    view_similarity: list[list[float]]
    frames: list[FrameProposal]
    note: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["frames"] = [asdict(f) for f in self.frames]
        return d


# ---------------------------------------------------------------------- view check
def _structure(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    g = cv2.cvtColor(cv2.resize(img, (WORK_WIDTH, max(8, round(h * WORK_WIDTH / w))), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.equalizeHist(g.astype(np.uint8)).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    mag = cv2.GaussianBlur(mag, (0, 0), 3)
    return (mag - mag.mean()) / (mag.std() + 1e-6)


def view_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of blurred gradient maps: ~1 same view (lighting/people changes tolerated), low when moved."""
    sa, sb = _structure(a), _structure(b)
    if sa.shape != sb.shape:
        sb = cv2.resize(sb, (sa.shape[1], sa.shape[0]))
    return float(np.clip((sa * sb).mean(), -1.0, 1.0))


# ---------------------------------------------------------------------- polygons
def _clean(binary: np.ndarray) -> np.ndarray:
    k = max(3, (binary.shape[1] // 80) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)


def largest_polygon(binary: np.ndarray) -> Optional[list[list[float]]]:
    """Largest region (>= MIN_REGION of the frame) as a simplified, normalised polygon (<= MAX_POINTS)."""
    h, w = binary.shape[:2]
    m = _clean(binary)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_REGION * h * w:
        return None
    eps = 0.005 * cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, eps, True)
    while len(poly) > MAX_POINTS:
        eps *= 1.4
        poly = cv2.approxPolyDP(c, eps, True)
    if len(poly) < 3:
        return None
    return [[round(float(x) / (w - 1), 4), round(float(y) / (h - 1), 4)] for x, y in poly.reshape(-1, 2)]


def polygon_mask(poly: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    pts = np.array([[x * (w - 1), y * (h - 1)] for x, y in poly], np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def _zones(ground: np.ndarray, indoor: np.ndarray) -> tuple[list[dict], Optional[np.ndarray]]:
    """One include polygon around the largest ground region after carving out the shop interior (dilated a little
    so shelving edges never fall inside the ROI)."""
    k = max(3, (indoor.shape[1] // 60) | 1)
    interior = cv2.dilate((indoor >= 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    inc = largest_polygon((ground >= 0.5) & (interior == 0))
    if not inc:
        return [], None
    inc_mask = polygon_mask(inc, ground.shape) & (interior == 0)
    return [{"name": "outdoor-ground (auto)", "type": "sidewalk", "mode": "include", "polygon": inc}], inc_mask


def _score(ground: np.ndarray, indoor: np.ndarray, inc_mask: Optional[np.ndarray], certainty: float,
           agreement: float, outdoor: Optional[np.ndarray] = None) -> float:
    if inc_mask is None:
        return 0.0
    share = float(inc_mask.mean())
    size = 1.0 if 0.05 <= share <= 0.85 else max(0.0, 1 - abs(share - (0.05 if share < 0.05 else 0.85)) * 6)
    overlap = float((indoor[inc_mask] >= 0.5).mean()) if inc_mask.any() else 1.0
    separation = max(0.0, 1.0 - 4.0 * overlap)
    fill = float((ground[inc_mask] >= 0.5).mean()) if inc_mask.any() else 0.0     # polygon really is ground
    specific = 1.0 if outdoor is None else min(1.0, float((outdoor[inc_mask] >= 0.5).mean()) / 0.5)
    return round(float(np.clip(agreement * size * separation * min(1.0, fill / 0.8) * certainty * specific, 0, 1)), 3)


# ---------------------------------------------------------------------- main entry
def unavailable(note: str) -> Proposal:
    return Proposal(False, "none", [], None, "unavailable", None, [], [], note)


def propose(images: dict[int, np.ndarray], segmenter: Optional[Segmenter]) -> Proposal:
    """``images``: readable screenshots by sample index (1..4)."""
    if segmenter is None:
        return unavailable("Automatic outdoor-ROI identification is not available on this server; "
                           "draw the outdoor ground area on a screenshot.")
    if not images:
        return unavailable("No readable screenshot to identify the outdoor area from.")
    order = sorted(images)
    sim = [[round(view_similarity(images[a], images[b]), 3) if a != b else 1.0 for b in order] for a in order]
    static = all(sim[i][j] >= STATIC_VIEW for i in range(len(order)) for j in range(len(order)))
    shape = images[order[0]].shape[:2]
    per: dict[int, Masks] = {}
    for i in order:
        m = segmenter.masks(images[i])
        if m.ground.shape != shape:     # different resolution between frames: compare at the first frame's size
            rs = lambda a: None if a is None else cv2.resize(a, (shape[1], shape[0]))   # noqa: E731
            m = Masks(rs(m.ground), rs(m.indoor), m.certainty, m.labels, rs(m.outdoor))
        per[i] = m

    frames: list[FrameProposal] = []
    for i in order:
        m = per[i]
        z, inc = _zones(m.ground, m.indoor)
        frames.append(FrameProposal(i, z, _score(m.ground, m.indoor, inc, m.certainty, 1.0, m.outdoor),
                                    round(float((m.ground >= 0.5).mean()), 3), round(float((m.indoor >= 0.5).mean()), 3),
                                    m.labels))
    if not static:
        conf = min((f.confidence for f in frames), default=0.0)
        return Proposal(True, segmenter.name, [], conf, "uncertain", False, sim, frames,
                        "The camera view differs between screenshots: confirm the outdoor area for each screenshot.")

    ground = np.mean([per[i].ground for i in order], axis=0)
    indoor = np.mean([per[i].indoor for i in order], axis=0)
    binaries = [per[i].ground >= 0.5 for i in order]
    pairs = [(a, b) for a in range(len(binaries)) for b in range(a + 1, len(binaries))]
    agreement = float(np.mean([_iou(binaries[a], binaries[b]) for a, b in pairs])) if pairs else 1.0
    zones, inc = _zones(ground, indoor)
    certainty = float(np.mean([per[i].certainty for i in order]))
    outs = [per[i].outdoor for i in order]
    outdoor = None if any(o is None for o in outs) else np.mean(outs, axis=0)
    conf = _score(ground, indoor, inc, certainty, agreement, outdoor)
    if not any(z["mode"] == "include" for z in zones):
        return Proposal(True, segmenter.name, zones, 0.0, "uncertain", True, sim, frames,
                        "No outdoor ground area was found in the screenshots. The camera may not show the area "
                        "outside the store: confirm or draw the ROI manually.")
    status = "proposed" if conf >= AUTO_ACCEPT else "uncertain"
    note = ("Outdoor area identified consistently in all screenshots." if status == "proposed" else
            "The outdoor area could not be separated from the shop interior with enough confidence: please check "
            "and adjust the proposed ROI.")
    return Proposal(True, segmenter.name, zones, conf, status, True, sim, frames, note)
