from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

REJECTED, WEAK, CANDIDATE, CONFIRMED = "rejected", "weak", "candidate", "confirmed"


@dataclass
class Detection:
    """One detected object. Coordinates are pixels in the original frame."""
    bbox: list[float]                 # x1, y1, x2, y2
    cls_id: int
    cls_name: str
    confidence: float
    status: str = CANDIDATE           # rejected | weak | candidate | confirmed
    reason: Optional[str] = None      # why rejected (roi, size, class, moving, ...)
    track_id: Optional[int] = None
    source: str = "full"              # "full" or "tile:<i>" (diagnostics for tiled inference)

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])

    def reject(self, reason: str) -> None:
        self.status, self.reason = REJECTED, reason

    def to_dict(self, priority: str | None = None) -> dict[str, Any]:
        d = {"class": self.cls_name, "confidence": round(self.confidence, 4),
             "bbox": [round(v, 1) for v in self.bbox], "status": self.status}
        if self.reason:
            d["reason"] = self.reason
        if self.track_id is not None:
            d["track_id"] = self.track_id
        if priority:
            d["priority"] = priority
        return d


@dataclass
class InferenceInfo:
    inference_ms: float = 0.0
    tiles: int = 0
    tiled: bool = False
    raw_before_merge: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
