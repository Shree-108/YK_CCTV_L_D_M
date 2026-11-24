"""Runtime detection settings (validated with Pydantic, persisted in the DB).

Global settings are stored under the ``global`` key of the ``settings`` table.
Each camera may override any subset via ``cameras.settings_json`` (deep-merged).
All thresholds below are DEFAULTS; the operating threshold should be chosen from
the validation threshold sweep (training/evaluate.py), not guessed.
"""
from __future__ import annotations

import copy
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConfidenceSettings(_M):
    weak_threshold: float = Field(0.30, gt=0, lt=1)       # below -> discarded by the detector
    candidate_threshold: float = Field(0.45, gt=0, lt=1)  # >= counts as a temporal "hit"
    confirmed_threshold: float = Field(0.60, gt=0, lt=1)  # mean hit conf needed to confirm

    @model_validator(mode="after")
    def _order(self) -> "ConfidenceSettings":
        if not (self.weak_threshold <= self.candidate_threshold <= self.confirmed_threshold):
            raise ValueError("thresholds must satisfy weak <= candidate <= confirmed")
        return self


class InferenceSettings(_M):
    image_size: int = Field(640, ge=160, le=4096)
    iou_threshold: float = Field(0.50, gt=0, lt=1)
    max_detections: int = Field(300, ge=1, le=3000)
    device: str = "auto"                 # auto | cpu | cuda | cuda:0 | mps
    half: bool = True                    # FP16 on CUDA only
    agnostic_nms: bool = False           # False: two different classes may overlap (packet on cardboard)


class TileSettings(_M):
    enabled: bool = True
    mode: Literal["auto", "always"] = "auto"   # auto: only tile frames larger than tile_size * auto_min_ratio
    tile_size: int = Field(1024, ge=256, le=4096)
    overlap: float = Field(0.20, ge=0, lt=0.9)
    auto_min_ratio: float = Field(1.25, ge=1.0)
    include_full_frame: bool = True      # also run on the whole (downscaled) frame for large objects
    merge_iou: float = Field(0.50, gt=0, lt=1)
    cross_tile_ios: float = Field(0.75, gt=0, le=1)  # suppress a cut-off partial box mostly inside another


class FilterSettings(_M):
    class_allowlist: Optional[list[str]] = None      # None = all model classes
    class_blocklist: list[str] = []
    min_box_area_px: float = Field(36.0, ge=0)        # 6x6 px; tiny litter must survive
    max_box_area_frac: float = Field(0.25, gt=0, le=1)
    max_aspect_ratio: float = Field(12.0, ge=1)


class TrackingSettings(_M):
    iou_match: float = Field(0.30, gt=0, lt=1)
    low_iou_match: float = Field(0.50, gt=0, lt=1)
    max_missed_frames: int = Field(15, ge=1)          # sampled frames a track survives unseen
    temporal_window: int = Field(5, ge=1, le=100)
    min_confirmations: int = Field(3, ge=1, le=100)
    stationarity_enabled: bool = True
    max_center_shift_frac: float = Field(0.05, gt=0, le=1)  # of frame diagonal; litter doesn't walk away

    @model_validator(mode="after")
    def _window(self) -> "TrackingSettings":
        if self.min_confirmations > self.temporal_window:
            raise ValueError("min_confirmations cannot exceed temporal_window")
        return self


class SeverityRule(_M):
    priorities: list[str] = []       # class priorities from class_metadata.json
    classes: list[str] = []
    min_confidence: float = 0.0
    min_active_objects: int = 0      # confirmed objects currently visible on that camera


class SeveritySettings(_M):
    high: list[SeverityRule] = [SeverityRule(priorities=["critical"], min_confidence=0.75),
                                SeverityRule(min_active_objects=5)]
    medium: list[SeverityRule] = [SeverityRule(priorities=["critical", "high"])]
    # anything not matching high/medium is LOW


class AlertSettings(_M):
    enabled: bool = True
    one_alert_per_track: bool = True
    dedup_window_sec: float = Field(900, ge=0)
    dedup_iou: float = Field(0.30, ge=0, le=1)
    dedup_center_frac: float = Field(0.04, ge=0, le=1)
    class_match: Literal["exact", "group", "any"] = "group"
    snapshot_jpeg_quality: int = Field(90, ge=30, le=100)
    crop_padding_frac: float = Field(0.25, ge=0, le=2)
    alert_on_image_uploads: bool = True


class VideoSettings(_M):
    sample_fps: float = Field(2.0, gt=0, le=60)       # frames/s analysed from files (litter is static)
    tiling: bool = False                              # tiled inference for video (~6x slower on CPU)
    skip_static_frames: bool = True                   # reuse detections while a fixed camera's view is unchanged
    static_diff_threshold: float = Field(2.0, ge=0, le=255)  # mean abs grey-level change (96x54 thumbnail)
    max_reuse_sec: float = Field(5.0, ge=0)           # re-run the model at least this often
    batch_size: int = Field(8, ge=1, le=64)           # frames per model call on GPU (CPU always uses 1)
    live_process_fps: float = Field(2.0, gt=0, le=60) # frames/s analysed per live camera (PROCESSING_FPS)
    live_frame_skip: int = Field(0, ge=0, le=100)     # FRAME_SKIP: offer only every (n+1)-th captured frame
    live_log_interval_sec: float = Field(1.0, ge=0)   # throttle raw detection rows for live cameras
    annotate_all_frames: bool = True                  # output video keeps original fps
    keep_original_video: bool = True                  # keep the uploaded video in LDM_OUTPUT/.../input (owner brief)
    max_video_seconds: float = Field(3600, gt=0)
    screenshot_seed: Optional[int] = Field(None, ge=0)  # four evidence screenshots; None = derived from the analysis id


class ViewQualitySettings(_M):
    """When a screenshot counts as ``view_blocked`` in the dashboard import rows (backend/services/dashboard_rows.py).
    Reporting only: detection never reads these. Measured on a 640-px-wide grey copy of the screenshot. Defaults
    were set well clear of 156 real CCTV screenshots in storage/ (2026-11-24: Laplacian variance >= 1880, mean
    brightness 63-129, grey std >= 45), so ordinary day and night views are never flagged."""
    blur_min_laplacian_var: float = Field(100.0, ge=0)    # below -> "blurred" (out of focus, smeared lens, fog)
    dark_max_mean: float = Field(30.0, ge=0, le=255)      # mean brightness below -> "too dark"
    bright_min_mean: float = Field(230.0, ge=0, le=255)   # mean brightness above -> "too bright" (glare, overexposed)
    uniform_max_std: float = Field(12.0, ge=0, le=255)    # grey std below -> "mostly uniform" (covered lens, no signal)

    @model_validator(mode="after")
    def _order(self) -> "ViewQualitySettings":
        if self.dark_max_mean >= self.bright_min_mean:
            raise ValueError("dark_max_mean must be below bright_min_mean")
        return self


class DetectionSettings(_M):
    confidence: ConfidenceSettings = ConfidenceSettings()
    inference: InferenceSettings = InferenceSettings()
    tiling: TileSettings = TileSettings()
    filters: FilterSettings = FilterSettings()
    tracking: TrackingSettings = TrackingSettings()
    severity: SeveritySettings = SeveritySettings()
    alerts: AlertSettings = AlertSettings()
    video: VideoSettings = VideoSettings()
    view_quality: ViewQualitySettings = ViewQualitySettings()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# Keys of the removed ROI / pavement-only features (owner decision 2026-11-19: ROI removed from the application).
# Settings stored by older versions may still contain them; they are dropped when read so they can never fail
# validation (extra="forbid") or influence detection. The stored JSON itself is left as it is (history).
LEGACY_TOP_KEYS = ("pavement",)
LEGACY_FILTER_KEYS = ("roi_enabled", "roi_anchor", "roi_margin_frac", "roi_crop_inference", "roi_crop_pad_frac")


def strip_legacy(raw: dict[str, Any] | None) -> dict[str, Any]:
    out = {k: v for k, v in (raw or {}).items() if k not in LEGACY_TOP_KEYS}
    if isinstance(out.get("filters"), dict):
        out["filters"] = {k: v for k, v in out["filters"].items() if k not in LEGACY_FILTER_KEYS}
    return out


def merged_settings(global_json: dict[str, Any] | None, camera_json: dict[str, Any] | None = None) -> DetectionSettings:
    base = DetectionSettings().model_dump()
    return DetectionSettings.model_validate(deep_merge(deep_merge(base, strip_legacy(global_json)),
                                                       strip_legacy(camera_json)))
