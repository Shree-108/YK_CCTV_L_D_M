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
    roi_enabled: bool = True
    roi_anchor: Literal["footpoint", "center"] = "footpoint"
    roi_margin_frac: float = Field(0.02, ge=0, le=0.2)  # tolerance so the ROI edge doesn't drop real litter
    roi_crop_inference: bool = True                     # run the model only on the ROI's bounding rectangle
    roi_crop_pad_frac: float = Field(0.08, ge=0, le=0.5)  # padding around it (objects standing on the edge)


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
    max_video_seconds: float = Field(3600, gt=0)


class DetectionSettings(_M):
    confidence: ConfidenceSettings = ConfidenceSettings()
    inference: InferenceSettings = InferenceSettings()
    tiling: TileSettings = TileSettings()
    filters: FilterSettings = FilterSettings()
    tracking: TrackingSettings = TrackingSettings()
    severity: SeveritySettings = SeveritySettings()
    alerts: AlertSettings = AlertSettings()
    video: VideoSettings = VideoSettings()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def merged_settings(global_json: dict[str, Any] | None, camera_json: dict[str, Any] | None = None) -> DetectionSettings:
    base = DetectionSettings().model_dump()
    return DetectionSettings.model_validate(deep_merge(deep_merge(base, global_json or {}), camera_json or {}))
