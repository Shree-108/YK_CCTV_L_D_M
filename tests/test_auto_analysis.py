"""Zero-click video analysis: one upload with auto_analyze=true -> four screenshots -> detection -> reports.

Synthetic videos only. A passing test shows the workflow and the bookkeeping are right; it says nothing about
real-world detection accuracy (see tests/fixtures/v3/README.md for what the litter fixture is).
"""
from __future__ import annotations

import csv
import io
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.sampling import auto_roi, detection_report
from backend.sampling.extract import extract
from backend.sampling.plan import FIXED_FRACTIONS, TooShort, plan_fixed
from backend.sampling.reports import lifecycle
from tests.conftest import FIXTURE_IMAGE

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------- synthetic videos
def plain(idx: int, w: int = 320, h: int = 192) -> np.ndarray:
    """A litter-free 'paved forecourt' (coloured, textured, static camera) with the frame number written on it.
    Coloured and textured on purpose: a flat neutral picture is - correctly - rejected as decoder grey fill."""
    rng = np.random.default_rng(3)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (95, 120, 150)                                          # warm paving colour (BGR)
    img = np.clip(img + rng.normal(0, 14, (h, w, 3)), 0, 255).astype(np.uint8)
    for x in range(0, w, 24):
        cv2.line(img, (x, 0), (x, h), (70, 85, 105), 2)
    for y in range(0, h, 16):
        cv2.line(img, (0, y), (w, y), (70, 85, 105), 2)
    cv2.putText(img, str(idx), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def test_synthetic_frames_pass_the_quality_check(tmp_path):
    """Guard for the tests below: the 'plain' frames must be readable pictures after encoding and decoding."""
    from backend.sampling import frame_quality
    cap = cv2.VideoCapture(str(video(tmp_path / "p.mp4", 30)))
    try:
        frames = [cap.read()[1] for _ in range(30)]
    finally:
        cap.release()
    assert all(frame_quality.problem(f) is None for f in frames)


GREY = np.full((192, 320, 3), 128, np.uint8)       # what a decoder paints where data is missing


def video(path: Path, n: int, fps: float = 10.0, grey: set[int] = frozenset(), frame=plain) -> Path:
    first = frame(0)
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        vw.write(cv2.resize(GREY, (w, h)) if i in grey else frame(i))
    vw.release()
    return path


# ---------------------------------------------------------------------- 4/5. sampling positions and timestamps
def test_fixed_screenshots_at_10_35_65_90_percent_with_disjoint_windows():
    s = plan_fixed(1000, 25.0)
    assert [x.frame for x in s] == [100, 350, 650, 900]
    assert [x.seconds for x in s] == pytest.approx([4.0, 14.0, 26.0, 36.0])
    assert plan_fixed(1000, 25.0) == s                                   # deterministic
    for x in s:
        lo, hi = x.segment
        assert lo <= x.frame <= hi and x.frame - lo <= 25                # at most ~1 s of adjustment
    assert all(a.segment[1] < b.segment[0] for a, b in zip(s, s[1:]))    # can never pick the same frame twice
    with pytest.raises(TooShort):
        plan_fixed(5, 10.0)
    with pytest.raises(TooShort):
        plan_fixed(100, 0)
    assert FIXED_FRACTIONS == (0.10, 0.35, 0.65, 0.90)


def test_undecodable_planned_frame_is_replaced_by_the_nearest_readable_one_and_recorded(tmp_path):
    v = video(tmp_path / "v.mp4", 120, grey={12})                       # 10 % of 120 = frame 12
    out = extract(v, 10.0, plan_fixed(120, 10.0), tmp_path / "f", "v", adjust=True)
    assert [e.status for e in out] == ["saved"] * 4
    first = out[0]
    # the encoder smears the grey frame into its neighbours, so the nearest *readable* one may be a few frames away
    assert first.frame != 12 and abs(first.frame - 12) <= 10 and first.sample.frame == 12
    from backend.sampling import frame_quality
    assert frame_quality.problem(cv2.imread(str(first.path))) is None
    assert "planned frame 12 was not usable" in first.note and "grey fill" in first.note
    assert f"_f{first.frame:07d}_" in first.path.name                   # file name carries the frame really used
    assert first.seconds == pytest.approx(first.frame / 10.0, abs=0.11)
    assert all(e.note is None and e.frame == e.sample.frame for e in out[1:])


def test_without_a_readable_frame_nearby_the_screenshot_fails_and_nothing_is_substituted(tmp_path):
    v = video(tmp_path / "v.mp4", 120, grey=set(range(28, 58)))          # 35 % = frame 42, window 32..52
    out = extract(v, 10.0, plan_fixed(120, 10.0), tmp_path / "f", "v", adjust=True)
    assert out[1].status == "unreadable" and out[1].path is None and "no readable frame" in out[1].error
    assert [e.status for i, e in enumerate(out) if i != 1] == ["saved"] * 3


def test_random_operator_sampling_still_never_substitutes(tmp_path):
    from backend.sampling.plan import Sample
    v = video(tmp_path / "v.mp4", 60, grey={20})
    out = extract(v, 10.0, [Sample(1, 20, 2.0, (15, 25))], tmp_path / "f", "v")   # adjust=False (default)
    assert out[0].status == "unreadable" and out[0].frame == 20


# ---------------------------------------------------------------------- 13/14. ROI policy
INCLUDE = {"name": "pavement", "type": "sidewalk", "mode": "include", "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}
EXCLUDE = {"name": "shelf", "type": "custom", "mode": "exclude", "polygon": [[0.6, 0], [1, 0], [1, 0.5], [0.6, 0.5]]}


def test_roi_policy_priorities_and_safe_fallback():
    d = auto_roi.decide("CAM-1", {"id": "CAM-1", "roi": [INCLUDE, EXCLUDE]}, None)
    assert (d.status, d.source, d.approved, d.name) == ("APPROVED", "REGISTERED_CAMERA", True, "pavement")
    assert d.configuration_id.startswith("camera:CAM-1#") and len(d.zones) == 2
    d = auto_roi.decide("CAM-2", None, [INCLUDE])
    assert (d.status, d.source, d.approved) == ("APPROVED", "SAVED_CAMERA_ROI", True)
    for cam_id, cam, saved in ((None, None, None), ("CAM-3", None, None)):
        d = auto_roi.decide(cam_id, cam, saved)
        assert (d.status, d.source, d.approved, d.zones) == ("NOT_CONFIGURED", "FULL_FRAME_FALLBACK", False, [])
        assert "whole frame" in d.note
    d = auto_roi.decide("CAM-4", None, [EXCLUDE])                        # no ground area: never "approved"
    assert d.status == "NOT_CONFIGURED" and [z["name"] for z in d.zones] == ["shelf"] and "exclude zone" in d.note
    d = auto_roi.decide("CAM-1", {"id": "CAM-1", "roi": [INCLUDE]}, None, roi_enabled=False)
    assert d.status == "DISABLED_IN_SETTINGS" and d.zones == [] and not d.approved
    d = auto_roi.decide("CAM-5", None, [{"mode": "include", "polygon": [[0, 0]]}])   # damaged stored ROI
    assert d.status == "NOT_CONFIGURED"


# ---------------------------------------------------------------------- linking and report rows
def _frame(idx, dets, outcome="LITTER_DETECTED", **kw):
    return {"idx": idx, "frame_idx": idx * 10, "video_time_sec": idx * 1.0, "status": "saved", "outcome": outcome,
            "original_path": f"/x/v_s{idx}.jpg", "annotated_path": f"/x/v_s{idx}_annotated.jpg",
            "result": {"detections": dets, "image_size": [1000, 500], "model_version": "v3_store_cctv"}, **kw}


def _det(cls, box, conf=0.9, status="confirmed", tid=1, **kw):
    return {"class": cls, "confidence": conf, "bbox": box, "status": status, "track_id": tid, **kw}


def test_the_same_object_in_several_screenshots_is_one_estimated_object():
    frames = [_frame(1, [_det("plastic_bottle", [100, 300, 140, 380], severity="MEDIUM"),
                         _det("plastic_bottle", [110, 300, 150, 380], 0.8, tid=2, severity="MEDIUM")]),
              _frame(2, [_det("plastic_bottle", [102, 301, 142, 381], 0.95, severity="HIGH", alert_id=7)]),
              _frame(3, [_det("paper", [700, 100, 760, 150], 0.5, status="candidate")], outcome="NO_LITTER"),
              _frame(4, [], outcome="NO_LITTER")]
    objs = detection_report.link_objects(frames, iou_min=0.3, center_frac=0.04, similar=lambda a, b: a == b)
    ids = [d["object_id"] for d in frames[0]["result"]["detections"]]
    assert len(set(ids)) == 2                                            # two boxes in one picture = two objects
    assert frames[1]["result"]["detections"][0]["object_id"] == ids[0]  # best match wins in screenshot 2
    o = objs[ids[0]]
    assert o["sightings"] == [1, 2] and o["severity"] == "HIGH" and o["alert_id"] == 7
    assert o["max_confidence"] == 0.95
    assert len(objs) == 3 and objs["OBJ-003"]["confirmed_sightings"] == 0      # candidate: possible litter only


JOB = {"id": "sv_1", "store_id": "S1", "camera_id": "CAM-1", "started_at": "2026-11-15T10:00:00.000+00:00",
       "mode": "auto", "sampling": "fixed_pct", "sample_count": 4, "model_version": "v3_store_cctv"}
MEDIA = {"filename": "clip.mp4", "width": 1000, "height": 500, "fps": 10.0, "frames": 400, "duration_sec": 40.0}
APPROVED = auto_roi.decide("CAM-1", {"roi": [INCLUDE]}, None).as_dict()
FALLBACK = auto_roi.decide(None, None, None).as_dict()


def _report(frames, roi=APPROVED, failed=False):
    objs = detection_report.link_objects(frames, iou_min=0.3, center_frac=0.04, similar=lambda a, b: a == b)
    return detection_report.build(JOB, MEDIA, frames, objs, roi, completed_at="2026-11-15T10:00:12.500+00:00",
                                  failed=failed)


def _csv(rows, fields=detection_report.DETECTION_FIELDS):
    return list(csv.DictReader(io.StringIO(detection_report.to_csv(rows, fields))))


def test_csv_columns_values_and_no_repeated_totals():
    frames = [_frame(1, [_det("plastic_bottle", [100, 300, 140, 380], severity="MEDIUM", alert_id=5)]),
              _frame(2, [_det("plastic_bottle", [101, 300, 141, 380], severity="MEDIUM"),
                         _det("cup", [600, 300, 620, 340], 0.7, tid=2, status="rejected", reason="outside_roi")]),
              _frame(3, [], outcome="NO_LITTER"), _frame(4, [], outcome="NO_LITTER")]
    rows, summary = _report(frames)
    out = _csv(rows)
    required = ("job_id video_filename video_duration_seconds video_resolution video_fps total_video_frames "
                "analysis_started_at analysis_completed_at processing_duration_seconds store_id camera_id "
                "screenshot_id screenshot_number screenshot_timestamp_seconds frame_number screenshot_filename "
                "annotated_screenshot_filename detection_id object_track_id litter_class confidence_score "
                "bounding_box_x1 bounding_box_y1 bounding_box_x2 bounding_box_y2 bounding_box_width "
                "bounding_box_height object_area_pixels object_area_percentage roi_status roi_source roi_name "
                "roi_membership roi_configuration_id detection_status severity alert_id stationary_confirmation_status "
                "duplicate_detection_status final_classification review_required screenshot_analysis_status "
                "overall_analysis_status").split()
    assert set(required) <= set(out[0])
    shots = [r for r in out if r["record_type"] == "SCREENSHOT"]
    assert [r["screenshot_number"] for r in shots] == ["1", "2", "3", "4"]          # always one per screenshot
    assert [r["screenshot_analysis_status"] for r in shots] == ["LITTER_OBSERVED", "LITTER_OBSERVED",
                                                                 "NO_LITTER_OBSERVED", "NO_LITTER_OBSERVED"]
    d1 = next(r for r in out if r["record_type"] == "DETECTION" and r["screenshot_number"] == "1")
    assert (d1["litter_class"], d1["confidence_score"], d1["bounding_box_width"], d1["object_area_pixels"]) == \
        ("plastic_bottle", "0.9", "40.0", "3200.0")
    assert d1["object_area_percentage"] == "0.64" and d1["video_resolution"] == "1000x500"
    assert (d1["roi_status"], d1["roi_source"], d1["roi_membership"], d1["roi_name"]) == \
        ("APPROVED", "REGISTERED_CAMERA", "INSIDE", "pavement")
    assert (d1["detection_status"], d1["severity"], d1["alert_id"], d1["final_classification"], d1["review_required"]) == \
        ("CONFIRMED", "MEDIUM", "5", "CONFIRMED_LITTER", "FALSE")
    assert (d1["duplicate_detection_status"], d1["stationary_confirmation_status"]) == ("FIRST_SIGHTING", "REPEATED_AT_SAME_POSITION")
    d2 = next(r for r in out if r["record_type"] == "DETECTION" and r["litter_class"] == "plastic_bottle"
              and r["screenshot_number"] == "2")
    assert d2["object_track_id"] == d1["object_track_id"] and d2["duplicate_detection_status"] == "REPEAT_SIGHTING"
    assert d2["alert_id"] == "5"                                         # the repeat belongs to the same alert
    outside = next(r for r in out if r["litter_class"] == "cup")
    assert (outside["roi_membership"], outside["final_classification"], outside["object_track_id"]) == \
        ("OUTSIDE", "EXCLUDED_OUTSIDE_ROI", "")
    assert d1["processing_duration_seconds"] == "12.5"
    assert d1["overall_analysis_status"] == "LITTER_OBSERVED"
    # totals live only in the summary, never on the detection rows
    assert not {"total_objects_detected", "unique_objects_estimated", "total_high_severity"} & set(out[0])
    assert (summary["total_objects_detected"], summary["unique_objects_estimated"], summary["total_medium_severity"],
            summary["outside_roi_detections"], summary["alerts_created"]) == (2, 1, 1, 1, 1)
    s = _csv([summary], detection_report.SUMMARY_FIELDS)
    assert len(s) == 1 and s[0]["unique_objects_estimated"] == "1"


def test_no_litter_video_is_reported_as_no_litter_observed():
    rows, summary = _report([_frame(i, [], outcome="NO_LITTER") for i in range(1, 5)])
    assert summary["overall_analysis_status"] == "NO_LITTER_OBSERVED" and summary["total_objects_detected"] == 0
    assert len(rows) == 4 and {r["screenshot_analysis_status"] for r in rows} == {"NO_LITTER_OBSERVED"}
    assert {r["final_classification"] for r in rows} == {"NO_LITTER_OBSERVED"}


def test_unreadable_screenshots_make_the_analysis_incomplete_never_clean():
    frames = [_frame(1, [], outcome="NO_LITTER"), _frame(2, [], outcome="UNREADABLE", status="unreadable"),
              _frame(3, [], outcome="NO_LITTER"), _frame(4, [], outcome="FAILED", status="failed")]
    rows, summary = _report(frames)
    assert summary["overall_analysis_status"] == "INCOMPLETE_NO_LITTER_IN_ANALYSED_SCREENSHOTS"
    assert summary["screenshots_failed"] == 2 and summary["review_required"] == "TRUE"
    assert [r["review_required"] for r in rows] == ["FALSE", "TRUE", "FALSE", "TRUE"]
    _, summary = _report(frames, failed=True)
    assert summary["overall_analysis_status"] == "FAILED"


def test_full_frame_fallback_marks_litter_for_review():
    rows, summary = _report([_frame(1, [_det("cup", [10, 10, 30, 40], severity="LOW")])] +
                            [_frame(i, [], outcome="NO_LITTER") for i in (2, 3, 4)], roi=FALLBACK)
    det = next(r for r in rows if r["record_type"] == "DETECTION")
    assert (det["roi_status"], det["roi_source"], det["roi_membership"], det["review_required"]) == \
        ("NOT_CONFIGURED", "FULL_FRAME_FALLBACK", "NOT_APPLICABLE", "TRUE")
    assert summary["review_required"] == "TRUE" and "whole frame" in summary["note"]


def test_csv_formula_injection_is_neutralised():
    job = {**JOB, "store_id": "=HYPERLINK(\"http://x\")", "id": "sv_2"}
    media = {**MEDIA, "filename": "+cmd|' /C calc'!A0.mp4"}
    frames = [_frame(i, [], outcome="NO_LITTER", sampling_note="-2+3 adjusted" if i == 1 else None) for i in range(1, 5)]
    rows, summary = detection_report.build(job, media, frames, {}, APPROVED, completed_at="2026-11-15T10:00:01+00:00")
    out = _csv(rows)
    assert out[0]["store_id"].startswith("'=") and out[0]["video_filename"].startswith("'+")
    assert out[0]["sampling_adjustment"].startswith("'-")
    s = _csv([summary], detection_report.SUMMARY_FIELDS)[0]
    assert s["store_id"].startswith("'=") and s["video_filename"].startswith("'+")


def test_lifecycle_names():
    j = lambda state, stage: lifecycle({"state": state, "stage": stage})   # noqa: E731
    assert [j("queued", "received"), j("running", "validating"), j("running", "sampling"), j("running", "analysing"),
            j("running", "generating_images"), j("running", "generating_csv"), j("done", "completed"),
            j("failed", "sampling"), j("interrupted", "analysing"), j("cancelled", "analysing")] == \
        ["QUEUED", "PROCESSING", "EXTRACTING_SCREENSHOTS", "RUNNING_DETECTION", "RUNNING_DETECTION",
         "GENERATING_REPORT", "COMPLETED", "FAILED", "FAILED", "CANCELLED"]


# ---------------------------------------------------------------------- API: the zero-click journey
@pytest.fixture
def client(cfg):
    cfg.roi_segmenter = "off"
    with TestClient(create_app(cfg)) as c:
        if c.app.state.app_state.engine is None:
            pytest.skip("no detection model loaded")
        yield c


def _drop(client, path: Path, **form):
    """What the page sends when a video is dropped: one multipart request, nothing else."""
    data = {"auto_analyze": "true", **form}
    return client.post("/api/sampled/media", files={"file": (path.name, path.read_bytes(), "video/mp4")}, data=data)


def _wait(client, job_id, timeout=180):
    seen, end = [], time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/sampled/jobs/{job_id}").json()             # read-only polling, as the page does
        if not seen or seen[-1] != j["lifecycle"]:
            seen.append(j["lifecycle"])
        if j["lifecycle"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return j, seen
        time.sleep(0.1)
    raise AssertionError(f"job stuck: {seen}")


def litter_frame(idx: int) -> np.ndarray:
    img = cv2.imread(str(FIXTURE_IMAGE))
    return cv2.resize(img, (640, int(img.shape[0] * 640 / img.shape[1])))


def test_drop_starts_the_whole_analysis_and_reports_exist_at_completion(client, tmp_path, cfg):
    if not FIXTURE_IMAGE.is_file():
        pytest.skip("litter fixture image not present")
    v = video(tmp_path / "shopfront.mp4", 100, fps=10, frame=litter_frame)
    r = _drop(client, v)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["job_id"] and body["duplicate_of_active"] is False         # the upload created the job
    j, seen = _wait(client, body["job_id"])
    assert j["lifecycle"] == "COMPLETED", (j["error"], seen)
    assert j["mode"] == "auto" and j["sampling"] == "fixed_pct" and j["state"] == "done"
    # four screenshots at 10/35/65/90 % (100 frames -> 10, 35, 65, 90) with their decoded time
    assert [f["frame_idx"] for f in j["frames"]] == [10, 35, 65, 90]
    assert [f["video_time_sec"] for f in j["frames"]] == pytest.approx([1.0, 3.5, 6.5, 9.0], abs=0.11)
    assert all(f["original_url"] and f["annotated_url"] for f in j["frames"])
    # no camera given -> whole frame, clearly NOT_CONFIGURED, no alerts from unverified full-frame detections
    assert j["roi_policy"]["status"] == "NOT_CONFIGURED" and j["roi_policy"]["source"] == "FULL_FRAME_FALLBACK"
    assert j["create_alerts"] is False and j["counts"]["alerts"] == 0
    # the fixture holds 19 visually verified litter boxes: the model must find litter in every screenshot, but
    # without an approved ROI that is never pavement-litter confirmation: ROI_NOT_CONFIGURED, listed for review
    assert all(f["outcome"] == "ROI_NOT_CONFIGURED" and f["detection_count"] > 0 for f in j["frames"])
    assert j["counts"]["litter"] == 0 and j["counts"]["unverified_full_frame"] >= 4
    s = j["summary"]
    assert s["total_objects_detected"] >= 4 and 1 <= s["unique_objects_estimated"] < s["total_objects_detected"]
    assert s["overall_analysis_status"] == "PAVEMENT_ROI_NOT_CONFIGURED" and s["review_required"] == "TRUE"
    assert s["total_high_severity"] + s["total_medium_severity"] + s["total_low_severity"] == s["unique_objects_estimated"]
    # reports were written before COMPLETED and are downloadable
    assert set(j["reports"]) == {"detection_csv", "summary_csv", "summary_json"}
    det = client.get(j["reports"]["detection_csv"])
    assert det.status_code == 200 and "attachment" in det.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(det.text)))
    assert sum(r["record_type"] == "SCREENSHOT" for r in rows) == 4
    confirmed = [r for r in rows if r["detection_status"] == "CONFIRMED"]
    assert len(confirmed) == s["total_objects_detected"]
    assert all(r["review_required"] == "TRUE" and r["roi_membership"] == "NOT_APPLICABLE" for r in confirmed)
    assert {r["screenshot_timestamp_seconds"] for r in rows if r["record_type"] == "SCREENSHOT"} == {"1.0", "3.5", "6.5", "9.0"}
    assert any(r["duplicate_detection_status"] == "REPEAT_SIGHTING" for r in confirmed)
    assert client.get(j["reports"]["summary_json"]).json()["summary"]["job_id"] == j["id"]
    assert client.get(j["reports"]["summary_csv"]).status_code == 200
    assert client.get(f"/api/sampled/jobs/{j['id']}/reports/../../x.csv").status_code == 404
    assert client.get(f"/api/sampled/jobs/{j['id']}/reports/other.csv").status_code == 404
    img = client.get(j["frames"][0]["download_annotated_url"])
    assert img.status_code == 200 and "attachment" in img.headers["content-disposition"]
    out = Path(cfg.sampled_dir) / j["id"]
    assert (out / "reports" / "detection_report.csv").is_file() and len(list((out / "annotated").glob("*.jpg"))) == 4


def test_no_litter_video_and_saved_camera_roi(client, tmp_path):
    client.put("/api/roi/CAM-9", json={"zones": [INCLUDE]})                # an operator saved this camera's ROI
    r = _drop(client, video(tmp_path / "quiet.mp4", 80), camera_id="CAM-9", store_id="Store 3")
    j, _ = _wait(client, r.json()["job_id"])
    assert j["lifecycle"] == "COMPLETED", j["error"]
    assert (j["roi_policy"]["status"], j["roi_policy"]["source"]) == ("APPROVED", "SAVED_CAMERA_ROI")
    assert j["roi"] == [INCLUDE] and j["create_alerts"] is True
    assert (j["store_id"], j["camera_id"]) == ("Store 3", "CAM-9")
    assert j["summary"]["overall_analysis_status"] == "NO_LITTER_OBSERVED"
    assert all(f["outcome"] == "NO_LITTER" for f in j["frames"])
    assert all(f["sampling_note"] is None and f["planned_frame_idx"] is None for f in j["frames"])   # no adjustment
    rows = list(csv.DictReader(io.StringIO(client.get(j["reports"]["detection_csv"]).text)))
    assert len(rows) == 4 and {r["final_classification"] for r in rows} == {"NO_LITTER_OBSERVED"}
    assert {r["roi_status"] for r in rows} == {"APPROVED"} and rows[0]["roi_name"] == "pavement"


def test_invalid_corrupted_and_too_short_videos_create_no_job(client, tmp_path):
    jobs = lambda: client.get("/api/sampled/jobs").json()["total"]          # noqa: E731
    bad = client.post("/api/sampled/media", files={"file": ("x.exe", b"MZ", "application/octet-stream")},
                      data={"auto_analyze": "true"})
    assert bad.status_code == 415
    corrupt = client.post("/api/sampled/media", files={"file": ("x.mp4", b"\x00\x00\x00 ftypisom broken", "video/mp4")},
                          data={"auto_analyze": "true"})
    assert corrupt.status_code == 422
    short = _drop(client, video(tmp_path / "short.mp4", 6))
    assert short.status_code == 422 and "too short" in short.json()["detail"]
    assert jobs() == 0


def test_unreadable_video_fails_with_a_reason_and_no_clean_result(client, tmp_path):
    v = video(tmp_path / "grey.mp4", 80, grey=set(range(80)))
    j, _ = _wait(client, _drop(client, v).json()["job_id"])
    assert j["lifecycle"] == "FAILED" and "None of the four screenshots" in j["error"]
    assert j["reports"] is None and j["summary"] is None                    # never reported as NO_LITTER


def test_partly_unreadable_video_completes_as_incomplete(client, tmp_path):
    v = video(tmp_path / "gap.mp4", 120, grey=set(range(28, 58)))           # screenshot 2 and its window are grey
    j, _ = _wait(client, _drop(client, v).json()["job_id"])
    assert j["lifecycle"] == "COMPLETED"
    assert j["frames"][1]["outcome"] == "UNREADABLE" and j["frames"][1]["error"]
    # no approved ROI: never clean (the old INCOMPLETE_NO_LITTER_... status no longer applies to an unverified job)
    assert j["summary"]["overall_analysis_status"] == "PAVEMENT_ROI_NOT_CONFIGURED"
    rows = list(csv.DictReader(io.StringIO(client.get(j["reports"]["detection_csv"]).text)))
    shots = [r for r in rows if r["record_type"] == "SCREENSHOT"]
    assert [r["screenshot_analysis_status"] for r in shots][1] == "UNREADABLE" and len(shots) == 4   # failed one kept
    assert all(r["overall_analysis_status"] != "NO_LITTER_OBSERVED" for r in rows)


# ---------------------------------------------------------------------- camera association and ROI safety
ADMIN = {"Authorization": "Bearer test-admin-token"}
BOTTOM = {"name": "forecourt", "type": "sidewalk", "mode": "include", "polygon": [[0, 0.55], [1, 0.55], [1, 1], [0, 1]]}


def _alerts(client) -> int:
    return client.get("/api/alerts", params={"limit": 1}).json()["total"]


def _rows(client, j):
    return list(csv.DictReader(io.StringIO(client.get(j["reports"]["detection_csv"]).text)))


def test_registered_camera_resolves_and_its_approved_roi_is_applied(client, tmp_path):
    if not FIXTURE_IMAGE.is_file():
        pytest.skip("litter fixture image not present")
    r = client.post("/api/cameras", headers=ADMIN, json={"id": "CAM-REG", "name": "Forecourt", "source": "rtsp://127.0.0.1:9/none",
                                                         "enabled": False, "roi": [BOTTOM]})
    assert r.status_code == 201, r.text
    before = _alerts(client)
    j, _ = _wait(client, _drop(client, video(tmp_path / "reg.mp4", 100, frame=litter_frame), camera_id="CAM-REG").json()["job_id"])
    assert j["lifecycle"] == "COMPLETED", j["error"]
    p = j["roi_policy"]
    assert (p["status"], p["source"], p["name"]) == ("APPROVED", "REGISTERED_CAMERA", "forecourt")
    assert p["configuration_id"].startswith("camera:CAM-REG#") and j["create_alerts"] is True
    rows = [x for x in _rows(client, j) if x["record_type"] == "DETECTION"]
    confirmed = [x for x in rows if x["detection_status"] == "CONFIRMED"]
    assert confirmed and all(x["roi_membership"] == "INSIDE" and x["review_required"] == "FALSE" for x in confirmed)
    h = j["media"]["height"]                                                             # foot point inside (2 % margin)
    assert all(float(x["bounding_box_y2"]) >= (0.55 - 0.021) * h for x in confirmed)
    assert any(x["final_classification"] == "EXCLUDED_OUTSIDE_ROI" for x in rows)        # upper area filtered, still listed
    assert _alerts(client) > before and all(x["alert_id"] for x in confirmed)            # approved ROI -> alerts


def test_unknown_camera_or_store_alone_never_selects_another_cameras_roi(client, tmp_path):
    if not FIXTURE_IMAGE.is_file():
        pytest.skip("litter fixture image not present")
    client.put("/api/roi/CAM-OTHER", json={"zones": [BOTTOM]})                          # some other camera's ROI exists
    client.post("/api/cameras", headers=ADMIN, json={"id": "CAM-REG2", "name": "x", "source": "rtsp://127.0.0.1:9/none",
                                                     "location": "Store 1", "enabled": False, "roi": [BOTTOM]})
    v = video(tmp_path / "unk.mp4", 100, frame=litter_frame)
    before = _alerts(client)
    for form in ({"camera_id": "CAM-NOPE", "store_id": "Store 1"}, {"store_id": "Store 1"}):
        j, _ = _wait(client, _drop(client, v, **form).json()["job_id"])
        assert j["lifecycle"] == "COMPLETED", j["error"]
        assert (j["roi_policy"]["status"], j["roi_policy"]["source"], j["roi"]) == ("NOT_CONFIGURED", "FULL_FRAME_FALLBACK", [])
        assert j["create_alerts"] is False and j["counts"]["alerts"] == 0
        dets = [x for x in _rows(client, j) if x["record_type"] == "DETECTION"]
        assert dets and all(x["review_required"] == "TRUE" and x["roi_status"] == "NOT_CONFIGURED" for x in dets)
        assert all(x["alert_id"] == "" for x in dets)
    assert _alerts(client) == before                                                    # no alert from the fallback


def test_reports_are_on_disk_before_the_job_is_marked_completed(client, tmp_path, monkeypatch):
    from backend.sampling.repo import SampledRepo
    real, seen = SampledRepo.update_job, []

    def spy(self, job_id, **fields):
        if fields.get("state") == "done":
            out = Path(self.job(job_id)["output_dir"]) / "reports"
            seen.append(all((out / n).is_file() for n in ("detection_report.csv", "analysis_summary.csv",
                                                          "analysis_summary.json")))
        return real(self, job_id, **fields)
    monkeypatch.setattr(SampledRepo, "update_job", spy)
    j, _ = _wait(client, _drop(client, video(tmp_path / "order.mp4", 80)).json()["job_id"])
    assert j["lifecycle"] == "COMPLETED" and seen == [True]


@pytest.fixture
def held(monkeypatch):
    """Hold the worker inside extraction until released (to observe an active job deterministically)."""
    import backend.sampling.runner as runner_mod
    gate, entered = threading.Event(), threading.Event()
    real = runner_mod.extract

    def slow(*a, **kw):
        entered.set()
        gate.wait(30)
        return real(*a, **kw)
    monkeypatch.setattr(runner_mod, "extract", slow)
    yield gate, entered
    gate.set()


def test_the_same_drop_twice_creates_one_job(client, tmp_path, held):
    gate, entered = held
    v = video(tmp_path / "twice.mp4", 80)
    first = _drop(client, v).json()
    assert entered.wait(10)
    second = _drop(client, v)
    assert second.status_code == 200
    assert second.json()["job_id"] == first["job_id"] and second.json()["duplicate_of_active"] is True
    assert client.get("/api/sampled/reports/overview").json()["summary"]["videos_received"] == 1   # copy dropped
    gate.set()
    j, _ = _wait(client, first["job_id"])
    assert j["lifecycle"] == "COMPLETED"
    third = _drop(client, v).json()                                         # finished: a new drop is a new run
    assert third["job_id"] != first["job_id"] and third["duplicate_of_active"] is False
    _wait(client, third["job_id"])


def test_cancel_during_processing(client, tmp_path, held):
    gate, entered = held
    jid = _drop(client, video(tmp_path / "c.mp4", 80)).json()["job_id"]
    assert entered.wait(10)
    assert client.get(f"/api/sampled/jobs/{jid}").json()["lifecycle"] == "EXTRACTING_SCREENSHOTS"
    assert client.post(f"/api/sampled/jobs/{jid}/cancel").status_code == 200
    gate.set()
    j, _ = _wait(client, jid)
    assert j["lifecycle"] == "CANCELLED" and j["reports"] is None


def test_health_reports_the_upload_limit(client, cfg):
    lim = client.get("/api/health").json()["limits"]
    assert lim["max_video_mb"] == cfg.max_video_mb and ".mp4" in lim["video_extensions"]
