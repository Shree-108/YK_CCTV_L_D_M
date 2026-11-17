"""Operator ROI calibration of automatic four-screenshot analyses (backend/sampling/roi_calibration.py).

Synthetic videos and a temporary database only (the cfg fixture): nothing here reads or writes production data, and a
passing test says nothing about real-camera accuracy - the litter fixture is a synthetic street frame.
"""
from __future__ import annotations

import csv
import io

import cv2
import numpy as np
import pytest

from backend.detection.filters import ROISet
from backend.sampling import auto_roi
from backend.sampling.roi_calibration import InvalidPolygon, aspect_mismatch, tag_membership, validate_zones
from tests.conftest import FIXTURE_IMAGE
from tests.test_auto_analysis import ADMIN, _alerts, _drop, _wait, client, litter_frame, video  # noqa: F401

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# the monitored ground: the lower part of the picture, drawn with perspective (narrower at the top)
GROUND = {"name": "monitored-ground", "type": "ground", "mode": "include",
          "polygon": [[0.08, 0.55], [0.92, 0.55], [1.0, 1.0], [0.0, 1.0]]}
BOWTIE = {"name": "bad", "type": "ground", "mode": "include", "polygon": [[0.1, 0.5], [0.9, 1.0], [0.9, 0.5], [0.1, 1.0]]}
TINY = {"name": "tiny", "type": "ground", "mode": "include", "polygon": [[0.5, 0.5], [0.52, 0.5], [0.51, 0.52]]}
WHOLE = {"name": "all", "type": "ground", "mode": "include", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}


def _approve(client, job_id, zones, ref=1, **kw):
    return client.post(f"/api/sampled/jobs/{job_id}/roi/approve", json={"zones": zones, "reference_idx": ref, **kw})


def _rows(client, j):
    return list(csv.DictReader(io.StringIO(client.get(j["reports"]["detection_csv"]).text)))


def _db(client):
    return client.app.state.app_state.db


def _needs_fixture():
    if not FIXTURE_IMAGE.is_file():
        pytest.skip("litter fixture image not present")


# ---------------------------------------------------------------------- unit: polygon rules and membership
def test_polygon_validation_rejects_invalid_shapes():
    validate_zones([GROUND], 640, 360)                                         # valid
    for zones, why in (([], "Draw"), ([BOWTIE], "crosses itself"), ([TINY], "too small"),
                       ([WHOLE], "whole screenshot"),
                       ([{**GROUND, "mode": "exclude"}], "include zone"),
                       ([{**GROUND, "polygon": [[0.1, 0.6], [0.1, 0.6], [0.5, 0.9], [0.2, 0.95]]}], "on top of each other")):
        with pytest.raises(InvalidPolygon, match=why):
            validate_zones(zones, 640, 360)


def test_membership_uses_the_ground_contact_point_and_keeps_every_detection():
    roi = ROISet.from_json([GROUND])
    w, h = 640, 360
    dets = [
        {"bbox": [300, 250, 340, 300], "status": "confirmed"},     # foot (320, 300) inside the ground
        {"bbox": [300, 120, 340, 180], "status": "confirmed"},     # foot (320, 180): on the shop front, outside
        {"bbox": [300, 150, 340, 215], "status": "candidate"},     # top outside, foot (320, 215) inside -> member
        {"bbox": [5, 205, 25, 215], "status": "weak"},             # foot left of the slanted edge -> outside
    ]
    c = tag_membership(dets, roi, w, h, "footpoint", 0.0)
    assert [d["roi_member"] for d in dets] == [True, False, True, False]
    assert c == {"raw": 4, "inside_roi": 2, "outside_roi": 2, "review_required": 1, "confirmed": 2}
    assert len(dets) == 4                                                     # nothing discarded


def test_the_normalised_polygon_stays_aligned_at_any_resolution():
    roi = ROISet.from_json([GROUND])
    for w, h in ((640, 360), (1920, 1080), (320, 180)):                      # same view, other resolutions
        d = [{"bbox": [0.45 * w, 0.7 * h, 0.55 * w, 0.8 * h], "status": "confirmed"},
             {"bbox": [0.45 * w, 0.2 * h, 0.55 * w, 0.3 * h], "status": "confirmed"}]
        tag_membership(d, roi, w, h, "footpoint", 0.0)
        assert [x["roi_member"] for x in d] == [True, False]
    assert not aspect_mismatch((1920, 1080), (640, 360), 0.02)               # same aspect: no revalidation
    assert aspect_mismatch((640, 480), (640, 360), 0.02)                      # other aspect: revalidate


def test_policy_requires_revalidation_when_the_view_shape_changes():
    d = auto_roi.decide("CAM-1", None, [GROUND], frame_size=(640, 480), reference_size=[640, 360])
    assert (d.status, d.approved, d.zones) == ("REVALIDATION_REQUIRED", False, [])
    d = auto_roi.decide("CAM-1", None, [GROUND], frame_size=(1280, 720), reference_size=[640, 360])
    assert d.status == "APPROVED" and d.reference_size == [640, 360]


# ---------------------------------------------------------------------- API journeys (real model, temp DB)
def test_no_camera_is_roi_not_configured_and_never_a_clean_result(client, tmp_path):
    # quiet (litter-free) video, no camera: before this change it was reported NO_LITTER_OBSERVED
    j, _ = _wait(client, _drop(client, video(tmp_path / "quiet.mp4", 80)).json()["job_id"])
    assert j["lifecycle"] == "COMPLETED", j["error"]
    assert j["roi_policy"]["status"] == "NOT_CONFIGURED" and not j["roi_policy"]["approved"]
    assert all(f["outcome"] == "ROI_NOT_CONFIGURED" for f in j["frames"])
    s = j["summary"]
    assert s["overall_analysis_status"] == "PAVEMENT_ROI_NOT_CONFIGURED" and s["review_required"] == "TRUE"
    rows = _rows(client, j)
    assert all(r["final_classification"] != "NO_LITTER_OBSERVED" for r in rows)
    ov = client.get("/api/sampled/reports/overview").json()
    assert ov["summary"]["total_litter"] == 0


def test_operator_draws_and_approves_roi_applied_to_all_four_screenshots(client, tmp_path):
    _needs_fixture()
    before_alerts = _alerts(client)
    j, _ = _wait(client, _drop(client, video(tmp_path / "shop.mp4", 100, frame=litter_frame)).json()["job_id"])
    assert all(f["outcome"] == "ROI_NOT_CONFIGURED" for f in j["frames"])
    assert j["counts"]["litter"] == 0 and j["counts"]["unverified_full_frame"] > 0     # full frame never counted
    hist = client.get("/api/sampled/reports/detections").json()
    assert hist["total"] == 0                                                          # not in detection history

    r = _approve(client, j["id"], [GROUND], ref=2)
    assert r.status_code == 202, r.text
    j2, seen = _wait(client, j["id"])
    assert j2["lifecycle"] == "COMPLETED", (j2["error"], seen)
    p = j2["roi_policy"]
    assert (p["status"], p["source"], p["approved"], p["reference_idx"]) == ("APPROVED", "OPERATOR_APPROVED_FOR_JOB", True, 2)
    assert p["reference_size"] == j2["frames"][1]["image_size"] and p["configuration_id"].startswith(f"job:{j['id']}#")
    # the SAME normalised polygon on all four screenshots
    assert j2["roi"] == [GROUND]
    assert all(f["roi_status"] == "approved" and f["roi"] is None for f in j2["frames"])
    assert all(f["outcome"] in ("LITTER_DETECTED", "NO_LITTER") for f in j2["frames"])
    for f in j2["frames"]:
        rc = f["roi_counts"]
        assert rc["raw"] == len(f["detections"]) == rc["inside_roi"] + rc["outside_roi"]
        w, h = f["image_size"]
        for d in f["detections"]:
            foot_y = d["bbox"][3] / h
            if d["status"] == "confirmed":                       # retained: inside the ROI (2 % margin)
                assert d["roi_member"] and foot_y >= 0.55 - 0.021
            if not d["roi_member"]:                              # outside: never confirmed, still listed
                assert d["status"] != "confirmed"
    assert any(not d["roi_member"] for f in j2["frames"] for d in f["detections"])      # some were outside
    assert j2["counts"]["litter"] > 0 and j2["summary"]["overall_analysis_status"] == "LITTER_OBSERVED"
    rows = [x for x in _rows(client, j2) if x["record_type"] == "DETECTION"]
    assert any(x["final_classification"] == "EXCLUDED_OUTSIDE_ROI" for x in rows)
    assert all(x["roi_membership"] == "INSIDE" for x in rows if x["detection_status"] == "CONFIRMED")
    # one set of detection rows (the full-frame rows were replaced), no alerts from this upload
    n = _db(client)._one("SELECT COUNT(*) AS n FROM detections WHERE job_id=?", (j["id"],))["n"]
    assert n == sum(f["roi_counts"]["raw"] for f in j2["frames"])
    assert _alerts(client) == before_alerts
    # the approved polygon is visible (green) on the annotated picture at its scaled position
    img = cv2.imdecode(np.frombuffer(client.get(j2["frames"][0]["annotated_url"]).content, np.uint8), cv2.IMREAD_COLOR)
    hh, ww = img.shape[:2]
    y = int(round(0.55 * hh))
    band = img[y - 2:y + 3, int(0.3 * ww):int(0.7 * ww)].reshape(-1, 3).astype(int)
    greenish = (band[:, 1] > band[:, 0] + 40) & (band[:, 1] > band[:, 2] + 40)
    assert greenish.mean() > 0.5
    # re-approval of this job's own ROI is allowed; the original stays traceable
    assert _approve(client, j["id"], [GROUND], ref=1).status_code == 202
    j3, _ = _wait(client, j["id"])
    assert j3["roi_policy"]["replaces"]["source"] == "OPERATOR_APPROVED_FOR_JOB"


def test_invalid_polygons_and_wrong_states_are_rejected(client, tmp_path):
    j, _ = _wait(client, _drop(client, video(tmp_path / "q.mp4", 80)).json()["job_id"])
    for zones in ([BOWTIE], [TINY], [WHOLE]):
        r = _approve(client, j["id"], zones)
        assert r.status_code == 422 and "Invalid ROI polygon" in r.json()["detail"]
    assert _approve(client, j["id"], [{**GROUND, "polygon": [[0, 0], [1.5, 0], [1, 1]]}]).status_code == 422
    assert _approve(client, j["id"], [GROUND], ref=7).status_code == 422
    assert _approve(client, "sv_missing", [GROUND]).status_code == 404
    assert _approve(client, j["id"], [GROUND], save_for_camera=True).status_code == 422   # no camera on this job
    j2 = client.get(f"/api/sampled/jobs/{j['id']}").json()
    assert j2["roi_policy"]["status"] == "NOT_CONFIGURED" and j2["state"] == "done"       # nothing changed


def test_registered_camera_roi_is_reused_and_not_reconfigured(client, tmp_path):
    _needs_fixture()
    assert client.post("/api/cameras", headers=ADMIN, json={
        "id": "CAM-A", "name": "Front", "source": "rtsp://127.0.0.1:9/none", "enabled": False, "roi": [GROUND]}).status_code == 201
    j, _ = _wait(client, _drop(client, video(tmp_path / "a.mp4", 100, frame=litter_frame), camera_id="CAM-A").json()["job_id"])
    p = j["roi_policy"]
    assert (p["status"], p["source"], p["approved"]) == ("APPROVED", "REGISTERED_CAMERA", True)
    assert all(f["roi_counts"] and f["outcome"] in ("LITTER_DETECTED", "NO_LITTER") for f in j["frames"])
    r = _approve(client, j["id"], [GROUND])
    assert r.status_code == 422 and "Cameras page" in r.json()["detail"]                  # never asked again


def test_unknown_camera_never_inherits_and_saved_roi_needs_revalidation_on_a_new_view(client, tmp_path):
    _needs_fixture()
    v = video(tmp_path / "x.mp4", 100, frame=litter_frame)
    # job for CAM-X: approve WITHOUT saving -> another job of CAM-X (or an unknown camera) gets no ROI
    j, _ = _wait(client, _drop(client, v, camera_id="CAM-X").json()["job_id"])
    assert _approve(client, j["id"], [GROUND]).status_code == 202
    _wait(client, j["id"])
    for cam in ("CAM-X", "CAM-UNKNOWN"):
        k, _ = _wait(client, _drop(client, v, camera_id=cam, store_id="Store 1").json()["job_id"])
        assert k["roi_policy"]["status"] == "NOT_CONFIGURED" and k["roi"] == []
    # now approve WITH "save for camera": later videos of CAM-X reuse it, other camera IDs still do not
    j, _ = _wait(client, _drop(client, v, camera_id="CAM-X").json()["job_id"])
    assert _approve(client, j["id"], [GROUND], save_for_camera=True).status_code == 202
    j, _ = _wait(client, j["id"])
    assert _db(client).get_setting("roi:CAM-X") == [GROUND]
    assert _db(client).get_setting("roi_ref:CAM-X") == j["roi_policy"]["reference_size"] == j["frames"][0]["image_size"]
    assert _db(client).get_setting("roi:CAM-UNKNOWN") is None
    n, _ = _wait(client, _drop(client, v, camera_id="CAM-X").json()["job_id"])
    assert (n["roi_policy"]["status"], n["roi_policy"]["source"]) == ("APPROVED", "SAVED_CAMERA_ROI")
    o, _ = _wait(client, _drop(client, v, camera_id="CAM-Y").json()["job_id"])
    assert o["roi_policy"]["status"] == "NOT_CONFIGURED"
    # same camera ID, different picture shape (camera changed / other stream) -> revalidation, nothing confirmed
    sq = video(tmp_path / "sq.mp4", 100, frame=lambda i: cv2.resize(litter_frame(i), (480, 480)))
    m, _ = _wait(client, _drop(client, sq, camera_id="CAM-X").json()["job_id"])
    assert m["roi_policy"]["status"] == "REVALIDATION_REQUIRED" and not m["roi_policy"]["approved"]
    assert all(f["outcome"] == "ROI_FRAME_MISMATCH" for f in m["frames"])
    assert m["counts"]["litter"] == 0 and m["summary"]["review_required"] == "TRUE"
