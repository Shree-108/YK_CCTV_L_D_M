"""Results view (backend/services/results_view.py + /api/results): one row per camera check, importer CSV fields."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.database.db import Database
from backend.main import create_app
from backend.services import garbage_export as ge
from backend.services import results_view as rv
from tests.test_result_bundle import CLASSES, Job, write_cfg

MIXED = ["LITTER_DETECTED", "NO_LITTER", "UNREADABLE", "LITTER_DETECTED"]


def settings(tmp: Path, cfg: Path) -> ge.Settings:
    return ge.Settings(cfg, tmp / "exports", tmp / "ledger.db")


def test_one_row_per_check_with_the_csv_fields(tmp_path):
    cfg = write_cfg(tmp_path)
    job = Job(tmp_path, MIXED, errors={3: "blank frame: covered lens"})
    out = rv.view(job.query, settings(tmp_path, cfg), rv.Filters(), allowed_classes=CLASSES)
    rows = out["rows"]
    assert out["total"] == 4 and [r["idx"] for r in rows] == [1, 2, 3, 4]
    assert out["summary"] == {"total": 4, "clean": 1, "litter_detected": 2, "view_blocked": 1, "time_unknown": 0,
                              "no_camera": 0, "import_ready": 4}
    lit, clean, blocked = rows[0], rows[1], rows[2]
    assert {k: lit[k] for k in ge.COLUMNS} == {
        "timestamp": "2026-11-19T16:30:41+05:30", "camera_id": "TEST_CAM_01",
        "image_name": f"{job.job_id}_s1_annotated.jpg", "litter_class": "food_wrapper", "confidence_score": "78.27",
        "detection_status": "litter_detected", "remark": lit["remark"], "cart_id": "901",
        "store_name": "TEST Store (not a real store)"}
    assert "time source: filename" in lit["remark"] and "under review, not counted: cardboard 40.00" in lit["remark"]
    assert clean["litter_class"] == clean["confidence_score"] == "" and clean["detection_status"] == "clean"
    assert blocked["detection_status"] == "view_blocked" and "covered lens" in blocked["remark"]
    assert lit["image_url"] == f"/api/results/checks/{job.job_id}/1/image.jpg" and lit["job_id"] == job.job_id
    assert all(r["time_known"] and r["camera_flag"] is None and r["import_ready"] for r in rows)
    assert out["options"] == {"stores": ["TEST_STORE_01"], "cameras": ["TEST_CAM_01"],
                              "classes": ["food_wrapper"]}


def test_unknown_time_is_flagged_never_replaced_by_processing_time(tmp_path):
    job = Job(tmp_path, ["NO_LITTER"] * 4, filename="20260723094200.mp4")
    out = rv.view(job.query, settings(tmp_path, write_cfg(tmp_path)), rv.Filters())
    r = out["rows"][0]
    assert r["timestamp"] is None and not r["time_known"] and r["time_warning"].startswith("time unknown")
    assert "time unknown" in r["remark"] and not r["import_ready"] and "no recording time" in r["import_issue"]
    assert out["summary"]["time_unknown"] == 4 and out["summary"]["import_ready"] == 0
    # a date filter can never match a row whose recording date is unknown
    assert rv.view(job.query, settings(tmp_path, write_cfg(tmp_path)), rv.Filters(date_from="2000-01-01"))["total"] == 0


@pytest.mark.parametrize("camera", [None, "upload"])
def test_no_camera_assigned_is_flagged(tmp_path, camera):
    job = Job(tmp_path, ["NO_LITTER"] * 4, camera=camera)
    out = rv.view(job.query, settings(tmp_path, write_cfg(tmp_path)), rv.Filters())
    assert all(r["camera_flag"] == "no camera assigned" and r["camera_id"] == "" for r in out["rows"])
    assert all(not r["import_ready"] and r["cart_id"] == r["store_name"] == "" for r in out["rows"])
    assert out["summary"]["no_camera"] == 4


def test_filters_and_summary(tmp_path):
    cfg = write_cfg(tmp_path)
    job = Job(tmp_path, MIXED)
    q, s = job.query, settings(tmp_path, cfg)
    assert rv.view(q, s, rv.Filters(status="view_blocked"))["total"] == 1
    assert rv.view(q, s, rv.Filters(litter_class="food_wrapper"))["total"] == 2
    assert rv.view(q, s, rv.Filters(min_confidence=78.27))["total"] == 2
    assert rv.view(q, s, rv.Filters(min_confidence=79))["total"] == 0
    assert rv.view(q, s, rv.Filters(store="TEST_STORE_01", camera="TEST_CAM_01"))["total"] == 4
    assert rv.view(q, s, rv.Filters(camera="OTHER"))["total"] == 0
    assert rv.view(q, s, rv.Filters(date_from="2026-11-19", date_to="2026-11-19"))["total"] == 4
    assert rv.view(q, s, rv.Filters(date_from="2026-11-20"))["total"] == 0
    assert rv.view(q, s, rv.Filters(status="clean"))["summary"] == {
        "total": 1, "clean": 1, "litter_detected": 0, "view_blocked": 0, "time_unknown": 0, "no_camera": 0,
        "import_ready": 1}


def test_dashboard_csv_equals_the_bundle_and_the_exporter(tmp_path):
    cfg = write_cfg(tmp_path)
    job = Job(tmp_path, MIXED)
    s = settings(tmp_path, cfg)
    text, n, left_out = rv.importer_csv(job.query, s, rv.Filters(job_id=job.job_id), allowed_classes=CLASSES)
    assert (n, left_out) == (4, 0)
    bundle = job.build(cfg)                                                          # the result bundle
    (bundle_csv,) = Path(bundle["folder"]).glob("*.csv")
    assert text.encode("utf-8") == bundle_csv.read_bytes()
    flat = ge.export_source(job.query, kind="sampled", source_id=job.job_id, allowed_classes=CLASSES,  # flat exporter
                            settings=ge.Settings(cfg, tmp_path / "flat", tmp_path / "flat_ledger.db"))
    (f,) = flat["files"]
    assert text.encode("utf-8") == Path(f["path"]).read_bytes()
    assert ge.validate_text(text, allowed_classes=CLASSES, stores=ge.load_store_config(cfg),
                            store_key="TEST_STORE_01") == []


def test_dashboard_csv_leaves_out_rows_that_are_not_import_ready(tmp_path):
    cfg = write_cfg(tmp_path)
    a = Job(tmp_path, ["NO_LITTER"] * 4)
    Job(tmp_path, ["NO_LITTER"] * 4, job_id="sv_20261121_130000_bbbbbb", camera=None, db=a.db)
    text, n, left_out = rv.importer_csv(a.query, settings(tmp_path, cfg), rv.Filters())
    assert (n, left_out) == (4, 4)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 4 and {r["camera_id"] for r in rows} == {"TEST_CAM_01"}


# ------------------------------------------------------------------------------------------ API
@pytest.fixture
def app_client(cfg):
    with TestClient(create_app(cfg)) as c:
        yield c


def test_api_rows_csv_and_image(app_client, cfg, tmp_path):
    cfg.store_config_path = write_cfg(tmp_path)
    job = Job(Path(cfg.storage_dir), MIXED, db=Database(cfg.db_path))
    r = app_client.get("/api/results/checks", params={"status": "litter_detected"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 2 and body["columns"] == ge.COLUMNS and not any("_import" in x for x in body["rows"])
    csv_r = app_client.get("/api/results/checks.csv")
    assert csv_r.status_code == 200 and csv_r.headers["x-rows-written"] == "4"
    assert csv_r.content.startswith(b"timestamp,camera_id,image_name,") and not csv_r.content.startswith(b"\xef\xbb\xbf")
    img = app_client.get(body["rows"][0]["image_url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert app_client.get(f"/api/results/checks/{job.job_id}/9/image.jpg").status_code == 404
    assert app_client.get(f"/api/results/checks/{job.job_id}/3/image.jpg").status_code == 404   # blocked, no bundle
    assert app_client.get("/api/results/checks", params={"min_confidence": 101}).status_code == 422
