"""Dashboard import rows of a four-screenshot Video Analysis (backend/services/dashboard_rows.py) and
GET /api/detect/video/{job_id}/report?format=table|csv|csv_full&per_object=."""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.core.settings import DetectionSettings, ViewQualitySettings
from backend.database.db import Database
from backend.main import create_app
from backend.services import dashboard_rows as dr
from backend.services import garbage_export as ge

NVR = "20261119_20261119163013_20261119163458_163015.mp4"     # span 285 s = duration -> filename time source
TIMES = (28.596, 99.284, 184.3, 256.1)
VQ = DetectionSettings().view_quality
HEADER = b"timestamp,camera_id,image_name,litter_class,confidence_score,detection_status,remark\r\n"
FULL_HEADER = HEADER[:-2] + b",store_name,source_file\r\n"
CONFIRMED = [{"class": "food_wrapper", "confidence": 0.7827, "bbox": [0, 0, 10, 10], "status": "confirmed"},
             {"class": "plastic_piece", "confidence": 0.6101, "bbox": [50, 50, 70, 70], "status": "confirmed"}]
POSSIBLE = [{"class": "food_wrapper", "confidence": 0.5478, "bbox": [90, 90, 99, 99], "status": "candidate"},
            {"class": "other_litter", "confidence": 0.4628, "bbox": [20, 90, 29, 99], "status": "candidate"}]
WEAK = [{"class": "paper", "confidence": 0.33, "bbox": [5, 60, 9, 66], "status": "weak"}]
REAL_JOB = "sv_20261121_182108_d04b08"          # same video/plan as the brief's sv_20261124_105006_0e6287


def scene(seed: int = 0) -> np.ndarray:
    """A sharp, normally lit picture (random texture) that passes every view-quality check."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(30, 220, (45, 80, 3), dtype=np.uint8)          # 8x8-pixel patches: edges + contrast
    return cv2.resize(blocks, (640, 360), interpolation=cv2.INTER_NEAREST)


class Job:
    """A finished four-screenshot job in a throw-away database, with real image files."""

    def __init__(self, tmp: Path, frames: list[dict], *, job_id="sv_20261124_105006_0e6287", filename=NVR,
                 duration=284.659, store=None, camera=None, recording_start=None, db=None,
                 uploaded="2026-11-24T05:20:06+00:00"):
        self.job_id = job_id
        self.db = db or Database(tmp / "ldm.db")
        out = tmp / "sampled" / job_id
        (out / "annotated").mkdir(parents=True)
        (out / "frames").mkdir(parents=True)
        media_id = job_id[-6:] + "0123456789"
        with self.db.connect() as c:
            c.execute("INSERT INTO sampled_media(id,created_at,filename,path,sha256,size_bytes,duration_sec) "
                      "VALUES (?,?,?,?,?,?,?)", (media_id, uploaded, filename, str(tmp / "missing.mp4"), "x", 1,
                                                 duration))
            c.execute("INSERT INTO sampled_jobs(id,media_id,created_at,stage,state,store_id,camera_id,recording_start,"
                      "seed,sample_count,roi_mode,output_dir,model_version,counts_json) VALUES "
                      "(?,?,?,?,?,?,?,?,0,4,'none',?,?,?)",
                      (job_id, media_id, uploaded, "completed", "done", store, camera, recording_start, str(out),
                       "v3_store_cctv", "{}"))
            for i, f in enumerate(frames, start=1):
                orig, ann = out / "frames" / f"clip_s{i}.jpg", out / "annotated" / f"clip_s{i}_annotated.jpg"
                outcome = f.get("outcome", "LITTER_DETECTED" if any(d["status"] == "confirmed" for d in f.get("dets", []))
                                else "NO_LITTER")
                if f.get("image", True) is not None:
                    cv2.imwrite(str(orig), scene(i) if f.get("image", True) is True else f["image"])
                    cv2.imwrite(str(ann), scene(i))
                c.execute("INSERT INTO sampled_frames(job_id,idx,frame_idx,video_time_sec,status,outcome,original_path,"
                          "annotated_path,result_json,error) VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (job_id, i, 100 * i, TIMES[i - 1], "saved", outcome, str(orig), str(ann),
                           json.dumps({"detections": f.get("dets", [])}), f.get("error")))
        self.query = ge.sqlite_query(self.db.path)

    def job(self) -> dict:
        return self.query("SELECT * FROM sampled_jobs WHERE id=?", (self.job_id,))[0]

    def rows(self, per_object=False, vq=VQ):
        return dr.build_dashboard_rows(self.job(), query=self.query, view_quality=vq, per_object=per_object)


ACCEPTANCE = [{"dets": [CONFIRMED[0], *POSSIBLE, *WEAK]},                     # litter + possible
              {"dets": [{**CONFIRMED[0], "confidence": 0.7599}, CONFIRMED[1]]},   # two confirmed
              {"dets": POSSIBLE},                                               # possible only -> clean
              {"dets": []}]                                                     # nothing -> clean


def check_row(r: dict) -> None:
    assert r["detection_status"] in dr.STATUSES
    for k in dr.FULL_HEADER:
        assert isinstance(r[k], str) and r[k] not in ("—", "-", "null", "None", "nan")
    if r["confidence_score"]:
        assert re.fullmatch(r"\d{1,3}\.\d", r["confidence_score"]) and 0 <= float(r["confidence_score"]) <= 100
    if r["detection_status"] == "litter_detected":
        assert r["litter_class"] and r["confidence_score"]
    else:
        assert r["litter_class"] == r["confidence_score"] == ""
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", r["timestamp"])
    assert "\n" not in r["remark"] and len(r["remark"]) <= dr.REMARK_MAX + 40


# ------------------------------------------------------------------------------------------ rows
def test_one_row_per_screenshot_with_primary_detection_and_remark(tmp_path):
    rows = Job(tmp_path, ACCEPTANCE).rows()
    assert len(rows) == 4 and [r["screenshot"] for r in rows] == [1, 2, 3, 4]
    for r in rows:
        check_row(r)
    s1, s2, s3, s4 = rows
    assert (s1["detection_status"], s1["litter_class"], s1["confidence_score"]) == ("litter_detected", "food_wrapper", "78.3")
    assert (s2["detection_status"], s2["litter_class"], s2["confidence_score"]) == ("litter_detected", "food_wrapper", "76.0")
    # confirmed litter lists its possible detections; weak detections are not counted
    assert s1["remark"] == "2 possible, review required: food_wrapper 54.8%, other_litter 46.3%; t=28.6s"
    assert s2["remark"] == "also plastic_piece 61.0%; t=99.3s"
    assert (s3["detection_status"], s3["remark"]) == ("clean", "2 possible, review required; t=184.3s")
    assert (s4["detection_status"], s4["remark"]) == ("clean", "t=256.1s")
    assert [r["timestamp"] for r in rows] == ["2026-11-19 16:30:41", "2026-11-19 16:31:52", "2026-11-19 16:33:17",
                                              "2026-11-19 16:34:29"]
    assert s1["image_name"] == "sv_20261124_105006_0e6287_s1_annotated.jpg" == ge.bundle_image_name(s1["job_id"], 1)
    assert s1["camera_id"] == "VIDEO-UPLOAD" and s1["store_name"] == "" and s1["source_file"] == NVR


def test_per_object_mode_lists_every_confirmed_and_possible_object(tmp_path):
    rows = Job(tmp_path, ACCEPTANCE).rows(per_object=True)
    for r in rows:
        check_row(r)
    # s1: 1 confirmed + 2 possible, s2: 2 confirmed, s3: 2 possible, s4: no object -> its screenshot row
    assert [(r["screenshot"], r["row_kind"]) for r in rows] == [
        (1, "object"), (1, "object"), (1, "object"), (2, "object"), (2, "object"), (3, "object"), (3, "object"),
        (4, "screenshot")]
    assert [r["detection_status"] for r in rows] == ["litter_detected", "clean", "clean", "litter_detected",
                                                     "litter_detected", "clean", "clean", "clean"]
    assert rows[4]["litter_class"] == "plastic_piece" and rows[4]["confidence_score"] == "61.0"
    assert rows[0]["remark"] == "object 1/3; t=28.6s"
    assert rows[1]["remark"] == "possible: food_wrapper 54.8%, review required; object 2/3; t=28.6s"
    assert all(r["litter_class"] == r["confidence_score"] == "" for r in rows if r["detection_status"] == "clean")


def test_possible_object_rows_are_always_clean_whatever_the_confidence(tmp_path):
    sure = [{**POSSIBLE[0], "confidence": 0.99}]
    rows = Job(tmp_path, [{"dets": [CONFIRMED[0], *sure]}, {"dets": sure}, {}, {}]).rows(per_object=True)
    possible = [r for r in rows if "possible:" in r["remark"]]
    assert len(possible) == 2 and all(r["detection_status"] == "clean" for r in possible)
    assert all(r["litter_class"] == r["confidence_score"] == "" for r in possible)
    assert possible[0]["remark"].startswith("possible: food_wrapper 99.0%, review required")
    assert "view_blocked" not in {r["detection_status"] for r in rows}


def test_possible_litter_alone_never_makes_litter_detected(tmp_path):
    rows = Job(tmp_path, [{"dets": POSSIBLE * 3}] * 4).rows()
    assert {r["detection_status"] for r in rows} == {"clean"}
    assert all(r["remark"].startswith("6 possible, review required") for r in rows)


@pytest.mark.parametrize("image,reason", [
    (np.full((360, 640, 3), 8, np.uint8), "blank frame"),                                    # existing check
    (np.clip(np.random.default_rng(1).normal(12, 8, (360, 640, 3)), 0, 255).astype(np.uint8), "too dark"),
    (np.clip(np.random.default_rng(2).normal(245, 8, (360, 640, 3)), 0, 255).astype(np.uint8), "too bright"),
    (cv2.GaussianBlur(scene(3), (41, 41), 0), "blurred"),
])
def test_view_blocked_rules_and_reason_in_remark(tmp_path, image, reason):
    rows = Job(tmp_path, [{"dets": POSSIBLE, "image": image}, {}, {}, {}]).rows()
    r = rows[0]
    check_row(r)
    assert r["detection_status"] == "view_blocked" and f"view blocked: {reason}" in r["remark"]
    assert [x["detection_status"] for x in rows[1:]] == ["clean"] * 3


def test_mostly_uniform_view_is_blocked(tmp_path):
    img = np.full((360, 640, 3), (40, 150, 70), np.uint8)       # flat green, not decoder grey
    img[:, :40] = np.random.default_rng(4).integers(0, 255, (360, 40, 3), dtype=np.uint8)   # not "blank" (98 %)
    vq = ViewQualitySettings(uniform_max_std=60)
    r = Job(tmp_path, [{"image": img}, {}, {}, {}]).rows(vq=vq)[0]
    assert r["detection_status"] == "view_blocked" and "mostly uniform" in r["remark"]


def test_unreadable_and_failed_screenshots_are_view_blocked(tmp_path):
    rows = Job(tmp_path, [{"outcome": "UNREADABLE", "error": "corrupted frame\n(grey fill)", "image": None},
                          {"outcome": "FAILED", "error": "inference error", "image": None}, {}, {}]).rows()
    assert [r["detection_status"] for r in rows[:2]] == ["view_blocked", "view_blocked"]
    assert "unreadable: corrupted frame (grey fill)" in rows[0]["remark"]
    assert "not analysed (failed: inference error)" in rows[1]["remark"]
    for r in rows:
        check_row(r)


def test_thresholds_come_from_settings(tmp_path):
    job = Job(tmp_path, [{"dets": [CONFIRMED[0]]}, {}, {}, {}])
    assert job.rows()[1]["detection_status"] == "clean"
    strict = ViewQualitySettings(blur_min_laplacian_var=1e9)
    rows = job.rows(vq=strict)
    assert rows[1]["detection_status"] == "view_blocked" and "blurred" in rows[1]["remark"]
    # confirmed litter wins; the quality problem is still reported
    assert rows[0]["detection_status"] == "litter_detected" and "view quality: blurred" in rows[0]["remark"]


def test_operator_rejected_detection_does_not_count(tmp_path, monkeypatch):
    job = Job(tmp_path, [{"dets": [CONFIRMED[0]]}, {}, {}, {}])
    with job.db.connect() as c:
        c.execute("UPDATE sampled_jobs SET counts_json=?", (json.dumps({"standard_outputs": {"analysis_id": "a1"}}),))
    seen = []
    monkeypatch.setattr(ge, "_rejected_boxes",
                        lambda q, aid: seen.append(aid) or [("food_wrapper", [0, 0, 10, 10])])
    r = job.rows()[0]
    assert r["detection_status"] == "clean" and r["remark"].startswith("1 rejected by operator") and seen == ["a1"]


def test_long_remark_is_shortened(tmp_path):
    many = [{**CONFIRMED[0], "confidence": 0.99 - i / 100, "bbox": [i, i, i + 5, i + 5]} for i in range(30)]
    r = Job(tmp_path, [{"dets": many + POSSIBLE}, {}, {}, {}]).rows()[0]
    assert len(r["remark"]) <= dr.REMARK_MAX and "more" in r["remark"] and r["remark"].endswith("t=28.6s")


# ------------------------------------------------------------------------------------------ fields
def test_camera_store_and_source_file(tmp_path):
    job = Job(tmp_path, ACCEPTANCE, camera="CAM_A", store="STORE_X")
    with job.db.connect() as c:
        c.execute("INSERT INTO stores(id,name,created_at,updated_at) VALUES ('STORE_1','Store One','t','t')")
        c.execute("INSERT INTO stores(id,name,created_at,updated_at) VALUES ('STORE_X','Store X','t','t')")
        c.execute("INSERT INTO cameras(id,name,source,created_at,updated_at,store_id) "
                  "VALUES ('CAM_A','A','file','t','t','STORE_1')")
    r = job.rows()[0]
    assert (r["camera_id"], r["store_name"], r["source_file"]) == ("CAM_A", "Store One", NVR)   # camera's store wins
    other = Job(tmp_path / "b", ACCEPTANCE, camera=None, store="STORE_X")
    with other.db.connect() as c:
        c.execute("INSERT INTO stores(id,name,created_at,updated_at) VALUES ('STORE_X','Store X','t','t')")
    assert (other.rows()[0]["camera_id"], other.rows()[0]["store_name"]) == ("VIDEO-UPLOAD", "Store X")


def test_recording_start_order(tmp_path):
    # (a) the form input wins over the file name
    a = Job(tmp_path / "a", ACCEPTANCE, recording_start="2026-11-20T09:00:00")
    assert a.rows()[0]["timestamp"] == "2026-11-20 09:00:28" and "estimated" not in a.rows()[0]["remark"]
    # (c) compact date/time in a non-NVR file name
    c = Job(tmp_path / "c", ACCEPTANCE, filename="20260723094200.mp4", duration=900)
    assert c.rows()[0]["timestamp"] == "2026-07-23 09:42:28"
    # (d) nothing -> upload time (UTC 05:20:06 = 10:50:06 IST) + offset, flagged as estimated
    d = Job(tmp_path / "d", ACCEPTANCE, filename="clip.mp4")
    r = d.rows()[0]
    assert r["timestamp"] == "2026-11-24 10:50:34" and r["remark"].endswith(dr.ESTIMATED)


def test_container_creation_time_is_used_before_the_file_name(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "ffprobe_creation_time",
                        lambda p: dt.datetime(2026, 11, 19, 12, 0, 0, tzinfo=dt.timezone.utc))
    assert Job(tmp_path, ACCEPTANCE).rows()[0]["timestamp"] == "2026-11-19 17:30:28"


# ------------------------------------------------------------------------------------------ CSV
def test_exact_header_utf8_without_bom_and_round_trip(tmp_path):
    rows = Job(tmp_path, ACCEPTANCE).rows(per_object=True)
    for full, header in ((False, HEADER), (True, FULL_HEADER)):
        data = dr.to_csv(rows, full=full).encode("utf-8")
        assert data.startswith(header) and not data.startswith(b"\xef\xbb\xbf")
        back = list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))
        cols = dr.FULL_HEADER if full else dr.HEADER
        assert [list(b) for b in back] == [cols] * len(rows)
        assert back == [{k: r[k] for k in cols} for r in rows]
    assert "store_name" not in dr.to_csv(rows).splitlines()[0]


def test_formula_text_cannot_reach_a_spreadsheet(tmp_path):
    job = Job(tmp_path, ACCEPTANCE, filename="=HYPERLINK(evil).mp4")
    row = list(csv.DictReader(io.StringIO(dr.to_csv(job.rows(), full=True))))[0]
    assert row["source_file"].startswith("'=")


# ------------------------------------------------------------------------------------------ API
@pytest.fixture
def app_client(cfg):
    with TestClient(create_app(cfg)) as c:
        yield c


def test_api_table_and_both_csv_formats_use_the_same_rows(app_client, cfg):
    job = Job(Path(cfg.storage_dir), ACCEPTANCE, db=Database(cfg.db_path))
    base = f"/api/detect/video/{job.job_id}/report"
    t = app_client.get(base)
    assert t.status_code == 200, t.text
    body = t.json()
    assert body["total"] == 4 and body["columns"] == dr.FULL_HEADER
    assert body["summary"] == {"clean": 2, "litter_detected": 2, "view_blocked": 0}
    assert app_client.get(base, params={"per_object": "true"}).json()["total"] == 8
    c1 = app_client.get(base, params={"format": "csv"})
    assert c1.status_code == 200 and c1.headers["content-type"].startswith("text/csv")
    assert c1.content.startswith(HEADER) and len(c1.content.decode().splitlines()) == 5
    back = list(csv.DictReader(io.StringIO(c1.content.decode("utf-8"), newline="")))
    assert back == [{k: r[k] for k in dr.HEADER} for r in body["rows"]]
    c2 = app_client.get(base, params={"format": "csv_full", "per_object": "true"})
    assert c2.content.startswith(FULL_HEADER) and len(c2.content.decode().splitlines()) == 9
    assert app_client.get("/api/detect/video/sv_nope/report").status_code == 404


def test_image_name_is_the_name_of_the_downloaded_annotated_jpg(app_client, cfg):
    job = Job(Path(cfg.storage_dir), ACCEPTANCE, db=Database(cfg.db_path))
    for r in app_client.get(f"/api/detect/video/{job.job_id}/report").json()["rows"]:
        img = app_client.get(r["image_url"])
        assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg" and len(img.content) > 100
        assert f'filename="{r["image_name"]}"' in img.headers["content-disposition"]


# ------------------------------------------------------------------------------------------ real job (acceptance)
REAL_DB = Path(__file__).resolve().parents[1] / "storage" / "litter_monitor.db"


@pytest.mark.skipif(not REAL_DB.is_file(), reason="no local application database")
def test_acceptance_on_the_real_job_when_present():
    q = ge.sqlite_query(REAL_DB)
    jobs = q("SELECT * FROM sampled_jobs WHERE id IN (?, ?) AND state='done' ORDER BY id DESC",
             ("sv_20261124_105006_0e6287", REAL_JOB))
    if not jobs:
        pytest.skip("acceptance job not in the local database")
    job = jobs[0]
    rows = dr.build_dashboard_rows(job, query=q, view_quality=VQ)
    objs = dr.build_dashboard_rows(job, query=q, view_quality=VQ, per_object=True)
    detailed = Path(job["output_dir"]) / "reports" / "detection_report.csv"
    old = list(csv.DictReader(io.StringIO(detailed.read_text(encoding="utf-8"), newline="")))
    assert len(old) == 26 and len(old[0]) == 45
    assert len(rows) == 4 and len(objs) == sum(1 for r in old if r["record_type"] == "DETECTION") == 22
    assert (rows[0]["detection_status"], rows[0]["litter_class"], rows[0]["confidence_score"]) == (
        "litter_detected", "food_wrapper", "78.3")
    assert (rows[1]["detection_status"], rows[1]["litter_class"], rows[1]["confidence_score"]) == (
        "litter_detected", "food_wrapper", "76.0")
    assert "t=28.6s" in rows[0]["remark"]
    for r in rows + objs:
        check_row(r)
    frames = {f["idx"]: f for f in q("SELECT idx, annotated_path FROM sampled_frames WHERE job_id=?", (job["id"],))}
    for r in rows:                         # the image behind every image_name exists (served under that name)
        assert Path(frames[r["screenshot"]]["annotated_path"]).is_file()
