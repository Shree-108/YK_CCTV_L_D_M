"""Four-screenshot result bundle (backend/services/result_bundle.py + garbage_export.validate_bundle)."""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.database.db import Database
from backend.services import export_guard
from backend.services import garbage_export as ge
from backend.services import result_bundle as rb

CLASSES = {"food_wrapper", "plastic_piece", "paper", "cardboard"}
NVR = "20261119_20261119163013_20261119163458_163015.mp4"      # span 285 s = duration -> filename time source
HEADER = b"timestamp,camera_id,image_name,litter_class,confidence_score,detection_status,remark,cart_id,store_name\r\n"
TIMES = (28.596, 99.284, 184.3, 256.1)
LITTER = [{"class": "food_wrapper", "confidence": 0.7827, "bbox": [0, 0, 10, 10], "status": "confirmed"},
          {"class": "paper", "confidence": 0.61, "bbox": [50, 50, 70, 70], "status": "confirmed"},
          {"class": "cardboard", "confidence": 0.40, "bbox": [90, 90, 99, 99], "status": "candidate"}]


def write_cfg(tmp: Path, **over) -> Path:
    entry = {"storeKey": "TEST_STORE_01", "storeName": "TEST Store (not a real store)", "cart_id": 901,
             "camera_ids": ["TEST_CAM_01"], **over}
    p = tmp / "store_config.TEST.json"
    p.write_text(json.dumps({"stores": [entry]}), encoding="utf-8")
    return p


class Job:
    """A finished four-screenshot job in a throw-away database, with real annotated image files."""

    def __init__(self, tmp: Path, outcomes, *, job_id="sv_20261121_120000_abcdef", filename=NVR, duration=284.659,
                 store="TEST_STORE_01", camera="TEST_CAM_01", recording_start=None, errors=None, db=None):
        self.tmp, self.job_id = tmp, job_id
        self.db = db or Database(tmp / "ldm.db")
        out = tmp / "sampled" / job_id
        (out / "annotated").mkdir(parents=True)
        media_id = job_id[-6:] + "0123456789"
        with self.db.connect() as c:
            c.execute("INSERT INTO sampled_media(id,created_at,filename,path,sha256,size_bytes,duration_sec) "
                      "VALUES (?,?,?,?,?,?,?)", (media_id, "t", filename, str(tmp / "missing.mp4"), "x", 1, duration))
            c.execute("INSERT INTO sampled_jobs(id,media_id,created_at,stage,state,store_id,camera_id,recording_start,"
                      "seed,sample_count,roi_mode,output_dir,model_version,counts_json) VALUES "
                      "(?,?,?,?,?,?,?,?,0,4,'none',?,?,?)",
                      (job_id, media_id, "t", "completed", "done", store, camera, recording_start, str(out),
                       "v3_store_cctv", "{}"))
            for i, outcome in enumerate(outcomes, start=1):
                orig = out / "frames" / f"clip_s{i}.jpg"
                ann = out / "annotated" / f"clip_s{i}_annotated.jpg"
                dets = LITTER if outcome == "LITTER_DETECTED" else []
                if outcome in rb.ANALYSED:
                    cv2.imwrite(str(ann), np.full((120, 160, 3), 30 * i, np.uint8))
                c.execute("INSERT INTO sampled_frames(job_id,idx,frame_idx,video_time_sec,status,outcome,original_path,"
                          "annotated_path,result_json,error) VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (job_id, i, 100 * i, TIMES[i - 1], "saved" if outcome in rb.ANALYSED else "unreadable",
                           outcome, str(orig), str(ann) if outcome in rb.ANALYSED else None,
                           json.dumps({"detections": dets}), (errors or {}).get(i)))
        self.query = ge.sqlite_query(self.db.path)

    def build(self, cfg: Path, frames=None, **kw):
        return rb.build_bundle(self.query, job_id=self.job_id,
                               settings=ge.Settings(cfg, self.tmp / "exports", self.tmp / "ledger.db"),
                               allowed_classes=CLASSES, frame_reader=lambda p, idx: frames or {}, **kw)


def rows_of(folder: Path) -> list[dict[str, str]]:
    (f,) = folder.glob("*.csv")
    return list(csv.DictReader(io.StringIO(f.read_text(encoding="utf-8"), newline="")))


def test_mixed_bundle_has_four_images_one_csv_and_a_manifest(tmp_path):
    job = Job(tmp_path, ["LITTER_DETECTED", "NO_LITTER", "UNREADABLE", "LITTER_DETECTED"],
              errors={3: "blank frame: the picture is one flat colour\n(covered lens)"})
    m = job.build(write_cfg(tmp_path), frames={300: np.full((120, 160, 3), 5, np.uint8)})
    folder = Path(m["folder"])
    assert folder == tmp_path / "exports" / "TEST_STORE_01" / "2026-11-19" / job.job_id
    assert m["import_ready"] and m["validator"] == {"ok": True, "issues": []}
    images = sorted(p.name for p in folder.glob("*.jpg"))
    assert images == [rb.image_name(job.job_id, i) for i in range(1, 5)]
    assert (tmp_path / "sampled" / job.job_id / "annotated" / "clip_s1_annotated.jpg").is_file()   # copied
    (csv_file,) = folder.glob("*.csv")
    assert csv_file.name == f"TEST_STORE_01_2026-11-19_TEST_CAM_01_{job.job_id}.csv"
    assert csv_file.read_bytes().startswith(HEADER)                       # exact header, UTF-8, no BOM
    rows = rows_of(folder)
    assert [r["image_name"] for r in rows] == images
    assert [r["detection_status"] for r in rows] == ["litter_detected", "clean", "view_blocked", "litter_detected"]
    assert [r["timestamp"] for r in rows] == ["2026-11-19T16:30:41+05:30", "2026-11-19T16:31:52+05:30",
                                              "2026-11-19T16:33:17+05:30", "2026-11-19T16:34:29+05:30"]
    lit, clean, blocked = rows[0], rows[1], rows[2]
    assert (lit["litter_class"], lit["confidence_score"]) == ("food_wrapper", "78.27")
    assert lit["remark"].startswith("2 litter objects: food_wrapper 78.27, paper 61.00; 1 possible object under "
                                    "review, not counted: cardboard 40.00; time source: filename")
    assert clean["litter_class"] == clean["confidence_score"] == "" and "no litter detected" in clean["remark"]
    assert blocked["litter_class"] == blocked["confidence_score"] == ""
    assert "\n" not in blocked["remark"] and "flat colour (covered lens)" in blocked["remark"]
    assert all((r["cart_id"], r["store_name"]) == ("901", "TEST Store (not a real store)") for r in rows)
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert man["job_id"] == job.job_id and man["store"]["storeKey"] == "TEST_STORE_01"
    assert man["camera_id"] == "TEST_CAM_01" and man["time"]["source"] == "filename"
    assert man["model_version"] == "v3_store_cctv"
    assert man["counts"] == {"screenshots": 4, "images": 4, "rows": 4, "clean": 1, "litter_detected": 2,
                             "view_blocked": 1, "litter_objects": 4, "under_review_not_counted": 2,
                             "weak_not_counted": 0}
    assert ge.validate_bundle(folder, allowed_classes=CLASSES, stores=ge.load_store_config(write_cfg(tmp_path))) == []
    assert not list(folder.parent.glob(".*"))                             # temporary folder gone


def test_all_clean_bundle(tmp_path):
    m = Job(tmp_path, ["NO_LITTER"] * 4).build(write_cfg(tmp_path))
    assert m["import_ready"]
    assert {r["detection_status"] for r in rows_of(Path(m["folder"]))} == {"clean"}


def test_all_blocked_bundle_uses_a_card_when_the_frame_cannot_be_decoded(tmp_path):
    m = Job(tmp_path, ["UNREADABLE"] * 4, errors={i: "corrupted frame" for i in range(1, 5)}).build(write_cfg(tmp_path))
    assert m["import_ready"] and m["counts"]["view_blocked"] == 4
    assert all(s["source"] == "frame not decodable: placeholder card" for s in m["screenshots"])
    assert len(list(Path(m["folder"]).glob("*.jpg"))) == 4


def test_missing_recording_time_gives_images_and_manifest_but_no_csv(tmp_path):
    job = Job(tmp_path, ["LITTER_DETECTED", "NO_LITTER", "NO_LITTER", "NO_LITTER"], filename="20260723094200.mp4")
    m = job.build(write_cfg(tmp_path))
    folder = Path(m["folder"])
    assert folder.parent.name == rb.TIME_UNKNOWN and not m["import_ready"]
    assert not list(folder.glob("*.csv")) and len(list(folder.glob("*.jpg"))) == 4
    assert m["time"]["source"] is None and "no recording start time" in m["time"]["reason"]
    assert any("processing time is never used" in r for r in m["csv"]["reasons"])
    assert all(s["timestamp"] is None for s in m["screenshots"])
    assert any("expected exactly 1" in i for i in m["validator"]["issues"])
    # the operator-given start (the last source in the order) makes the same job exportable
    m2 = job.build(write_cfg(tmp_path), recording_start="2026-07-23T09:42:00")
    assert m2["import_ready"] and m2["time"]["source"] == "recording-start argument"
    assert rows_of(Path(m2["folder"]))[0]["timestamp"] == "2026-07-23T09:42:28+05:30"


@pytest.mark.parametrize("over,msg", [({"cart_id": None}, "cart_id"),
                                      ({"storeName": "REPLACE_WITH_STORE_NAME"}, "placeholder"),
                                      ({"camera_ids": ["OTHER_CAM"]}, "not listed for store")])
def test_missing_or_bad_store_config_gives_no_csv(tmp_path, over, msg):
    m = Job(tmp_path, ["NO_LITTER"] * 4).build(write_cfg(tmp_path, **over))
    folder = Path(m["folder"])
    assert folder.parent.parent.name == rb.UNASSIGNED and not m["import_ready"]
    assert not list(folder.glob("*.csv")) and len(list(folder.glob("*.jpg"))) == 4
    assert any(msg in r for r in m["csv"]["reasons"])


def test_missing_store_config_file_and_no_camera(tmp_path):
    m = Job(tmp_path, ["NO_LITTER"] * 4).build(tmp_path / "nope.json")
    assert any("store config not found" in r for r in m["csv"]["reasons"]) and not m["import_ready"]
    m = Job(tmp_path / "b", ["NO_LITTER"] * 4, camera=None).build(write_cfg(tmp_path))
    assert m["camera_id"] is None and m["camera_note"] == "no camera assigned"
    assert any("no camera assigned" in r for r in m["csv"]["reasons"])


def test_missing_annotated_image_gives_no_csv(tmp_path):
    job = Job(tmp_path, ["LITTER_DETECTED", "NO_LITTER", "NO_LITTER", "NO_LITTER"])
    (tmp_path / "sampled" / job.job_id / "annotated" / "clip_s2_annotated.jpg").unlink()
    m = job.build(write_cfg(tmp_path))
    assert not m["import_ready"] and len(list(Path(m["folder"]).glob("*.jpg"))) == 3
    assert any("screenshot 2: no annotated image" in r for r in m["csv"]["reasons"])
    assert any("3 image(s)" in i for i in m["validator"]["issues"])


def test_failed_screenshot_is_not_a_check_so_no_csv(tmp_path):
    m = Job(tmp_path, ["NO_LITTER", "NO_LITTER", "FAILED", "NO_LITTER"]).build(write_cfg(tmp_path))
    assert not m["import_ready"]
    assert "screenshot 3: processing failed (not a camera check)" in m["csv"]["reasons"]


def test_a_check_is_never_exported_twice_and_a_bundle_never_overwritten(tmp_path):
    cfg = write_cfg(tmp_path)
    a = Job(tmp_path, ["NO_LITTER"] * 4)
    assert a.build(cfg)["import_ready"]
    with pytest.raises(ge.ExportError, match="already exists"):
        a.build(cfg)
    b = Job(tmp_path, ["NO_LITTER"] * 4, job_id="sv_20261121_130000_bbbbbb", db=a.db)   # same video analysed again
    m = b.build(cfg)
    assert not m["import_ready"] and any("already exported in" in r for r in m["csv"]["reasons"])


def test_validate_bundle_catches_tampering(tmp_path):
    cfg = write_cfg(tmp_path)
    m = Job(tmp_path, ["LITTER_DETECTED", "NO_LITTER", "NO_LITTER", "NO_LITTER"]).build(cfg)
    folder = Path(m["folder"])
    (csv_file,) = folder.glob("*.csv")
    good = csv_file.read_bytes()

    def issues_after(data: bytes) -> str:
        csv_file.write_bytes(data)
        return " | ".join(map(str, ge.validate_bundle(folder, allowed_classes=CLASSES,
                                                      stores=ge.load_store_config(cfg))))
    assert "BOM" in issues_after(b"\xef\xbb\xbf" + good)
    assert "header must be exactly" in issues_after(good.replace(b"remark,", b"remarks,", 1))
    lines = good.split(b"\r\n")
    assert "expected one per screenshot" in issues_after(b"\r\n".join(lines[:-2] + [b""]))
    assert "not a file in the bundle folder" in issues_after(good.replace(b"_s2_annotated.jpg", b"_s9_annotated.jpg"))
    assert "duplicate check" in issues_after(b"\r\n".join(lines[:2] + [lines[1]] + lines[3:]))
    bad_status = lines[2].replace(b",clean,", b",dirty,")
    assert "not one of" in issues_after(b"\r\n".join(lines[:2] + [bad_status] + lines[3:]))
    filled = lines[2].replace(b"_annotated.jpg,,,clean,", b"_annotated.jpg,paper,50.00,clean,")
    assert filled != lines[2]
    assert "must be empty for clean" in issues_after(b"\r\n".join(lines[:2] + [filled] + lines[3:]))
    csv_file.write_bytes(good)
    (folder / rb.image_name(m["job_id"], 4)).unlink()
    assert "image(s) in the folder" in " ".join(map(str, ge.validate_bundle(folder)))


def test_guard_never_raises_and_records_the_outcome(tmp_path):
    st = SimpleNamespace(cfg=SimpleNamespace(store_config_path=tmp_path / "x.json", garbage_export_dir=tmp_path / "e",
                                             garbage_export_ledger=tmp_path / "l.db", trust_container_time=False),
                         db=SimpleNamespace(_query=lambda *a: []), engine=None)
    res = export_guard.safe_result_bundle(st, job_id="sv_missing", out_dir=tmp_path / "reports")
    assert res["ok"] is False and "not found" in res["error"]
    assert json.loads((tmp_path / "reports" / "result_bundle.json").read_text())["ok"] is False


# ------------------------------------------------------------------------------------------ the real runner
from tests.test_auto_analysis import _drop, _wait, client, video  # noqa: E402,F401 - reuse the app fixtures


def test_a_finished_job_produces_its_bundle_through_the_runner(client, tmp_path, cfg):
    """Upload -> four screenshots -> detection -> bundle, in the app. Store config = a labelled TEST file."""
    cfg.store_config_path = write_cfg(tmp_path)
    v = video(tmp_path / "20261121_20261121100000_20261121100008_100000.mp4", 80)       # 8 s, NVR name span 8 s
    r = _drop(client, v, store_id="TEST_STORE_01", camera_id="TEST_CAM_01")
    j, _ = _wait(client, r.json()["job_id"])
    assert j["lifecycle"] == "COMPLETED", j["error"]
    status_file = Path(cfg.sampled_dir) / j["id"] / "reports" / "result_bundle.json"
    end = time.time() + 30                   # the bundle is made right after the job is marked done
    while not status_file.is_file() and time.time() < end:
        time.sleep(0.1)
    status = json.loads(status_file.read_text())
    assert status["ok"] and status["import_ready"], status
    folder = Path(status["folder"])
    assert folder == Path(cfg.garbage_export_dir) / "TEST_STORE_01" / "2026-11-21" / j["id"]
    rows = rows_of(folder)
    start = ge.parse_manual_start("2026-11-21T10:00:00")                 # from the NVR file name
    assert len(rows) == 4 and [r["timestamp"] for r in rows] == [
        ge.format_ts(start + dt.timedelta(seconds=f["video_time_sec"])) for f in j["frames"]]
    assert [r["image_name"] for r in rows] == [rb.image_name(j["id"], i) for i in range(1, 5)]
    assert {r["detection_status"] for r in rows} == {"clean"}
    assert ge.validate_bundle(folder, stores=ge.load_store_config(cfg.store_config_path)) == []
