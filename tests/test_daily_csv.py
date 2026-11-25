"""Daily store CSV (backend/services/daily_csv.py): storage/daily_csv/YYYY-MM-DD.csv, one row per store per day."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

from backend.database.db import Database
from backend.services import daily_csv as dc
from backend.services import garbage_export as ge

CLASSES = {"food_wrapper", "paper", "cardboard"}
NVR = "20261119_20261119163013_20261119163458_163015.mp4"      # filename time: 2026-11-19 16:30:13 +05:30
TIMES = (28.596, 99.284, 184.3, 256.1)
LITTER = [{"class": "food_wrapper", "confidence": 0.7827, "bbox": [0, 0, 10, 10], "status": "confirmed"}]
HEADER = ("storeKey,cartId,storeName,date,entries,totalEntries,sourceFiles,parserVersion,lastSyncedAt,createdAt,"
          "updatedAt\r\n")


def write_cfg(tmp: Path) -> Path:
    p = tmp / "store_config.TEST.json"
    p.write_text(json.dumps({"stores": [
        {"storeKey": "TEST_STORE_01", "storeName": "TEST Store, Pune", "cart_id": 901, "camera_ids": ["CAM_A"]},
        {"storeKey": "TEST_STORE_02", "storeName": "TEST Store 2", "cart_id": 902, "camera_ids": ["CAM_B"]}]}),
        encoding="utf-8")
    return p


class Env:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.db = Database(tmp / "ldm.db")
        self.query = ge.sqlite_query(self.db.path)
        self.settings = ge.Settings(write_cfg(tmp), tmp / "exports", tmp / "ledger.db")
        self.storage = tmp / "storage"

    def job(self, job_id: str, *, filename=NVR, store="TEST_STORE_01", camera="CAM_A", recording_start=None,
            outcomes=("LITTER_DETECTED", "NO_LITTER", "UNREADABLE", "NO_LITTER"), video_bytes=b"video-1"):
        video = self.tmp / f"{job_id}.mp4"
        video.write_bytes(video_bytes)
        media_id = job_id[-6:] + "0123456789"
        with self.db.connect() as c:
            c.execute("INSERT INTO sampled_media(id,created_at,filename,path,sha256,size_bytes,duration_sec) "
                      "VALUES (?,?,?,?,?,?,?)", (media_id, "t", filename, str(video), "x", 1, 284.659))
            c.execute("INSERT INTO sampled_jobs(id,media_id,created_at,stage,state,store_id,camera_id,recording_start,"
                      "seed,sample_count,roi_mode,output_dir,model_version,counts_json) VALUES "
                      "(?,?,?,?,?,?,?,?,0,4,'none',?,?,?)",
                      (job_id, media_id, "t", "completed", "done", store, camera, recording_start,
                       str(self.tmp / job_id), "v3_store_cctv", "{}"))
            for i, outcome in enumerate(outcomes, start=1):
                c.execute("INSERT INTO sampled_frames(job_id,idx,frame_idx,video_time_sec,status,outcome,original_path,"
                          "annotated_path,result_json,error) VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (job_id, i, 100 * i, TIMES[i - 1], "saved", outcome, None, None,
                           json.dumps({"detections": LITTER if outcome == "LITTER_DETECTED" else []}),
                           "blank frame" if outcome == "UNREADABLE" else None))
        return job_id

    def append(self, job_id: str, now="2026-11-25T10:00:00+05:30"):
        return dc.append_job(self.query, self.settings, self.storage, job_id, CLASSES, now=now)

    def rows(self, day: str) -> list[dict]:
        return dc.read_rows(self.storage / "daily_csv" / f"{day}.csv")


def test_file_name_format_and_header_order(tmp_path):
    env = Env(tmp_path)
    res = env.append(env.job("sv_1"))
    p = tmp_path / "storage" / "daily_csv" / "2026-11-19.csv"
    assert res["files"] == [str(p)] and res["added"] == 4
    text = p.read_bytes().decode("utf-8")
    assert text.startswith(HEADER) and text.count("storeKey,cartId") == 1     # header once, no BOM
    assert dc.file_name("2026-11-10") == "2026-11-10.csv"
    raw = list(csv.DictReader(io.StringIO(text, newline="")))
    assert len(raw) == 1
    r = raw[0]
    assert (r["storeKey"], r["cartId"], r["storeName"]) == ("TEST_STORE_01", "901", "TEST Store, Pune")
    assert r["date"] == "2026-11-19T00:00:00+05:30" and r["totalEntries"] == "4" and r["parserVersion"] == "1"
    assert r["createdAt"] == r["updatedAt"] == r["lastSyncedAt"] == "2026-11-25T10:00:00+05:30"
    entries = json.loads(r["entries"])                                      # valid JSON inside the CSV cell
    assert [list(e) for e in entries] == [dc.ENTRY_FIELDS] * 4
    assert [e["imageName"] for e in entries] == [f"{NVR[:-4]}_s{i}.jpg" for i in range(1, 5)]
    assert [e["detectionStatus"] for e in entries] == ["litter_detected", "clean", "view_blocked", "clean"]
    assert entries[0]["capturedAt"] == "2026-11-19T16:30:41+05:30"
    assert (entries[0]["litterClass"], entries[0]["confidenceScore"]) == ("food_wrapper", "78.27")
    assert entries[2]["remark"].startswith("view blocked: blank frame")
    assert all(e["storeName"] == "TEST Store, Pune" and e["sourceFile"] == NVR and e["cameraId"] == "CAM_A"
               for e in entries)
    (src,) = json.loads(r["sourceFiles"])
    assert src["key"] == NVR and len(src["etag"]) == 32 and src["lastModified"].endswith("+05:30")


def test_second_video_same_store_and_day_updates_the_same_row(tmp_path):
    env = Env(tmp_path)
    env.append(env.job("sv_1"), now="2026-11-25T10:00:00+05:30")
    other = "20261119_20261119090000_20261119090445_090000.mp4"            # earlier the same day
    env.append(env.job("sv_2", filename=other, video_bytes=b"video-2"), now="2026-11-25T11:00:00+05:30")
    (row,) = env.rows("2026-11-19")
    assert len(row["entries"]) == 8 and [s["key"] for s in row["sourceFiles"]] == [NVR, other]
    caps = [e["capturedAt"] for e in row["entries"]]
    assert caps == sorted(caps) and row["entries"][0]["sourceFile"] == other        # sorted by capturedAt
    assert row["createdAt"] == "2026-11-25T10:00:00+05:30"                          # kept
    assert row["updatedAt"] == row["lastSyncedAt"] == "2026-11-25T11:00:00+05:30"
    text = (tmp_path / "storage" / "daily_csv" / "2026-11-19.csv").read_bytes().decode("utf-8")
    assert text.count("storeKey,cartId") == 1 and '"totalEntries"' not in text
    assert list(csv.DictReader(io.StringIO(text, newline="")))[0]["totalEntries"] == "8"


def test_a_different_store_on_the_same_day_gets_its_own_row(tmp_path):
    env = Env(tmp_path)
    env.append(env.job("sv_1"))
    env.append(env.job("sv_2", filename="cam_b_" + NVR, store="TEST_STORE_02", camera="CAM_B"))
    rows = env.rows("2026-11-19")
    assert [(r["storeKey"], r["cartId"], len(r["entries"])) for r in rows] == [("TEST_STORE_01", "901", 4),
                                                                              ("TEST_STORE_02", "902", 4)]


def test_rescan_of_the_same_video_is_not_duplicated(tmp_path):
    env = Env(tmp_path)
    env.append(env.job("sv_1"))
    res = env.append(env.job("sv_1_rescan"), now="2026-11-25T12:00:00+05:30")    # same video name, new job
    assert res["added"] == 0 and res["duplicates"] == 4
    (row,) = env.rows("2026-11-19")
    assert len(row["entries"]) == 4 and len(row["sourceFiles"]) == 1
    assert row["lastSyncedAt"] == "2026-11-25T12:00:00+05:30"


def test_duplicate_entries_are_skipped_by_source_file_and_image_name(tmp_path):
    rows: list[dict] = []
    e = {"capturedAt": "2026-11-10T10:00:00+05:30", "cameraId": "CAM_A", "imageName": "v_s1.jpg",
         "litterClass": "", "confidenceScore": "", "detectionStatus": "clean", "remark": "", "storeName": "S",
         "sourceFile": "v.mp4"}
    dc.merge(rows, store_key="S1", cart_id="1", store_name="S", day="2026-11-10", entries=[e],
             source={"key": "v.mp4", "etag": "a", "lastModified": ""}, now="2026-11-10T11:00:00+05:30")
    res = dc.merge(rows, store_key="S1", cart_id="1", store_name="S", day="2026-11-10",
                   entries=[e, {**e, "imageName": "v_s2.jpg"}],
                   source={"key": "v_copy.mp4", "etag": "b", "lastModified": ""}, now="2026-11-10T12:00:00+05:30")
    assert res == {"added": 1, "duplicates": 1, "skipped_source": 0}
    assert [x["imageName"] for x in rows[0]["entries"]] == ["v_s1.jpg", "v_s2.jpg"]


def test_recording_after_midnight_goes_to_the_next_days_file(tmp_path):
    env = Env(tmp_path)
    res = env.append(env.job("sv_1", filename="clip.mp4", recording_start="2026-11-10T23:58:00+05:30"))
    d = tmp_path / "storage" / "daily_csv"
    assert res["files"] == [str(d / "2026-11-10.csv"), str(d / "2026-11-11.csv")]
    (a,) = env.rows("2026-11-10")
    (b,) = env.rows("2026-11-11")
    assert [e["imageName"] for e in a["entries"]] == ["clip_s1.jpg", "clip_s2.jpg"]       # 23:58:28, 23:59:39
    assert [e["imageName"] for e in b["entries"]] == ["clip_s3.jpg", "clip_s4.jpg"]       # 00:01:04, 00:02:16
    assert b["date"] == "2026-11-11T00:00:00+05:30"
    assert b["entries"][0]["capturedAt"] == "2026-11-11T00:01:04+05:30"


def test_scan_without_a_store_goes_to_unassigned(tmp_path):
    env = Env(tmp_path)
    res = env.append(env.job("sv_1", store=None, camera="upload"))
    assert res["store_key"] == "UNASSIGNED" and res["unassigned"] == 4
    (row,) = env.rows("2026-11-19")
    assert (row["storeKey"], row["cartId"], row["storeName"]) == ("UNASSIGNED", "", "")


def test_a_csv_error_never_raises_from_the_runner_hook(tmp_path):
    state = SimpleNamespace(db=None, cfg=None, engine=None)               # broken on purpose
    res = dc.safe_append_job(state, "sv_x")
    assert res["ok"] is False and "error" in res


def test_a_file_with_a_foreign_header_is_never_overwritten(tmp_path):
    env = Env(tmp_path)
    p = tmp_path / "storage" / "daily_csv" / "2026-11-19.csv"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"something,else\r\n1,2\r\n")
    try:
        env.append(env.job("sv_1"))
    except dc.DailyCsvError:
        pass
    assert p.read_bytes().decode("utf-8") == "something,else\r\n1,2\r\n"
