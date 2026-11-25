"""S3 layout + safe writes + job upload, against an in-memory fake S3 (no network, no credentials)."""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
from pathlib import Path

import pytest

from backend.core import s3 as s3l
from backend.services import daily_csv as dc
from backend.services import s3_sync

B = "bucket"
DAY = dt.date(2026, 11, 25)


class ClientError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    def __init__(self):
        self.objs: dict[str, bytes] = {}
        self.etags: dict[str, str] = {}
        self.puts = 0

    def put_object(self, *, Bucket, Key, Body, ContentType, ChecksumSHA256, IfNoneMatch=None, IfMatch=None):
        data = Body if isinstance(Body, bytes) else Body.read()
        if base64.b64encode(hashlib.sha256(data).digest()).decode() != ChecksumSHA256:
            raise ClientError("BadDigest")
        if IfNoneMatch == "*" and Key in self.objs:
            raise ClientError("PreconditionFailed")
        if IfMatch is not None and self.etags.get(Key) != IfMatch:
            raise ClientError("PreconditionFailed")
        self.puts += 1
        self.objs[Key] = data
        self.etags[Key] = f'"{hashlib.md5(data).hexdigest()}"'
        return {"ETag": self.etags[Key]}

    def head_object(self, *, Bucket, Key, ChecksumMode=None):
        if Key not in self.objs:
            raise ClientError("404")
        d = self.objs[Key]
        return {"ContentLength": len(d), "ChecksumSHA256": base64.b64encode(hashlib.sha256(d).digest()).decode()}

    def get_object(self, *, Bucket, Key):
        if Key not in self.objs:
            raise ClientError("NoSuchKey")
        return {"Body": io.BytesIO(self.objs[Key]), "ETag": self.etags[Key]}

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        return {"Contents": [{"Key": k} for k in sorted(self.objs) if k.startswith(Prefix)], "IsTruncated": False}


# ============================================================================================ layout
def test_layout_uses_cart_everywhere():
    eid = s3l.entity_id("50")
    assert eid == "cart_50" and s3l.entity_id("cart_50") == "cart_50"
    assert s3l.input_key(DAY, eid, "NVR 1/ch2 19:21.mp4") == "11-2026/25-11-2026/cart_50/input_data/ch2_19_21.mp4"
    assert s3l.daily_csv_key(DAY, eid) == "11-2026/25-11-2026/cart_50/output_data/cart50_2026_11_25.csv"
    assert s3l.image_key(DAY, eid, 3) == "11-2026/25-11-2026/cart_50/output_data/cart50_2026_11_25_pic3.jpg"
    with pytest.raises(s3l.S3Error):
        s3l.entity_id("../x")
    assert s3l.daily_csv_key(DAY, "cart_unassigned") == \
        "11-2026/25-11-2026/cart_unassigned/output_data/cart_unassigned_2026_11_25.csv"
    assert s3l.image_key(DAY, "cart_unassigned", 1).endswith("/cart_unassigned_2026_11_25_pic1.jpg")


def test_config_requires_all_env(monkeypatch):
    with pytest.raises(s3l.S3Error, match="LDM_S3_BUCKET"):
        s3l.S3Config.from_env()


# ============================================================================================ safe writes
def test_put_new_never_overwrites():
    s3 = FakeS3()
    s3l.put_new(s3, B, "a.jpg", b"one", "image/jpeg")
    with pytest.raises(s3l.Exists):
        s3l.put_new(s3, B, "a.jpg", b"two", "image/jpeg")
    assert s3.objs["a.jpg"] == b"one"


def test_put_new_or_same(tmp_path):
    s3 = FakeS3()
    f = tmp_path / "v.mp4"
    f.write_bytes(b"video")
    assert s3l.put_new_or_same(s3, B, "v.mp4", f, "video/mp4")["existed"] is False
    assert s3l.put_new_or_same(s3, B, "v.mp4", f, "video/mp4")["existed"] is True
    f.write_bytes(b"other")
    with pytest.raises(s3l.S3Error, match="different content"):
        s3l.put_new_or_same(s3, B, "v.mp4", f, "video/mp4")
    assert s3.objs["v.mp4"] == b"video"


def test_append_csv_appends_and_keeps_existing_bytes():
    s3 = FakeS3()
    cols = ["pic", "x"]
    s3l.append_csv(s3, B, "d.csv", cols, lambda ex: [{"pic": "pic1", "x": "a"}])
    before = s3.objs["d.csv"]
    s3l.append_csv(s3, B, "d.csv", cols, lambda ex: [{"pic": f"pic{len(ex) + 1}", "x": "b"}])
    after = s3.objs["d.csv"]
    assert after.startswith(before)
    assert after.decode() == "pic,x\r\npic1,a\r\npic2,b\r\n"


def test_append_csv_retries_on_concurrent_writer():
    s3 = FakeS3()
    cols = ["pic"]
    s3l.append_csv(s3, B, "d.csv", cols, lambda ex: [{"pic": "pic1"}])
    calls = []

    def rows(ex):
        calls.append(len(ex))
        if len(calls) == 1:                         # another writer appends between our read and write
            s3.objs["d.csv"] += b"pic2\r\n"
            s3.etags["d.csv"] = '"changed"'
        return [{"pic": f"pic{len(ex) + 1}"}]

    s3l.append_csv(s3, B, "d.csv", cols, rows)
    assert calls == [1, 2]
    assert s3.objs["d.csv"].decode() == "pic\r\npic1\r\npic2\r\npic3\r\n"


def test_append_csv_refuses_foreign_header():
    s3 = FakeS3()
    s3.put_object(Bucket=B, Key="d.csv", Body=b"other\r\n1\r\n", ContentType="text/csv",
                  ChecksumSHA256=s3l.sha256_b64(b"other\r\n1\r\n"))
    with pytest.raises(s3l.S3Error, match="different header"):
        s3l.append_csv(s3, B, "d.csv", ["pic"], lambda ex: [{"pic": "pic1"}])


# ============================================================================================ job upload
def _entries(video: str, n: int) -> list[dict]:
    return [{"capturedAt": f"2026-11-24T23:5{i}:00+05:30", "cameraId": "CAM_A", "imageName": dc.image_name(video, i),
             "litterClass": "packet" if i == 1 else "", "confidenceScore": "0.71" if i == 1 else "",
             "detectionStatus": "litter_detected" if i == 1 else "clean", "remark": "r", "storeName": "Store",
             "sourceFile": video} for i in range(1, n + 1)]


@pytest.fixture
def job(tmp_path, monkeypatch):
    video = tmp_path / "rec.mp4"
    video.write_bytes(b"mp4-bytes")
    imgs = {}
    for i in (1, 2, 3, 4):
        imgs[i] = tmp_path / f"s{i}.jpg"
        imgs[i].write_bytes(f"jpg{i}".encode())

    def fake_entries(query, settings, job_id, classes=None):
        info = {"store_key": "S1", "cart_id": "50", "store_name": "Store", "video": f"{job_id}.mp4",
                "video_path": video}
        return info, _entries(f"{job_id}.mp4", 4), 0

    monkeypatch.setattr(dc, "job_entries", fake_entries)
    return lambda j, i: imgs.get(i)


def _run(s3, job_id, image_path):
    return s3_sync.upload_job(None, None, job_id, s3=s3, bucket=B, image_path=image_path, processing_day=DAY)


def test_upload_job_layout_numbering_and_idempotence(job):
    s3 = FakeS3()
    r1 = _run(s3, "job1", job)
    assert r1["appended"] == 4
    # input under the RECORDING day (24th), outputs under the PROCESSING day (25th)
    assert "11-2026/24-11-2026/cart_50/input_data/job1.mp4" in s3.objs
    csv_key = "11-2026/25-11-2026/cart_50/output_data/cart50_2026_11_25.csv"
    assert r1["images"][-1] == "11-2026/25-11-2026/cart_50/output_data/cart50_2026_11_25_pic4.jpg"

    r2 = _run(s3, "job2", job)
    assert r2["images"][0].endswith("_pic5.jpg") and r2["appended"] == 4
    rows = s3l.csv_rows(s3.objs[csv_key].decode())
    assert [r["pic"] for r in rows] == [f"pic{i}" for i in range(1, 9)]
    assert {r["jobId"] for r in rows} == {"job1", "job2"}

    puts = s3.puts
    r3 = _run(s3, "job1", job)                       # re-run of the same job: nothing written
    assert r3["skipped"] == "job already in the CSV" and s3.puts == puts


def test_upload_job_skips_taken_pic_number(job):
    s3 = FakeS3()
    orphan = s3l.image_key(DAY, "cart_50", 1)       # an image left by a run whose CSV append failed
    s3.put_object(Bucket=B, Key=orphan, Body=b"x", ContentType="image/jpeg", ChecksumSHA256=s3l.sha256_b64(b"x"))
    r = _run(s3, "job1", job)
    assert r["images"][0].endswith("_pic2.jpg")
    assert s3.objs[orphan] == b"x"


def test_unassigned_cart():
    assert s3_sync.cart_of("") == "cart_unassigned"
    assert s3_sync.cart_of("7") == "cart_7"


def test_submit_job_is_noop_when_off():
    class St:
        class cfg:
            s3_upload_enabled = False
    assert s3_sync.submit_job(St, "job1") is None
