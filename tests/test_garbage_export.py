"""ai_garbage_detection import CSV (backend/services/garbage_export.py)."""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.services import garbage_export as ge

CLASSES = {"food_wrapper", "plastic_piece", "paper", "cardboard"}
STORE = ge.Store("STORE_KEY_1", "Test Store, Pune", 7, ("CAM_A", "CAM_B"))
T0 = dt.datetime(2026, 11, 19, 16, 30, 13, tzinfo=ge.IST)


def check(status, litter=(), review=(), cam="CAM_A", at=T0, reason=""):
    return ge.Check(cam, at, "frame_s1.jpg", status, list(litter), list(review), reason=reason,
                    time_note="time source: filename (x) + 0.000 s")


def write_cfg(tmp_path, stores):
    p = tmp_path / "store_config.json"
    p.write_text(json.dumps({"stores": stores}), encoding="utf-8")
    return p


GOOD = {"storeKey": "STORE_KEY_1", "storeName": "Test Store, Pune", "cart_id": 7, "camera_ids": ["CAM_A", "CAM_B"]}


# ------------------------------------------------------------------------------------------ rows
def test_litter_row():
    r = ge.build_row(check(ge.LITTER, [("food_wrapper", 0.7827)]), STORE, CLASSES)
    assert r == {"timestamp": "2026-11-19T16:30:13+05:30", "camera_id": "CAM_A", "image_name": "frame_s1.jpg",
                 "litter_class": "food_wrapper", "confidence_score": "78.27", "detection_status": "litter_detected",
                 "remark": "1 litter object: food_wrapper 78.27; time source: filename (x) + 0.000 s",
                 "cart_id": "7", "store_name": "Test Store, Pune"}


def test_clean_row_has_empty_class_and_confidence():
    r = ge.build_row(check(ge.CLEAN, review=[("paper", 0.48)]), STORE, CLASSES)
    assert r["detection_status"] == "clean" and r["litter_class"] == "" and r["confidence_score"] == ""
    assert "under review, not counted: paper 48.00" in r["remark"]


def test_blocked_row():
    r = ge.build_row(check(ge.BLOCKED, reason="blank frame: covered lens"), STORE, CLASSES)
    assert r["detection_status"] == "view_blocked" and r["litter_class"] == "" and r["confidence_score"] == ""
    assert r["remark"].startswith("view blocked: blank frame: covered lens")


@pytest.mark.parametrize("conf,out", [(0.7827, "78.27"), (1.0, "100.00"), (0.0, "0.00"), (0.12345, "12.35"),
                                      (0.5, "50.00")])
def test_confidence_converted_to_percent(conf, out):
    assert ge.to_percent(conf) == out


@pytest.mark.parametrize("bad", [78.27, -0.1, 1.0001])
def test_confidence_not_on_unit_scale_is_refused(bad):
    with pytest.raises(ge.ExportError):
        ge.to_percent(bad)


def test_multiple_detections_one_row_highest_confidence():
    r = ge.build_row(check(ge.LITTER, [("plastic_piece", 0.61), ("food_wrapper", 0.7599), ("food_wrapper", 0.633)]),
                     STORE, CLASSES)
    assert (r["litter_class"], r["confidence_score"]) == ("food_wrapper", "75.99")
    assert r["remark"].startswith("3 litter objects: food_wrapper 75.99, food_wrapper 63.30, plastic_piece 61.00")


def test_status_outside_the_three_values_fails():
    with pytest.raises(ge.ExportError, match="not one of"):
        ge.build_row(check("DETECTED", [("food_wrapper", 0.7)]), STORE, CLASSES)
    with pytest.raises(ge.ExportError):
        ge.build_row(check(ge.LITTER, []), STORE, CLASSES)          # litter without a detection
    with pytest.raises(ge.ExportError):
        ge.build_row(check(ge.CLEAN, [("paper", 0.7)]), STORE, CLASSES)
    with pytest.raises(ge.ExportError, match="class"):
        ge.build_row(check(ge.LITTER, [("glass", 0.7)]), STORE, CLASSES)   # never invents a class


# ------------------------------------------------------------------------------------------ time zone
def test_timestamps_are_india_time_with_offset():
    assert ge.format_ts(dt.datetime(2026, 11, 19, 11, 0, 13, tzinfo=dt.timezone.utc)) == "2026-11-19T16:30:13+05:30"
    assert ge.format_ts(ge.to_ist(dt.datetime(2026, 11, 19, 23, 59, 59, 900000))) == "2026-11-19T23:59:59+05:30"
    with pytest.raises(ge.ExportError):
        ge.format_ts(dt.datetime(2026, 11, 19, 16, 30))             # a naive time never reaches the CSV


def test_frame_offset_crossing_midnight_changes_date():
    tb = ge.resolve_time_base(video_path=None, original_name=None, duration_sec=None,
                              manual_start="2026-11-19T23:59:50")
    r = ge.build_row(check(ge.CLEAN, at=tb.start + dt.timedelta(seconds=15.4)), STORE)
    assert r["timestamp"] == "2026-11-20T00:00:05+05:30"


# ------------------------------------------------------------------------------------------ recording start
NVR = "20261119_20261119163013_20261119163458_163015.mp4"


def test_filename_start_needs_matching_duration():
    tb = ge.resolve_time_base(video_path=None, original_name=NVR, duration_sec=284.659, manual_start=None)
    assert tb.source == "filename" and tb.start == T0
    with pytest.raises(ge.NoRecordingTime, match="does not match the video duration"):
        ge.resolve_time_base(video_path=None, original_name=NVR, duration_sec=14.3, manual_start=None)


def test_no_time_source_refuses_and_never_uses_processing_time():
    with pytest.raises(ge.NoRecordingTime, match="--recording-start"):
        ge.resolve_time_base(video_path=None, original_name="quick_default_test.mp4", duration_sec=14.3,
                             manual_start=None)


def test_manual_start_used_last_and_conflict_fails():
    tb = ge.resolve_time_base(video_path=None, original_name="clip.mp4", duration_sec=10, manual_start="2026-11-19T10:00:00")
    assert tb.source == "recording-start argument" and tb.start.utcoffset() == dt.timedelta(hours=5, minutes=30)
    with pytest.raises(ge.ExportError, match="contradicts"):
        ge.resolve_time_base(video_path=None, original_name=NVR, duration_sec=284.659,
                             manual_start="2026-11-19T17:00:00")


def _mp4_with_mvhd(path: Path, when: dt.datetime) -> Path:
    secs = int((when - dt.datetime(1904, 1, 1, tzinfo=dt.timezone.utc)).total_seconds())
    mvhd = struct.pack(">I4sB3sI", 8 + 1 + 3 + 4, b"mvhd", 0, b"\0\0\0", secs)
    path.write_bytes(struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd)
    return path


def test_container_time_ignored_unless_trusted(tmp_path):
    p = _mp4_with_mvhd(tmp_path / "x.mp4", dt.datetime(2026, 11, 19, 11, 0, 13, tzinfo=dt.timezone.utc))
    assert ge.container_creation_time(p) == dt.datetime(2026, 11, 19, 11, 0, 13, tzinfo=dt.timezone.utc)
    with pytest.raises(ge.NoRecordingTime, match="ignored"):
        ge.resolve_time_base(video_path=p, original_name="x.mp4", duration_sec=5, manual_start=None)
    tb = ge.resolve_time_base(video_path=p, original_name="x.mp4", duration_sec=5, manual_start=None,
                              trust_container_time=True)
    assert tb.source == "container metadata" and tb.start == T0


# ------------------------------------------------------------------------------------------ store config
def test_store_config_loads(tmp_path):
    stores = ge.load_store_config(write_cfg(tmp_path, [GOOD]))
    assert stores["STORE_KEY_1"] == STORE


@pytest.mark.parametrize("patch,msg", [({"cart_id": None}, "cart_id"), ({"cart_id": "7"}, "cart_id"),
                                       ({"cart_id": True}, "cart_id"), ({"storeKey": "REPLACE_ME"}, "storeKey"),
                                       ({"camera_ids": []}, "camera_ids"), ({"storeName": ""}, "storeName"),
                                       ({"camera_ids": ["REPLACE_WITH_CAMERA_ID"]}, "camera_ids")])
def test_store_config_rejects_missing_or_placeholder(tmp_path, patch, msg):
    with pytest.raises(ge.StoreConfigError, match=msg):
        ge.load_store_config(write_cfg(tmp_path, [{**GOOD, **patch}]))


@pytest.mark.parametrize("value", ["REPLACE_WITH_S3_STORE_FOLDER", "REPLACE_WITH_STORE_NAME", "REPLACE_WITH_CAMERA_ID",
                                   "replace with store name", "TODO", "tbd", "CHANGE_ME", "changeme", "PLACEHOLDER",
                                   "FILL_IN", "fill me", "XXX", "???"])
def test_placeholder_values_detected(value):
    assert ge.PLACEHOLDER_RE.match(value)


@pytest.mark.parametrize("value", ["FILLING_STATION_07", "Fillmore Mart", "Todoroki Store", "Replacement Parts Depot",
                                   "Changemen Bazaar", "PlaceholderCo", "Pune Store 12", "TODOS_SANTOS"])
def test_real_names_are_not_mistaken_for_placeholders(value, tmp_path):
    assert not ge.PLACEHOLDER_RE.match(value)
    key = value.replace(" ", "_")
    stores = ge.load_store_config(write_cfg(tmp_path, [{**GOOD, "storeKey": key, "storeName": value}]))
    assert stores[key].name == value


def test_shipped_placeholder_config_is_rejected():
    with pytest.raises(ge.StoreConfigError):
        ge.load_store_config(Path(__file__).resolve().parents[1] / "config" / "store_config.json")


def test_missing_camera_id_or_unknown_store_fails():
    stores = {STORE.key: STORE}
    with pytest.raises(ge.ExportError, match="camera_id missing"):
        ge.store_for(stores, STORE.key, None)
    with pytest.raises(ge.ExportError, match="unknown store"):
        ge.store_for(stores, "OTHER", "CAM_A")
    with pytest.raises(ge.ExportError, match="not listed"):
        ge.store_for(stores, STORE.key, "CAM_Z")
    with pytest.raises(ge.ExportError, match="camera_id missing"):
        ge.build_row(check(ge.CLEAN, cam=""), STORE)
    with pytest.raises(ge.ExportError, match="contradicts"):
        ge.pick("camera_id", "CAM_A", "CAM_B")


# ------------------------------------------------------------------------------------------ validator
def _csv(rows):
    return ge.rows_to_csv(rows)


def _good_rows():
    return [ge.build_row(check(ge.LITTER, [("food_wrapper", 0.7827)]), STORE, CLASSES),
            ge.build_row(check(ge.CLEAN, at=T0 + dt.timedelta(minutes=1)), STORE, CLASSES),
            ge.build_row(check(ge.BLOCKED, reason="blank", at=T0 + dt.timedelta(minutes=2)), STORE, CLASSES)]


def test_validator_accepts_good_rows():
    assert ge.validate_text(_csv(_good_rows()), allowed_classes=CLASSES, stores={STORE.key: STORE},
                            store_key=STORE.key) == []


@pytest.mark.parametrize("col,val,msg", [
    ("detection_status", "Clean", "not one of"), ("detection_status", "DETECTED", "not one of"),
    ("timestamp", "2026-11-19 16:30:13", "+05:30"), ("timestamp", "2026-11-19T16:30:13Z", "+05:30"),
    ("confidence_score", "0.78", None), ("confidence_score", "178.20", "0-100"), ("confidence_score", "", "0-100"),
    ("litter_class", "", "required"), ("litter_class", "glass", "approved"), ("cart_id", "", "whole number"),
    ("cart_id", "7.5", "whole number"), ("camera_id", "", "missing"), ("store_name", "", "missing"),
    ("remark", "=HYPERLINK(1)", "formula")])
def test_validator_reports_row_errors(col, val, msg):
    rows = _good_rows()
    rows[0][col] = val
    issues = ge.validate_text(_csv(rows), allowed_classes=CLASSES)
    if col == "confidence_score" and val == "0.78":
        assert issues == []          # 0.78 is a legal 0-100 value; a 0-1 value cannot be told apart in a file
        return
    assert any(i.line == 2 and i.column == col and msg in i.message for i in issues), issues


@pytest.mark.parametrize("val", ["None", "null", "0"])
def test_validator_requires_empty_class_and_confidence_for_clean(val):
    rows = _good_rows()
    rows[1]["confidence_score"] = val
    issues = ge.validate_text(_csv(rows))
    assert any(i.line == 3 and i.column == "confidence_score" for i in issues)


def test_validator_header_bom_duplicates_and_store_match(tmp_path):
    assert ge.validate_text("a,b\n1,2\n")[0].message.startswith("header must be")
    p = tmp_path / "STORE_KEY_1" / "2026-11-19" / "f.csv"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"\xef\xbb\xbf" + _csv(_good_rows()).encode())
    assert "BOM" in ge.validate_file(p)[0].message
    rows = _good_rows()
    rows[1]["timestamp"] = rows[0]["timestamp"]
    rows[2]["cart_id"] = "8"
    p.write_text(_csv(rows), encoding="utf-8", newline="")
    msgs = [str(i) for i in ge.validate_file(p, stores={STORE.key: STORE})]
    assert any("duplicate check" in m for m in msgs) and any("does not match the store config (7)" in m for m in msgs)
    other = tmp_path / "STORE_KEY_1" / "2026-11-20" / "f.csv"
    other.parent.mkdir(parents=True)
    other.write_text(_csv(_good_rows()), encoding="utf-8", newline="")
    assert any("folder date" in str(i) for i in ge.validate_file(other))


# ------------------------------------------------------------------------------------------ writer + ledger
def test_export_writes_new_files_and_never_repeats_a_check(tmp_path):
    root, ledger = tmp_path / "out", tmp_path / "ledger.db"
    checks = [check(ge.LITTER, [("food_wrapper", 0.7827)]), check(ge.CLEAN, at=T0 + dt.timedelta(minutes=1)),
              check(ge.CLEAN, cam="CAM_B")]
    r1 = ge.export_checks(checks, STORE, source_id="sv_1", export_root=root, ledger_path=ledger,
                          allowed_classes=CLASSES)
    assert r1["rows_written"] == 3 and len(r1["files"]) == 2                     # one file per camera
    files = sorted(root.rglob("*.csv"))
    assert all(f.parent.name == "2026-11-19" and f.parent.parent.name == "STORE_KEY_1" for f in files)
    assert all(ge.validate_file(f, allowed_classes=CLASSES, stores={STORE.key: STORE}) == [] for f in files)
    text = files[0].read_bytes()
    assert not text.startswith(b"\xef\xbb\xbf") and text.splitlines()[0] == ",".join(ge.COLUMNS).encode()
    assert '"Test Store, Pune"' in text.decode()                                 # quoted because of the comma
    before = {f: f.read_bytes() for f in files}
    r2 = ge.export_checks(checks + [check(ge.CLEAN, at=T0 + dt.timedelta(minutes=5))], STORE, source_id="sv_2",
                          export_root=root, ledger_path=ledger, allowed_classes=CLASSES)
    assert r2["rows_written"] == 1 and len(r2["skipped_duplicates"]) == 3
    assert all(f.read_bytes() == b for f, b in before.items())                   # uploaded files never rewritten
    r3 = ge.export_checks(checks, STORE, source_id="sv_3", export_root=root, ledger_path=ledger)
    assert r3["files"] == [] and len(sorted(root.rglob("*.csv"))) == 3


def test_export_is_all_or_nothing(tmp_path):
    with pytest.raises(ge.ExportError):
        ge.export_checks([check(ge.CLEAN), check(ge.LITTER, [("food_wrapper", 7.0)])], STORE, source_id="x",
                         export_root=tmp_path / "o", ledger_path=tmp_path / "l.db")
    assert not list((tmp_path).rglob("*.csv"))


def test_dry_run_writes_nothing(tmp_path):
    r = ge.export_checks([check(ge.CLEAN)], STORE, source_id="x", export_root=tmp_path / "o",
                         ledger_path=tmp_path / "l.db", dry_run=True)
    assert r["rows_written"] == 1 and not list(tmp_path.rglob("*.csv"))
    r = ge.export_checks([check(ge.CLEAN)], STORE, source_id="x", export_root=tmp_path / "o",
                         ledger_path=tmp_path / "l.db")
    assert r["rows_written"] == 1                                                # the dry run reserved nothing


# ------------------------------------------------------------------------------------------ sources
def fake_query(tables):
    def q(sql, params=()):
        for key, fn in tables.items():
            if key in sql:
                return fn(params)
        raise AssertionError(sql)
    return q


def _frame(idx, outcome, t, dets=(), error=None):
    return {"idx": idx, "outcome": outcome, "video_time_sec": t, "error": error,
            "original_path": f"C:/x/frames/clip_s{idx}.jpg", "result_json": json.dumps({"detections": list(dets)})}


def test_sampled_checks_map_outcomes_and_operator_rejections():
    frames = [_frame(1, "LITTER_DETECTED", 28.596, [
                  {"class": "food_wrapper", "confidence": 0.7827, "bbox": [0, 0, 10, 10], "status": "confirmed"},
                  {"class": "paper", "confidence": 0.5, "bbox": [50, 50, 60, 60], "status": "candidate"}]),
              _frame(2, "NO_LITTER", 99.284),
              _frame(3, "UNREADABLE", 184.3, error="blank frame"),
              _frame(4, "FAILED", 250.0),
              _frame(5, "LITTER_DETECTED", 260.0, [
                  {"class": "cardboard", "confidence": 0.71, "bbox": [100, 100, 200, 200], "status": "confirmed"}])]
    q = fake_query({"FROM litter_events": lambda p: [{"class": "cardboard", "x1": 101, "y1": 99, "x2": 201,
                                                      "y2": 200}],
                    "FROM sampled_frames": lambda p: frames})
    src = ge.Source("sv_1", None, None, NVR, None, 284.659, None, "qck_1")
    tb = ge.resolve_time_base(video_path=None, original_name=NVR, duration_sec=284.659, manual_start=None)
    checks, excluded = ge.sampled_checks(q, src, "CAM_A", tb)
    assert [c.status for c in checks] == [ge.LITTER, ge.CLEAN, ge.BLOCKED, ge.CLEAN]
    assert excluded == ["screenshot 4: processing failed (not a camera check)"]
    rows = [ge.build_row(c, STORE, CLASSES) for c in checks]
    assert rows[0]["timestamp"] == "2026-11-19T16:30:41+05:30" and rows[0]["image_name"] == "sv_1_s1_annotated.jpg"
    assert "rejected by an operator" in rows[3]["remark"] and rows[3]["litter_class"] == ""
    assert "time source: filename" in rows[1]["remark"]


def test_sampled_unknown_outcome_fails_loudly():
    q = fake_query({"FROM litter_events": lambda p: [], "FROM sampled_frames": lambda p: [_frame(1, "WEIRD", 1.0)]})
    tb = ge.TimeBase(T0, "filename", "x")
    with pytest.raises(ge.ExportError, match="unknown outcome"):
        ge.sampled_checks(q, ge.Source("sv", None, None, None, None, None, None, None), "CAM_A", tb)


def test_video_checks_use_frame_quality_and_review_status():
    shots = {"screenshots": [
        {"file": "screenshot_01.png", "frame_idx": 0, "video_time_sec": 0.0, "analysed_frame": True,
         "detections": [{"class": "food_wrapper", "confidence": 0.8, "detection_status": "DETECTED", "event_id": "E1"},
                        {"class": "paper", "confidence": 0.9, "detection_status": "DETECTED", "event_id": "E2"}]},
        {"file": "screenshot_02.png", "frame_idx": 10, "video_time_sec": 1.0, "detections": []}]}
    q = fake_query({"FROM litter_events": lambda p: [{"event_id": "E1", "review_status": "DETECTED"},
                                                     {"event_id": "E2", "review_status": "REJECTED"}]})
    from backend.sampling.frame_quality import problem
    readable = np.random.default_rng(0).integers(0, 255, (360, 640, 3), np.uint8)
    covered = np.full((360, 640, 3), 12, np.uint8)
    src = ge.Source("vid_1", None, None, None, None, None, None, "vid_1")
    checks, _ = ge.video_checks(q, src, shots, "CAM_A", ge.TimeBase(T0, "filename", "x"), frame_problem=problem,
                                frames={0: readable, 10: None})
    assert [c.status for c in checks] == [ge.LITTER, ge.BLOCKED]
    assert checks[0].litter == [("food_wrapper", 0.8)] and "rejected by an operator" in checks[0].notes[0]
    assert "no usable image data" in checks[1].reason
    checks, _ = ge.video_checks(q, src, shots, "CAM_A", ge.TimeBase(T0, "filename", "x"), frame_problem=problem,
                                frames={0: covered, 10: readable})
    assert [c.status for c in checks] == [ge.BLOCKED, ge.CLEAN] and checks[0].litter == []
    assert "covered lens" in checks[0].reason


# ------------------------------------------------------------------------------------------ hook
def test_auto_export_is_off_by_default_and_never_raises(tmp_path):
    st = SimpleNamespace(cfg=SimpleNamespace(garbage_export_enabled=False))
    assert ge.auto_export(st, kind="sampled", source_id="x") is None
    cfg = SimpleNamespace(garbage_export_enabled=True, store_config_path=tmp_path / "missing.json",
                          garbage_export_dir=tmp_path / "o", garbage_export_ledger=tmp_path / "l.db",
                          trust_container_time=False)
    st = SimpleNamespace(cfg=cfg, db=SimpleNamespace(_query=lambda *a: []), engine=None)
    res = ge.auto_export(st, kind="sampled", source_id="x", out_dir=tmp_path / "job")
    assert res["ok"] is False and "store config not found" in res["error"]
    assert json.loads((tmp_path / "job" / "ai_garbage_export.json").read_text())["ok"] is False
