"""Import-ready CSV export for the dashboard importer's ``ai_garbage_detection`` schema (owner brief 2026-11-20).

The ONLY place this format is defined. Detection, thresholds, model and class names are not touched: a row is built
from results the pipeline already produced. The older CSV formats (LDM_OUTPUT detections/summary, daily six-column,
sampled reports) are unchanged; this export is additive.

    storage/exports/ai_garbage_detection/<storeKey>/<YYYY-MM-DD>/<storeKey>_<YYYY-MM-DD>_<camera>_<source>_<stamp>.csv

Contract
  * columns, fixed order: timestamp, camera_id, image_name, litter_class, confidence_score, detection_status, remark,
    cart_id, store_name. UTF-8 without BOM, comma-delimited, RFC 4180 quoting (CRLF line ends).
  * ONE ROW = ONE CHECK of one camera (one analysed screenshot). Several litter objects in a frame: the
    highest-confidence object fills litter_class / confidence_score; all objects are listed in ``remark``.
  * detection_status: exactly clean | litter_detected | view_blocked. Anything else raises.
      litter_detected  >= 1 detection that passed the existing confirmed rule and was not rejected by an operator
      clean            the frame was analysed and no such detection exists (under-review candidates -> remark only)
      view_blocked     the frame is not a usable picture: backend/sampling/frame_quality.problem() (>= 98 % flat:
                       covered lens / no signal / darkness; >= 25 % decoder grey fill) or it could not be decoded.
                       Partial obstruction (a vehicle or person in front of the lens) is NOT detected.
  * litter_class / confidence_score are EMPTY for clean and view_blocked. confidence_score = model 0-1 x 100, 2 dp.
  * timestamp = recording start + frame offset, Asia/Kolkata, ISO 8601 with +05:30, second precision. Processing
    time is NEVER used. Recording start, in order: (1) container metadata (only when LDM_TRUST_CONTAINER_TIME=on,
    because exports seen so far carry an encode time there), (2) NVR file name <day>_<start>_<end>_... whose span
    matches the video duration, (3) burned-in OSD clock (not implemented: no OCR engine on this PC -> always
    unavailable), (4) an operator-given start (--recording-start, or the one typed when the analysis was started).
    The source is written in ``remark``. No source -> NoRecordingTime, and no file is written.
  * store: config/store_config.json (storeKey = S3 folder, storeName, cart_id, camera_ids). Unknown store, a camera
    not listed for the store, a missing cart_id or a placeholder entry -> error. There are no defaults.
  * a camera check is exported at most once: a ledger (SQLite) keyed (storeKey, camera_id, timestamp) is checked
    and filled in the same transaction as the file is written; a new file is never written over an existing one.

CLI (read-only towards the detection DB; writes only the export folder + ledger):
    py -m backend.services.garbage_export check-config
    py -m backend.services.garbage_export export --sampled-job <id> --store-key <key> --camera-id <cam> [--recording-start ISO]
    py -m backend.services.garbage_export export --video-analysis <id> ...
    py -m backend.services.garbage_export validate <file.csv> [...]
    py -m backend.services.garbage_export validate-bundle <bundle folder> [...]   (four-screenshot result bundles)

Also used by backend/services/result_bundle.py (per-job bundles: 4 annotated images + this CSV + manifest.json) and
backend/services/results_view.py (dashboard Results view and its importer CSV download): the row text is made by
describe()/build_row() only, so all three always agree.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import logging
import os
import re
import secrets
import sqlite3
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

SCHEMA = "ai_garbage_detection"
COLUMNS = ["timestamp", "camera_id", "image_name", "litter_class", "confidence_score", "detection_status", "remark",
           "cart_id", "store_name"]
CLEAN, LITTER, BLOCKED = "clean", "litter_detected", "view_blocked"
STATUSES = (CLEAN, LITTER, BLOCKED)
IST = ZoneInfo("Asia/Kolkata")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+05:30$")
CONFIDENCE_RE = re.compile(r"^\d{1,3}(?:\.\d{1,2})?$")
CART_RE = re.compile(r"^\d+$")
STORE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")      # safe as an S3 folder name
CAMERA_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")          # never starts with a formula character
# Whole-value placeholders only: a real name such as "FILLING_STATION_07" or "Replacement Parts Depot" must pass.
PLACEHOLDER_RE = re.compile(r"^(?:REPLACE[_ ]WITH(?:[_ ].*)?|REPLACE[_ ]?ME|TODO|TBD|CHANGE[_ ]?ME|PLACEHOLDER|"
                            r"FILL[_ ]?(?:ME|IN)|XXX+|\?+)$", re.I)
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")
NVR_NAME_RE = re.compile(r"(?:^|[^0-9])(\d{8})_(\d{14})_(\d{14})(?:[^0-9]|$)")
DURATION_TOLERANCE_SEC, DURATION_TOLERANCE_FRAC = 3.0, 0.02
MANUAL_CONFLICT_SEC = 2.0

# Sampled-screenshot outcomes (backend/sampling/report.py). Mapped explicitly; an unknown value raises.
OUTCOME_STATUS = {"LITTER_DETECTED": LITTER, "NO_LITTER": CLEAN, "UNREADABLE": BLOCKED}
OUTCOME_NOT_A_CHECK = {"FAILED": "processing failed (not a camera check)",
                       "ROI_NOT_CONFIGURED": "not analysed (legacy ROI outcome, before ROI removal)"}
LITTER_DETECTION_STATUSES = ("DETECTED", "VERIFIED")      # screenshots.json of full-video analyses


class ExportError(ValueError):
    """Any reason an import-ready CSV cannot be produced. The message says what is missing."""


class StoreConfigError(ExportError):
    pass


class NoRecordingTime(ExportError):
    pass


# ============================================================================================ store config
@dataclass(frozen=True)
class Store:
    key: str
    name: str
    cart_id: int
    camera_ids: tuple[str, ...]


def load_store_config(path: Path) -> dict[str, Store]:
    """storeKey -> Store. Every problem in the file is reported at once; nothing is defaulted."""
    path = Path(path)
    if not path.is_file():
        raise StoreConfigError(f"store config not found: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StoreConfigError(f"{path.name} is not valid JSON: {exc}") from exc
    entries = doc.get("stores") if isinstance(doc, dict) else None
    if not isinstance(entries, list) or not entries:
        raise StoreConfigError(f"{path.name}: 'stores' must be a non-empty list")
    errors: list[str] = []
    stores: dict[str, Store] = {}
    owner: dict[str, str] = {}
    for i, e in enumerate(entries):
        where = f"stores[{i}]"
        if not isinstance(e, dict):
            errors.append(f"{where}: must be an object")
            continue
        key, name, cart, cams = e.get("storeKey"), e.get("storeName"), e.get("cart_id"), e.get("camera_ids")
        bad = len(errors)
        if not isinstance(key, str) or not STORE_KEY_RE.match(key) or PLACEHOLDER_RE.match(key):
            errors.append(f"{where}.storeKey: {key!r} is missing, a placeholder or not a valid S3 folder name")
        if not isinstance(name, str) or not name.strip() or PLACEHOLDER_RE.match(name.strip()):
            errors.append(f"{where}.storeName: {name!r} is missing or a placeholder")
        elif name != name.strip() or name.startswith(FORMULA_START):
            errors.append(f"{where}.storeName: {name!r} has surrounding spaces or starts with a formula character")
        if isinstance(cart, bool) or not isinstance(cart, int) or cart < 0:
            errors.append(f"{where}.cart_id: {cart!r} must be a whole number >= 0 (no default is used)")
        if not isinstance(cams, list) or not cams:
            errors.append(f"{where}.camera_ids: must be a non-empty list")
        else:
            for c in cams:
                if not isinstance(c, str) or not CAMERA_RE.match(c) or PLACEHOLDER_RE.match(c):
                    errors.append(f"{where}.camera_ids: {c!r} is a placeholder or not a valid camera id")
                elif c in owner:
                    errors.append(f"{where}.camera_ids: {c} is already listed for store {owner[c]}")
                else:
                    owner[c] = str(key)
        if isinstance(key, str) and key in stores:
            errors.append(f"{where}.storeKey: {key} is listed twice")
        if len(errors) == bad:
            stores[key] = Store(key, name, cart, tuple(cams))
    if errors:
        raise StoreConfigError(f"{path.name} is not usable:\n  - " + "\n  - ".join(errors))
    return stores


def store_for(stores: dict[str, Store], store_key: Optional[str], camera_id: Optional[str]) -> Store:
    if not store_key:
        raise ExportError("store key missing: pass --store-key (the S3 folder name) or set the store on the analysis")
    if not camera_id:
        raise ExportError("camera_id missing: pass --camera-id or set the camera on the analysis")
    st = stores.get(store_key)
    if st is None:
        raise ExportError(f"unknown store {store_key!r}: it is not in the store config")
    if camera_id not in st.camera_ids:
        raise ExportError(f"camera {camera_id!r} is not listed for store {store_key} in the store config")
    return st


# ============================================================================================ time
@dataclass(frozen=True)
class TimeBase:
    start: dt.datetime            # aware, Asia/Kolkata
    source: str                   # container metadata | filename | OSD | recording-start argument | ...
    detail: str


def to_ist(t: dt.datetime) -> dt.datetime:
    """A naive time is camera-local India time; an aware time is converted."""
    return t.replace(tzinfo=IST) if t.tzinfo is None else t.astimezone(IST)


def format_ts(t: dt.datetime) -> str:
    if t.tzinfo is None:
        raise ExportError("internal error: a naive datetime reached the CSV writer")
    return t.astimezone(IST).replace(microsecond=0).isoformat(timespec="seconds")


def _mp4_boxes(f: Any, start: int, end: int) -> Iterable[tuple[bytes, int, int]]:
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        head = f.read(8)
        if len(head) < 8:
            return
        size, typ = struct.unpack(">I4s", head)
        hdr = 8
        if size == 1:
            size, hdr = struct.unpack(">Q", f.read(8))[0], 16
        elif size == 0:
            size = end - pos
        if size < hdr:
            return
        yield typ, pos + hdr, pos + size
        pos += size


def container_creation_time(path: Optional[Path]) -> Optional[dt.datetime]:
    """MP4/MOV ``moov/mvhd`` creation time (UTC by the spec), or None when absent/zero/unreadable."""
    if not path or not Path(path).is_file():
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            for typ, s, e in _mp4_boxes(f, 0, n):
                if typ != b"moov":
                    continue
                for t2, s2, _e2 in _mp4_boxes(f, s, e):
                    if t2 == b"mvhd":
                        f.seek(s2)
                        version = f.read(4)[0]
                        raw = f.read(8 if version == 1 else 4)
                        secs = struct.unpack(">Q" if version == 1 else ">I", raw)[0]
                        if not secs:
                            return None
                        return dt.datetime(1904, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=secs)
    except (OSError, struct.error, IndexError):
        return None
    return None


def filename_start(name: Optional[str], duration_sec: Optional[float]) -> tuple[Optional[dt.datetime], str]:
    """Recording start from an NVR export name <day>_<start>_<end>_..., accepted only when end - start matches the
    decoded duration (proves the two numbers are the recording span, not e.g. an export time)."""
    if not name:
        return None, "no file name"
    m = NVR_NAME_RE.search(Path(name).name)
    if not m:
        return None, f"file name {Path(name).name!r} has no <day>_<start>_<end> NVR pattern"
    try:
        start = dt.datetime.strptime(m.group(2), "%Y%m%d%H%M%S")
        end = dt.datetime.strptime(m.group(3), "%Y%m%d%H%M%S")
    except ValueError:
        return None, "file name timestamps are not valid dates"
    if m.group(1) != m.group(2)[:8]:
        return None, "file name day does not match its start time"
    if not duration_sec or duration_sec <= 0:
        return None, "video duration unknown, so the file name span cannot be checked"
    span = (end - start).total_seconds()
    if abs(span - duration_sec) > max(DURATION_TOLERANCE_SEC, DURATION_TOLERANCE_FRAC * duration_sec):
        return None, f"file name span {span:.0f} s does not match the video duration {duration_sec:.1f} s"
    return start, f"{m.group(2)}, span {span:.0f} s = duration {duration_sec:.1f} s"


def parse_manual_start(value: str) -> dt.datetime:
    try:
        return to_ist(dt.datetime.fromisoformat(value.strip()))
    except ValueError as exc:
        raise ExportError(f"recording start {value!r} is not an ISO 8601 date-time") from exc


def resolve_time_base(*, video_path: Optional[Path], original_name: Optional[str], duration_sec: Optional[float],
                      manual_start: Optional[str], manual_label: str = "recording-start argument",
                      trust_container_time: bool = False) -> TimeBase:
    """Recording start of a video, in the approved order. Raises NoRecordingTime when nothing usable exists and
    ExportError when an operator-given start contradicts a stronger source."""
    tried: list[str] = []
    found: Optional[TimeBase] = None
    meta = container_creation_time(video_path)
    if meta is None:
        tried.append("container metadata: none")
    elif not trust_container_time:
        tried.append(f"container metadata {meta.isoformat()} ignored (LDM_TRUST_CONTAINER_TIME is off)")
    else:
        found = TimeBase(meta.astimezone(IST), "container metadata", f"mvhd {meta.isoformat()}")
    if found is None:
        start, why = filename_start(original_name, duration_sec)
        if start is not None:
            found = TimeBase(to_ist(start), "filename", why)
        else:
            tried.append(f"filename: {why}")
    if found is None:
        tried.append("OSD: not available (no OCR engine installed; not implemented)")
    if manual_start:
        manual = parse_manual_start(manual_start)
        if found is not None:
            gap = abs((manual - found.start).total_seconds())
            if gap > MANUAL_CONFLICT_SEC:
                raise ExportError(f"{manual_label} {manual.isoformat()} contradicts the {found.source} "
                                  f"({found.start.isoformat()}, {gap:.0f} s apart); resolve it before exporting")
        else:
            found = TimeBase(manual, manual_label, manual.isoformat())
    if found is None:
        raise NoRecordingTime("no recording start time: " + "; ".join(tried)
                              + ". Pass --recording-start (ISO 8601) to export this run; processing time is never "
                                "used.")
    return found


# ============================================================================================ checks + rows
@dataclass
class Check:
    camera_id: str
    captured_at: Optional[dt.datetime]               # aware; None = no trustworthy recording time (never exported)
    image_name: str
    status: str                                      # clean | litter_detected | view_blocked
    litter: list[tuple[str, float]] = field(default_factory=list)        # (class, model confidence 0-1)
    under_review: list[tuple[str, float]] = field(default_factory=list)
    reason: str = ""                                 # why the view is blocked
    notes: list[str] = field(default_factory=list)
    time_note: str = ""
    idx: Optional[int] = None                        # screenshot number (1-4) of a four-screenshot job


def to_percent(conf: float) -> str:
    """Model confidence (0-1) -> 0-100 with 2 decimals. The model never reports above 1, so a larger value means a
    caller passed an already-converted or wrong number: refuse it instead of guessing."""
    v = float(conf)
    if not 0.0 <= v <= 1.0:
        raise ExportError(f"confidence {conf!r} is not on the model's 0-1 scale")
    return f"{round(v * 100.0, 2):.2f}"


def _objects(items: Sequence[tuple[str, float]]) -> str:
    return ", ".join(f"{c} {to_percent(p)}" for c, p in sorted(items, key=lambda x: -x[1]))


def one_line(text: str) -> str:
    """CSV cells of this format never contain line breaks (error texts from decoders sometimes do)."""
    return re.sub(r"\s*[\r\n]+\s*", " ", text).strip()


def describe(check: Check, allowed_classes: Optional[set[str]] = None) -> tuple[str, str, str]:
    """(litter_class, confidence_score, remark) of one check - the same text in the import CSV and on the dashboard.
    Raises ExportError when the check contradicts the status rules."""
    if check.status not in STATUSES:
        raise ExportError(f"detection_status {check.status!r} is not one of {', '.join(STATUSES)}")
    for cls, _ in (*check.litter, *check.under_review):
        if allowed_classes is not None and cls not in allowed_classes:
            raise ExportError(f"class {cls!r} is not one of the model's class names")
    parts: list[str] = []
    cls_out = conf_out = ""
    if check.status == LITTER:
        if not check.litter:
            raise ExportError(f"{check.image_name}: litter_detected without a litter detection")
        top = max(check.litter, key=lambda x: x[1])
        cls_out, conf_out = top[0], to_percent(top[1])
        n = len(check.litter)
        parts.append(f"{n} litter object{'s' if n != 1 else ''}: {_objects(check.litter)}")
    else:
        if check.litter:
            raise ExportError(f"{check.image_name}: {check.status} row cannot carry litter detections")
        parts.append("no litter detected by the model" if check.status == CLEAN
                     else f"view blocked: {check.reason or 'frame not usable'}")
    if check.under_review:
        k = len(check.under_review)
        parts.append(f"{k} possible object{'s' if k != 1 else ''} under review, not counted: "
                     f"{_objects(check.under_review)}")
    parts.extend(check.notes)
    if check.time_note:
        parts.append(check.time_note)
    return cls_out, conf_out, one_line("; ".join(parts))


def build_row(check: Check, store: Store, allowed_classes: Optional[set[str]] = None) -> dict[str, str]:
    """One import-ready row. Strict: store, camera and recording time must all be known."""
    if not check.camera_id:
        raise ExportError("camera_id missing")
    if check.camera_id not in store.camera_ids:
        raise ExportError(f"camera {check.camera_id!r} is not listed for store {store.key}")
    if check.captured_at is None:
        raise NoRecordingTime(f"{check.image_name}: no recording time ({check.time_note or 'unknown'})")
    cls_out, conf_out, remark = describe(check, allowed_classes)
    return {"timestamp": format_ts(check.captured_at), "camera_id": check.camera_id,
            "image_name": check.image_name, "litter_class": cls_out, "confidence_score": conf_out,
            "detection_status": check.status, "remark": remark, "cart_id": str(store.cart_id),
            "store_name": store.name}


def rows_to_csv(rows: Sequence[dict[str, str]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="raise", quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


# ============================================================================================ validator
@dataclass(frozen=True)
class Issue:
    line: int            # 1 = header
    column: str
    message: str

    def __str__(self) -> str:
        return f"line {self.line} [{self.column}]: {self.message}"


def validate_text(text: str, *, allowed_classes: Optional[set[str]] = None, stores: Optional[dict[str, Store]] = None,
                  store_key: Optional[str] = None, folder_date: Optional[str] = None) -> list[Issue]:
    """Row-level check of an import CSV against the contract. Empty list = valid."""
    issues: list[Issue] = []
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        return [Issue(1, "*", "file is empty (no header)")]
    if rows[0] != COLUMNS:
        return [Issue(1, "*", f"header must be exactly {','.join(COLUMNS)} (found {','.join(rows[0])})")]
    st: Optional[Store] = None
    if stores is not None and store_key is not None:
        st = stores.get(store_key)
        if st is None:
            issues.append(Issue(1, "*", f"store folder {store_key!r} is not in the store config"))
    seen: dict[tuple[str, str], int] = {}
    for n, raw in enumerate(rows[1:], start=2):
        if len(raw) != len(COLUMNS):
            issues.append(Issue(n, "*", f"{len(raw)} fields, expected {len(COLUMNS)}"))
            continue
        r = dict(zip(COLUMNS, raw))

        def bad(col: str, msg: str) -> None:
            issues.append(Issue(n, col, msg))

        for col, v in r.items():
            if v.startswith(FORMULA_START):
                bad(col, f"value starts with a formula character: {v[:20]!r}")
            if v.strip().lower() in ("none", "null", "nan") :
                bad(col, f"literal {v!r} instead of an empty value")
            if "\n" in v or "\r" in v:
                bad(col, "contains a line break")
        ts = r["timestamp"]
        if not TIMESTAMP_RE.match(ts):
            bad("timestamp", f"{ts!r} is not YYYY-MM-DDTHH:MM:SS+05:30")
        else:
            try:
                parsed = dt.datetime.fromisoformat(ts)
                if folder_date and parsed.date().isoformat() != folder_date:
                    bad("timestamp", f"date {parsed.date()} is not the folder date {folder_date}")
            except ValueError:
                bad("timestamp", f"{ts!r} is not a real date-time")
        if not r["camera_id"]:
            bad("camera_id", "missing")
        elif not CAMERA_RE.match(r["camera_id"]):
            bad("camera_id", f"{r['camera_id']!r} is not a valid camera id")
        status, cls, conf = r["detection_status"], r["litter_class"], r["confidence_score"]
        if status not in STATUSES:
            bad("detection_status", f"{status!r} is not one of {', '.join(STATUSES)}")
        elif status == LITTER:
            if not cls:
                bad("litter_class", "required for litter_detected")
            elif allowed_classes is not None and cls not in allowed_classes:
                bad("litter_class", f"{cls!r} is not an approved class name")
            if not CONFIDENCE_RE.match(conf):
                bad("confidence_score", f"{conf!r} must be a number 0-100 with at most 2 decimals")
            elif not 0.0 <= float(conf) <= 100.0:
                bad("confidence_score", f"{conf} is outside 0-100")
        else:
            if cls:
                bad("litter_class", f"must be empty for {status} (found {cls!r})")
            if conf:
                bad("confidence_score", f"must be empty for {status} (found {conf!r})")
        if not CART_RE.match(r["cart_id"]):
            bad("cart_id", f"{r['cart_id']!r} must be a whole number")
        if not r["store_name"].strip():
            bad("store_name", "missing")
        if st is not None:
            if r["camera_id"] and r["camera_id"] not in st.camera_ids:
                bad("camera_id", f"{r['camera_id']} is not listed for store {st.key}")
            if CART_RE.match(r["cart_id"]) and int(r["cart_id"]) != st.cart_id:
                bad("cart_id", f"{r['cart_id']} does not match the store config ({st.cart_id})")
            if r["store_name"] != st.name:
                bad("store_name", f"{r['store_name']!r} does not match the store config ({st.name!r})")
        key = (r["camera_id"], ts)
        if key in seen:
            bad("timestamp", f"duplicate check of camera {key[0]} at {ts} (also line {seen[key]})")
        else:
            seen[key] = n
    return issues


def validate_file(path: Path, *, allowed_classes: Optional[set[str]] = None,
                  stores: Optional[dict[str, Store]] = None) -> list[Issue]:
    """Validate a written file, including encoding and (from its folder) the store key and date."""
    data = Path(path).read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return [Issue(1, "*", "file starts with a UTF-8 BOM (the importer expects plain UTF-8)")]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [Issue(1, "*", f"not valid UTF-8: {exc}")]
    p = Path(path).resolve()
    folder_date = p.parent.name if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.parent.name) else None
    store_key = p.parent.parent.name if folder_date else None
    return validate_text(text, allowed_classes=allowed_classes, stores=stores,
                         store_key=store_key if stores is not None else None, folder_date=folder_date)


# ============================================================================================ result bundle
BUNDLE_IMAGES = 4                 # one annotated image per screenshot of a four-screenshot job
BUNDLE_MANIFEST = "manifest.json"


def bundle_image_name(job_id: str, idx: int) -> str:
    """image_name of a four-screenshot check - the same in the bundle, the flat export and the dashboard."""
    return f"{job_id}_s{idx}_annotated.jpg"


def validate_bundle(folder: Path, *, allowed_classes: Optional[set[str]] = None,
                    stores: Optional[dict[str, Store]] = None, store_key: Optional[str] = None,
                    folder_date: Optional[str] = None, job_id: Optional[str] = None) -> list[Issue]:
    """Check a four-screenshot result bundle (backend/services/result_bundle.py): exactly 4 images, one import CSV
    whose rows pass validate_text, one row per screenshot, every image_name a file in the folder, no image without
    a row. ``store_key`` / ``folder_date`` / ``job_id`` default to the folder layout
    <export_root>/<storeKey>/<YYYY-MM-DD>/<job_id>/. Empty list = import-ready."""
    folder = Path(folder)
    job_id = job_id or folder.name
    if folder_date is None and re.fullmatch(r"\d{4}-\d{2}-\d{2}", folder.parent.name):
        folder_date = folder.parent.name
    if store_key is None and folder_date:
        store_key = folder.parent.parent.name
    issues: list[Issue] = []
    man_p = folder / BUNDLE_MANIFEST
    try:
        manifest = json.loads(man_p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [Issue(0, BUNDLE_MANIFEST, f"missing or not valid JSON: {exc}")]
    if manifest.get("job_id") != job_id:
        issues.append(Issue(0, BUNDLE_MANIFEST, f"job_id {manifest.get('job_id')!r} is not the folder's {job_id!r}"))
    images = sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(images) != BUNDLE_IMAGES:
        issues.append(Issue(0, "images", f"{len(images)} image(s) in the folder, expected exactly {BUNDLE_IMAGES}"))
    csvs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if len(csvs) != 1:
        why = (manifest.get("csv") or {}).get("reasons") or []
        issues.append(Issue(0, "csv", f"{len(csvs)} CSV file(s), expected exactly 1"
                                      + (f" (not produced: {'; '.join(why)})" if why else "")))
        return issues
    data = csvs[0].read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return issues + [Issue(1, "*", "file starts with a UTF-8 BOM (the importer expects plain UTF-8)")]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return issues + [Issue(1, "*", f"not valid UTF-8: {exc}")]
    # row dates are checked against the folder below (first screenshot), not row by row: a recording that runs past
    # midnight keeps all four checks of the job in one bundle
    issues += validate_text(text, allowed_classes=allowed_classes, stores=stores,
                            store_key=store_key if stores is not None else None)
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or rows[0] != COLUMNS:
        return issues
    body = [dict(zip(COLUMNS, r)) for r in rows[1:] if len(r) == len(COLUMNS)]
    if len(rows) - 1 != BUNDLE_IMAGES:
        issues.append(Issue(0, "csv", f"{len(rows) - 1} row(s), expected one per screenshot ({BUNDLE_IMAGES})"))
    names = [r["image_name"] for r in body]
    for n, name in enumerate(names, start=2):
        if not name:
            issues.append(Issue(n, "image_name", "missing"))
        elif name not in images:
            issues.append(Issue(n, "image_name", f"{name!r} is not a file in the bundle folder"))
    dup = sorted({x for x in names if names.count(x) > 1})
    if dup:
        issues.append(Issue(0, "image_name", f"image(s) used by more than one row: {dup}"))
    unused = sorted(set(images) - set(names))
    if unused:
        issues.append(Issue(0, "images", f"image(s) without a row: {unused}"))
    shots = [s.get("image") for s in manifest.get("screenshots") or []]
    if sorted(x for x in shots if x) != sorted(names):
        issues.append(Issue(0, BUNDLE_MANIFEST, "the manifest's screenshots do not match the CSV rows"))
    if folder_date and body and TIMESTAMP_RE.match(body[0]["timestamp"]) \
            and body[0]["timestamp"][:10] != folder_date:
        issues.append(Issue(2, "timestamp", f"first screenshot date {body[0]['timestamp'][:10]} is not the folder "
                                            f"date {folder_date}"))
    return issues


# ============================================================================================ writer + ledger
def _ledger(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    con.execute("CREATE TABLE IF NOT EXISTS exported_checks (store_key TEXT NOT NULL, camera_id TEXT NOT NULL, "
                "timestamp TEXT NOT NULL, file TEXT NOT NULL, source_id TEXT NOT NULL, exported_at TEXT NOT NULL, "
                "PRIMARY KEY (store_key, camera_id, timestamp))")
    return con


def export_checks(checks: Sequence[Check], store: Store, *, source_id: str, export_root: Path, ledger_path: Path,
                  allowed_classes: Optional[set[str]] = None, dry_run: bool = False,
                  now: Optional[dt.datetime] = None) -> dict[str, Any]:
    """Write the checks as new import CSV file(s) (one per date and camera). All rows are built and validated first:
    one bad row and nothing is written. Checks already exported (ledger) are skipped, never written twice."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", source_id):
        raise ExportError(f"source id {source_id!r} is not safe in a file name")
    rows = [build_row(c, store, allowed_classes) for c in checks]
    problems = validate_text(rows_to_csv(rows), allowed_classes=allowed_classes, stores={store.key: store},
                             store_key=store.key)
    if problems:
        raise ExportError("rows failed validation: " + "; ".join(map(str, problems[:10])))
    stamp = (now or dt.datetime.now(IST)).astimezone(IST).strftime("%Y%m%dT%H%M%S")
    con = _ledger(Path(ledger_path))
    written: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    created: list[Path] = []
    try:
        con.execute("BEGIN IMMEDIATE")
        groups: dict[tuple[str, str], list[dict[str, str]]] = {}
        for r in rows:
            hit = con.execute("SELECT file FROM exported_checks WHERE store_key=? AND camera_id=? AND timestamp=?",
                              (store.key, r["camera_id"], r["timestamp"])).fetchone()
            key = (r["timestamp"][:10], r["camera_id"])
            if hit or any(x["timestamp"] == r["timestamp"] for x in groups.get(key, [])):
                skipped.append({"camera_id": r["camera_id"], "timestamp": r["timestamp"],
                                "already_in": hit[0] if hit else "this run"})
                continue
            groups.setdefault(key, []).append(r)
        for (day, cam), grp in sorted(groups.items()):
            folder = Path(export_root) / store.key / day
            path = folder / f"{store.key}_{day}_{cam}_{source_id}_{stamp}_{secrets.token_hex(2)}.csv"
            written.append({"path": str(path), "rows": len(grp), "date": day, "camera_id": cam,
                            "statuses": {s: sum(1 for r in grp if r["detection_status"] == s) for s in STATUSES}})
            if dry_run:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise ExportError(f"refusing to overwrite {path}")
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(rows_to_csv(grp), encoding="utf-8", newline="")
            os.replace(tmp, path)
            created.append(path)
            issues = validate_file(path, allowed_classes=allowed_classes, stores={store.key: store})
            if issues:
                raise ExportError(f"{path.name} failed validation after writing: " + "; ".join(map(str, issues[:10])))
            at = dt.datetime.now(IST).isoformat(timespec="seconds")
            con.executemany("INSERT INTO exported_checks VALUES (?,?,?,?,?,?)",
                            [(store.key, r["camera_id"], r["timestamp"], path.name, source_id, at) for r in grp])
        if dry_run:
            con.execute("ROLLBACK")
        else:
            con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        for p in created:                               # a file whose checks are not in the ledger must not stay
            Path(p).unlink(missing_ok=True)
        raise
    finally:
        con.close()
    return {"schema": SCHEMA, "store_key": store.key, "source_id": source_id, "dry_run": dry_run,
            "files": written, "rows_written": sum(f["rows"] for f in written), "skipped_duplicates": skipped}


# ============================================================================================ sources
Query = Callable[[str, tuple], list[dict[str, Any]]]


def sqlite_query(db_path: Path) -> Query:
    """Read-only access to the detection database (the CLI never writes it)."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"

    def q(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        con = sqlite3.connect(uri, uri=True)
        con.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in con.execute(sql, params).fetchall()]
        finally:
            con.close()
    return q


def _known(v: Optional[str]) -> Optional[str]:
    v = (v or "").strip()
    return None if not v or v.upper() == "UNKNOWN" else v


def pick(label: str, recorded: Optional[str], given: Optional[str]) -> Optional[str]:
    """Value recorded on the analysis vs the one passed now: either may be missing; a contradiction is an error."""
    recorded, given = _known(recorded), _known(given)
    if recorded and given and recorded != given:
        raise ExportError(f"{label} {given!r} contradicts the analysis' own {label} {recorded!r}")
    return recorded or given


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _rejected_boxes(query: Query, analysis_id: Optional[str]) -> list[tuple[str, list[float]]]:
    if not analysis_id:
        return []
    return [(r["class"], [r["x1"], r["y1"], r["x2"], r["y2"]]) for r in query(
        "SELECT class, x1, y1, x2, y2 FROM litter_events WHERE analysis_id=? AND review_status='REJECTED'",
        (analysis_id,))]


@dataclass
class Source:
    source_id: str
    store_key: Optional[str]
    camera_id: Optional[str]
    original_name: Optional[str]
    video_path: Optional[Path]
    duration_sec: Optional[float]
    manual_start: Optional[str]
    analysis_id: Optional[str]


def sampled_source(query: Query, job_id: str) -> Source:
    jobs = query("SELECT * FROM sampled_jobs WHERE id=?", (job_id,))
    if not jobs:
        raise ExportError(f"sampled job {job_id} not found")
    job = jobs[0]
    if job.get("state") != "done":
        raise ExportError(f"sampled job {job_id} is not finished (state {job.get('state')})")
    media = (query("SELECT * FROM sampled_media WHERE id=?", (job["media_id"],)) or [{}])[0]
    counts = json.loads(job.get("counts_json") or "{}")
    return Source(job_id, job.get("store_id"), job.get("camera_id"), media.get("filename"),
                  Path(media["path"]) if media.get("path") else None, media.get("duration_sec"),
                  job.get("recording_start"), (counts.get("standard_outputs") or {}).get("analysis_id"))


def sampled_checks(query: Query, src: Source, camera_id: str, tb: Optional[TimeBase],
                   time_reason: str = "") -> tuple[list[Check], list[str]]:
    """One Check per screenshot that is a camera check. ``tb`` None: no recording time (dashboard rows only;
    ``captured_at`` stays None and the remark says why) - such checks are never exported."""
    rejected = _rejected_boxes(query, src.analysis_id)
    checks, excluded = [], []
    for f in query("SELECT * FROM sampled_frames WHERE job_id=? ORDER BY idx", (src.source_id,)):
        outcome = f.get("outcome")
        name = bundle_image_name(src.source_id, f["idx"])     # the annotated image's name in the result bundle
        if outcome in OUTCOME_NOT_A_CHECK:
            excluded.append(f"screenshot {f['idx']}: {OUTCOME_NOT_A_CHECK[outcome]}")
            continue
        if outcome not in OUTCOME_STATUS:
            raise ExportError(f"screenshot {f['idx']} of {src.source_id} has an unknown outcome {outcome!r}")
        if f.get("video_time_sec") is None:
            raise ExportError(f"screenshot {f['idx']} of {src.source_id} has no video time")
        status = OUTCOME_STATUS[outcome]
        res = json.loads(f.get("result_json") or "{}")
        litter, review, notes = [], [], []
        n_rej = 0
        for d in res.get("detections", []):
            if d.get("status") == "confirmed":
                if any(c == d["class"] and _iou(b, d["bbox"]) >= 0.5 for c, b in rejected):
                    n_rej += 1
                else:
                    litter.append((d["class"], float(d["confidence"])))
            elif d.get("status") == "candidate":
                review.append((d["class"], float(d["confidence"])))
        if n_rej:
            notes.append(f"{n_rej} detection{'s' if n_rej != 1 else ''} excluded: rejected by an operator")
        if status != BLOCKED:
            from backend.services.roi_service import roi_remarks
            notes.extend(roi_remarks(res.get("roi")))
        if status == LITTER and not litter:
            status = CLEAN
        if status == CLEAN and litter:
            raise ExportError(f"screenshot {f['idx']}: outcome {outcome} but confirmed detections exist")
        if status == BLOCKED:
            litter, review = [], []
        offset = float(f["video_time_sec"])
        at = tb.start + dt.timedelta(seconds=offset) if tb else None
        tnote = (f"time source: {tb.source} ({tb.detail}) + {offset:.3f} s" if tb
                 else f"time unknown: {time_reason or 'no recording start'} (video offset {offset:.3f} s)")
        checks.append(Check(camera_id, at, name, status, litter, review, reason=f.get("error") or "", notes=notes,
                            time_note=tnote, idx=f.get("idx")))
    return checks, excluded


def video_source(query: Query, analysis_id: str) -> tuple[Source, dict[str, Any]]:
    rows = query("SELECT * FROM analyses WHERE id=? AND kind='video'", (analysis_id,))
    if not rows:
        raise ExportError(f"video analysis {analysis_id} not found")
    a = rows[0]
    files = json.loads(a.get("files_json") or "{}")
    meta = json.loads(Path(files["metadata"]).read_text(encoding="utf-8")) if files.get("metadata") else {}
    shots_p = files.get("screenshots_json")
    if not shots_p or not Path(shots_p).is_file():
        raise ExportError(f"video analysis {analysis_id} has no screenshots.json (no checks to export)")
    shots = json.loads(Path(shots_p).read_text(encoding="utf-8"))
    video = meta.get("video") or {}
    stored = video.get("stored_path") or files.get("original")
    duration = (meta.get("validation") or {}).get("duration_sec") or shots.get("duration_sec")
    return Source(analysis_id, a.get("store_id"), a.get("camera_id"), video.get("original_name"),
                  Path(stored) if stored else None, duration, meta.get("recording_start"), analysis_id), shots


def read_frames(path: Path, indices: Iterable[int]) -> dict[int, Any]:
    """Decode the wanted frames by reading the video in order (seeking HEVC/H.264 can return grey-filled frames
    that would wrongly look 'blocked'). A frame that cannot be decoded maps to None."""
    import cv2
    want = sorted(set(int(i) for i in indices))
    out: dict[int, Any] = {i: None for i in want}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ExportError(f"cannot open the original video {path} to check the camera view")
    try:
        i = 0
        for target in want:
            while i < target:
                if not cap.grab():
                    return out
                i += 1
            ok, frame = cap.read()
            i += 1
            out[target] = frame if ok else None
    finally:
        cap.release()
    return out


def video_checks(query: Query, src: Source, shots: dict[str, Any], camera_id: str, tb: TimeBase,
                 frame_problem: Optional[Callable[[Any], Optional[str]]] = None,
                 frames: Optional[dict[int, Any]] = None) -> tuple[list[Check], list[str]]:
    if frame_problem is None:
        from backend.sampling.frame_quality import problem as frame_problem
    statuses = {r["event_id"]: r["review_status"] for r in query(
        "SELECT event_id, review_status FROM litter_events WHERE analysis_id=?", (src.source_id,))}
    shot_list = shots.get("screenshots") or []
    if frames is None:
        if not src.video_path or not src.video_path.is_file():
            raise ExportError(f"the original video of {src.source_id} was not kept, so the camera view of its "
                              "screenshots cannot be checked")
        frames = read_frames(src.video_path, [s["frame_idx"] for s in shot_list])
    checks: list[Check] = []
    for s in shot_list:
        offset = float(s["video_time_sec"])
        name = f"{src.source_id}/{s['file']}"
        why = frame_problem(frames.get(int(s["frame_idx"])))
        tnote = f"time source: {tb.source} ({tb.detail}) + {offset:.3f} s"
        at = tb.start + dt.timedelta(seconds=offset)
        if why:
            checks.append(Check(camera_id, at, name, BLOCKED, reason=why, time_note=tnote))
            continue
        litter, review, n_rej = [], [], 0
        for d in s.get("detections", []):
            st = d.get("detection_status")
            if st in LITTER_DETECTION_STATUSES:
                if statuses.get(d.get("event_id")) == "REJECTED":
                    n_rej += 1
                else:
                    litter.append((d["class"], float(d["confidence"])))
            elif st == "UNDER_REVIEW":
                review.append((d["class"], float(d["confidence"])))
            else:
                raise ExportError(f"{name}: unknown detection status {st!r}")
        notes = [f"{n_rej} detection{'s' if n_rej != 1 else ''} excluded: rejected by an operator"] if n_rej else []
        if not s.get("analysed_frame", True):
            notes.append("frame outside the analysis sampling: model run without temporal confirmation")
        from backend.services.roi_service import roi_remarks
        notes.extend(roi_remarks(s.get("roi")))
        checks.append(Check(camera_id, at, name, LITTER if litter else CLEAN, litter, review, notes=notes,
                            time_note=tnote))
    return checks, []


# ============================================================================================ orchestration
@dataclass
class Settings:
    store_config: Path
    export_root: Path
    ledger: Path
    trust_container_time: bool = False


def settings_from_config(cfg: Any) -> Settings:
    return Settings(cfg.store_config_path, cfg.garbage_export_dir, cfg.garbage_export_ledger,
                    cfg.trust_container_time)


def export_source(query: Query, *, kind: str, source_id: str, settings: Settings, store_key: Optional[str] = None,
                  camera_id: Optional[str] = None, recording_start: Optional[str] = None,
                  allowed_classes: Optional[set[str]] = None, dry_run: bool = False) -> dict[str, Any]:
    """Build the checks of one finished run and write them. ``kind``: sampled | video."""
    stores = load_store_config(settings.store_config)
    if kind == "sampled":
        src, shots = sampled_source(query, source_id), None
    elif kind == "video":
        src, shots = video_source(query, source_id)
    else:
        raise ExportError(f"unknown source kind {kind!r}")
    store = store_for(stores, pick("store key", src.store_key, store_key), pick("camera_id", src.camera_id, camera_id))
    cam = pick("camera_id", src.camera_id, camera_id) or ""
    manual = recording_start or src.manual_start
    tb = resolve_time_base(video_path=src.video_path, original_name=src.original_name, duration_sec=src.duration_sec,
                           manual_start=manual,
                           manual_label="recording-start argument" if recording_start else "operator-entered start",
                           trust_container_time=settings.trust_container_time)
    if kind == "sampled":
        checks, excluded = sampled_checks(query, src, cam, tb)
    else:
        checks, excluded = video_checks(query, src, shots or {}, cam, tb)
    if not checks:
        raise ExportError(f"{source_id}: no camera checks to export ({'; '.join(excluded) or 'no screenshots'})")
    out = export_checks(checks, store, source_id=source_id, export_root=settings.export_root,
                        ledger_path=settings.ledger, allowed_classes=allowed_classes, dry_run=dry_run)
    out.update(time_source=tb.source, time_detail=tb.detail, recording_start=tb.start.isoformat(),
               excluded_screenshots=excluded)
    return out


def auto_export(state: Any, *, kind: str, source_id: str, out_dir: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Hook after a finished run (LDM_GARBAGE_EXPORT=on only). Never raises: an export problem must not fail the
    analysis. The outcome is logged and written to ``out_dir/ai_garbage_export.json``."""
    cfg = state.cfg
    if not getattr(cfg, "garbage_export_enabled", False):
        return None
    try:
        classes = set(getattr(state.engine, "class_names", []) or []) or None
        res = {"ok": True, **export_source(state.db._query, kind=kind, source_id=source_id,
                                           settings=settings_from_config(cfg), allowed_classes=classes)}
    except Exception as exc:  # noqa: BLE001
        log.warning("ai_garbage_detection export of %s %s not produced: %s", kind, source_id, exc)
        res = {"ok": False, "schema": SCHEMA, "source_id": source_id, "error": str(exc)}
    if out_dir is not None:
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            Path(out_dir, "ai_garbage_export.json").write_text(json.dumps(res, indent=2, default=str),
                                                                encoding="utf-8")
        except OSError:
            log.exception("could not write the export status of %s", source_id)
    return res


# ============================================================================================ CLI
def _active_classes(query: Query) -> Optional[set[str]]:
    rows = query("SELECT classes_json FROM models WHERE active=1", ())
    return set(json.loads(rows[0]["classes_json"])) if rows and rows[0].get("classes_json") else None


def main(argv: Optional[list[str]] = None) -> int:
    from backend.core.config import AppConfig
    cfg = AppConfig()
    ap = argparse.ArgumentParser(prog="python -m backend.services.garbage_export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-config", type=Path, default=cfg.store_config_path)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-config", help="validate the store config")
    ex = sub.add_parser("export", help="export one finished run")
    g = ex.add_mutually_exclusive_group(required=True)
    g.add_argument("--sampled-job")
    g.add_argument("--video-analysis")
    ex.add_argument("--store-key")
    ex.add_argument("--camera-id")
    ex.add_argument("--recording-start", help="ISO 8601; required when no other recording-time source exists")
    ex.add_argument("--out", type=Path, default=cfg.garbage_export_dir)
    ex.add_argument("--ledger", type=Path, default=cfg.garbage_export_ledger)
    ex.add_argument("--db", type=Path, default=cfg.db_path)
    ex.add_argument("--dry-run", action="store_true")
    va = sub.add_parser("validate", help="validate import CSV file(s)")
    va.add_argument("files", nargs="+", type=Path)
    va.add_argument("--db", type=Path, default=cfg.db_path, help="active model's class names are taken from here")
    va.add_argument("--no-store-check", action="store_true", help="skip store config / folder checks")
    vb = sub.add_parser("validate-bundle", help="validate four-screenshot result bundle folder(s)")
    vb.add_argument("folders", nargs="+", type=Path)
    vb.add_argument("--db", type=Path, default=cfg.db_path, help="active model's class names are taken from here")
    vb.add_argument("--no-store-check", action="store_true", help="skip store config checks")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check-config":
            stores = load_store_config(a.store_config)
            print(f"OK: {len(stores)} store(s): " + ", ".join(f"{s.key} ({len(s.camera_ids)} camera(s), cart_id "
                                                              f"{s.cart_id})" for s in stores.values()))
            return 0
        if a.cmd == "export":
            q = sqlite_query(a.db)
            res = export_source(q, kind="sampled" if a.sampled_job else "video",
                                source_id=a.sampled_job or a.video_analysis,
                                settings=Settings(a.store_config, a.out, a.ledger, cfg.trust_container_time),
                                store_key=a.store_key, camera_id=a.camera_id, recording_start=a.recording_start,
                                allowed_classes=_active_classes(q), dry_run=a.dry_run)
            print(json.dumps(res, indent=2, default=str))
            return 0
        classes = _active_classes(sqlite_query(a.db)) if a.db and Path(a.db).is_file() else None
        stores = None if a.no_store_check else load_store_config(a.store_config)
        failed = 0
        if a.cmd == "validate-bundle":
            for d in a.folders:
                issues = validate_bundle(d, allowed_classes=classes, stores=stores)
                print(f"{'OK  ' if not issues else 'FAIL'} {d}" + ("" if not issues else f" ({len(issues)} issue(s))"))
                for i in issues:
                    print(f"     {i}")
                failed += bool(issues)
            return 1 if failed else 0
        for f in a.files:
            issues = validate_file(f, allowed_classes=classes, stores=stores)
            print(f"{'OK  ' if not issues else 'FAIL'} {f}" + ("" if not issues else f" ({len(issues)} issue(s))"))
            for i in issues:
                print(f"     {i}")
            failed += bool(issues)
        return 1 if failed else 0
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
