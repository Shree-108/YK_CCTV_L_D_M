"""Dashboard import rows of a four-screenshot Video Analysis job (owner brief 2026-11-24).

``build_dashboard_rows(job, ...)`` is the ONE place these rows are made; the Video Analysis table and both CSV
downloads (GET /api/detect/video/{job_id}/report?format=table|csv|csv_full) all use it. Detection is not touched:
rows are built from the detections the pipeline already stored for each screenshot (after its filters; ROI was
removed from the application on 2026-11-19). The older reports are unchanged, including the 45-column
reports/detection_report.csv ("Detailed CSV") and the strict ai_garbage_detection exporter in garbage_export.py.

Importer CSV (``csv``):   timestamp,camera_id,image_name,litter_class,confidence_score,detection_status,remark
Full CSV (``csv_full``):  the same + store_name,source_file
UTF-8 without BOM, RFC 4180 quoting, CRLF. Empty cells for nulls (never "-", "—" or "null").

Rows
  * default: one row per screenshot. ``per_object``: one row per listed object (confirmed litter or possible
    litter) of each screenshot; a screenshot with no listed object (or a blocked view) keeps one screenshot row.
  * detection_status: exactly clean | litter_detected | view_blocked.
      litter_detected  >= 1 CONFIRMED detection not rejected by an operator
      view_blocked     otherwise, when the screenshot shows no usable view: unreadable / not analysed, or
                       frame_quality.view_blocked_reason() (blank or grey-fill frame, mostly uniform, too dark, too
                       bright, blurred; thresholds in settings ``view_quality``). The reason is in remark.
      clean            everything else. Possible litter (candidate) never makes a row litter_detected.
    view_blocked is decided per screenshot only. Per-object mode: a CONFIRMED_LITTER row is litter_detected with
    its class and confidence; a POSSIBLE_LITTER row is always clean, class/confidence empty, remark
    "possible: <class> <conf>%, review required"; a blocked screenshot keeps its one view_blocked row.
    Remark (per screenshot): a screenshot with confirmed litter lists its possible detections ("N possible, review
    required: <class> <conf>%, ..."); a screenshot with only possible detections says "N possible, review required".
  * litter_class / confidence_score: the primary detection = highest-confidence CONFIRMED detection (per-object
    rows: that object). Empty when the status is clean or view_blocked. confidence = model 0-1 x 100, 1 decimal.
  * timestamp = recording start + screenshot offset, Asia/Kolkata, "YYYY-MM-DD HH:MM:SS" (the repo has no sample
    import file to copy a format from). Recording start, in order: (a) the "Recording start" typed on the upload
    form, (b) the video's container creation_time (ffprobe when installed, else the MP4 mvhd box), (c) a date/time
    in the file name, (d) the upload time -> remark "capturedAt estimated".
  * camera_id: the camera chosen on the upload form, else "VIDEO-UPLOAD".
  * image_name: the annotated screenshot's name, identical to the downloaded JPG (garbage_export.bundle_image_name).
  * store_name: the store of the camera (cameras.store_id), else the store picked on the form; empty if unknown.
  * source_file: the original uploaded video file name.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2

from backend.sampling import frame_quality
from backend.sampling.plan import recording_start_from_name
from backend.services import garbage_export as ge

Query = Callable[[str, tuple], list[dict[str, Any]]]

HEADER = ["timestamp", "camera_id", "image_name", "litter_class", "confidence_score", "detection_status", "remark"]
FULL_HEADER = HEADER + ["store_name", "source_file"]
CLEAN, LITTER, BLOCKED = "clean", "litter_detected", "view_blocked"
STATUSES = (CLEAN, LITTER, BLOCKED)
FALLBACK_CAMERA = "VIDEO-UPLOAD"
ESTIMATED = "capturedAt estimated"
TS_FORMAT = "%Y-%m-%d %H:%M:%S"
REMARK_MAX = 200
UNKNOWN = {"", "unknown", "upload"}


# ---------------------------------------------------------------------------------------------- time
@dataclass(frozen=True)
class RecordingStart:
    start: dt.datetime              # aware, Asia/Kolkata
    source: str                     # form | creation_time | filename | upload_time
    estimated: bool


def ffprobe_creation_time(path: Optional[Path]) -> Optional[dt.datetime]:
    """``format_tags=creation_time`` via ffprobe when it is installed; otherwise the same tag read directly from the
    MP4/MOV mvhd box (garbage_export.container_creation_time). None when absent."""
    if not path or not Path(path).is_file():
        return None
    exe = shutil.which("ffprobe")
    if exe:
        try:
            out = subprocess.run([exe, "-v", "error", "-show_entries", "format_tags=creation_time",
                                  "-of", "default=nw=1:nk=1", str(path)],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return dt.datetime.fromisoformat(out.replace("Z", "+00:00"))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return None
    return ge.container_creation_time(Path(path))


def recording_start(job: dict[str, Any], media: dict[str, Any]) -> RecordingStart:
    """(a) form input -> (b) creation_time -> (c) file name -> (d) upload time (estimated)."""
    if job.get("recording_start"):
        try:
            return RecordingStart(ge.to_ist(dt.datetime.fromisoformat(str(job["recording_start"]).strip())), "form",
                                  False)
        except ValueError:
            pass
    path = Path(media["path"]) if media.get("path") else None
    created = ffprobe_creation_time(path)
    if created is not None:
        return RecordingStart(ge.to_ist(created), "creation_time", False)
    name = media.get("filename") or ""
    start, _why = ge.filename_start(name, media.get("duration_sec"))   # NVR <day>_<start>_<end>, span checked
    if start is None and (s := recording_start_from_name(name)):
        start = dt.datetime.fromisoformat(s)
    if start is not None:
        return RecordingStart(ge.to_ist(start), "filename", False)
    uploaded = media.get("created_at") or job.get("created_at")
    return RecordingStart(ge.to_ist(dt.datetime.fromisoformat(uploaded)), "upload_time", True)


def format_ts(t: dt.datetime) -> str:
    return t.astimezone(ge.IST).strftime(TS_FORMAT)


# ---------------------------------------------------------------------------------------------- helpers
def percent(conf: float) -> str:
    v = float(conf)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"confidence {conf!r} is not on the model's 0-1 scale")
    return f"{v * 100:.1f}"


def _known(v: Optional[str]) -> str:
    v = (v or "").strip()
    return "" if v.lower() in UNKNOWN else v


def _store_name(query: Query, job: dict[str, Any], camera_id: str) -> str:
    store_id = ""
    if camera_id:
        cam = query("SELECT store_id FROM cameras WHERE id=?", (camera_id,))
        store_id = _known((cam[0].get("store_id") if cam else "") or "")
    store_id = store_id or _known(job.get("store_id"))
    if not store_id:
        return ""
    st = query("SELECT name FROM stores WHERE id=?", (store_id,))
    return (st[0].get("name") if st else None) or store_id


def _fmt_objs(items: list[tuple[str, float]]) -> str:
    return ", ".join(f"{c} {percent(p)}%" for c, p in items)


def _remark(parts: list[str], others: list[tuple[str, float]] = (), possible: list[tuple[str, float]] = (),
            list_possible: bool = False) -> str:
    """Remark text, parts joined with "; ":
        also <other confirmed detections>; N possible, review required[: <possible detections>]; <parts>
    The two detection lists are shortened ("+k more"), the longer first, so the text stays within about REMARK_MAX
    characters; the other parts are short and always kept."""
    def lst(items: list[tuple[str, float]], n: int) -> str:
        return _fmt_objs(items[:n]) + (f" +{len(items) - n} more" if n < len(items) else "")

    def text(n_o: int, n_p: int) -> str:
        bits = []
        if others:
            bits.append(f"also {lst(others, n_o)}")
        if possible:
            bits.append(f"{len(possible)} possible, review required"
                        + (f": {lst(possible, n_p)}" if list_possible else ""))
        return ge.one_line("; ".join(bits + list(parts)))

    n_o, n_p = len(others), len(possible)
    out = text(n_o, n_p)
    while len(out) > REMARK_MAX and (n_o > 1 or (list_possible and n_p > 1)):
        if list_possible and n_p > 1 and (n_p >= n_o or n_o <= 1):
            n_p -= 1
        else:
            n_o -= 1
        out = text(n_o, n_p)
    return out


def _screen_frame(path: Optional[str]) -> Any:
    if not path or not Path(path).is_file():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------------------------- rows
def build_dashboard_rows(job: dict[str, Any], *, query: Query, view_quality: Any,
                         per_object: bool = False) -> list[dict[str, Any]]:
    """Rows of one finished four-screenshot job (a ``sampled_jobs`` row). Each row has the FULL_HEADER keys (all
    strings) plus ``job_id``, ``screenshot`` (1-4), ``row_kind`` (screenshot | object) and ``image_url``."""
    media = (query("SELECT * FROM sampled_media WHERE id=?", (job["media_id"],)) or [{}])[0]
    counts = json.loads(job.get("counts_json") or "{}")
    rejected = ge._rejected_boxes(query, (counts.get("standard_outputs") or {}).get("analysis_id"))
    rs = recording_start(job, media)
    camera_id = _known(job.get("camera_id"))
    store_name = _store_name(query, job, camera_id)
    common = {"camera_id": camera_id or FALLBACK_CAMERA, "store_name": store_name,
              "source_file": media.get("filename") or ""}
    rows: list[dict[str, Any]] = []
    for f in query("SELECT * FROM sampled_frames WHERE job_id=? ORDER BY idx", (job["id"],)):
        idx = f["idx"]
        res = json.loads(f.get("result_json") or "{}")
        offset = float(f.get("video_time_sec") or 0.0)
        base = {**common, "job_id": job["id"], "screenshot": idx,
                "timestamp": format_ts(rs.start + dt.timedelta(seconds=offset)),
                "image_name": ge.bundle_image_name(job["id"], idx),
                "image_url": f"/api/sampled/jobs/{job['id']}/frames/{idx}/annotated.jpg?download=1"
                if f.get("annotated_path") else ""}
        confirmed: list[tuple[str, float]] = []
        possible: list[tuple[str, float]] = []
        n_rejected = 0
        for d in res.get("detections", []):
            if d.get("status") == "confirmed":
                if any(c == d["class"] and ge._iou(b, d["bbox"]) >= 0.5 for c, b in rejected):
                    n_rejected += 1
                else:
                    confirmed.append((d["class"], float(d["confidence"])))
            elif d.get("status") == "candidate":
                possible.append((d["class"], float(d["confidence"])))
        confirmed.sort(key=lambda x: -x[1])
        possible.sort(key=lambda x: -x[1])

        # view check: analysed screenshots are measured; unreadable / failed ones are blocked by definition
        outcome = f.get("outcome")
        if outcome in ("LITTER_DETECTED", "NO_LITTER"):
            reason = frame_quality.view_blocked_reason(_screen_frame(f.get("original_path")), view_quality) \
                if f.get("original_path") and Path(f["original_path"]).is_file() else None
        elif outcome == "UNREADABLE":
            reason = f"unreadable: {f.get('error') or 'frame not usable'}"
        else:
            reason = f"not analysed ({(outcome or f.get('status') or 'unknown').lower()}" \
                     f"{': ' + f['error'] if f.get('error') else ''})"
            confirmed, possible = [], []
        if outcome == "UNREADABLE":
            confirmed, possible = [], []
        status = LITTER if confirmed else BLOCKED if reason else CLEAN

        tail: list[str] = []                       # parts every row of this screenshot carries
        if n_rejected:
            tail.append(f"{n_rejected} rejected by operator")
        tail.append(f"t={offset:.1f}s")
        if reason:
            tail.append(f"view blocked: {reason}" if status == BLOCKED else f"view quality: {reason}")
        if rs.estimated:
            tail.append(ESTIMATED)

        def row(status_: str, cls: str, conf: str, kind: str, remark: str) -> dict[str, Any]:
            return {**base, "row_kind": kind, "detection_status": status_, "litter_class": cls,
                    "confidence_score": conf, "remark": remark}

        if not per_object or status == BLOCKED or not (confirmed or possible):
            # per screenshot: confirmed litter lists its possible detections; otherwise only their count
            if status == LITTER:
                top = confirmed[0]
                rows.append(row(LITTER, top[0], percent(top[1]), "screenshot",
                                _remark(tail, confirmed[1:], possible, list_possible=True)))
            else:
                rows.append(row(status, "", "", "screenshot", _remark(tail, possible=possible)))
            continue
        # per object: CONFIRMED_LITTER -> litter_detected; POSSIBLE_LITTER -> always clean (class/confidence empty)
        n_obj = len(confirmed) + len(possible)
        for k, (cls, conf) in enumerate(confirmed, start=1):
            rows.append(row(LITTER, cls, percent(conf), "object", _remark([f"object {k}/{n_obj}", *tail])))
        for k, (cls, conf) in enumerate(possible, start=len(confirmed) + 1):
            rows.append(row(CLEAN, "", "", "object",
                            _remark([f"possible: {cls} {percent(conf)}%, review required", f"object {k}/{n_obj}",
                                     *tail])))
    return rows


# ---------------------------------------------------------------------------------------------- CSV
def to_csv(rows: list[dict[str, Any]], *, full: bool = False) -> str:
    """UTF-8 (no BOM is added; encode with "utf-8"), RFC 4180 quoting, CRLF. Cells that could start a spreadsheet
    formula are neutralised by garbage_export's rules (ids and names are validated upstream)."""
    fields = FULL_HEADER if full else HEADER
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL,
                       lineterminator="\r\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _safe(r.get(k)) for k in fields})
    return buf.getvalue()


def _safe(v: Any) -> str:
    s = "" if v is None else str(v)
    return "'" + s if s.startswith(ge.FORMULA_START) and not _is_number(s) else s


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = {s: sum(1 for r in rows if r["detection_status"] == s) for s in STATUSES}
    return {"columns": FULL_HEADER, "rows": rows, "total": len(rows), "summary": by_status}
