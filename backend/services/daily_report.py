"""One consolidated six-column litter report per day (owner brief 2026-11-20).

    storage/analysis_outputs/daily_reports/litter_report_<YYYY-MM-DD>.csv

The first video analysed on a date creates the file (header + its rows). Every later video that day reads the same
file back, appends its rows and saves it again (temp file + rename, so a crash never leaves a half-written report).
By the end of the day the file holds the rows of every video analysed that day, in the order they finished.

The date is the PROCESSING date (the PC's local date when the analysis finished), not the recording date: the
``timestamp`` column inside the rows is still the recording time when a recording start was given.

The columns are exactly video_evidence.REPORT_COLUMNS. Rows are a snapshot at analysis time; an operator's later
VERIFIED / REJECTED review is reflected in the per-analysis download (/api/analyses/{id}/files/report_csv), not here.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

from backend.core.csv_safety import safe_rows
from backend.services.video_evidence import EVIDENCE_DIR, REPORT_COLUMNS

DAILY_DIR = "daily_reports"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LOCK = threading.Lock()          # appends are read-modify-write: one at a time per process


def daily_dir(storage_dir: Path) -> Path:
    return Path(storage_dir) / EVIDENCE_DIR / DAILY_DIR


def path_for(storage_dir: Path, day: dt.date | str) -> Path:
    d = day.isoformat() if isinstance(day, dt.date) else str(day)
    if not DATE_RE.match(d):
        raise ValueError(f"not a YYYY-MM-DD date: {d!r}")
    dt.date.fromisoformat(d)                                     # rejects 2026-13-40
    return daily_dir(storage_dir) / f"litter_report_{d}.csv"


def _rows_text(rows: Sequence[dict[str, Any]], header: bool) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=REPORT_COLUMNS, extrasaction="raise", lineterminator="\n")
    if header:
        w.writeheader()
    w.writerows(safe_rows(rows))
    return buf.getvalue()


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8-sig")


def append(storage_dir: Path, day: dt.date, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fetch the day's CSV (or start it), append ``rows`` and save it back. Returns date, path, rows appended and
    the day's total. An existing file whose header is not the six columns is never overwritten (ValueError)."""
    path = path_for(storage_dir, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        if path.is_file():
            existing = read_text(path)
            head = next(csv.reader(io.StringIO(existing)), None)
            if head != REPORT_COLUMNS:
                raise ValueError(f"{path.name} does not have the six report columns (found {head}); "
                                 "it was not modified")
            if existing and not existing.endswith("\n"):
                existing += "\n"
            text = existing + _rows_text(rows, header=False)
        else:
            text = _rows_text(rows, header=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8-sig", newline="")
        os.replace(tmp, path)
        total = len(list(csv.DictReader(io.StringIO(text))))
    return {"date": day.isoformat(), "path": str(path), "rows_appended": len(rows), "total_rows": total,
            "url": f"/api/reports/daily/{day.isoformat()}.csv"}


def list_days(storage_dir: Path) -> list[dict[str, Any]]:
    """Every daily report on disk, newest first, with its row count."""
    out = []
    d = daily_dir(storage_dir)
    for p in sorted(d.glob("litter_report_*.csv"), reverse=True) if d.is_dir() else []:
        day = p.stem.removeprefix("litter_report_")
        if not DATE_RE.match(day):
            continue
        rows = len(list(csv.DictReader(io.StringIO(read_text(p)))))
        out.append({"date": day, "rows": rows, "size_bytes": p.stat().st_size,
                    "url": f"/api/reports/daily/{day}.csv"})
    return out


def find(storage_dir: Path, day: str) -> Optional[Path]:
    try:
        p = path_for(storage_dir, day)
    except ValueError:
        return None
    return p if p.is_file() else None
