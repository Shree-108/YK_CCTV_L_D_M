"""CSV output (UTF-8 with BOM so Excel shows ₹ and non-English text correctly)."""
from __future__ import annotations

import csv
import datetime as dt
import io
import logging
from pathlib import Path

from invoice_extractor.models import ITEM_FIELDS, InvoiceResult

log = logging.getLogger(__name__)

PRODUCT_COLUMNS = (["source_file", "invoice_number", "invoice_date", "vendor_name", "vendor_gstin", "line_no"]
                   + ITEM_FIELDS + ["arithmetic_check", "ocr_confidence", "needs_review", "review_reasons",
                                    "extraction_method", "raw_text", "processed_at"])
ISSUE_COLUMNS = ["source_file", "invoice_number", "line_no", "severity", "field", "issue", "detail", "invoice_status"]
SUMMARY_COLUMNS = ["source_file", "status", "invoice_number", "invoice_date", "vendor_name", "vendor_gstin",
                   "line_items", "items_needing_review", "sum_line_totals", "subtotal_printed", "invoice_total_printed",
                   "blur_score", "mean_ocr_confidence", "ocr_engine", "ocr_pass", "table_method", "duplicate_of",
                   "processed_at", "sha256"]
_TEXT_COLUMNS = {"source_file", "invoice_number", "vendor_name", "product_name", "sku", "unit", "review_reasons",
                 "raw_text", "detail", "duplicate_of"}


def _safe(col: str, value) -> str:
    """Neutralise spreadsheet formula injection in free-text cells (=, +, @ prefixes)."""
    s = "" if value is None else str(value)
    if col in _TEXT_COLUMNS and s[:1] in ("=", "+", "@", "\t", "\r"):
        return "'" + s
    return s


def product_rows(results: list[InvoiceResult], keep_duplicates: bool = False) -> list[dict]:
    rows = []
    for r in results:
        if r.status == "duplicate" and not keep_duplicates:
            continue
        inv_reasons = [f"invoice: {i.detail}" for i in r.issues if i.severity in ("error", "review")]
        for it in r.items:
            reasons = [i.detail for i in it.issues if i.severity in ("error", "review")] + inv_reasons
            row = {"source_file": r.source_file, "invoice_number": r.invoice_number, "invoice_date": r.invoice_date,
                   "vendor_name": r.vendor_name, "vendor_gstin": r.vendor_gstin, "line_no": it.line_no,
                   **{k: it.values.get(k, "") for k in ITEM_FIELDS},
                   "arithmetic_check": it.arithmetic_check,
                   "ocr_confidence": "" if it.ocr_confidence is None else f"{it.ocr_confidence:.2f}",
                   "needs_review": "YES" if reasons else "NO",
                   "review_reasons": "; ".join(reasons),
                   "extraction_method": it.method, "raw_text": it.raw_text, "processed_at": r.processed_at}
            rows.append(row)
    return rows


def issue_rows(results: list[InvoiceResult], include_info: bool = False) -> list[dict]:
    rows = []
    for r in results:
        for i in r.all_issues():
            if i.severity == "info" and not include_info:
                continue
            rows.append({"source_file": r.source_file, "invoice_number": r.invoice_number,
                         "line_no": "" if i.line_no is None else i.line_no, "severity": i.severity,
                         "field": i.field, "issue": i.issue, "detail": i.detail, "invoice_status": r.status})
    order = {"error": 0, "review": 1, "warning": 2, "info": 3}
    rows.sort(key=lambda x: (x["source_file"], order.get(x["severity"], 9), str(x["line_no"])))
    return rows


def summary_rows(results: list[InvoiceResult]) -> list[dict]:
    out = []
    for r in results:
        q = r.quality or {}
        out.append({"source_file": r.source_file, "status": r.status, "invoice_number": r.invoice_number,
                    "invoice_date": r.invoice_date, "vendor_name": r.vendor_name, "vendor_gstin": r.vendor_gstin,
                    "line_items": len(r.items), "items_needing_review": sum(it.needs_review for it in r.items),
                    "sum_line_totals": q.get("sum_line_totals", ""), "subtotal_printed": r.subtotal,
                    "invoice_total_printed": r.invoice_total, "blur_score": q.get("blur_score", ""),
                    "mean_ocr_confidence": q.get("mean_confidence", ""), "ocr_engine": r.engine,
                    "ocr_pass": q.get("ocr_pass", ""), "table_method": q.get("table_method", ""),
                    "duplicate_of": r.duplicate_of, "processed_at": r.processed_at, "sha256": r.sha256})
    return out


def to_csv_text(columns: list[str], rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\r\n")
    w.writeheader()
    for row in rows:
        w.writerow({c: _safe(c, row.get(c, "")) for c in columns})
    return buf.getvalue()


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> Path:
    """Write atomically. If the file is locked (open in Excel), write a timestamped copy instead."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = to_csv_text(columns, rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8-sig", newline="")
    try:
        tmp.replace(path)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{dt.datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        tmp.replace(alt)
        log.warning("%s is open in another program (Excel?); wrote %s instead", path.name, alt.name)
        return alt
