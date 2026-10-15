"""Invoice image -> product CSV extraction (local OCR via the invoice_extractor package).

Uploads are processed in memory and not stored; the response carries the extracted rows and the
CSV text so the browser can offer products.csv / review report downloads.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.api.deps import safe_display_name

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/invoices", tags=["invoices"])

ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}
MAX_FILES = 25
MAX_MB = int(os.environ.get("INVOICE_MAX_MB", "15"))

_extractor = None
_extractor_error: Optional[str] = None
_lock = threading.Lock()


def _get_extractor():
    """Create the OCR engine once (loading models takes a few seconds)."""
    global _extractor, _extractor_error
    with _lock:
        if _extractor is None:
            try:
                from invoice_extractor import ExtractSettings, InvoiceExtractor
                _extractor = InvoiceExtractor(ExtractSettings(engine=os.environ.get("INVOICE_OCR_ENGINE", "auto")))
                _extractor_error = None
            except Exception as exc:   # OCR engine missing / failed to load
                _extractor_error = str(exc)
                log.error("invoice OCR unavailable: %s", exc)
                raise HTTPException(503, f"Invoice OCR engine unavailable: {exc}. "
                                         "Install it with: pip install -r invoice_extractor/requirements.txt") from exc
        return _extractor


@router.get("/engine")
def engine_status() -> dict[str, Any]:
    if _extractor is not None:
        return {"available": True, "engine": _extractor.engine.name, "loaded": True}
    try:
        import importlib.util
        if importlib.util.find_spec("rapidocr") and importlib.util.find_spec("onnxruntime"):
            return {"available": True, "engine": "rapidocr", "loaded": False}
        from invoice_extractor.ocr import find_tesseract
        if importlib.util.find_spec("pytesseract") and find_tesseract():
            return {"available": True, "engine": "tesseract", "loaded": False}
    except Exception as exc:
        return {"available": False, "engine": None, "loaded": False, "error": str(exc)}
    return {"available": False, "engine": None, "loaded": False,
            "error": _extractor_error or "No OCR engine installed (pip install -r invoice_extractor/requirements.txt)"}


@router.post("/extract")
async def extract(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    from invoice_extractor.dedupe import mark_duplicates
    from invoice_extractor.pipeline import sha256_bytes
    from invoice_extractor.writer import (ISSUE_COLUMNS, PRODUCT_COLUMNS, SUMMARY_COLUMNS, issue_rows,
                                          product_rows, summary_rows, to_csv_text)

    if not files:
        raise HTTPException(422, "No files uploaded")
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"Too many files (max {MAX_FILES} per request)")
    payloads: list[tuple[str, bytes]] = []
    for f in files:
        name = safe_display_name(f.filename)
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(415, f"'{name}': unsupported type '{ext or '?'}'. Allowed: JPG, JPEG, PNG")
        data = await f.read(MAX_MB * 1024 * 1024 + 1)
        if len(data) > MAX_MB * 1024 * 1024:
            raise HTTPException(413, f"'{name}' is larger than {MAX_MB} MB")
        if not data:
            raise HTTPException(422, f"'{name}' is empty")
        payloads.append((name, data))

    extractor = await asyncio.to_thread(_get_extractor)
    results, seen = [], {}
    for name, data in payloads:
        sha = sha256_bytes(data)
        if sha in seen:   # identical file uploaded twice in one batch
            log.info("%s is byte-identical to %s; skipped", name, seen[sha])
            continue
        seen[sha] = name
        results.append(await asyncio.to_thread(extractor.extract_bytes, data, name))
    mark_duplicates(results)

    skipped = [n for n, d in payloads if seen.get(sha256_bytes(d)) != n]
    products = product_rows(results)
    return {
        "engine": extractor.engine.name,
        "invoices": [_invoice_out(r) for r in results],
        "skipped_identical": skipped,
        "totals": {
            "invoices": len(results),
            "rows": len(products),
            "rows_needing_review": sum(r["needs_review"] == "YES" for r in products),
            "failed": sum(r.status == "failed" for r in results),
            "duplicates": sum(r.status == "duplicate" for r in results),
        },
        "products": products,
        "csv": {
            "products": to_csv_text(PRODUCT_COLUMNS, products),
            "review": to_csv_text(ISSUE_COLUMNS, issue_rows(results)),
            "summary": to_csv_text(SUMMARY_COLUMNS, summary_rows(results)),
        },
        "disclaimer": "OCR extraction is not 100% accurate. Verify every row marked needs_review against the image.",
    }


def _invoice_out(r) -> dict[str, Any]:
    d = r.to_dict()
    d["needs_review"] = r.needs_review
    d["issues"] = [i.to_dict() for i in r.all_issues()]
    for it in d["items"]:
        it.pop("y0", None)
        it.pop("y1", None)
    return d
