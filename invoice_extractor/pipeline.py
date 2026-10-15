"""Image -> InvoiceResult. One OCR engine instance is reused for every image."""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from invoice_extractor import preprocess
from invoice_extractor.fields import extract_fields
from invoice_extractor.layout import deskew, group_lines
from invoice_extractor.models import InvoiceResult, Issue, Word
from invoice_extractor.ocr import OcrEngine, create_engine
from invoice_extractor.table import TableResult, parse_table
from invoice_extractor.validate import validate_invoice

log = logging.getLogger(__name__)


@dataclass
class ExtractSettings:
    engine: str = "auto"
    low_confidence: float = 0.75     # word confidence below this => field flagged for review
    blur_threshold: float = 60.0     # Laplacian variance below this => image flagged as blurry
    retry_enhanced: bool = True      # second OCR pass with contrast/sharpening on poor results
    try_rotations: bool = True       # try 90/270 degree rotation when text looks vertical


@dataclass
class _Attempt:
    label: str
    words: list[Word]
    table: TableResult
    lines: list
    angle: float

    @property
    def score(self) -> float:
        ok = sum(1 for it in self.table.items if it.arithmetic_check == "ok")
        conf = statistics.fmean(w.conf for w in self.words) if self.words else 0.0
        bonus = 2 if self.table.method == "table" else 0
        return ok * 3 + len(self.table.items) + bonus + conf


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class InvoiceExtractor:
    def __init__(self, settings: Optional[ExtractSettings] = None, engine: Optional[OcrEngine] = None) -> None:
        self.settings = settings or ExtractSettings()
        self.engine = engine or create_engine(self.settings.engine)

    # ------------------------------------------------------------ public API
    def extract_file(self, path: Union[str, Path]) -> InvoiceResult:
        path = Path(path)
        try:
            data = path.read_bytes()
        except OSError as exc:
            return self._failed(path.name, "", f"cannot read file: {exc}")
        return self.extract_bytes(data, path.name)

    def extract_bytes(self, data: bytes, source_name: str) -> InvoiceResult:
        sha = sha256_bytes(data)
        t0 = time.perf_counter()
        try:
            img = preprocess.load_image(data)
        except preprocess.ImageLoadError as exc:
            return self._failed(source_name, sha, str(exc))
        try:
            res = self._extract(img, source_name, sha)
        except Exception as exc:   # one bad image must never stop a batch
            log.exception("extraction crashed for %s", source_name)
            return self._failed(source_name, sha, f"unexpected error: {exc}")
        res.quality["seconds"] = round(time.perf_counter() - t0, 2)
        log.info("%s: %d item(s), status=%s, %.1fs", source_name, len(res.items), res.status, res.quality["seconds"])
        return res

    # ------------------------------------------------------------ internals
    def _failed(self, name: str, sha: str, msg: str) -> InvoiceResult:
        log.error("%s: %s", name, msg)
        res = InvoiceResult(source_file=name, sha256=sha, processed_at=_now(), engine=self.engine.name,
                            issues=[Issue("error", "extraction_failed", msg)])
        res.refresh_status()
        return res

    def _attempt(self, label: str, img: np.ndarray) -> _Attempt:
        words = self.engine.read(img)
        words, angle = deskew(words)
        lines = group_lines(words)
        return _Attempt(label, words, parse_table(lines, self.settings.low_confidence), lines, angle)

    def _extract(self, original: np.ndarray, name: str, sha: str) -> InvoiceResult:
        s = self.settings
        blur = preprocess.blur_score(original)
        img = preprocess.normalize_size(original)
        attempts = [self._attempt("original", img)]
        best = attempts[0]

        def poor(a: _Attempt) -> bool:
            return (not a.table.items or a.table.method != "table"
                    or any(it.arithmetic_check == "mismatch" for it in a.table.items))

        if s.retry_enhanced and (poor(best) or blur < s.blur_threshold):
            attempts.append(self._attempt("enhanced", preprocess.enhance(img)))
        if s.try_rotations and not any(a.table.items for a in attempts) and _looks_vertical(best.words):
            for cw in (True, False):
                attempts.append(self._attempt(f"rotated_{'cw' if cw else 'ccw'}", preprocess.rotate90(img, cw)))
        best = max(attempts, key=lambda a: a.score)
        if len(attempts) > 1:
            log.info("%s: OCR passes %s -> using '%s'", name,
                     ", ".join(f"{a.label}={a.score:.1f}" for a in attempts), best.label)

        res = InvoiceResult(source_file=name, sha256=sha, processed_at=_now(), engine=self.engine.name)
        if not best.words:
            res.issues.append(Issue("error", "no_text", "OCR found no text in the image"))
        f = extract_fields(best.lines, best.table.top, best.table.bottom)
        res.invoice_number, res.invoice_date = f.invoice_number, f.invoice_date
        res.vendor_name, res.vendor_gstin = f.vendor_name, f.vendor_gstin
        res.subtotal, res.invoice_total = f.subtotal, f.invoice_total
        res.issues.extend(f.issues)
        res.issues.extend(best.table.issues)
        res.items = best.table.items
        res.quality = {
            "blur_score": round(blur, 1),
            "mean_confidence": round(statistics.fmean(w.conf for w in best.words), 3) if best.words else None,
            "words": len(best.words),
            "image_size": [int(original.shape[1]), int(original.shape[0])],
            "ocr_pass": best.label,
            "deskew_degrees": round(best.angle, 2),
            "table_method": best.table.method,
            "columns": best.table.columns,
        }
        if best.label != "original":
            res.issues.append(Issue("info", "ocr_retry", f"results taken from the '{best.label}' OCR pass"))
        validate_invoice(res, s.blur_threshold, s.low_confidence)
        res.refresh_status()
        return res


def _looks_vertical(words: list[Word]) -> bool:
    multi = [w for w in words if len(w.text) >= 3]
    return bool(multi) and sum(w.h > 1.5 * w.w for w in multi) > 0.5 * len(multi)


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")
