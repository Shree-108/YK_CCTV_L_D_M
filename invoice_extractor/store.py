"""Output folder state: one cached JSON result per processed image (keyed by SHA-256).

The CSV reports are always regenerated from the cache, so re-running never duplicates rows and
a re-scanned image replaces the earlier result for the same filename.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from invoice_extractor.dedupe import mark_duplicates
from invoice_extractor.models import InvoiceResult
from invoice_extractor.writer import (ISSUE_COLUMNS, PRODUCT_COLUMNS, SUMMARY_COLUMNS, issue_rows, product_rows,
                                      summary_rows, write_csv)

log = logging.getLogger(__name__)


class OutputStore:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.cache_dir = self.out_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sha: str) -> Path:
        return self.cache_dir / f"{sha}.json"

    def has(self, sha: str) -> bool:
        return self._path(sha).is_file()

    def get(self, sha: str) -> InvoiceResult | None:
        try:
            return InvoiceResult.from_dict(json.loads(self._path(sha).read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, res: InvoiceResult) -> None:
        if not res.sha256:
            return
        # a new version of the same file name supersedes the old cached result
        for old in self.load_all():
            if old.source_file == res.source_file and old.sha256 != res.sha256:
                log.info("%s changed since last run; replacing previous result", res.source_file)
                self._path(old.sha256).unlink(missing_ok=True)
        tmp = self._path(res.sha256).with_suffix(".tmp")
        tmp.write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self._path(res.sha256))

    def load_all(self) -> list[InvoiceResult]:
        out = []
        for p in self.cache_dir.glob("*.json"):
            try:
                out.append(InvoiceResult.from_dict(json.loads(p.read_text(encoding="utf-8"))))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                log.warning("ignoring unreadable cache file %s: %s", p.name, exc)
        out.sort(key=lambda r: (r.processed_at, r.source_file))
        return out

    def write_reports(self, keep_duplicates: bool = False) -> dict[str, Path]:
        results = self.load_all()
        mark_duplicates(results)
        return {
            "products": write_csv(self.out_dir / "products.csv", PRODUCT_COLUMNS, product_rows(results, keep_duplicates)),
            "review": write_csv(self.out_dir / "extraction_errors.csv", ISSUE_COLUMNS, issue_rows(results)),
            "summary": write_csv(self.out_dir / "invoices_summary.csv", SUMMARY_COLUMNS, summary_rows(results)),
        }
