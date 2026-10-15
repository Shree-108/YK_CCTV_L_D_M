"""Local invoice image -> CSV extraction (OCR + layout analysis, no cloud services).

Extraction is best-effort: OCR and layout heuristics can misread values, so every row
carries ``needs_review`` / ``review_reasons`` and must be checked when flagged.
"""
from invoice_extractor.models import InvoiceResult, Issue, LineItem
from invoice_extractor.pipeline import ExtractSettings, InvoiceExtractor

__all__ = ["ExtractSettings", "InvoiceExtractor", "InvoiceResult", "Issue", "LineItem"]
__version__ = "1.0.0"
