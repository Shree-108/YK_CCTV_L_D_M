"""Duplicate-invoice detection.

* Same file bytes (SHA-256) are never processed twice (handled by the output store).
* A *different* file that carries the same invoice number from the same vendor (or with the same
  date when the vendor is unreadable) is marked ``duplicate`` — e.g. a re-photographed invoice.
* Products are never merged: two identical rows on one invoice are two legitimate line items.
"""
from __future__ import annotations

import re

from invoice_extractor.models import InvoiceResult, Issue


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def invoice_key(r: InvoiceResult) -> tuple[str, str] | None:
    inv = _norm(r.invoice_number)
    if not inv:
        return None
    other = _norm(r.vendor_gstin) or _norm(r.vendor_name) or _norm(r.invoice_date)
    return (inv, other) if other else None


def mark_duplicates(results: list[InvoiceResult]) -> list[InvoiceResult]:
    """Mark later results whose invoice key matches an earlier one. ``results`` must be in
    processing order. Returns the list of results marked as duplicates."""
    first: dict[tuple[str, str], InvoiceResult] = {}
    dups = []
    for r in results:
        r.duplicate_of = ""
        r.issues = [i for i in r.issues if i.issue != "duplicate_invoice"]
        if r.status == "failed":
            r.refresh_status()
            continue
        key = invoice_key(r)
        if key is None:
            r.refresh_status()
            continue
        if key in first and first[key].sha256 != r.sha256:
            orig = first[key]
            r.duplicate_of = orig.source_file
            r.issues.append(Issue("review", "duplicate_invoice",
                                  f"possible duplicate: invoice {r.invoice_number} was already extracted from "
                                  f"'{orig.source_file}' (rows left out of products.csv unless duplicates are kept)"))
            dups.append(r)
        else:
            first.setdefault(key, r)
        r.refresh_status()
    return dups
