"""Invoice-level fields: invoice number, date, vendor, GSTIN and printed totals.

Values are copied as printed (dates are NOT reformatted: 03/04/2024 is ambiguous between
day-first and month-first). Anything inferred without an explicit label is flagged for review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from invoice_extractor.models import Issue, Line
from invoice_extractor.numbers import fmt, parse_number

_INV_LABEL = r"(?:tax\s+|proforma\s+)?(?:invoice|inv|bill|receipt|voucher|memo)"
_INV_VALUE = r"([A-Za-z0-9][A-Za-z0-9\-/_.]{0,29})"
INV_RES = [
    re.compile(rf"\b{_INV_LABEL}\s*(?:no|number|num|nbr|#)\b\.?\s*[:#\-]?\s*{_INV_VALUE}", re.I),
    re.compile(rf"\b{_INV_LABEL}\s*#\s*{_INV_VALUE}", re.I),
    re.compile(rf"\b{_INV_LABEL}\s*[:\-]\s*{_INV_VALUE}", re.I),
]
INV_LABEL_ONLY = re.compile(rf"\b{_INV_LABEL}\s*(?:no|number|num|#)\b\.?\s*[:#\-]?\s*$", re.I)

_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_RE = re.compile(
    rf"\b(\d{{1,2}}[/\-.]\d{{1,2}}[/\-.](?:\d{{4}}|\d{{2}})"
    rf"|\d{{4}}[/\-.]\d{{1,2}}[/\-.]\d{{1,2}}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?[\s\-.,]*{_MON}[\s\-.,]*(?:\d{{4}}|\d{{2}})"
    rf"|{_MON}\s*\d{{1,2}}(?:st|nd|rd|th)?,?\s*\d{{4}})(?![\d])", re.I)
DATE_LABEL = re.compile(r"\b(?:invoice\s*date|inv\.?\s*date|bill\s*date|date\s*of\s*(?:invoice|issue)|dated|date)\b", re.I)
OTHER_DATE_LABEL = re.compile(r"\b(?:due|order|po|p\.o|delivery|challan|dispatch|supply|ack|eway|e-way)\b", re.I)

GSTIN_RE = re.compile(r"(?<![0-9A-Z])(\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]Z[0-9A-Z])(?![0-9A-Z])")
VENDOR_LABEL = re.compile(r"^\s*(?:vendor|supplier|seller|sold\s*by|billed\s*by|from|issued\s*by)\s*(?:name)?\s*[:\-]\s*(.{3,})$", re.I)
BUYER_LABEL = re.compile(r"\b(?:bill(?:ed)?\s*to|ship(?:ped)?\s*to|buyer|customer|consignee|sold\s*to|deliver\s*to|recipient)\b", re.I)
NOT_VENDOR = re.compile(r"\b(?:tax\s*invoice|invoice|bill\s*of\s*supply|cash\s*memo|receipt|estimate|quotation|original|duplicate|"
                        r"triplicate|copy|gstin|gst|pan|phone|ph|mob|mobile|tel|email|e-mail|www|http|date|page|state|code|"
                        r"address|road|street|pin|no\.)\b", re.I)
COMPANY_HINT = re.compile(r"\b(?:pvt|private|ltd|limited|llp|inc|corp|co\.|company|traders|trading|enterprises?|industries|"
                          r"agencies|agency|distributors?|stores?|\w*mart|\w*market|wholesale|retail|suppliers?|solutions|"
                          r"exports?|imports?|sons|brothers|bros|& co|mills|pharma|electricals?|hardware)\b", re.I)

GRAND_TOTAL_RE = re.compile(r"\b(?:grand\s*total|invoice\s*total|total\s*amount|net\s*(?:amount|payable|total)|amount\s*payable|"
                            r"total\s*payable|total\s*invoice\s*value|bill\s*amount)\b", re.I)
SUBTOTAL_RE = re.compile(r"\b(?:sub\s*-?\s*total|total\s*before\s*tax|total\s*taxable\s*(?:value|amount)|taxable\s*total)\b", re.I)
PLAIN_TOTAL_RE = re.compile(r"^\W*total\b", re.I)


@dataclass
class InvoiceFields:
    invoice_number: str = ""
    invoice_date: str = ""
    vendor_name: str = ""
    vendor_gstin: str = ""
    subtotal: str = ""
    invoice_total: str = ""
    issues: list[Issue] = field(default_factory=list)


def _valid_date(s: str) -> bool:
    nums = [int(n) for n in re.findall(r"\d+", s)]
    if len(nums) == 3:
        a, b, c = nums
        if a > 31:                       # yyyy-mm-dd
            return 1 <= b <= 12 and 1 <= c <= 31
        return 1 <= a <= 31 and 1 <= b <= 31 and (a <= 12 or b <= 12)
    return True                          # textual month


def _first_date(text: str, start: int = 0) -> Optional[str]:
    for m in DATE_RE.finditer(text, start):
        if _valid_date(m.group(1)):
            return m.group(1).strip(" ,.")
    return None


def _below(lines: list[Line], i: int, x0: float, x1: float) -> Optional[Line]:
    """Next row under line i that horizontally overlaps [x0, x1]."""
    for ln in lines[i + 1:i + 3]:
        part = [w for w in ln.words if w.x1 > x0 - 20 and w.x0 < x1 + 200]
        if part:
            return Line(part)
    return None


def _last_amount(line: Line) -> str:
    for w in reversed(line.words):
        n = parse_number(w.text, ocr_fix=False)
        if n.ok and not n.percent:
            return fmt(n.value)
    return ""


def extract_fields(lines: list[Line], table_top: Optional[float], table_bottom: Optional[float]) -> InvoiceFields:
    f = InvoiceFields()
    texts = [ln.text for ln in lines]

    # ---------------------------------------------------------------- invoice number
    for i, t in enumerate(texts):
        for rx in INV_RES:
            m = rx.search(t)
            if m and re.search(r"\d", m.group(1)) and m.group(1).lower() not in ("date", "dated"):
                f.invoice_number = m.group(1).rstrip(".:,")
                break
        if f.invoice_number:
            break
        lab = INV_LABEL_ONLY.search(t)
        if lab:
            below = _below(lines, i, lines[i].x0, lines[i].x1)
            tok = next((w.text for w in (below.words if below else []) if re.search(r"\d", w.text)), "")
            if tok:
                f.invoice_number = tok.rstrip(".:,")
                f.issues.append(Issue("review", "invoice_number_below_label",
                                      f"invoice number '{tok}' taken from the line under the label", "invoice_number"))
                break
    if not f.invoice_number:
        f.issues.append(Issue("review", "missing_invoice_number", "invoice number not found", "invoice_number"))

    # ---------------------------------------------------------------- invoice date
    labelled = None
    for i, t in enumerate(texts):
        for m in DATE_LABEL.finditer(t):
            before = t[max(0, m.start() - 12):m.start()]
            if OTHER_DATE_LABEL.search(before):
                continue
            d = _first_date(t, m.end())
            if not d:
                below = _below(lines, i, lines[i].x0, lines[i].x1)
                d = _first_date(below.text) if below else None
            if d:
                labelled = d
                break
        if labelled:
            break
    if labelled:
        f.invoice_date = labelled
    else:
        top = [t for ln, t in zip(lines, texts) if table_top is None or ln.cy < table_top]
        d = next((x for x in (_first_date(t) for t in top if not OTHER_DATE_LABEL.search(t)) if x), None)
        if d:
            f.invoice_date = d
            f.issues.append(Issue("review", "invoice_date_unlabelled", f"date '{d}' found without an 'Invoice Date' label",
                                  "invoice_date"))
        else:
            f.issues.append(Issue("review", "missing_invoice_date", "invoice date not found", "invoice_date"))

    # ---------------------------------------------------------------- vendor name + GSTIN
    header_lines = [ln for ln in lines if table_top is None or ln.cy < table_top]
    if table_top is None:
        header_lines = lines[: max(3, len(lines) // 3)]
    buyer_zone: set[int] = set()
    for ln in header_lines:
        if BUYER_LABEL.search(ln.text):
            # buyer block: the label row and the rows directly under it, in the same horizontal band
            idx = lines.index(ln)
            for j in range(idx, min(idx + 5, len(lines))):
                buyer_zone.add(id(lines[j]))
    for ln in header_lines:
        m = VENDOR_LABEL.match(ln.text)
        if m and id(ln) not in buyer_zone:
            f.vendor_name = m.group(1).strip()
            break
    if not f.vendor_name:
        med_h = sorted(ln.h for ln in header_lines)[len(header_lines) // 2] if header_lines else 0
        best, best_score = None, 0.0
        for rank, ln in enumerate(header_lines[:12]):
            t = ln.text.strip()
            letters = sum(c.isalpha() for c in t)
            if id(ln) in buyer_zone or letters < 3 or letters / max(1, len(t.replace(" ", ""))) < 0.6:
                continue
            if NOT_VENDOR.search(t) and not COMPANY_HINT.search(t):
                continue
            score = 1.0 + (3.0 if COMPANY_HINT.search(t) else 0.0) + (2.0 if med_h and ln.h > 1.25 * med_h else 0.0)
            score += max(0.0, 1.5 - 0.25 * rank)
            if score > best_score:
                best, best_score = ln, score
        if best is not None:
            f.vendor_name = best.text.strip()
            if not COMPANY_HINT.search(f.vendor_name):
                f.issues.append(Issue("review", "vendor_name_inferred",
                                      f"vendor name '{f.vendor_name}' inferred from the document heading", "vendor_name"))
        else:
            f.issues.append(Issue("review", "missing_vendor_name", "vendor name not found", "vendor_name"))
    for ln in header_lines:
        if id(ln) in buyer_zone:
            continue
        m = GSTIN_RE.search(ln.text.upper())
        if m:
            f.vendor_gstin = m.group(1)
            break

    # ---------------------------------------------------------------- printed totals (below the table)
    tail = [ln for ln in lines if table_bottom is None or ln.cy >= table_bottom - 1]
    for ln in tail:
        if not f.invoice_total and GRAND_TOTAL_RE.search(ln.text):
            f.invoice_total = _last_amount(ln)
        if not f.subtotal and SUBTOTAL_RE.search(ln.text):
            f.subtotal = _last_amount(ln)
    if not f.invoice_total:
        for ln in tail:
            if PLAIN_TOTAL_RE.search(ln.text) and not SUBTOTAL_RE.search(ln.text):
                f.invoice_total = _last_amount(ln)
                if f.invoice_total:
                    break
    return f
