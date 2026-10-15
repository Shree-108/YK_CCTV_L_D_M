"""Data structures shared by the OCR, parsing, validation and output stages."""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# Product-level fields written to products.csv, in column order.
ITEM_FIELDS = ["product_name", "sku", "hsn_code", "quantity", "unit", "unit_price", "discount",
               "taxable_value", "tax_rate", "tax_amount", "line_total"]
NUMERIC_FIELDS = {"quantity", "unit_price", "discount", "taxable_value", "tax_rate", "tax_amount", "line_total"}

# Issue severities, most to least serious. "error" = extraction failed; "review" = a human must
# verify the value; "warning" = something looks off but the value is probably usable; "info" = note.
SEVERITIES = ("error", "review", "warning", "info")


@dataclass
class Word:
    """One OCR token with its axis-aligned box in (pre-processed) image pixels."""
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float
    seg: int = 0          # OCR segment/line id the token came from
    angle: float = 0.0    # text baseline angle in degrees (for deskew)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return max(1.0, self.y1 - self.y0)

    @property
    def w(self) -> float:
        return max(1.0, self.x1 - self.x0)


@dataclass
class Line:
    """Words that share a text row, sorted left to right."""
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def y0(self) -> float:
        return min(w.y0 for w in self.words)

    @property
    def y1(self) -> float:
        return max(w.y1 for w in self.words)

    @property
    def cy(self) -> float:
        return statistics.median(w.cy for w in self.words)

    @property
    def h(self) -> float:
        return statistics.median(w.h for w in self.words)

    @property
    def x0(self) -> float:
        return min(w.x0 for w in self.words)

    @property
    def x1(self) -> float:
        return max(w.x1 for w in self.words)

    @property
    def conf(self) -> float:
        return statistics.fmean(w.conf for w in self.words) if self.words else 0.0


@dataclass
class Issue:
    severity: str            # one of SEVERITIES
    issue: str               # short machine-friendly code, e.g. "arithmetic_mismatch"
    detail: str              # human explanation
    field: str = ""          # affected field ("" = whole record)
    line_no: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Issue":
        return cls(**{k: d.get(k) for k in ("severity", "issue", "detail", "field", "line_no")})


@dataclass
class LineItem:
    line_no: int
    values: dict[str, str] = field(default_factory=dict)   # field -> value as written to CSV
    raw_text: str = ""                                      # OCR text of the row(s), unmodified
    ocr_confidence: Optional[float] = None
    method: str = "table"                                   # "table" (header columns) or "arithmetic" (fallback)
    arithmetic_check: str = "not_checked"                   # ok | mismatch | not_checked
    issues: list[Issue] = field(default_factory=list)
    y0: float = 0.0
    y1: float = 0.0

    @property
    def needs_review(self) -> bool:
        return any(i.severity in ("error", "review") for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["issues"] = [i.to_dict() for i in self.issues]
        d["needs_review"] = self.needs_review
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LineItem":
        return cls(line_no=d["line_no"], values=dict(d.get("values") or {}), raw_text=d.get("raw_text", ""),
                   ocr_confidence=d.get("ocr_confidence"), method=d.get("method", "table"),
                   arithmetic_check=d.get("arithmetic_check", "not_checked"),
                   issues=[Issue.from_dict(i) for i in d.get("issues") or []],
                   y0=d.get("y0", 0.0), y1=d.get("y1", 0.0))


@dataclass
class InvoiceResult:
    source_file: str
    sha256: str = ""
    processed_at: str = ""
    engine: str = ""
    status: str = "ok"                       # ok | needs_review | failed | duplicate
    invoice_number: str = ""
    invoice_date: str = ""
    vendor_name: str = ""
    vendor_gstin: str = ""
    subtotal: str = ""
    invoice_total: str = ""
    items: list[LineItem] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)   # invoice-level issues
    quality: dict[str, Any] = field(default_factory=dict)
    duplicate_of: str = ""

    @property
    def needs_review(self) -> bool:
        return self.status != "ok" or any(i.needs_review for i in self.items)

    def all_issues(self) -> list[Issue]:
        out = list(self.issues)
        for it in self.items:
            out.extend(it.issues)
        return out

    def refresh_status(self) -> None:
        if self.duplicate_of:
            self.status = "duplicate"
        elif any(i.severity == "error" for i in self.issues) or not self.items:
            self.status = "failed"
        elif any(i.severity == "review" for i in self.issues) or any(it.needs_review for it in self.items):
            self.status = "needs_review"
        else:
            self.status = "ok"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["items"] = [i.to_dict() for i in self.items]
        d["issues"] = [i.to_dict() for i in self.issues]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InvoiceResult":
        simple = {k: d.get(k, "") for k in ("source_file", "sha256", "processed_at", "engine", "status",
                                             "invoice_number", "invoice_date", "vendor_name", "vendor_gstin",
                                             "subtotal", "invoice_total", "duplicate_of")}
        return cls(**simple, items=[LineItem.from_dict(i) for i in d.get("items") or []],
                   issues=[Issue.from_dict(i) for i in d.get("issues") or []], quality=dict(d.get("quality") or {}))
