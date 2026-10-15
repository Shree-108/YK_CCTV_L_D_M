"""Conservative number parsing for OCR'd invoice values.

Rules:
* Currency symbols, thousands separators (incl. Indian 1,25,000.00) and a trailing "/-" are
  formatting and are removed.
* Anything that needs an *interpretation* (OCR letter->digit fixes such as "l0" -> 10, or a
  comma read as a decimal point) is reported through ``Num.note`` so the caller can flag the
  value for review instead of silently trusting it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

_CURRENCY = re.compile(r"(?i)(₹|rs\.?|inr|usd|eur|\$|€|£)")
_OCR_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1", "i": "1",
                          "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2"})
_NUMERIC_CHARS = re.compile(r"^[\d.,]+$")


@dataclass(frozen=True)
class Num:
    value: Optional[Decimal]
    raw: str
    note: str = ""          # non-empty when an interpretation was needed (=> needs review)
    percent: bool = False

    @property
    def ok(self) -> bool:
        return self.value is not None


def _strict_decimal(t: str) -> tuple[Optional[Decimal], str]:
    """Parse digits with ',' / '.' separators. Returns (value, note)."""
    if not t or not _NUMERIC_CHARS.match(t) or not re.search(r"\d", t):
        return None, ""
    note = ""
    if re.fullmatch(r"\d{1,3}(\.\d{3})+,\d{1,2}", t):            # 1.234,56 (European)
        t, note = t.replace(".", "").replace(",", "."), "European number format (1.234,56) interpreted"
    elif re.fullmatch(r"\d+,\d{1,2}", t):                         # 12,50: decimal comma or misread '.'
        t, note = t.replace(",", "."), "comma interpreted as decimal point"
    else:
        if "," in t and not re.fullmatch(r"\d{1,3}(,\d{2,3})*(,\d{3})?(\.\d+)?", t):
            return None, ""
        t = t.replace(",", "")
        if t.count(".") > 1:
            return None, ""
    try:
        return Decimal(t), note
    except InvalidOperation:
        return None, ""


def parse_number(raw: str, ocr_fix: bool = True) -> Num:
    s = (raw or "").strip()
    if not s:
        return Num(None, raw or "")
    percent = "%" in s
    t = _CURRENCY.sub("", s).replace("%", "").replace("/-", "").replace(" ", "").strip(":=")
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg, t = True, t[1:-1]
    if t.startswith("-"):
        neg, t = True, t[1:]
    t = t.rstrip(".") if re.search(r"\d\.$", t) else t
    value, note = _strict_decimal(t)
    if value is None and ocr_fix and t:
        digits = sum(c.isdigit() for c in t)
        if digits and digits / len(t) >= 0.5:
            fixed = t.translate(_OCR_FIX)
            value, note2 = _strict_decimal(fixed)
            if value is not None:
                note = f"OCR characters corrected: '{raw.strip()}' read as {fixed}" + (f"; {note2}" if note2 else "")
    if value is None:
        return Num(None, raw, percent=percent)
    return Num(-value if neg else value, raw, note, percent)


def fmt(value: Decimal) -> str:
    """Plain decimal string (no exponent, keeps printed precision)."""
    s = format(value, "f")
    return "0" if s in ("-0", "-0.0") else s


def looks_numeric(text: str) -> bool:
    return parse_number(text, ocr_fix=False).ok


def close(a: Decimal, b: Decimal, rel: Decimal = Decimal("0.01"), abs_tol: Decimal = Decimal("1.00")) -> bool:
    """Equal within rounding: 1 currency unit or 1 % of the larger value, whichever is bigger."""
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))
