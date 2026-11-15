"""CSV formula-injection protection for every CSV the application exports.

A spreadsheet (Excel, LibreOffice, Google Sheets) treats a cell that starts with ``=``, ``+``, ``-`` or ``@`` (or
with a tab / carriage return followed by one) as a formula. Free text that reaches an export - an operator's alert
note, an uploaded video's file name, an error message - could therefore run a formula when an administrator opens the
file. Such text cells are prefixed with a single quote, which spreadsheets display as plain text (OWASP guidance).

Only text is changed. Numbers (int/float) and text that is a plain number (e.g. "-0.5", "+3", "1e-3") are written
unchanged, so coordinates, confidences and counts keep their existing format. Applying it twice changes nothing.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_PLAIN_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def safe_cell(value: Any) -> Any:
    """``value`` made safe to open in a spreadsheet; anything that is not risky text is returned unchanged."""
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES) and not _PLAIN_NUMBER.fullmatch(value):
        return "'" + value
    return value


def safe_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: safe_cell(v) for k, v in row.items()}


def safe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [safe_row(r) for r in rows]
