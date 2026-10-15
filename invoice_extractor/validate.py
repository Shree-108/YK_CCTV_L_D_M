"""Numeric validation and cross-checks. Never changes a value silently: any interpretation is
recorded as a review issue and the cell keeps the text exactly as OCR read it."""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from invoice_extractor.models import NUMERIC_FIELDS, InvoiceResult, Issue, LineItem
from invoice_extractor.numbers import Num, close, fmt, parse_number

HUNDRED = Decimal(100)


def _num(item: LineItem, key: str) -> Optional[Num]:
    raw = item.values.get(key, "").strip()
    return parse_number(raw) if raw else None


def validate_item(item: LineItem) -> None:
    n = item.line_no
    parsed: dict[str, Decimal] = {}
    for key in sorted(NUMERIC_FIELDS):
        x = _num(item, key)
        if x is None:
            continue
        if not x.ok:
            item.issues.append(Issue("review", "not_a_number", f"{key} '{x.raw}' is not a readable number", key, n))
            continue
        parsed[key] = x.value
        if x.note:
            # keep OCR text as-is in the CSV; the interpretation goes to the review report only
            item.issues.append(Issue("review", "value_interpreted", f"{key}: {x.note}", key, n))
        elif key == "discount" and x.percent:
            item.values[key] = f"{fmt(x.value)}%"
        else:
            item.values[key] = fmt(x.value)      # formatting only: currency symbol / thousands separators removed

    if not item.values.get("product_name", "").strip():
        item.issues.append(Issue("review", "missing_product_name", "product name not found", "product_name", n))
    for key in ("quantity", "line_total"):
        if key not in parsed and not item.values.get(key):
            item.issues.append(Issue("review", f"missing_{key}", f"{key} not found", key, n))
    q = parsed.get("quantity")
    if q is not None and q <= 0:
        item.issues.append(Issue("review", "non_positive_quantity", f"quantity {fmt(q)} is not positive", "quantity", n))
    for key in ("unit_price", "line_total", "tax_amount", "taxable_value"):
        if parsed.get(key) is not None and parsed[key] < 0:
            item.issues.append(Issue("review", "negative_value", f"{key} {fmt(parsed[key])} is negative", key, n))
    rate = parsed.get("tax_rate")
    if rate is not None and not (0 <= rate <= 100):
        item.issues.append(Issue("review", "tax_rate_out_of_range", f"tax rate {fmt(rate)} outside 0-100 %", "tax_rate", n))

    _check_arithmetic(item, parsed)


def _check_arithmetic(item: LineItem, p: dict[str, Decimal]) -> None:
    n = item.line_no
    q, price, total = p.get("quantity"), p.get("unit_price"), p.get("line_total")
    taxable, tax_amt, rate = p.get("taxable_value"), p.get("tax_amount"), p.get("tax_rate")
    disc_raw = item.values.get("discount", "")
    disc = p.get("discount")
    checks: list[bool] = []

    if q is not None and price is not None and (total is not None or taxable is not None):
        base = q * price
        nets = [base]
        if disc is not None:
            nets += [base - abs(disc)]                                   # discount as amount
            if abs(disc) <= 100:
                nets += [base * (1 - abs(disc) / HUNDRED)]               # discount as percent
            if "%" in disc_raw:
                nets = [base, base * (1 - abs(disc) / HUNDRED)]
        if taxable is not None:
            ok = any(close(x, taxable) for x in nets)
            checks.append(ok)
            if not ok:
                item.issues.append(Issue("review", "arithmetic_mismatch",
                                         f"qty x price ({fmt(q)} x {fmt(price)} = {fmt(base)}) does not match taxable value {fmt(taxable)}",
                                         "taxable_value", n))
            nets = [taxable]
        if total is not None:
            cands = list(nets)
            for x in nets:
                if tax_amt is not None:
                    cands.append(x + tax_amt)
                if rate is not None:
                    cands.append(x * (1 + rate / HUNDRED))
            ok = any(close(c, total) for c in cands)
            checks.append(ok)
            if not ok:
                item.issues.append(Issue("review", "arithmetic_mismatch",
                                         f"line total {fmt(total)} does not match qty x price ({fmt(q)} x {fmt(price)} = {fmt(base)})"
                                         + (" incl. discount/tax" if (disc or tax_amt or rate) else ""), "line_total", n))
    if tax_amt is not None and rate is not None:
        basis = taxable if taxable is not None else (q * price if q is not None and price is not None else None)
        if basis is not None and basis > 0:
            expected = basis * rate / HUNDRED
            ok = close(expected, tax_amt) or close(expected, tax_amt * 2) or close(expected / 2, tax_amt)
            checks.append(ok)
            if not ok:
                item.issues.append(Issue("review", "tax_mismatch",
                                         f"tax amount {fmt(tax_amt)} != {fmt(rate)}% of {fmt(basis)} ({expected:.2f})",
                                         "tax_amount", n))
    if checks:
        item.arithmetic_check = "ok" if all(checks) else "mismatch"
    elif item.arithmetic_check != "ok":
        item.arithmetic_check = "not_checked"
    if item.arithmetic_check == "ok" and item.method == "table":
        # an inferred column whose numbers reconcile (qty x price = total) is confirmed by the data
        for i in item.issues:
            if i.issue == "inferred_column":
                i.severity, i.detail = "warning", i.detail + " (confirmed by arithmetic check)"


def validate_invoice(res: InvoiceResult, blur_threshold: float, low_conf: float) -> None:
    for it in res.items:
        validate_item(it)
    q = res.quality
    if q.get("blur_score") is not None and q["blur_score"] < blur_threshold:
        res.issues.append(Issue("review", "blurry_image",
                                f"image looks blurry (sharpness {q['blur_score']:.0f} < {blur_threshold:.0f}); "
                                "verify all values or rescan"))
    if q.get("mean_confidence") is not None and q["mean_confidence"] < low_conf:
        res.issues.append(Issue("review", "low_overall_confidence",
                                f"average OCR confidence {q['mean_confidence']:.2f} is low"))
    if not res.items:
        res.issues.append(Issue("error", "no_line_items", "no product line items could be extracted"))
        return
    totals = [parse_number(it.values.get("line_total", "")) for it in res.items]
    if all(t.ok for t in totals):
        s = sum((t.value for t in totals), Decimal(0))
        res.quality["sum_line_totals"] = fmt(s)
        printed = [(lbl, parse_number(v)) for lbl, v in (("subtotal", res.subtotal), ("invoice total", res.invoice_total)) if v]
        printed = [(lbl, x.value) for lbl, x in printed if x.ok]
        if printed and not any(close(s, v, rel=Decimal("0.005")) for _, v in printed):
            # line totals are often pre-tax while the grand total is post-tax: compare against each
            # printed figure, and also against grand total minus summed line tax
            tax = sum((parse_number(it.values.get("tax_amount", "")).value or Decimal(0)
                       for it in res.items if parse_number(it.values.get("tax_amount", "")).ok), Decimal(0))
            if not any(close(s + tax, v, rel=Decimal("0.005")) for _, v in printed):
                shown = ", ".join(f"{lbl} {fmt(v)}" for lbl, v in printed)
                res.issues.append(Issue("review", "totals_mismatch",
                                        f"sum of line totals {fmt(s)} does not match printed {shown} "
                                        "(possible missed/misread line, or charges not listed as items)"))
