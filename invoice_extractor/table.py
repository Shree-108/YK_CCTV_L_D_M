"""Product table extraction.

Primary method ("table"): find the column-header row (Description / HSN / Qty / Rate / Amount ...),
use the header words' x-positions as column anchors, assign every word of each following row to
a column, and merge wrapped description lines into their item. Parsing stops at the totals block.

Fallback method ("arithmetic"), used only when no header row is recognised: a row is accepted as
a product line only if two of its numbers multiply to its last number (qty x rate = amount).
Rows that look like products but fail that check are reported for review, never invented.
"""
from __future__ import annotations

import itertools
import re
import statistics
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from invoice_extractor.layout import median_line_height
from invoice_extractor.models import ITEM_FIELDS, Issue, Line, LineItem, Word
from invoice_extractor.numbers import close, fmt, parse_number

TAX_ANY = "tax_any"

_SYNONYMS: dict[str, list[str]] = {
    "sl": ["sl", "slno", "sno", "srno", "sr", "sn", "si", "sino", "no", "#", "serial", "serialno", "itemno", "s1", "sr#"],
    "product_name": ["description", "descriptionofgoods", "descriptionofgoodsservices", "descriptionofservices",
                     "item", "items", "itemname", "itemdescription", "itemdetails", "product", "productname",
                     "productdescription", "productdetails", "particulars", "goods", "name", "details", "material",
                     "article", "service", "services", "nameofproduct", "nameofitem", "goodsdescription"],
    "sku": ["sku", "skucode", "code", "itemcode", "productcode", "partno", "partnumber", "artno", "articleno",
            "model", "modelno", "barcode", "ean", "catno", "styleno", "mfrcode", "prodcode"],
    "hsn_code": ["hsn", "hsncode", "hsnsac", "sac", "hsnsaccode", "hsnno", "sachsn", "hsnsacno"],
    "quantity": ["qty", "quantity", "qnty", "qtty", "billedqty", "noofunits", "quan", "qtypcs", "qtynos",
                 "aty", "oty", "qly", "qtv", "qiy", "gty"],   # last row: frequent OCR misreads of "Qty"
    "unit": ["unit", "uom", "units", "per", "um", "unitofmeasure"],
    "unit_price": ["rate", "price", "unitprice", "unitrate", "rateperunit", "unitcost", "cost", "priceperunit",
                   "rateunit", "saleprice", "sellingprice", "listprice", "mrp", "rateinr", "raters", "pricers",
                   "basicrate", "netrate", "unitpricers", "unitpriceinr"],
    "discount": ["disc", "discount", "disc%", "discount%", "dis", "discamt", "discountamt", "discountamount", "dis%",
                 "less"],
    "taxable_value": ["taxablevalue", "taxableamount", "taxableamt", "taxable", "assessablevalue", "taxableval",
                      "amountbeforetax"],
    "tax_rate": ["gst%", "gstrate", "tax%", "taxrate", "vat%", "vatrate", "igst%", "cgst%", "sgst%", "rate%",
                 "igstrate", "cgstrate", "sgstrate", "%"],
    "tax_amount": ["taxamount", "taxamt", "gstamount", "gstamt", "vatamount", "vatamt", "igstamt", "cgstamt",
                   "sgstamt", "igstamount", "cgstamount", "sgstamount", "totaltax", "taxvalue", "gstvalue"],
    TAX_ANY: ["gst", "tax", "vat", "igst", "cgst", "sgst", "utgst"],
    "line_total": ["amount", "total", "linetotal", "netamount", "amt", "totalamount", "value", "netamt", "amountrs",
                   "amountinr", "totalvalue", "grossamount", "totalrs", "totalamt", "itemtotal", "nettotal",
                   "lineamount"],
}
SYN = {s: k for k, v in _SYNONYMS.items() for s in v}
NUMERIC_KEYS = {"quantity", "unit_price", "discount", "taxable_value", "tax_rate", "tax_amount", "line_total", TAX_ANY}
ANCHOR_KEYS = ("quantity", "unit_price", "line_total", "taxable_value")

UNITS = set("""pcs pc piece pieces nos no no's nos. kg kgs g gm gms gram grams ltr ltrs l lt litre liter ml mtr mtrs m
mm cm ft feet inch box boxes set sets pkt pkts pack packs pair pairs prs dozen doz dz bag bags roll rolls sqft sqm sq.ft
ea each unit units bottle bottles btl can cans ctn ctns carton cartons jar tin tube sheet sheets bundle bundles coil drum
ream hrs hr hour hours day days month months lot job qtl ton tons tonne mt cft rft rmt km strip strips cup tray card
tab tabs cap caps vial pkg""".split())

STOP_RE = re.compile(r"^\W*(?:\d{1,3}\W+)?(?:sub\s*-?\s*total|grand\s*total|total|net\s*(?:amount|payable|total)|"
                     r"amount\s*(?:in\s*words|chargeable|payable)|round(?:ed|ing)?\s*off|taxable\s*(?:value|amount)|"
                     r"less\s*:|add\s*:|cgst|sgst|igst|utgst|tax\s*(?:summary|amount)|bank|terms|declaration|rupees|"
                     r"in\s*words|e\.?\s*&\s*o\.?\s*e|thank|authori[sz]ed|for\s+[A-Z]|balance|amount\s*due|paid)", re.I)
NON_ITEM_RE = re.compile(r"\b(?:invoice|inv\s*no|date|gstin|gst\s*no|pan|phone|ph|mob|mobile|tel|fax|email|www|http|"
                         r"total|sub\s*total|cgst|sgst|igst|round|balance|paid|due|words|bank|ifsc|a/c|account|pin|"
                         r"address|state|code\s*:|order|challan|vehicle|place\s*of\s*supply)\b", re.I)


@dataclass
class Column:
    key: Optional[str]      # canonical field, or None/"extra" for ignored columns
    x0: float
    x1: float
    label: str
    inferred: bool = False  # header text unreadable; column placed from the data rows

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class TableResult:
    items: list[LineItem] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    method: str = "none"
    columns: list[str] = field(default_factory=list)
    top: Optional[float] = None       # y of header row (fields above it are invoice header)
    bottom: Optional[float] = None    # y where the totals block starts


def _norm(tok: str) -> str:
    t = tok.lower().replace("₹", "")
    t = re.sub(r"[()\[\]{}:.,'\"`/\\|_\-*]", "", t)
    return t.strip()


# ---------------------------------------------------------------- header detection
def _match_header(line: Line) -> tuple[list[Column], float]:
    """Greedy longest-phrase match of header synonyms. Returns (columns, fraction of words matched)."""
    words = line.words
    cols: list[Column] = []
    matched = counted = 0
    i = 0
    while i < len(words):
        if not _norm(words[i].text):
            if cols:                                   # "(₹)" etc: extend the previous column
                cols[-1].x1 = max(cols[-1].x1, words[i].x1)
            i += 1
            continue
        hit = None
        for k in (4, 3, 2, 1):
            if i + k > len(words):
                continue
            seg = words[i:i + k]
            # phrase words must be close together (same header cell)
            if any(b.x0 - a.x1 > 0.9 * max(a.h, b.h) for a, b in zip(seg, seg[1:])):
                continue
            key = SYN.get("".join(_norm(w.text) for w in seg))
            if key:
                hit = (key, k)
                break
        counted += 1
        if hit:
            key, k = hit
            seg = words[i:i + k]
            cols.append(Column(key, min(w.x0 for w in seg), max(w.x1 for w in seg), " ".join(w.text for w in seg)))
            matched += 1
            i += k
        else:
            i += 1
    return cols, matched / max(1, counted)


def _is_header(cols: list[Column], ratio: float) -> int:
    keys = {c.key for c in cols if c.key not in ("sl", None)}
    if ratio < 0.6 or len(keys) < 3:
        return 0
    if "product_name" not in keys and not ({"quantity", "line_total"} <= keys or {"quantity", "unit_price"} <= keys):
        return 0
    return len(keys)


def _merge_child_row(parents: list[Column], children: list[Column]) -> list[Column]:
    """Two-row headers, e.g. 'CGST' over 'Rate | Amount': children under a parent replace it."""
    out: list[Column] = []
    used = set()
    for p in parents:
        kids = [c for c in children if min(c.x1, p.x1) - max(c.x0, p.x0) > 0.3 * min(c.x1 - c.x0, p.x1 - p.x0)]
        if not kids:
            out.append(p)
            continue
        for c in kids:
            used.add(id(c))
            key = c.key
            if p.key in (TAX_ANY, "tax_amount", "tax_rate"):
                key = {"unit_price": "tax_rate", "line_total": "tax_amount", TAX_ANY: "tax_amount"}.get(key, key)
            out.append(Column(key, c.x0, c.x1, f"{p.label} {c.label}"))
    out.extend(c for c in children if id(c) not in used)
    return sorted(out, key=lambda c: c.x0)


def find_header(lines: list[Line]) -> tuple[Optional[int], list[Column]]:
    best_i, best_cols, best_score = None, [], 0
    for i, ln in enumerate(lines):
        cols, ratio = _match_header(ln)
        score = _is_header(cols, ratio)
        if score > best_score:
            best_i, best_cols, best_score = i, cols, score
    if best_i is None:
        return None, []
    # second header row directly underneath?
    if best_i + 1 < len(lines):
        nxt = lines[best_i + 1]
        if nxt.cy - lines[best_i].cy < 2.2 * lines[best_i].h:
            kids, ratio = _match_header(nxt)
            if kids and ratio >= 0.6 and not any(parse_number(w.text, ocr_fix=False).ok for w in nxt.words):
                best_cols = _merge_child_row(best_cols, kids)
                best_i += 1
    return best_i, _resolve_duplicates(sorted(best_cols, key=lambda c: c.x0))


def _resolve_duplicates(cols: list[Column]) -> list[Column]:
    seen: dict[str, int] = {}
    totals = [c for c in cols if c.key == "line_total"]
    for c in totals[:-1]:                    # rightmost "Amount"/"Total" is the line total
        c.key = "extra"
    for c in cols:
        if c.key in ("tax_amount", "tax_rate", TAX_ANY, "line_total", "extra", None):
            continue
        seen[c.key] = seen.get(c.key, 0) + 1
        if seen[c.key] > 1:
            c.key = "extra"
    return cols


# ---------------------------------------------------------------- cell assignment
def _fits(key: Optional[str], text: str) -> bool:
    t = text.strip()
    if key in (None, "extra", "product_name", "sku"):
        return True
    if key == "sl":
        return bool(re.fullmatch(r"\d{1,4}[.)]?", t))
    if key == "hsn_code":
        return bool(re.fullmatch(r"\d{2,8}", t))
    if key == "unit":
        return t.lower().strip(".") in UNITS or t.lower() in UNITS
    if t in ("%", "₹", "Rs", "Rs.", "/-", "-"):
        return True
    if key == "quantity" and re.fullmatch(r"[\d.,]+\s*[A-Za-z]{1,6}\.?", t):
        return True
    return parse_number(t).ok


def _assign(line: Line, cols: list[Column]) -> dict[int, list[Word]]:
    bounds = []
    for a, b in zip(cols, cols[1:]):
        bounds.append((a.x1 + b.x0) / 2 if a.x1 <= b.x0 else (a.cx + b.cx) / 2)

    def col_of(w: Word) -> int:
        for i, bd in enumerate(bounds):
            if w.cx < bd:
                return i
        return len(cols) - 1

    owner = {id(w): col_of(w) for w in line.words}
    desc = next((i for i, c in enumerate(cols) if c.key == "product_name"), None)
    if desc is not None:
        h = line.h
        changed = True
        while changed:
            changed = False
            desc_words = [w for w in line.words if owner[id(w)] == desc]
            for w in line.words:
                ci = owner[id(w)]
                if ci == desc:
                    continue
                gap = min((abs(w.x0 - d.x1) if w.x0 >= d.x1 else abs(d.x0 - w.x1)) for d in desc_words) if desc_words else None
                misfit = not _fits(cols[ci].key, w.text) and any(c.isalpha() for c in w.text)
                # description text spilling into a neighbouring column: tight word spacing, or a word
                # that does not fit the neighbour's data type
                tight = gap is not None and gap < 0.6 * h and ci > desc
                if tight or (gap is not None and misfit and gap < 1.5 * h) or \
                        (misfit and not desc_words and abs(ci - desc) == 1):
                    owner[id(w)] = desc
                    changed = True
                    break
    cells: dict[int, list[Word]] = {}
    for w in line.words:
        cells.setdefault(owner[id(w)], []).append(w)
    return cells


def _cell_text(words: list[Word], numeric: bool) -> str:
    """Numeric cells split by OCR ("1,250" ".00", "18" "%") are re-joined without spaces."""
    if numeric and all(re.fullmatch(r"[\d.,%₹()\-/]+", w.text) for w in words):
        return "".join(w.text for w in words)
    return " ".join(w.text for w in words)


# ---------------------------------------------------------------- items
@dataclass
class _Draft:
    cells: dict[int, list[Word]] = field(default_factory=dict)
    lines: list[Line] = field(default_factory=list)
    wrapped: bool = False

    def has_anchor(self, cols: list[Column]) -> bool:
        return any(cols[i].key in ANCHOR_KEYS and any(parse_number(w.text).ok for w in ws) for i, ws in self.cells.items())


def _row_has_anchor(cells: dict[int, list[Word]], cols: list[Column]) -> bool:
    return any(cols[i].key in ANCHOR_KEYS and parse_number(_cell_text(ws, True).split(" ")[0]).ok
               for i, ws in cells.items())


def _resolve_tax_any(cols: list[Column], drafts: list[_Draft]) -> None:
    for i, c in enumerate(cols):
        if c.key != TAX_ANY:
            continue
        vals = [_cell_text(d.cells[i], True) for d in drafts if i in d.cells]
        nums = [parse_number(v) for v in vals]
        pct = sum("%" in v for v in vals)
        small = all(n.ok and n.value <= 40 for n in nums) and nums
        c.key = "tax_rate" if vals and (pct >= 0.6 * len(vals) or small) else "tax_amount"


def _build_item(n: int, d: _Draft, cols: list[Column], low_conf: float) -> LineItem:
    item = LineItem(line_no=n, raw_text=" | ".join(ln.text for ln in d.lines), method="table",
                    y0=min(ln.y0 for ln in d.lines), y1=max(ln.y1 for ln in d.lines))
    by_key: dict[str, list[list[Word]]] = {}
    for i in sorted(d.cells):
        key = cols[i].key
        if key in (None, "extra", "sl"):
            continue
        by_key.setdefault(key, []).append(d.cells[i])
    all_words = [w for ws in d.cells.values() for w in ws]
    item.ocr_confidence = round(statistics.fmean(w.conf for w in all_words), 3) if all_words else None
    for key, groups in by_key.items():
        if key == "product_name":
            item.values[key] = " ".join(w.text for g in groups for w in g)   # cells are in reading order
        elif key in ("tax_amount", "tax_rate") and len(groups) > 1:
            texts = [_cell_text(g, True) for g in groups]
            nums = [parse_number(t) for t in texts]
            if all(x.ok and not x.note for x in nums):
                item.values[key] = fmt(sum((x.value for x in nums), Decimal(0)))
                item.issues.append(Issue("info", "tax_columns_summed",
                                         f"{key} = sum of {len(groups)} tax columns ({' + '.join(texts)})", key, n))
            else:
                item.values[key] = " + ".join(texts)
                item.issues.append(Issue("review", "tax_columns_unparsed",
                                         f"could not add tax columns: {' + '.join(texts)}", key, n))
        else:
            item.values[key] = " ".join(_cell_text(g, key in NUMERIC_KEYS) for g in groups)
        if any(cols[i].inferred and cols[i].key == key for i in d.cells):
            item.issues.append(Issue("review", "inferred_column",
                                     f"{key} column header was not readable; column position inferred from the data", key, n))
        low = [w for g in groups for w in g if w.conf < low_conf]
        if low:
            item.issues.append(Issue("review", "low_ocr_confidence",
                                     f"low OCR confidence ({min(w.conf for w in low):.2f}) reading "
                                     f"'{' '.join(w.text for w in low)}'", key, n))
    q = item.values.get("quantity", "")
    m = re.fullmatch(r"([\d.,]+)\s*([A-Za-z]{1,6}\.?)", q)
    if m and not item.values.get("unit"):
        item.values["quantity"], item.values["unit"] = m.group(1), m.group(2)
        item.issues.append(Issue("info", "unit_split_from_quantity", f"'{q}' split into quantity and unit", "unit", n))
    if d.wrapped:
        item.issues.append(Issue("info", "multi_line_row", "item text spans several printed lines and was merged",
                                 "product_name", n))
    return item


def parse_table(lines: list[Line], low_conf: float = 0.75) -> TableResult:
    hi, cols = find_header(lines)
    if hi is None:
        return parse_by_arithmetic(lines, low_conf)
    _infer_columns(lines[hi + 1:], cols)
    res = TableResult(method="table", columns=[f"{c.label}->{c.key}" for c in cols], top=lines[hi].y0)
    med_h = median_line_height(lines)
    drafts: list[_Draft] = []
    last_y1 = lines[hi].y1
    for ln in lines[hi + 1:]:
        cells = _assign(ln, cols)
        desc_i = next((i for i, c in enumerate(cols) if c.key == "product_name"), None)
        desc_text = " ".join(w.text for w in cells.get(desc_i, [])) if desc_i is not None else ""
        anchor = _row_has_anchor(cells, cols)
        if STOP_RE.match(ln.text) or (desc_text and STOP_RE.match(desc_text) and not anchor):
            res.bottom = ln.y0
            break
        if drafts and ln.y0 - last_y1 > 6 * med_h:
            res.bottom = ln.y0
            break
        sl_i = next((i for i, c in enumerate(cols) if c.key == "sl"), None)
        has_sl = sl_i is not None and sl_i in cells and _fits("sl", _cell_text(cells[sl_i], True))
        cur = drafts[-1] if drafts else None
        close_to_prev = cur is not None and ln.y0 - last_y1 < 2.5 * med_h
        if anchor:
            if cur is not None and not cur.has_anchor(cols) and not has_sl and close_to_prev:
                _merge(cur, cells, ln)
            else:
                drafts.append(_Draft(cells, [ln]))
        elif has_sl and desc_text:
            drafts.append(_Draft(cells, [ln]))
        elif desc_text and cur is not None and close_to_prev:
            _merge(cur, cells, ln)
        else:
            continue
        last_y1 = ln.y1
    if res.bottom is None:
        res.bottom = last_y1
    _resolve_tax_any(cols, drafts)
    res.items = [_build_item(n, d, cols, low_conf) for n, d in enumerate(drafts, start=1)]
    # a column that is filled on most rows but empty on this one usually means OCR missed the value
    keys = {c.key for c in cols if c.key in ITEM_FIELDS and c.key != "product_name"}
    for key in keys:
        filled = sum(1 for it in res.items if it.values.get(key, "").strip())
        if filled and filled >= 0.5 * len(res.items):
            for it in res.items:
                if not it.values.get(key, "").strip():
                    it.issues.append(Issue("review", "missing_cell",
                                           f"{key} is empty on this row but present on other rows (OCR may have missed it)",
                                           key, it.line_no))
    if not res.items:
        fb = parse_by_arithmetic(lines, low_conf)
        if fb.items:
            fb.issues.insert(0, Issue("warning", "header_without_rows",
                                      "a column header was found but no rows under it; used arithmetic fallback"))
            return fb
    return res


def _data_rows(rows: list[Line]) -> list[Line]:
    out = []
    for ln in rows[:60]:
        if STOP_RE.match(ln.text):
            break
        if sum(parse_number(w.text, ocr_fix=False).ok for w in ln.words) >= 2:
            out.append(ln)
    return out


def _infer_columns(rows: list[Line], cols: list[Column]) -> None:
    """Add columns whose header text OCR missed: a leading serial number left of the first header,
    and an unlabelled numeric column right of the last header (assumed line total, flagged)."""
    data = _data_rows(rows)
    if not data:
        return
    need = max(1, (len(data) + 1) // 2)
    first, last = cols[0], cols[-1]
    if not any(c.key == "sl" for c in cols):
        lead = [ln.words[0] for ln in data if _fits("sl", ln.words[0].text) and ln.words[0].x1 < first.x0]
        if len(lead) >= need:
            cols.insert(0, Column("sl", min(w.x0 for w in lead), max(w.x1 for w in lead), "(serial no.)", inferred=True))
    spans = []
    for ln in data:
        inside = [w for w in ln.words if last.x0 <= w.cx <= last.x1 and parse_number(w.text, ocr_fix=False).ok]
        right = [w for w in ln.words if w.cx > last.x1 + 0.2 * ln.h and parse_number(w.text, ocr_fix=False).ok]
        if inside and right:
            spans.append((min(w.x0 for w in right), max(w.x1 for w in right)))
    if len(spans) >= need:
        key = None if any(c.key == "line_total" for c in cols) else "line_total"
        cols.append(Column(key, min(a for a, _ in spans), max(b for _, b in spans), "(unlabelled)", inferred=True))


def _merge(d: _Draft, cells: dict[int, list[Word]], ln: Line) -> None:
    for i, ws in cells.items():
        d.cells.setdefault(i, []).extend(ws)
    d.lines.append(ln)
    d.wrapped = True


# ---------------------------------------------------------------- arithmetic fallback (no header)
def parse_by_arithmetic(lines: list[Line], low_conf: float = 0.75) -> TableResult:
    res = TableResult(method="arithmetic")
    res.issues.append(Issue("warning", "no_table_header",
                            "no product-table header recognised; rows accepted only where qty x price = amount"))
    n = 0
    for ln in lines:
        words = ln.words
        if len(words) < 3 or not any(c.isalpha() for c in ln.text):
            continue
        nums = [(j, parse_number(w.text, ocr_fix=False)) for j, w in enumerate(words)]
        nums = [(j, x) for j, x in nums if x.ok and not x.percent]
        if len(nums) < 3 or nums[-1][0] != len(words) - 1:
            continue
        if NON_ITEM_RE.search(ln.text) and not re.match(r"^\s*\d{1,3}\b", ln.text):
            continue
        total_j, total = nums[-1]
        found = None
        for (ja, a), (jb, b) in itertools.combinations(nums[:-1], 2):
            if a.value > 0 and b.value > 0 and close(a.value * b.value, total.value, abs_tol=Decimal("0.05")):
                found = (ja, a, jb, b)
                break
        if not found:
            res.issues.append(Issue("review", "unparsed_candidate_row",
                                    f"row looks like a product but the arithmetic did not match: '{ln.text}'"))
            continue
        ja, a, jb, b = found
        n += 1
        item = LineItem(line_no=n, raw_text=ln.text, method="arithmetic", y0=ln.y0, y1=ln.y1,
                        arithmetic_check="ok")
        item.ocr_confidence = round(ln.conf, 3)
        start = 1 if re.fullmatch(r"\d{1,3}[.)]?", words[0].text) and ja > 0 else 0
        desc_end = ja
        if ja - 1 > start and re.fullmatch(r"\d{4,8}", words[ja - 1].text):
            item.values["hsn_code"] = words[ja - 1].text
            desc_end = ja - 1
        item.values["product_name"] = " ".join(w.text for w in words[start:desc_end])
        item.values["quantity"], item.values["unit_price"] = words[ja].text, words[jb].text
        item.values["line_total"] = words[total_j].text
        unit = next((w.text for w in words[ja + 1:jb] if w.text.lower().strip(".") in UNITS), "")
        if unit:
            item.values["unit"] = unit
        item.issues.append(Issue("review", "columns_inferred",
                                 "columns inferred from arithmetic (no table header); verify quantity vs price", "", n))
        low = [w for w in words if w.conf < low_conf]
        if low:
            item.issues.append(Issue("review", "low_ocr_confidence",
                                     f"low OCR confidence reading '{' '.join(w.text for w in low)}'", "", n))
        if res.top is None:
            res.top = ln.y0
        res.bottom = ln.y1
        res.items.append(item)
    if not res.items:
        res.top = res.bottom = None
    return res
