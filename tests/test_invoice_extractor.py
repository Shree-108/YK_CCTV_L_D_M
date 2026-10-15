"""Invoice image -> CSV extraction: parsing/validation units (no OCR) + OCR end-to-end on generated samples."""
from __future__ import annotations

import csv
import importlib.util
import io
from decimal import Decimal
from pathlib import Path

import pytest

from invoice_extractor.dedupe import mark_duplicates
from invoice_extractor.fields import extract_fields
from invoice_extractor.layout import group_lines
from invoice_extractor.models import InvoiceResult, LineItem, Word
from invoice_extractor.numbers import parse_number
from invoice_extractor.store import OutputStore
from invoice_extractor.table import parse_table
from invoice_extractor.validate import validate_invoice, validate_item
from invoice_extractor.writer import PRODUCT_COLUMNS, product_rows, to_csv_text, write_csv

HAS_OCR = importlib.util.find_spec("rapidocr") is not None and importlib.util.find_spec("onnxruntime") is not None
needs_ocr = pytest.mark.skipif(not HAS_OCR, reason="RapidOCR not installed")


# ------------------------------------------------------------------ helpers
def row(y: float, cells: list[tuple[str, float]], h: float = 20, conf: float = 0.99) -> list[Word]:
    """Words of one printed line: (text, x0); width ~ 10 px per character."""
    return [Word(t, conf, x, y, x + 10 * len(t), y + h) for t, x in cells]


def lines_of(*rows: list[Word]):
    return group_lines([w for r in rows for w in r])


HEADER = [("Sl", 20), ("Description", 70), ("HSN", 330), ("Qty", 420), ("Unit", 480), ("Rate", 560), ("Amount", 680)]


def gst_like_lines():
    return lines_of(
        row(10, [("ACME", 20), ("TRADERS", 70), ("PVT", 150), ("LTD", 190)], h=30),
        row(60, [("Invoice", 20), ("No:", 100), ("INV-2024-0012", 140), ("Date:", 520), ("12/03/2024", 580)]),
        row(120, HEADER),
        row(160, [("1", 20), ("Steel", 70), ("Bolt", 125), ("M8", 170), ("7318", 330), ("10", 420), ("PCS", 480), ("12.50", 560), ("125.00", 680)]),
        row(200, [("2", 20), ("Copper", 70), ("Wire", 135), ("7408", 330), ("5", 420), ("KG", 480), ("450.00", 560), ("2,250.00", 680)]),
        row(225, [("2mm", 70), ("Insulated", 105)]),                       # wrapped description line
        row(265, [("3", 20), ("Steel", 70), ("Bolt", 125), ("M8", 170), ("7318", 330), ("10", 420), ("PCS", 480), ("12.50", 560), ("125.00", 680)]),
        row(320, [("Grand", 520), ("Total", 580), ("2500.00", 680)]),
    )


# ------------------------------------------------------------------ numbers
@pytest.mark.parametrize("raw,value,note", [
    ("1,250.00", "1250.00", False), ("₹ 1,25,000.00", "125000.00", False), ("Rs.450/-", "450", False),
    ("18%", "18", False), ("(25.00)", "-25.00", False), ("l0", "10", True), ("12,50", "12.50", True),
    ("1.234,56", "1234.56", True),
])
def test_parse_number(raw, value, note):
    n = parse_number(raw)
    assert n.ok and n.value == Decimal(value)
    assert bool(n.note) == note


@pytest.mark.parametrize("raw", ["", "abc", "1.2.3", "PCS", "12-03-2024"])
def test_parse_number_rejects(raw):
    assert not parse_number(raw).ok


# ------------------------------------------------------------------ table + fields
def test_table_columns_rows_and_wrapped_description():
    t = parse_table(gst_like_lines())
    assert t.method == "table"
    assert [it.values["product_name"] for it in t.items] == ["Steel Bolt M8", "Copper Wire 2mm Insulated", "Steel Bolt M8"]
    first = t.items[0].values
    assert (first["hsn_code"], first["quantity"], first["unit"], first["unit_price"], first["line_total"]) == \
        ("7318", "10", "PCS", "12.50", "125.00")
    assert t.items[1].values["line_total"] == "2,250.00"      # raw until validation normalises formatting


def test_repeated_products_are_not_merged():
    t = parse_table(gst_like_lines())
    same = [it for it in t.items if it.values["product_name"] == "Steel Bolt M8"]
    assert len(same) == 2 and {it.line_no for it in same} == {1, 3}


def test_invoice_fields():
    lines = gst_like_lines()
    t = parse_table(lines)
    f = extract_fields(lines, t.top, t.bottom)
    assert (f.invoice_number, f.invoice_date, f.vendor_name) == ("INV-2024-0012", "12/03/2024", "ACME TRADERS PVT LTD")
    assert f.invoice_total == "2500.00"
    assert not [i for i in f.issues if i.severity == "review"]


def test_missing_fields_are_flagged_not_invented():
    lines = lines_of(row(120, HEADER), row(160, [("1", 20), ("Widget", 70), ("3", 420), ("10.00", 560), ("30.00", 680)]))
    t = parse_table(lines)
    f = extract_fields(lines, t.top, t.bottom)
    assert f.invoice_number == "" and f.invoice_date == ""
    codes = {i.issue for i in f.issues}
    assert {"missing_invoice_number", "missing_invoice_date"} <= codes


def test_arithmetic_fallback_without_header():
    lines = lines_of(row(10, [("Kumar", 20), ("Hardware", 90), ("Stores", 180)]),
                     row(60, [("1", 20), ("Steel", 50), ("Bolt", 110), ("10", 400), ("12.50", 480), ("125.00", 600)]),
                     row(100, [("2", 20), ("Phone", 50), ("98765", 400), ("43210", 480), ("11", 600)]))
    t = parse_table(lines)
    assert t.method == "arithmetic" and len(t.items) == 1
    it = t.items[0]
    assert it.values["product_name"] == "Steel Bolt" and it.values["quantity"] == "10"
    assert it.needs_review                                   # inferred columns always need a human check


# ------------------------------------------------------------------ validation
def _item(**values) -> LineItem:
    return LineItem(line_no=1, values={"product_name": "X", **values})


def test_arithmetic_ok_with_discount_and_tax():
    it = _item(quantity="24", unit_price="135.00", discount="5", taxable_value="3078.00", tax_rate="5",
               tax_amount="153.90", line_total="3,231.90")
    validate_item(it)
    assert it.arithmetic_check == "ok" and not it.needs_review
    assert it.values["line_total"] == "3231.90"


def test_arithmetic_mismatch_flagged():
    it = _item(quantity="1", unit_price="140.00", line_total="150.00")
    validate_item(it)
    assert it.arithmetic_check == "mismatch" and it.needs_review


def test_ocr_corrected_value_kept_raw_and_flagged():
    it = _item(quantity="l0", unit_price="12.50", line_total="125.00")
    validate_item(it)
    assert it.values["quantity"] == "l0"                    # original text preserved in the CSV
    assert any(i.issue == "value_interpreted" for i in it.issues) and it.needs_review


def test_unreadable_number_flagged():
    it = _item(quantity="1O?", line_total="5")
    validate_item(it)
    assert any(i.issue == "not_a_number" and i.field == "quantity" for i in it.issues)


def test_no_items_is_an_error():
    r = InvoiceResult(source_file="a.png")
    validate_invoice(r, 60, 0.75)
    r.refresh_status()
    assert r.status == "failed"


def test_blurry_image_flagged():
    r = InvoiceResult(source_file="a.png", items=[_item(quantity="1", unit_price="5", line_total="5")],
                      quality={"blur_score": 12.0, "mean_confidence": 0.95})
    validate_invoice(r, 60, 0.75)
    assert any(i.issue == "blurry_image" for i in r.issues)


# ------------------------------------------------------------------ duplicates + output
def _res(name, sha, inv="INV-1", vendor="ACME"):
    return InvoiceResult(source_file=name, sha256=sha, invoice_number=inv, vendor_name=vendor, processed_at=name,
                         items=[_item(quantity="1", unit_price="5", line_total="5")])


def test_duplicate_invoice_detection():
    a, b, c = _res("a.png", "1"), _res("b.jpg", "2"), _res("c.jpg", "3", inv="INV-2")
    dups = mark_duplicates([a, b, c])
    assert dups == [b] and b.status == "duplicate" and b.duplicate_of == "a.png"
    assert len(product_rows([a, b, c])) == 2
    assert len(product_rows([a, b, c], keep_duplicates=True)) == 3


def test_same_invoice_number_different_vendor_is_not_duplicate():
    assert mark_duplicates([_res("a.png", "1", vendor="ACME"), _res("b.png", "2", vendor="Other Co")]) == []


def test_store_regenerates_without_duplicating_rows(tmp_path):
    store = OutputStore(tmp_path)
    store.save(_res("a.png", "1"))
    store.save(_res("a.png", "1"))                          # same content saved twice
    paths = store.write_reports()
    rows = list(csv.DictReader(io.StringIO(paths["products"].read_text(encoding="utf-8-sig"))))
    assert len(rows) == 1 and rows[0]["invoice_number"] == "INV-1"
    store.save(_res("a.png", "9"))                          # file re-scanned: replaces old result
    assert [r.sha256 for r in store.load_all()] == ["9"]


def test_csv_formula_injection_neutralised():
    text = to_csv_text(PRODUCT_COLUMNS, [{"product_name": "=HYPERLINK(1)", "discount": "-5"}])
    rec = next(csv.DictReader(io.StringIO(text)))
    assert rec["product_name"].startswith("'=") and rec["discount"] == "-5"


def test_locked_csv_falls_back_to_new_name(tmp_path, monkeypatch):
    target = tmp_path / "products.csv"
    real_replace = Path.replace

    def locked(self, dst):
        if Path(dst) == target:
            raise PermissionError("locked by Excel")
        return real_replace(self, dst)

    monkeypatch.setattr(Path, "replace", locked)
    out = write_csv(target, ["a"], [{"a": "1"}])
    assert out != target and out.exists()


# ------------------------------------------------------------------ OCR end-to-end on generated samples
@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    from invoice_extractor.samples import generate
    return {p.name: p for p in generate(tmp_path_factory.mktemp("samples"))}


@pytest.fixture(scope="module")
def extractor():
    if not HAS_OCR:
        pytest.skip("RapidOCR not installed")
    from invoice_extractor import InvoiceExtractor
    return InvoiceExtractor()


@needs_ocr
def test_ocr_gst_invoice(samples, extractor):
    r = extractor.extract_file(samples["sample1_gst_tax_invoice.png"])
    assert r.invoice_number == "SBT/2024/0457" and r.invoice_date == "14-03-2024"
    assert r.vendor_name == "SHREE BALAJI TRADERS PVT LTD" and r.vendor_gstin == "27AABCS1234F1Z5"
    assert len(r.items) == 5
    names = [it.values.get("product_name") for it in r.items]
    assert names.count("Basmati Rice 5kg Bag") == 2                      # repeated product kept twice
    oil = r.items[1].values
    assert oil["product_name"] == "Sunflower Oil 1L Refined, Pouch"
    assert (oil["quantity"], oil["unit_price"], oil["discount"], oil["tax_amount"], oil["line_total"]) == \
        ("24", "135.00", "5", "153.90", "3231.90")
    assert all(it.arithmetic_check == "ok" for it in r.items)
    assert r.quality["sum_line_totals"] == "12767.10"


@needs_ocr
def test_ocr_retail_bill_flags_wrong_total(samples, extractor):
    r = extractor.extract_file(samples["sample2_retail_bill.jpg"])
    assert r.invoice_number == "88213" and len(r.items) == 4
    tea = r.items[3]
    assert tea.arithmetic_check == "mismatch" and tea.needs_review
    assert not any(it.needs_review for it in r.items[:3])


@needs_ocr
def test_ocr_unreadable_file_fails_gracefully(tmp_path, extractor):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    r = extractor.extract_file(bad)
    assert r.status == "failed" and r.issues[0].issue == "extraction_failed"


@needs_ocr
def test_cli_end_to_end(samples, tmp_path):
    from invoice_extractor.cli import main
    in_dir = next(iter(samples.values())).parent
    out = tmp_path / "out"
    assert main(["--input", str(in_dir), "--output", str(out), "--log-level", "WARNING"]) == 0
    rows = list(csv.DictReader(io.StringIO((out / "products.csv").read_text(encoding="utf-8-sig"))))
    files = {r["source_file"] for r in rows}
    assert "sample3_blurry_rephoto_of_sample1.jpg" not in files         # duplicate of sample1
    assert {"sample1_gst_tax_invoice.png", "sample2_retail_bill.jpg"} <= files
    summary = {r["source_file"]: r for r in csv.DictReader(io.StringIO((out / "invoices_summary.csv").read_text(encoding="utf-8-sig")))}
    assert summary["sample3_blurry_rephoto_of_sample1.jpg"]["status"] == "duplicate"
    assert (out / "extraction_errors.csv").exists()
    # second run: nothing re-processed, same rows
    assert main(["--input", str(in_dir), "--output", str(out), "--log-level", "WARNING"]) == 0
    assert len(list(csv.DictReader(io.StringIO((out / "products.csv").read_text(encoding="utf-8-sig"))))) == len(rows)


@needs_ocr
def test_api_extract_returns_csv(samples):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.routes_invoice import router
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    assert c.get("/api/invoices/engine").json()["available"] is True
    p = samples["sample2_retail_bill.jpg"]
    r = c.post("/api/invoices/extract", files=[("files", (p.name, p.read_bytes(), "image/jpeg")),
                                                ("files", ("copy.jpg", p.read_bytes(), "image/jpeg"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["totals"]["rows"] == 4 and body["totals"]["rows_needing_review"] == 1
    assert body["skipped_identical"] == ["copy.jpg"]
    assert body["csv"]["products"].startswith("source_file,invoice_number")
    bad = c.post("/api/invoices/extract", files=[("files", ("x.pdf", b"%PDF", "application/pdf"))])
    assert bad.status_code == 415
