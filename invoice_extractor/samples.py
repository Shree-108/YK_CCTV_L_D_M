"""Generate synthetic sample invoices for testing (no real invoices are shipped with the project).

    python -m invoice_extractor.samples --out invoices/samples

Creates four layouts: a GST tax invoice with a ruled table (wrapped description, discount, a
legitimately repeated product), a plain retail bill with one deliberately wrong line total, a
blurry tilted re-photo of the first invoice (=> duplicate + blur checks), and a cash memo with no
column header (=> arithmetic fallback).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def gst_invoice() -> Image.Image:
    W, H = 1240, 1754
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f, fb, big = _font(20), _font(20, True), _font(34, True)
    d.text((60, 50), "SHREE BALAJI TRADERS PVT LTD", font=big, fill="black")
    d.text((60, 100), "Plot 14, MIDC Industrial Area, Pune 411019", font=f, fill="black")
    d.text((60, 128), "GSTIN: 27AABCS1234F1Z5    Phone: 020-2745 1180", font=f, fill="black")
    d.text((520, 180), "TAX INVOICE", font=_font(28, True), fill="black")
    d.text((60, 240), "Invoice No: SBT/2024/0457", font=fb, fill="black")
    d.text((760, 240), "Invoice Date: 14-03-2024", font=fb, fill="black")
    d.text((60, 290), "Bill To:", font=fb, fill="black")
    d.text((60, 318), "Yatrikart Retail LLP, Koramangala, Bengaluru", font=f, fill="black")
    d.text((60, 346), "GSTIN: 29AAACY9876K1Z3", font=f, fill="black")
    cols = [("Sl", 60), ("Description", 110), ("HSN", 430), ("Qty", 520), ("Unit", 590), ("Rate", 670),
            ("Disc%", 770), ("Taxable", 850), ("GST%", 960), ("GST Amt", 1030), ("Amount", 1130)]
    right = 1200
    top = 400
    d.rectangle((50, top, right, top + 44), outline="black", width=2)
    for name, x in cols:
        d.text((x, top + 11), name, font=fb, fill="black")
    rows = [
        ["1", ["Basmati Rice 5kg Bag"], "1006", "10", "BAG", "450.00", "0", "4500.00", "5", "225.00", "4725.00"],
        ["2", ["Sunflower Oil 1L", "Refined, Pouch"], "1512", "24", "PCS", "135.00", "5", "3078.00", "5", "153.90", "3231.90"],
        ["3", ["Toor Dal Premium"], "0713", "15", "KG", "160.00", "0", "2400.00", "5", "120.00", "2520.00"],
        ["4", ["Detergent Powder 1kg"], "3402", "12", "PCS", "95.00", "0", "1140.00", "18", "205.20", "1345.20"],
        ["5", ["Basmati Rice 5kg Bag"], "1006", "2", "BAG", "450.00", "0", "900.00", "5", "45.00", "945.00"],
    ]
    y = top + 44
    for r in rows:
        h = 40 + 28 * (len(r[1]) - 1)
        d.rectangle((50, y, right, y + h), outline="black", width=1)
        for (name, x), val in zip(cols, r):
            if isinstance(val, list):
                for k, part in enumerate(val):
                    d.text((x, y + 10 + 28 * k), part, font=f, fill="black")
            else:
                d.text((x, y + 10), val, font=f, fill="black")
        y += h
    for _, x in cols[1:]:
        d.line((x - 8, top, x - 8, y), fill="black", width=1)
    y += 30
    d.text((760, y), "Total Taxable Value", font=f, fill="black"); d.text((1100, y), "12018.00", font=f, fill="black")
    d.text((760, y + 34), "Total GST", font=f, fill="black"); d.text((1100, y + 34), "749.10", font=f, fill="black")
    d.text((760, y + 72), "Grand Total", font=fb, fill="black"); d.text((1100, y + 72), "12767.10", font=fb, fill="black")
    d.text((60, y + 140), "Amount in words: Rupees Twelve Thousand Seven Hundred Sixty Seven and Ten Paise Only", font=f, fill="black")
    d.text((60, y + 200), "Bank: HDFC Bank  A/c 50200012345678  IFSC HDFC0001234", font=f, fill="black")
    d.text((900, y + 300), "For SHREE BALAJI TRADERS", font=f, fill="black")
    return im


def retail_bill() -> Image.Image:
    W, H = 800, 1000
    im = Image.new("RGB", (W, H), (250, 248, 240))
    d = ImageDraw.Draw(im)
    f, fb = _font(22), _font(22, True)
    d.text((220, 40), "GREEN LEAF SUPERMART", font=_font(30, True), fill="black")
    d.text((240, 85), "MG Road, Bengaluru 560001", font=f, fill="black")
    d.text((60, 150), "Bill No. 88213", font=f, fill="black")
    d.text((470, 150), "Date: 02/04/2024", font=f, fill="black")
    d.line((50, 200, 750, 200), fill="black", width=2)
    xs = [60, 420, 530, 650]
    for name, x in zip(["Item", "Qty", "Price", "Total"], xs):
        d.text((x, 212), name, font=fb, fill="black")
    d.line((50, 250, 750, 250), fill="black", width=2)
    rows = [("Milk 1L", "2", "64.00", "128.00"), ("Bread Brown", "1", "45.00", "45.00"),
            ("Eggs Tray 12", "2", "84.00", "168.00"), ("Tea Powder 250g", "1", "140.00", "150.00")]
    y = 265
    for r in rows:
        for v, x in zip(r, xs):
            d.text((x, y), v, font=f, fill="black")
        y += 42
    d.line((50, y + 5, 750, y + 5), fill="black", width=2)
    d.text((420, y + 20), "Sub Total", font=f, fill="black"); d.text((650, y + 20), "491.00", font=f, fill="black")
    d.text((420, y + 60), "Grand Total", font=fb, fill="black"); d.text((650, y + 60), "491.00", font=fb, fill="black")
    d.text((250, y + 140), "Thank you! Visit again", font=f, fill="black")
    return im


def blurry_rephoto(base: Image.Image) -> Image.Image:
    im = base.rotate(1.8, resample=Image.BICUBIC, expand=True, fillcolor=(235, 235, 230))
    im = im.resize((im.width * 3 // 4, im.height * 3 // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1.6))
    arr = np.asarray(im).astype(np.int16)
    rng = np.random.default_rng(7)
    arr = np.clip(arr + rng.normal(0, 7, arr.shape) - 12, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def cash_memo() -> Image.Image:
    W, H = 900, 700
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f = _font(24)
    d.text((60, 40), "Kumar Hardware Stores", font=_font(32, True), fill="black")
    d.text((60, 100), "Cash Memo No: CM-1189      Date: 21 Mar 2024", font=f, fill="black")
    rows = [("1  Steel Bolt M8", "10", "12.50", "125.00"), ("2  PVC Pipe 1 inch", "4", "180.00", "720.00"),
            ("3  Wall Putty 5kg", "2", "310.00", "620.00")]
    y = 200
    for desc, q, p, t in rows:
        d.text((60, y), desc, font=f, fill="black")
        d.text((480, y), q, font=f, fill="black")
        d.text((580, y), p, font=f, fill="black")
        d.text((730, y), t, font=f, fill="black")
        y += 50
    d.text((480, y + 40), "Total 1465.00", font=_font(24, True), fill="black")
    return im


def generate(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    gst = gst_invoice()
    files = [(gst, "sample1_gst_tax_invoice.png"), (retail_bill(), "sample2_retail_bill.jpg"),
             (blurry_rephoto(gst), "sample3_blurry_rephoto_of_sample1.jpg"), (cash_memo(), "sample4_cash_memo_no_header.png")]
    paths = []
    for im, name in files:
        p = out / name
        im.save(p, quality=80) if p.suffix == ".jpg" else im.save(p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "invoices" / "samples")
    for p in generate(ap.parse_args().out):
        print(p)
