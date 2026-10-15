"""Command line: process every JPG/JPEG/PNG in the input folder and (re)write the CSV reports.

    python -m invoice_extractor                      # invoices/input -> invoices/output
    python -m invoice_extractor --input D:\\bills --output D:\\bills\\out
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

from invoice_extractor.dedupe import mark_duplicates
from invoice_extractor.ocr import OcrUnavailable
from invoice_extractor.pipeline import ExtractSettings, InvoiceExtractor, sha256_bytes
from invoice_extractor.preprocess import SUPPORTED_EXTS
from invoice_extractor.store import OutputStore
from invoice_extractor.writer import product_rows

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "invoices" / "input"
DEFAULT_OUTPUT = PROJECT_ROOT / "invoices" / "output"

log = logging.getLogger("invoice_extractor")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m invoice_extractor",
                                description="Extract product line items from invoice images into CSV (local OCR).")
    p.add_argument("--input", "-i", type=Path, default=Path(os.environ.get("INVOICE_INPUT_DIR", DEFAULT_INPUT)),
                   help="folder with invoice images (default: invoices/input or $INVOICE_INPUT_DIR)")
    p.add_argument("--output", "-o", type=Path, default=Path(os.environ.get("INVOICE_OUTPUT_DIR", DEFAULT_OUTPUT)),
                   help="folder for CSV reports (default: invoices/output or $INVOICE_OUTPUT_DIR)")
    p.add_argument("--engine", choices=["auto", "rapidocr", "tesseract"],
                   default=os.environ.get("INVOICE_OCR_ENGINE", "auto"), help="OCR engine (default: auto)")
    p.add_argument("--recursive", "-r", action="store_true", help="also scan sub-folders of the input folder")
    p.add_argument("--reprocess", action="store_true", help="re-run OCR even on images processed before")
    p.add_argument("--keep-duplicates", action="store_true",
                   help="include rows of invoices detected as duplicates in products.csv")
    p.add_argument("--low-confidence", type=float, default=0.75, help="word OCR confidence that triggers review (0-1)")
    p.add_argument("--blur-threshold", type=float, default=60.0, help="sharpness score below which an image is flagged")
    p.add_argument("--no-retry", action="store_true", help="disable the enhanced/rotated second OCR passes")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def setup_logging(out_dir: Path, level: str) -> Path:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"extraction_{dt.datetime.now():%Y%m%d}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(fh)
    for noisy in ("RapidOCR", "rapidocr", "onnxruntime", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file


def find_images(folder: Path, recursive: bool, exclude: Path) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    exclude = exclude.resolve()
    out = []
    for p in folder.glob(pattern):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and exclude not in p.resolve().parents:
            out.append(p)
    return sorted(out, key=lambda p: str(p).lower())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_dir, out_dir = args.input.resolve(), args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(out_dir, args.log_level)
    if not in_dir.is_dir():
        in_dir.mkdir(parents=True, exist_ok=True)
        log.warning("input folder %s did not exist; created it. Put invoice images there and run again.", in_dir)

    images = find_images(in_dir, args.recursive, out_dir)
    log.info("input: %s (%d image(s)), output: %s", in_dir, len(images), out_dir)
    store = OutputStore(out_dir)
    todo = []
    for p in images:
        try:
            sha = sha256_bytes(p.read_bytes())
        except OSError as exc:
            log.error("%s: cannot read (%s)", p.name, exc)
            continue
        if store.has(sha) and not args.reprocess:
            log.info("%s: already processed (same file content), skipping", p.name)
            continue
        todo.append(p)

    processed = 0
    if todo:
        settings = ExtractSettings(engine=args.engine, low_confidence=args.low_confidence,
                                   blur_threshold=args.blur_threshold, retry_enhanced=not args.no_retry,
                                   try_rotations=not args.no_retry)
        try:
            extractor = InvoiceExtractor(settings)
        except OcrUnavailable as exc:
            log.error("%s", exc)
            log.error("Install the OCR engine: pip install -r invoice_extractor/requirements.txt (see README)")
            return 2
        for n, p in enumerate(todo, start=1):
            log.info("[%d/%d] %s", n, len(todo), p.name)
            res = extractor.extract_file(p)
            res.source_file = str(p.relative_to(in_dir)) if args.recursive else p.name
            store.save(res)
            processed += 1

    paths = store.write_reports(keep_duplicates=args.keep_duplicates)
    results = store.load_all()
    mark_duplicates(results)
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    product = product_rows(results, args.keep_duplicates)
    rows, review_rows = len(product), sum(r["needs_review"] == "YES" for r in product)

    print()
    print("=" * 72)
    print(f" Processed this run : {processed} image(s)   (skipped already-processed: {len(images) - len(todo)})")
    print(f" Invoices in report : {len(results)}  " + "  ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    print(f" Product rows       : {rows}   rows flagged for manual review: {review_rows}")
    print(f" products.csv       : {paths['products']}")
    print(f" review report      : {paths['review']}")
    print(f" invoice summary    : {paths['summary']}")
    print(f" log                : {log_file}")
    print("-" * 72)
    print(" OCR extraction is not 100% accurate. Check every row with needs_review = YES")
    print(" against the original image before using the data.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
