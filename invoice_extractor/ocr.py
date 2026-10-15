"""OCR engines. All return word-level tokens with boxes so the table parser can use columns.

* ``rapidocr`` (default): PP-OCR models on ONNX Runtime. Pure ``pip install``, models ship with
  the package, runs fully offline on CPU. No admin installer needed on Windows.
* ``tesseract`` (optional): needs the Tesseract program installed separately (UB Mannheim build
  on Windows) plus ``pytesseract``.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from invoice_extractor.models import Word

log = logging.getLogger(__name__)

TESSERACT_WINDOWS_PATHS = [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                           r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                           os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")]


class OcrUnavailable(RuntimeError):
    pass


class OcrEngine(Protocol):
    name: str

    def read(self, img_bgr: np.ndarray) -> list[Word]: ...


def _quad_angle(quad) -> float:
    (x0, y0), (x1, y1) = quad[0], quad[1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0)) if x1 != x0 else 0.0


def _bbox(quad) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    return min(xs), min(ys), max(xs), max(ys)


class RapidOcrEngine:
    name = "rapidocr"

    def __init__(self) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise OcrUnavailable("RapidOCR is not installed: pip install rapidocr onnxruntime") from exc
        try:
            self._ocr = RapidOCR(params={"Global.log_level": "error"})
        except Exception:   # older/newer param schema
            self._ocr = RapidOCR()
        self._lock = threading.Lock()    # one inference at a time per engine instance

    def read(self, img_bgr: np.ndarray) -> list[Word]:
        with self._lock:
            r = self._ocr(img_bgr, return_word_box=True)
        if r is None or r.boxes is None or r.txts is None:
            return []
        word_results = getattr(r, "word_results", None) or ()
        words: list[Word] = []
        for seg, (quad, text, score) in enumerate(zip(r.boxes, r.txts, r.scores)):
            angle = _quad_angle(quad)
            seg_words = word_results[seg] if seg < len(word_results) else ()
            parsed = []
            for item in seg_words or ():
                try:
                    wtext, wscore, wquad = item
                except (TypeError, ValueError):
                    parsed = []
                    break
                if str(wtext).strip():
                    parsed.append(Word(str(wtext).strip(), float(wscore), *_bbox(wquad), seg=seg, angle=angle))
            if parsed and " ".join(w.text for w in parsed).replace(" ", "") == str(text).replace(" ", ""):
                words.extend(parsed)
            else:
                words.extend(_split_segment(str(text), float(score), _bbox(quad), seg, angle))
        return words


def _split_segment(text: str, score: float, box, seg: int, angle: float) -> list[Word]:
    """Fallback when word boxes are unavailable: split on spaces, place words proportionally."""
    x0, y0, x1, y1 = box
    tokens = text.split()
    if not tokens:
        return []
    total = max(1, len(text))
    out, pos = [], 0
    for tok in tokens:
        start = text.index(tok, pos)
        end = start + len(tok)
        pos = end
        out.append(Word(tok, score, x0 + (x1 - x0) * start / total, y0, x0 + (x1 - x0) * end / total, y1,
                        seg=seg, angle=angle))
    return out


def find_tesseract() -> str | None:
    cand = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if cand:
        return cand
    return next((p for p in TESSERACT_WINDOWS_PATHS if Path(p).is_file()), None)


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: str = "eng") -> None:
        try:
            import pytesseract
        except ImportError as exc:
            raise OcrUnavailable("pytesseract is not installed: pip install pytesseract") from exc
        cmd = find_tesseract()
        if not cmd:
            raise OcrUnavailable("Tesseract program not found. Install it from "
                                 "https://github.com/UB-Mannheim/tesseract/wiki or set TESSERACT_CMD")
        pytesseract.pytesseract.tesseract_cmd = cmd
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            raise OcrUnavailable(f"Tesseract at {cmd} does not run: {exc}") from exc
        self._pt, self.lang = pytesseract, lang

    def read(self, img_bgr: np.ndarray) -> list[Word]:
        rgb = img_bgr[:, :, ::-1]
        d = self._pt.image_to_data(rgb, lang=self.lang, config="--oem 1 --psm 4",
                                   output_type=self._pt.Output.DICT)
        words, seg_ids = [], {}
        for i, text in enumerate(d["text"]):
            text = (text or "").strip()
            conf = float(d["conf"][i])
            if not text or conf < 0:
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            seg = seg_ids.setdefault(key, len(seg_ids))
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            words.append(Word(text, conf / 100.0, x, y, x + w, y + h, seg=seg))
        return words


def create_engine(name: str = "auto") -> OcrEngine:
    name = (name or "auto").lower()
    if name == "rapidocr":
        return RapidOcrEngine()
    if name == "tesseract":
        return TesseractEngine()
    if name != "auto":
        raise ValueError(f"unknown OCR engine '{name}' (use auto, rapidocr or tesseract)")
    errors = []
    for cls in (RapidOcrEngine, TesseractEngine):
        try:
            eng = cls()
            log.info("OCR engine: %s", eng.name)
            return eng
        except OcrUnavailable as exc:
            errors.append(str(exc))
    raise OcrUnavailable("No OCR engine available. " + " | ".join(errors))
