"""Image loading and clean-up before OCR."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


class ImageLoadError(Exception):
    pass


def load_image(src: Union[str, Path, bytes]) -> np.ndarray:
    """Decode to BGR, honouring EXIF rotation (phone photos). Works with non-ASCII Windows paths."""
    try:
        data = src if isinstance(src, bytes) else Path(src).read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)
            rgb = np.asarray(im.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"cannot decode image: {exc}") from exc
    if rgb.size == 0 or min(rgb.shape[:2]) < 32:
        raise ImageLoadError(f"image too small ({rgb.shape[1]}x{rgb.shape[0]} px)")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def blur_score(img: np.ndarray) -> float:
    """Variance of the Laplacian at a fixed width; low values mean a blurry/out-of-focus image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = 1000 / gray.shape[1]
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalize_size(img: np.ndarray, min_long: int = 1600, max_long: int = 3500) -> np.ndarray:
    """Upscale small images (small text OCRs badly) and cap huge ones (speed)."""
    long = max(img.shape[:2])
    if long < min_long:
        s = min_long / long
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    if long > max_long:
        s = max_long / long
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def enhance(img: np.ndarray) -> np.ndarray:
    """Second-pass clean-up for faint/blurry scans: local contrast (CLAHE) + unsharp mask."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def rotate90(img: np.ndarray, clockwise: bool) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if clockwise else cv2.ROTATE_90_COUNTERCLOCKWISE)
