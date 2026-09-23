"""Vision data-URL handling via Presidio Image Redactor. Fail-closed without Tesseract."""

from __future__ import annotations

import base64
import io
import logging
import shutil
import threading
from typing import Any

from enigmatic.presidio_ops.mapping import SessionMapping

logger = logging.getLogger("enigmatic.images")

MISSING_TESSERACT = (
    "[image omitted: Tesseract is not installed; Enigmatic fail-closed on vision]"
)
IMAGE_TOO_LARGE = "[image omitted: larger than Enigmatic's image size cap]"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
_ENGINE: Any = None
_ENGINE_LOCK = threading.Lock()


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def tesseract_status() -> dict[str, Any]:
    path = shutil.which("tesseract")
    return {"ok": path is not None, "path": path}


def _image_engine() -> Any:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            from presidio_image_redactor import ImageAnalyzerEngine

            _ENGINE = ImageAnalyzerEngine()
        return _ENGINE


def redact_data_url(data_url: str, mapping: SessionMapping) -> str:
    """OCR + box overlay. On missing Tesseract, oversize input, or engine errors, drop the image."""
    if not data_url.startswith("data:image"):
        return data_url
    header, _, b64 = data_url.partition(",")
    if not b64 or (len(b64) * 3) // 4 > MAX_IMAGE_BYTES:
        return IMAGE_TOO_LARGE if b64 else MISSING_TESSERACT
    if not tesseract_available():
        return MISSING_TESSERACT
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        logger.warning("image redactor unavailable: %s", exc)
        return MISSING_TESSERACT

    try:
        raw = base64.b64decode(b64)
        if len(raw) > MAX_IMAGE_BYTES:
            return IMAGE_TOO_LARGE
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except Image.DecompressionBombError:
            return IMAGE_TOO_LARGE
        image = image.convert("RGB")
        engine = _image_engine()
        results = engine.analyze(image)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for result in results:
            entity_type = getattr(result, "entity_type", "ENTITY")
            left = int(getattr(result, "left", 0))
            top = int(getattr(result, "top", 0))
            width = int(getattr(result, "width", 0))
            height = int(getattr(result, "height", 0))
            ocr_text = getattr(result, "text", None)
            if not isinstance(ocr_text, str) or not ocr_text:
                ocr_text = entity_type
            token = mapping.placeholder_for(str(entity_type), ocr_text)
            box = [left, top, left + max(width, 8), top + max(height, 8)]
            draw.rectangle(box, fill=(20, 20, 20))
            draw.text((left + 2, top + 2), token, fill=(240, 240, 240), font=font)
        mime = "image/png"
        if "jpeg" in header or "jpg" in header:
            mime = "image/jpeg"
        buffer = io.BytesIO()
        fmt = "JPEG" if mime.endswith("jpeg") else "PNG"
        image.save(buffer, format=fmt)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception as exc:
        logger.warning("image redaction failed closed: %s", exc)
        return MISSING_TESSERACT
