"""Vision data-URL handling via Presidio Image Redactor. Fail-closed without Tesseract."""

from __future__ import annotations

import base64
import io
import logging
import shutil
from typing import Any

from enigmatic.presidio_ops.mapping import SessionMapping

logger = logging.getLogger("enigmatic.images")

MISSING_TESSERACT = (
    "[image omitted: Tesseract is not installed; Enigmatic fail-closed on vision]"
)


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def tesseract_status() -> dict[str, Any]:
    path = shutil.which("tesseract")
    return {"ok": path is not None, "path": path}


def redact_data_url(data_url: str, mapping: SessionMapping) -> str:
    """OCR + box overlay. On missing Tesseract or engine errors, drop the image."""
    if not data_url.startswith("data:image"):
        return data_url
    if not tesseract_available():
        return MISSING_TESSERACT
    try:
        from PIL import Image, ImageDraw, ImageFont
        from presidio_image_redactor import ImageAnalyzerEngine
    except Exception as exc:
        logger.warning("image redactor unavailable: %s", exc)
        return MISSING_TESSERACT

    try:
        header, _, b64 = data_url.partition(",")
        if not b64:
            return MISSING_TESSERACT
        raw = base64.b64decode(b64)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        engine = ImageAnalyzerEngine()
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
