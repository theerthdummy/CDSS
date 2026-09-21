"""document_ocr.py - Medical Document OCR & Text Extraction Service.

Adapted from PilotMaster DocPilot ingestion pipeline.
Supports PDF (text + OCR fallback), JPEG/PNG images, and DOCX.
Designed for clinical prescriptions and medical reports.
No PilotMaster dependencies required — self-contained.
"""

import logging
import mimetypes
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("orchestrator.clinical_assistant.document_ocr")

# ---------------------------------------------------------------------------
# Optional heavy dependencies – imported lazily so server starts even without
# them installed. Clear error messages are raised at call-time if missing.
# ---------------------------------------------------------------------------

def _require(pkg_name: str, import_path: Optional[str] = None):
    """Import a package or raise a clear ImportError."""
    import importlib
    try:
        return importlib.import_module(import_path or pkg_name)
    except ImportError:
        raise ImportError(
            f"'{pkg_name}' is required for document OCR. "
            f"Install it: pip install {pkg_name}"
        )


# Tesseract path resolution (Windows + Linux/Docker compatible)
def _configure_tesseract():
    try:
        import pytesseract
        binary = shutil.which("tesseract")
        if not binary and os.name == "nt":
            binary = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if binary:
            pytesseract.pytesseract.tesseract_cmd = binary
        return pytesseract
    except ImportError:
        return None


pytesseract_mod = _configure_tesseract()


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class TextExtractionError(Exception):
    """Raised when text extraction fails or yields empty content."""
    pass


@dataclass
class TextSection:
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalise whitespace and strip control characters."""
    if not text:
        return ""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_extension(file_path: str, mime_type: Optional[str] = None) -> str:
    """Return the lowercase file extension, falling back to mime detection."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext:
        return ext
    if mime_type:
        guessed = mimetypes.guess_extension(mime_type)
        return guessed or ""
    return ""


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _extract_pdf_sections(file_path: str) -> List[TextSection]:
    """
    Extract text from PDF pages.
    Primary: pypdf text layer.
    Fallback: pdf2image + pytesseract OCR (for scanned/image-only PDFs).
    """
    sections: List[TextSection] = []
    try:
        pypdf = _require("pypdf")
        reader = pypdf.PdfReader(file_path)
        text_layer_chars = 0

        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _clean_text(raw)
            if cleaned:
                text_layer_chars += len(cleaned)
                sections.append(
                    TextSection(
                        text=cleaned,
                        metadata={"page": page_num, "element_type": "text", "ocr": False},
                    )
                )

        if text_layer_chars >= 100:
            logger.info("PDF text layer extraction: %d chars across %d pages", text_layer_chars, len(sections))
            return sections

        # Insufficient text layer → fall back to OCR
        logger.info("PDF text layer sparse (%d chars). Falling back to OCR.", text_layer_chars)

    except Exception as exc:
        logger.warning("pypdf extraction failed: %s – trying OCR fallback.", exc)

    # OCR fallback via pdf2image
    return _extract_pdf_ocr_sections(file_path)


def _extract_pdf_ocr_sections(file_path: str) -> List[TextSection]:
    """Convert PDF pages to images then run Tesseract OCR."""
    if not pytesseract_mod:
        raise TextExtractionError(
            "PDF appears to be image-only (scanned) and pytesseract is not installed. "
            "Install pytesseract and Tesseract-OCR to process scanned PDFs."
        )
    try:
        pdf2image = _require("pdf2image")
        images = pdf2image.convert_from_path(file_path, dpi=300)
        sections = []
        for page_num, img in enumerate(images, start=1):
            raw = pytesseract_mod.image_to_string(img)
            cleaned = _clean_text(raw)
            if cleaned:
                sections.append(
                    TextSection(
                        text=cleaned,
                        metadata={"page": page_num, "element_type": "ocr_page", "ocr": True},
                    )
                )
        logger.info("PDF OCR extraction: %d pages processed", len(sections))
        return sections
    except TextExtractionError:
        raise
    except Exception as exc:
        logger.exception("PDF OCR extraction failed: %s", exc)
        raise TextExtractionError("Could not extract text from scanned PDF via OCR.") from exc


def _extract_image_sections(file_path: str) -> List[TextSection]:
    """Run Tesseract OCR on an image file (JPEG, PNG, WEBP)."""
    if not pytesseract_mod:
        raise TextExtractionError(
            "pytesseract is not installed. Install pytesseract and Tesseract-OCR to process image files."
        )
    try:
        PIL = _require("PIL.Image", "PIL")
        Image = getattr(PIL, "Image", None) or _require("PIL.Image", "PIL.Image")
        # Handle both `from PIL import Image` and `import PIL.Image`
        import PIL.Image as PILImage
        image = PILImage.open(file_path)
        raw = pytesseract_mod.image_to_string(image)
        cleaned = _clean_text(raw)
        logger.info("Image OCR extracted %d chars", len(cleaned))
        if not cleaned:
            raise TextExtractionError("OCR returned no text from this image. The image may be unclear or empty.")
        return [
            TextSection(
                text=cleaned,
                metadata={"element_type": "image", "ocr": True},
            )
        ]
    except TextExtractionError:
        raise
    except Exception as exc:
        logger.exception("Image OCR failed: %s", exc)
        raise TextExtractionError("Could not extract text from image via OCR.") from exc


def _extract_docx_sections(file_path: str) -> List[TextSection]:
    """Extract paragraphs from a DOCX file."""
    try:
        docx = _require("docx", "docx")
        Document = getattr(docx, "Document", None)
        if Document is None:
            from docx import Document  # type: ignore
        doc = Document(file_path)
        paragraphs = [_clean_text(p.text) for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
        if not full_text:
            raise TextExtractionError("DOCX document appears to be empty.")
        return [TextSection(text=full_text, metadata={"element_type": "text", "ocr": False})]
    except TextExtractionError:
        raise
    except Exception as exc:
        logger.exception("DOCX extraction failed: %s", exc)
        raise TextExtractionError("Could not read DOCX document.") from exc


def _extract_txt_sections(file_path: str) -> List[TextSection]:
    """Extract plain text files."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            content = _clean_text(fh.read())
        if not content:
            raise TextExtractionError("Text file is empty.")
        return [TextSection(text=content, metadata={"element_type": "text", "ocr": False})]
    except TextExtractionError:
        raise
    except Exception as exc:
        raise TextExtractionError("Could not read text file.") from exc


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_EXTRACTORS = {
    ".pdf":  _extract_pdf_sections,
    ".docx": _extract_docx_sections,
    ".txt":  _extract_txt_sections,
    ".md":   _extract_txt_sections,
    ".png":  _extract_image_sections,
    ".jpg":  _extract_image_sections,
    ".jpeg": _extract_image_sections,
    ".webp": _extract_image_sections,
}

SUPPORTED_EXTENSIONS = list(_EXTRACTORS.keys())


def extract_text_from_file(file_path: str, mime_type: Optional[str] = None) -> str:
    """
    Extract all text from a medical document file.

    Args:
        file_path: Absolute path to the uploaded file.
        mime_type: Optional MIME type hint (e.g. 'application/pdf').

    Returns:
        Extracted plain text string (may be multi-page concatenation).

    Raises:
        TextExtractionError: If extraction fails or yields no content.
    """
    ext = _detect_extension(file_path, mime_type)
    extractor = _EXTRACTORS.get(ext)

    if not extractor:
        raise TextExtractionError(
            f"Unsupported file type '{ext or mime_type}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    logger.info("Extracting text from '%s' (ext=%s)", os.path.basename(file_path), ext)
    sections = extractor(file_path)

    # Filter and join
    cleaned = [s.text for s in sections if s.text.strip()]
    if not cleaned:
        raise TextExtractionError("Extraction produced no usable text content.")

    full_text = "\n\n".join(cleaned)
    logger.info("Total extracted text length: %d chars", len(full_text))
    return full_text


# ---------------------------------------------------------------------------
# Prescription / Clinical Document Parser
# ---------------------------------------------------------------------------

# Regex patterns for common prescription fields
_RX_PATTERNS = {
    "patient_name":   re.compile(r"(?:patient|name|pt)[\s:]+([A-Za-z\s\.]+)", re.IGNORECASE),
    "date":           re.compile(r"(?:date|dt)[\s:]+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", re.IGNORECASE),
    "diagnosis":      re.compile(r"(?:diagnosis|dx|impression|assessment)[\s:]+([^\n]+)", re.IGNORECASE),
    "medications":    re.compile(r"(?:rx|medication|drug|medicine|tab|cap|inj|syrup)[\s:]+([^\n]+)", re.IGNORECASE),
    "dosage":         re.compile(r"(\d+\s*(?:mg|mcg|ml|g|IU|units?))[\s,]", re.IGNORECASE),
    "frequency":      re.compile(r"\b(once|twice|thrice|od|bd|tds|qid|prn|sos|stat|daily|weekly)\b", re.IGNORECASE),
    "doctor":         re.compile(r"(?:dr\.?|doctor|physician|consultant)[\s:]+([A-Za-z\s\.]+)", re.IGNORECASE),
    "hospital":       re.compile(r"(?:hospital|clinic|centre|center|institute)[\s:]+([^\n]+)", re.IGNORECASE),
    "complaints":     re.compile(r"(?:chief complaints?|c/o|complain|presenting|cc)[\s:]+([^\n]+)", re.IGNORECASE),
    "vitals":         re.compile(r"(?:bp|pulse|spo2|rr|temp|weight|height|bmi)[\s:]+([^\n]+)", re.IGNORECASE),
    "lab_values":     re.compile(r"(?:hb|wbc|rbc|plt|na|k|creatinine|urea|glucose|hba1c|ast|alt|tsh)[\s:]+([^\n]+)", re.IGNORECASE),
}


def parse_prescription(text: str) -> dict:
    """
    Parse common clinical fields from extracted prescription/report text.

    Returns a structured dict with parsed fields and the raw text.
    This dict is injected into the clinical assistant prompt to guide diagnosis.
    """
    parsed = {
        "raw_text": text,
        "patient_name": None,
        "date": None,
        "doctor": None,
        "hospital": None,
        "diagnosis": None,
        "chief_complaints": None,
        "medications": [],
        "dosages": [],
        "frequencies": [],
        "vitals": [],
        "lab_values": [],
    }

    for field_name, pattern in _RX_PATTERNS.items():
        matches = pattern.findall(text)
        if not matches:
            continue

        clean_matches = [m.strip().rstrip(".,;:") for m in matches if m.strip()]

        if field_name in ("medications", "dosage", "frequency", "vitals", "lab_values"):
            key_map = {
                "medications": "medications",
                "dosage": "dosages",
                "frequency": "frequencies",
                "vitals": "vitals",
                "lab_values": "lab_values",
            }
            parsed[key_map.get(field_name, field_name)].extend(clean_matches)
        else:
            # Take the first match for singular fields
            parsed[field_name] = clean_matches[0] if clean_matches else None

    # De-duplicate list fields
    for list_field in ("medications", "dosages", "frequencies", "vitals", "lab_values"):
        parsed[list_field] = list(dict.fromkeys(parsed[list_field]))

    return parsed


def build_clinical_context_from_document(parsed: dict) -> str:
    """
    Convert parsed prescription data into a structured prompt context string
    that can be injected into the clinical assistant's reasoning pipeline.
    """
    lines = ["[UPLOADED MEDICAL DOCUMENT — AUTHORITATIVE SOURCE]"]

    if parsed.get("patient_name"):
        lines.append(f"- Patient: {parsed['patient_name']}")
    if parsed.get("date"):
        lines.append(f"- Document Date: {parsed['date']}")
    if parsed.get("doctor"):
        lines.append(f"- Prescribing Physician: {parsed['doctor']}")
    if parsed.get("hospital"):
        lines.append(f"- Facility: {parsed['hospital']}")

    if parsed.get("diagnosis"):
        lines.append(f"- Stated Diagnosis/Impression: {parsed['diagnosis']}")

    if parsed.get("chief_complaints"):
        lines.append(f"- Chief Complaints: {parsed['chief_complaints']}")

    if parsed.get("vitals"):
        lines.append(f"- Vitals/Measurements: {'; '.join(parsed['vitals'][:6])}")

    if parsed.get("lab_values"):
        lines.append(f"- Lab Values: {'; '.join(parsed['lab_values'][:8])}")

    if parsed.get("medications"):
        meds_str = "; ".join(parsed["medications"][:10])
        lines.append(f"- Medications Listed: {meds_str}")
        if parsed.get("dosages"):
            lines.append(f"  Dosages: {', '.join(parsed['dosages'][:10])}")
        if parsed.get("frequencies"):
            lines.append(f"  Frequencies: {', '.join(parsed['frequencies'][:8])}")

    lines.append("")
    lines.append("[FULL DOCUMENT TEXT FOR DETAILED REVIEW]")
    # Truncate raw text to avoid blowing up the prompt
    raw = parsed.get("raw_text", "")
    lines.append(raw[:6000] + ("..." if len(raw) > 6000 else ""))

    return "\n".join(lines)
