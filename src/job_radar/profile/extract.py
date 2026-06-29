import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

# Plain-text formats are read as-is; anything else must be a PDF we can extract.
_TEXT_SUFFIXES = {".txt", ".md"}


def extract_text(path: Path) -> str:
    """Read a CV file into plain text.

    `.txt`/`.md` are passed through; `.pdf` is text-extracted with pdfplumber.
    Raises on an unsupported type, or when extraction yields nothing.
    An empty result usually means an image-only/scanned PDF with no text layer.
    """
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        with pdfplumber.open(path) as pdf:
            logger.debug("Extracting text from %d-page PDF %s", len(pdf.pages), path.name)
            # x_tolerance=1: pdfplumber's default merges adjacent glyphs that
            # have no explicit space between them ("Python,Kotlin" -> garbage),
            # which wrecks both LLM parsing and the embedding. A tighter
            # tolerance restores word boundaries from the glyph spacing.
            text = "\n".join(page.extract_text(x_tolerance=1) or "" for page in pdf.pages)
    else:
        raise ValueError(f"unsupported CV file type: {suffix or '(none)'}")

    text = text.strip()
    if not text:
        raise ValueError(f"no extractable text in {path}")
    return text
