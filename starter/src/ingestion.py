"""Read and normalize source data."""
import re
import unicodedata
from pathlib import Path
from pypdf import PdfReader
from .models import DocumentPage

_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+[A-Z]|[A-Z](?:\.\d+)+\s+[A-Z]|"
    r"(?:Abstract|Introduction|Conclusion|References|Appendix|Task\s+#\d+)\b|"
    r"(?:Table|Figure)\s+\d+:|•\s)"
)


def normalize_text(text: str) -> str:
    """Normalize Unicode, ligatures, and whitespace while inferring paragraph breaks."""
    if not text:
        return ""
    # NFKC fixes PDF ligatures (fi, fl) while normalizing compatibility characters
    text = unicodedata.normalize("NFKC", text).replace("\r", "")
    # Repair hyphenated line-wraps (e.g. "atten-\n tion" -> "attention")
    text = re.sub(r"([A-Za-z0-9])-\s*\n\s*([A-Za-z0-9])", r"\1\2", text)

    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    paragraphs: list[str] = []
    buf: list[str] = []

    for ln in lines:
        if not ln:
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
            continue

        starts_new_para = False
        if buf:
            prev = buf[-1]
            if _HEADING_RE.match(ln):
                starts_new_para = True
            elif prev.endswith((".", "!", "?")) and len(prev) < 68 and ln[:1].isupper():
                starts_new_para = True

        if starts_new_para:
            paragraphs.append(" ".join(buf))
            buf = [ln]
        else:
            buf.append(ln)

    if buf:
        paragraphs.append(" ".join(buf))

    return "\n\n".join(p for p in paragraphs if p)


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page; keep its page number and source."""
    pages: list[DocumentPage] = []
    reader = PdfReader(str(path))
    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        norm_text = normalize_text(raw_text)
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_idx,
                text=norm_text,
                metadata={
                    "document_id": document_id,
                    "source": source,
                    "page": page_idx,
                    "total_pages": len(reader.pages),
                    "has_text": len(norm_text.strip()) > 0,
                },
            )
        )
    return pages
