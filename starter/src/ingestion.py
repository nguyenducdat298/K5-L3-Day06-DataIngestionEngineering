"""Read and normalize source data."""
import unicodedata
import re
from pathlib import Path
from pypdf import PdfReader
from .models import DocumentPage


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding useful structure."""
    if not text:
        return ""
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Fix broken hyphenated linebreaks (e.g. "atten-\n tion" -> "attention")
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Replace carriage returns and excessive tabs
    text = text.replace('\r', '')
    # Normalize multiple whitespaces (preserve double newlines for paragraphs)
    paragraphs = text.split('\n\n')
    cleaned_paras = []
    for p in paragraphs:
        # replace single newlines inside paragraph with space
        cleaned_p = re.sub(r'\s*\n\s*', ' ', p)
        # normalize repeated spaces
        cleaned_p = re.sub(r'[ \t]+', ' ', cleaned_p).strip()
        if cleaned_p:
            cleaned_paras.append(cleaned_p)
    return '\n\n'.join(cleaned_paras)


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page; keep its page number and source."""
    pages: list[DocumentPage] = []
    reader = PdfReader(str(path))
    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        norm_text = normalize_text(raw_text)
        doc_page = DocumentPage(
            document_id=document_id,
            source=source,
            page=page_idx,
            text=norm_text,
            metadata={
                "document_id": document_id,
                "source": source,
                "page": page_idx,
                "total_pages": len(reader.pages),
                "has_text": len(norm_text.strip()) > 0
            }
        )
        pages.append(doc_page)
    return pages
