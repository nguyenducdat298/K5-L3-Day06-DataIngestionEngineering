"""Implement fixed-size and structure-aware chunking."""
import re
from .models import Chunk, DocumentPage


def _make_chunk(
    page: DocumentPage,
    chunk_id: str,
    text: str,
    chunk_type: str,
    idx: int,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata={
            **page.metadata,
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "chunk_index": idx,
            "char_length": len(text),
        },
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split text into overlapping windows and retain page metadata."""
    chunks: list[Chunk] = []
    text = page.text.strip()
    if not text:
        return chunks

    step = max(1, size - overlap)
    idx = 0
    for start in range(0, len(text), step):
        end = min(len(text), start + size)
        chunk_text = text[start:end].strip()
        if len(chunk_text) >= 20 or (start == 0 and len(chunk_text) > 0):
            chunk_id = f"{page.document_id}_p{page.page}_f{idx}"
            chunks.append(_make_chunk(page, chunk_id, chunk_text, "fixed", idx))
            idx += 1
        if end >= len(text):
            break
    return chunks


def _split_long_paragraph(para: str, max_size: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries first, windowing only if needed."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
    parts: list[str] = []
    cur = ""
    for sent in sentences:
        if len(sent) > max_size:
            if cur:
                parts.append(cur)
                cur = ""
            step = max(1, max_size - 100)
            for s in range(0, len(sent), step):
                sub = sent[s : s + max_size].strip()
                if sub:
                    parts.append(sub)
                if s + max_size >= len(sent):
                    break
        elif not cur:
            cur = sent
        elif len(cur) + 1 + len(sent) <= max_size:
            cur = f"{cur} {sent}"
        else:
            parts.append(cur)
            cur = sent
    if cur:
        parts.append(cur)
    return parts


def by_paragraph(page: DocumentPage, max_size: int = 800) -> list[Chunk]:
    """Group paragraphs cleanly up to max_size, splitting oversized blocks on sentence boundaries."""
    chunks: list[Chunk] = []
    raw_paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        return chunks

    # Expand oversized paragraphs into sentence-aligned units first
    units: list[tuple[str, str]] = []
    for para in raw_paragraphs:
        if len(para) > max_size:
            for sub in _split_long_paragraph(para, max_size):
                units.append((sub, "paragraph_split"))
        else:
            units.append((para, "paragraph"))

    idx = 0
    current_group: list[str] = []
    current_len = 0
    current_type = "paragraph"

    def flush() -> None:
        nonlocal idx, current_group, current_len, current_type
        if not current_group:
            return
        merged_text = "\n\n".join(current_group)
        chunk_id = f"{page.document_id}_p{page.page}_para{idx}"
        chunks.append(_make_chunk(page, chunk_id, merged_text, current_type, idx))
        idx += 1
        current_group = []
        current_len = 0
        current_type = "paragraph"

    for unit_text, unit_type in units:
        if unit_type == "paragraph_split":
            flush()
            chunk_id = f"{page.document_id}_p{page.page}_para{idx}"
            chunks.append(_make_chunk(page, chunk_id, unit_text, "paragraph_split", idx))
            idx += 1
        elif not current_group:
            current_group = [unit_text]
            current_len = len(unit_text)
        elif current_len + 2 + len(unit_text) <= max_size:
            current_group.append(unit_text)
            current_len += 2 + len(unit_text)
        else:
            flush()
            current_group = [unit_text]
            current_len = len(unit_text)

    flush()
    return chunks
