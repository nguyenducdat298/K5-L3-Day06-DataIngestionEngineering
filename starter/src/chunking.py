"""Implement fixed-size and structure-aware chunking."""
from .models import Chunk, DocumentPage


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
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=chunk_text,
                metadata={
                    **page.metadata,
                    "chunk_id": chunk_id,
                    "chunk_type": "fixed",
                    "chunk_index": idx,
                    "char_length": len(chunk_text)
                }
            )
            chunks.append(chunk)
            idx += 1
        if end >= len(text):
            break
    return chunks


def by_paragraph(page: DocumentPage, max_size: int = 800) -> list[Chunk]:
    """Group paragraphs, split oversized paragraphs and retain metadata."""
    chunks: list[Chunk] = []
    raw_paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        return chunks

    idx = 0
    current_group: list[str] = []
    current_len = 0

    for para in raw_paragraphs:
        # If single paragraph is oversized, split it
        if len(para) > max_size:
            # flush current group if any
            if current_group:
                merged_text = "\n\n".join(current_group)
                chunk_id = f"{page.document_id}_p{page.page}_para{idx}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=merged_text,
                    metadata={
                        **page.metadata,
                        "chunk_id": chunk_id,
                        "chunk_type": "paragraph",
                        "chunk_index": idx,
                        "char_length": len(merged_text)
                    }
                ))
                idx += 1
                current_group = []
                current_len = 0

            # Split oversized paragraph by sentences or fixed window
            for sub_start in range(0, len(para), max_size - 100):
                sub_end = min(len(para), sub_start + max_size)
                sub_text = para[sub_start:sub_end].strip()
                if sub_text:
                    chunk_id = f"{page.document_id}_p{page.page}_para{idx}"
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        document_id=page.document_id,
                        source=page.source,
                        page=page.page,
                        text=sub_text,
                        metadata={
                            **page.metadata,
                            "chunk_id": chunk_id,
                            "chunk_type": "paragraph_split",
                            "chunk_index": idx,
                            "char_length": len(sub_text)
                        }
                    ))
                    idx += 1
                if sub_end >= len(para):
                    break
        elif current_len + len(para) + 2 <= max_size:
            current_group.append(para)
            current_len += len(para) + 2
        else:
            merged_text = "\n\n".join(current_group)
            chunk_id = f"{page.document_id}_p{page.page}_para{idx}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=merged_text,
                metadata={
                    **page.metadata,
                    "chunk_id": chunk_id,
                    "chunk_type": "paragraph",
                    "chunk_index": idx,
                    "char_length": len(merged_text)
                }
            ))
            idx += 1
            current_group = [para]
            current_len = len(para)

    if current_group:
        merged_text = "\n\n".join(current_group)
        chunk_id = f"{page.document_id}_p{page.page}_para{idx}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=merged_text,
            metadata={
                **page.metadata,
                "chunk_id": chunk_id,
                "chunk_type": "paragraph",
                "chunk_index": idx,
                "char_length": len(merged_text)
            }
        ))

    return chunks
