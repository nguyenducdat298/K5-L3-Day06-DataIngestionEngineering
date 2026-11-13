"""Build day06_rag_benchmark.ipynb cleanly with embedded verified questions and zero quoting bugs."""
import ast
import json
from pathlib import Path


def make_code_cell(source: str) -> dict:
    # Validate Python syntax when cell contains pure Python (ignore %/! magics if any)
    ast.parse(source)
    lines = [line + "\n" for line in source.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def make_md_cell(source: str) -> dict:
    lines = [line + "\n" for line in source.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    questions_path = repo_root / "data" / "benchmark_questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    questions_literal = repr(questions)

    cell_md_intro = make_md_cell(
        "# Day 06 · Data Ingestion Engineering & RAG Benchmark (Kaggle + pgvector)\n"
        "- Corpus: Transformer (`arxiv:1706.03762`, 15 pages) + BERT (`arxiv:1810.04805`, 16 pages) = **31 pages**\n"
        "- Vector DB: PostgreSQL + `pgvector` v0.7.0\n"
        "- Embeddings: `google/embeddinggemma-2` (768-dim) via Kaggle Model Hub\n"
        "- Strategies: **A** Fixed-size (500/100), **B** Paragraph-aware (max 800), **C** Paragraph-aware + Metadata Filter\n"
        "- Benchmark: 20 verified questions (15 dev / 5 test, 4 metadata-filtered)"
    )

    cell_1_setup = make_code_cell(
        '''import os
import subprocess
import sys
import time

t0 = time.time()
print("[1/3] Installing PostgreSQL + build deps...")
subprocess.run(
    "sudo apt-get update -qq && DEBIAN_FRONTEND=noninteractive sudo apt-get install -y -qq "
    "postgresql postgresql-contrib postgresql-server-dev-all build-essential git",
    shell=True,
    check=True,
)

print("[2/3] Building pgvector v0.7.0 in /tmp...")
subprocess.run(
    "rm -rf /tmp/pgvector && "
    "git clone -q --branch v0.7.0 https://github.com/pgvector/pgvector.git /tmp/pgvector && "
    "make -s -C /tmp/pgvector && "
    "sudo make -s -C /tmp/pgvector install",
    shell=True,
    check=True,
)

subprocess.run(["sudo", "service", "postgresql", "restart"], check=True)
subprocess.run(
    ["sudo", "-u", "postgres", "psql", "-c", "CREATE USER kaggle WITH PASSWORD 'kaggle' SUPERUSER;"],
    check=False,
)
subprocess.run(
    ["sudo", "-u", "postgres", "psql", "-c", "CREATE DATABASE rag_db OWNER kaggle;"],
    check=False,
)
subprocess.run(
    ["sudo", "-u", "postgres", "psql", "-d", "rag_db", "-c", "CREATE EXTENSION IF NOT EXISTS vector;"],
    check=True,
)

print("[3/3] Installing Python packages (upgrading sentence-transformers for EmbeddingGemma)...")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "pgvector",
        "psycopg2-binary",
        "pypdf",
        "requests",
        "tabulate",
        "sentence-transformers>=6.1.0",
        "transformers",
    ],
    check=True,
)
print(f"[✓] Setup completed in {time.time() - t0:.2f}s")'''
    )

    cell_2_ingest = make_code_cell(
        '''import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import requests
from pypdf import PdfReader


@dataclass
class DocumentPage:
    document_id: str
    source: str
    page: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source: str
    page: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


HEADING_RE = re.compile(
    r"^(?:\\d+(?:\\.\\d+)*\\s+[A-Z]|[A-Z](?:\\.\\d+)+\\s+[A-Z]|"
    r"(?:Abstract|Introduction|Conclusion|References|Appendix|Task\\s+#\\d+)\\b|"
    r"(?:Table|Figure)\\s+\\d+:|•\\s)"
)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).replace("\\r", "")
    text = re.sub(r"([A-Za-z0-9])-\\s*\\n\\s*([A-Za-z0-9])", r"\\1\\2", text)
    lines = [re.sub(r"[ \\t]+", " ", ln).strip() for ln in text.split("\\n")]
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
            if HEADING_RE.match(ln):
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
    return "\\n\\n".join(p for p in paragraphs if p)


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        norm_text = normalize_text(page.extract_text() or "")
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


data_dir = Path("/tmp/day06_data")
data_dir.mkdir(parents=True, exist_ok=True)
pdfs_info = [
    ("attention_is_all_you_need.pdf", "https://arxiv.org/pdf/1706.03762.pdf", "transformer-2017", "arxiv:1706.03762"),
    ("bert_paper.pdf", "https://arxiv.org/pdf/1810.04805.pdf", "bert-2018", "arxiv:1810.04805"),
]

t_ingest = time.time()
all_pages: list[DocumentPage] = []
for fname, url, doc_id, src in pdfs_info:
    fpath = data_dir / fname
    if not fpath.exists():
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        fpath.write_bytes(resp.content)
    loaded = load_pdf(fpath, doc_id, src)
    print(f"Loaded {fname}: {len(loaded)} pages")
    all_pages.extend(loaded)

ingest_time_s = round(time.time() - t_ingest, 3)
print(f"Total corpus pages: {len(all_pages)} (Requirement >= 30) in {ingest_time_s}s")
assert len(all_pages) >= 30'''
    )

    cell_3_chunk = make_code_cell(
        '''def _make_chunk(page: DocumentPage, chunk_id: str, text: str, chunk_type: str, idx: int) -> Chunk:
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


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 100) -> list[Chunk]:
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
            chunks.append(_make_chunk(page, f"{page.document_id}_p{page.page}_f{idx}", chunk_text, "fixed", idx))
            idx += 1
        if end >= len(text):
            break
    return chunks


def _split_long_paragraph(para: str, max_size: int) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\\s+", para) if s.strip()]
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
    chunks: list[Chunk] = []
    raw_paragraphs = [p.strip() for p in page.text.split("\\n\\n") if p.strip()]
    if not raw_paragraphs:
        return chunks

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

    def flush() -> None:
        nonlocal idx, current_group, current_len
        if not current_group:
            return
        merged_text = "\\n\\n".join(current_group)
        chunks.append(_make_chunk(page, f"{page.document_id}_p{page.page}_para{idx}", merged_text, "paragraph", idx))
        idx += 1
        current_group = []
        current_len = 0

    for unit_text, unit_type in units:
        if unit_type == "paragraph_split":
            flush()
            chunks.append(_make_chunk(page, f"{page.document_id}_p{page.page}_para{idx}", unit_text, "paragraph_split", idx))
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


t0 = time.time()
chunks_a = [c for p in all_pages for c in fixed_size(p, 500, 100)]
time_chunk_a = round(time.time() - t0, 4)

t0 = time.time()
chunks_b = [c for p in all_pages for c in by_paragraph(p, 800)]
time_chunk_b = round(time.time() - t0, 4)

chunks_c = chunks_b
print(f"Strategy A (Fixed 500/100): {len(chunks_a)} chunks ({time_chunk_a}s)")
print(f"Strategy B (Paragraph <=800): {len(chunks_b)} chunks ({time_chunk_b}s)")
print(f"Strategy C (Paragraph + Metadata Filter): {len(chunks_c)} chunks")

# Sample 10 chunks for Coherence qualitative logging (5 Fixed + 5 Paragraph)
coherence_samples = []
for label, pool, indices in [("Fixed_500_100", chunks_a, [0, 35, 95, 160, 220]), ("Paragraph_800", chunks_b, [0, 25, 65, 110, 155])]:
    for idx in indices:
        if idx < len(pool):
            c = pool[idx]
            coherence_samples.append({
                "strategy": label,
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "page": c.page,
                "char_length": len(c.text),
                "head": c.text[:140],
                "tail": c.text[-140:],
            })
print(f"Collected {len(coherence_samples)} coherence inspection samples.")'''
    )

    cell_4_embed_store = make_code_cell(
        '''import glob
import json
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import torch
from sentence_transformers import SentenceTransformer

candidates = [
    "/kaggle/input/models/google/embeddinggemma-2/transformers/embeddinggemma-2/1",
    "/kaggle/input/embeddinggemma-2/transformers/embeddinggemma-2/1",
] + glob.glob("/kaggle/input/**/config.json", recursive=True)

model_dir = None
for cand in candidates:
    p = Path(cand)
    if p.name == "config.json":
        p = p.parent
    if (p / "config.json").exists():
        model_dir = str(p)
        break

if not model_dir:
    raise FileNotFoundError("Could not locate google/embeddinggemma-2 under /kaggle/input")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading EmbeddingGemma-2 from {model_dir} on device={device}...")
t_model = time.time()
st_model = SentenceTransformer(model_dir, device=device)
emb_dim = int(st_model.get_sentence_embedding_dimension())
print(f"Model loaded in {time.time() - t_model:.2f}s | embedding_dim={emb_dim}")


def embed_docs(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    if hasattr(st_model, "encode_document"):
        vecs = st_model.encode_document(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    else:
        vecs = st_model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_query(q: str) -> list[float]:
    if hasattr(st_model, "encode_query"):
        vec = st_model.encode_query(q, normalize_embeddings=True, show_progress_bar=False)
    else:
        vec = st_model.encode(q, normalize_embeddings=True, show_progress_bar=False)
    return vec.tolist()


class PgVectorStore:
    def __init__(self, dsn: str, dimensions: int) -> None:
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True
        self.dimensions = dimensions
        register_vector(self.conn)

    def create_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("DROP TABLE IF EXISTS rag_chunks;")
            cur.execute(f"""
                CREATE TABLE rag_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_id VARCHAR(100) NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    document_id VARCHAR(100) NOT NULL,
                    source TEXT NOT NULL,
                    page INT NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding vector({self.dimensions}) NOT NULL,
                    UNIQUE (strategy, chunk_id)
                );
                CREATE INDEX idx_rag_chunks_strat ON rag_chunks (strategy);
                CREATE INDEX idx_rag_chunks_doc ON rag_chunks (document_id);
                CREATE INDEX idx_rag_chunks_page ON rag_chunks (page);
            """)

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        rows = [
            (c.chunk_id, strategy, c.document_id, c.source, c.page, c.text, json.dumps(c.metadata), v)
            for c, v in zip(chunks, vectors)
        ]
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM rag_chunks WHERE strategy = %s;", (strategy,))
            execute_values(
                cur,
                """
                INSERT INTO rag_chunks (chunk_id, strategy, document_id, source, page, text, metadata, embedding)
                VALUES %s
                ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    source = EXCLUDED.source,
                    page = EXCLUDED.page,
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s, %s, %s::vector)",
            )

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        where = ["strategy = %s"]
        params: list[Any] = [query_vector, strategy]
        if filters:
            for k, val in filters.items():
                if k in ("document_id", "source"):
                    where.append(f"{k} = %s")
                    params.append(val)
                elif k == "page":
                    where.append("page = %s")
                    params.append(int(val))
                else:
                    where.append("metadata->>%s = %s")
                    params.extend([k, str(val)])
        params.extend([query_vector, top_k])
        sql = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM rag_chunks
            WHERE {" AND ".join(where)}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            {
                "chunk_id": r[0],
                "document_id": r[1],
                "source": r[2],
                "page": r[3],
                "text": r[4],
                "metadata": r[5],
                "score": float(r[6]),
            }
            for r in rows
        ]


DB_URL = "postgresql://kaggle:kaggle@localhost:5432/rag_db"
store = PgVectorStore(DB_URL, emb_dim)
store.create_schema()

t0 = time.time()
vecs_a = embed_docs([c.text for c in chunks_a])
time_embed_a = round(time.time() - t0, 3)
t0 = time.time()
store.upsert("A_fixed", chunks_a, vecs_a)
time_upsert_a = round(time.time() - t0, 3)

t0 = time.time()
vecs_b = embed_docs([c.text for c in chunks_b])
time_embed_b = round(time.time() - t0, 3)
t0 = time.time()
store.upsert("B_paragraph", chunks_b, vecs_b)
time_upsert_b = round(time.time() - t0, 3)

t0 = time.time()
store.upsert("C_meta_filter", chunks_c, vecs_b)
time_upsert_c = round(time.time() - t0, 3)

print(f"Indexed A_fixed: embed={time_embed_a}s, upsert={time_upsert_a}s")
print(f"Indexed B_paragraph: embed={time_embed_b}s, upsert={time_upsert_b}s")
print(f"Indexed C_meta_filter: embed={time_embed_b}s (reused B vecs), upsert={time_upsert_c}s")'''
    )

    cell_5_eval = make_code_cell(
        f'''from tabulate import tabulate

questions = {questions_literal}


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    gold_doc = question.get("gold_source_id")
    gold_page = question.get("gold_page")
    evidence_quote = question.get("evidence_quote", "").lower()
    quote_words = [w for w in evidence_quote.split() if len(w) > 3]

    hit_1 = False
    hit_3 = False
    found_at_rank = None

    for rank, h in enumerate(hits, start=1):
        doc_matches = h["document_id"] == gold_doc
        page_matches = h["page"] == gold_page
        clean_txt = h["text"].lower()
        if quote_words:
            matched_words = sum(1 for w in quote_words if w in clean_txt)
            text_matches = (matched_words / len(quote_words)) >= 0.6
        else:
            text_matches = False

        if doc_matches and (page_matches or text_matches):
            if rank == 1:
                hit_1 = True
            if rank <= 3:
                hit_3 = True
            if found_at_rank is None:
                found_at_rank = rank

    return {{
        "question_id": question["id"],
        "type": question["type"],
        "split": question["split"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold_source_id": gold_doc,
        "gold_page": gold_page,
        "evidence_quote": question["evidence_quote"],
        "filter": question.get("filter"),
        "hit@1": hit_1,
        "hit@3": hit_3,
        "found_at_rank": found_at_rank,
        "top_hits": [
            {{
                "rank": idx + 1,
                "chunk_id": h["chunk_id"],
                "document_id": h["document_id"],
                "page": h["page"],
                "score": round(h["score"], 4),
                "text_snippet": h["text"][:260],
            }}
            for idx, h in enumerate(hits[:3])
        ],
    }}


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {{}}
    for name, subset in [
        ("overall", traces),
        ("dev", [t for t in traces if t["split"] == "dev"]),
        ("test", [t for t in traces if t["split"] == "test"]),
        ("filtered_only", [t for t in traces if t.get("filter") is not None]),
    ]:
        total = len(subset)
        h1 = sum(1 for t in subset if t["hit@1"])
        h3 = sum(1 for t in subset if t["hit@3"])
        out[name] = {{
            "total_queries": total,
            "hit@1_count": h1,
            "hit@1_rate": round(h1 / total, 4) if total else 0.0,
            "hit@3_count": h3,
            "hit@3_rate": round(h3 / total, 4) if total else 0.0,
        }}
    return out


q_vecs = {{q["id"]: embed_query(q["question"]) for q in questions}}

configs = [
    ("Strategy_A_Fixed", "A_fixed", len(chunks_a), time_chunk_a, time_embed_a, time_upsert_a),
    ("Strategy_B_Paragraph", "B_paragraph", len(chunks_b), time_chunk_b, time_embed_b, time_upsert_b),
    ("Strategy_C_MetaFilter", "C_meta_filter", len(chunks_c), time_chunk_b, time_embed_b, time_upsert_c),
]

benchmark_results: dict[str, Any] = {{
    "meta": {{
        "model": "google/embeddinggemma-2",
        "embedding_dim": emb_dim,
        "corpus_pages": len(all_pages),
        "ingest_time_s": ingest_time_s,
    }},
    "coherence_samples": coherence_samples,
    "strategies": {{}},
}}
summary_rows = []
csv_lines = ["strategy,chunk_count,chunk_time_s,embed_time_s,upsert_time_s,query_time_s,hit1_overall,hit3_overall,hit3_dev,hit3_test,hit3_filtered"]

for strat_key, db_tag, n_chunks, t_chunk, t_emb, t_ups in configs:
    traces = []
    t0 = time.time()
    for q in questions:
        flt = q.get("filter") if strat_key == "Strategy_C_MetaFilter" else None
        hits = store.search(db_tag, q_vecs[q["id"]], top_k=3, filters=flt)
        traces.append(evaluate_query(q, hits))
    t_query = round(time.time() - t0, 4)
    summ = summarize(traces)
    summ["chunk_count"] = n_chunks
    summ["chunk_time_s"] = t_chunk
    summ["embed_time_s"] = t_emb
    summ["upsert_time_s"] = t_ups
    summ["query_time_seconds"] = t_query
    summ["traces"] = traces
    benchmark_results["strategies"][strat_key] = summ

    summary_rows.append([
        strat_key,
        n_chunks,
        f"{{t_emb:.2f}}s",
        f"{{t_query:.3f}}s",
        f"{{summ['overall']['hit@1_rate']*100:.1f}}% ({{summ['overall']['hit@1_count']}}/20)",
        f"{{summ['overall']['hit@3_rate']*100:.1f}}% ({{summ['overall']['hit@3_count']}}/20)",
        f"{{summ['dev']['hit@3_rate']*100:.1f}}% ({{summ['dev']['hit@3_count']}}/15)",
        f"{{summ['test']['hit@3_rate']*100:.1f}}% ({{summ['test']['hit@3_count']}}/5)",
        f"{{summ['filtered_only']['hit@3_rate']*100:.1f}}% ({{summ['filtered_only']['hit@3_count']}}/4)",
    ])
    csv_lines.append(
        f"{{db_tag}},{{n_chunks}},{{t_chunk}},{{t_emb}},{{t_ups}},{{t_query}},"
        f"{{summ['overall']['hit@1_rate']}},{{summ['overall']['hit@3_rate']}},"
        f"{{summ['dev']['hit@3_rate']}},{{summ['test']['hit@3_rate']}},{{summ['filtered_only']['hit@3_rate']}}"
    )

print("\\n=== DAY 06 BENCHMARK SUMMARY ===")
print(tabulate(summary_rows, headers=["Strategy", "Chunks", "Embed", "Query", "Hit@1 All", "Hit@3 All", "Hit@3 Dev", "Hit@3 Test", "Hit@3 Filtered"], tablefmt="github"))

with open("/kaggle/working/benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(benchmark_results, f, indent=2, ensure_ascii=False)

with open("/kaggle/working/experiment_log.csv", "w", encoding="utf-8") as f:
    f.write("\\n".join(csv_lines) + "\\n")

print("\\nSaved /kaggle/working/benchmark_results.json and /kaggle/working/experiment_log.csv")'''
    )

    nb = {
        "cells": [
            cell_md_intro,
            cell_1_setup,
            cell_2_ingest,
            cell_3_chunk,
            cell_4_embed_store,
            cell_5_eval,
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    out_path = Path(__file__).resolve().parent / "day06_rag_benchmark.ipynb"
    out_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {out_path} ({len(nb['cells'])} cells, all Python cells AST-validated)")


if __name__ == "__main__":
    main()
