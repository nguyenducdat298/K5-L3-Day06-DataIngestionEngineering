"""Suggested pipeline entry point and orchestrator."""
import os
import json
import time
from pathlib import Path

from .models import DocumentPage, Chunk
from .ingestion import load_pdf
from .chunking import fixed_size, by_paragraph
from .embeddings import OpenRouterEmbedder
from .store import PgVectorStore
from .evaluation import evaluate_query, summarize


def main() -> None:
    print("=" * 80)
    print("DAY 06 · DATA INGESTION & RAG BENCHMARK PIPELINE")
    print("=" * 80)
    
    # 1. Config & Paths
    root_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = root_dir / "data"
    results_dir = root_dir / "starter" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag_db")
    api_key = os.environ.get("OPENROUTER_API_KEY", "your_openrouter_api_key_here")
    
    # 2. Ingest Corpus (≥30 pages)
    pdf1 = data_dir / "attention_is_all_you_need.pdf"
    pdf2 = data_dir / "bert_paper.pdf"
    
    print(f"\n[1/5] Ingesting PDF documents from {data_dir}...")
    pages_doc1 = load_pdf(pdf1, document_id="transformer-2017", source="arxiv:1706.03762")
    pages_doc2 = load_pdf(pdf2, document_id="bert-2018", source="arxiv:1810.04805")
    all_pages = pages_doc1 + pages_doc2
    total_pages = len(all_pages)
    print(f"  ✓ Loaded {len(pages_doc1)} pages from Transformer paper")
    print(f"  ✓ Loaded {len(pages_doc2)} pages from BERT paper")
    print(f"  ✓ Total corpus content pages: {total_pages} (Requirement: ≥30 content pages)")
    assert total_pages >= 30, f"Corpus has {total_pages} pages, required at least 30!"

    # 3. Build 3 Strategies
    print("\n[2/5] Building Chunking Strategies...")
    # Strategy A: Fixed-size (500 chars, overlap 100)
    chunks_strat_a: list[Chunk] = []
    t0 = time.time()
    for page in all_pages:
        chunks_strat_a.extend(fixed_size(page, size=500, overlap=100))
    time_chunk_a = time.time() - t0
    print(f"  ✓ Strategy A (Fixed 500/100): {len(chunks_strat_a)} chunks in {time_chunk_a:.3f}s")

    # Strategy B: Paragraph-aware (max 800 chars)
    chunks_strat_b: list[Chunk] = []
    t0 = time.time()
    for page in all_pages:
        chunks_strat_b.extend(by_paragraph(page, max_size=800))
    time_chunk_b = time.time() - t0
    print(f"  ✓ Strategy B (Paragraph-aware): {len(chunks_strat_b)} chunks in {time_chunk_b:.3f}s")

    # Strategy C: Same chunks as B, but with metadata-filtered queries
    chunks_strat_c = chunks_strat_b

    # 4. Initialize Embedder & Vector Store
    print("\n[3/5] Initializing OpenRouter Embedder & PgVector Store...")
    embedder = OpenRouterEmbedder(model_name="text-embedding-3-small", api_key=api_key)
    print(f"  ✓ Embedder initialized: {embedder.model_name} (Dimensions: {embedder.dimensions})")
    
    store = PgVectorStore(database_url=db_url, dimensions=embedder.dimensions)
    store.create_schema()
    print("  ✓ PgVector database schema created and ready")

    # Embed and upsert Strategy A
    print(f"\n[4/5] Embedding and Indexing Chunks...")
    print(f"  * Embedding {len(chunks_strat_a)} chunks for Strategy A...")
    texts_a = [c.text for c in chunks_strat_a]
    t0 = time.time()
    vecs_a = embedder.embed_documents(texts_a)
    store.upsert(strategy="A_fixed", chunks=chunks_strat_a, vectors=vecs_a)
    time_ingest_a = time.time() - t0
    print(f"    ✓ Strategy A indexed in {time_ingest_a:.2f}s")

    # Embed and upsert Strategy B & C
    print(f"  * Embedding {len(chunks_strat_b)} chunks for Strategy B & C...")
    texts_b = [c.text for c in chunks_strat_b]
    t0 = time.time()
    vecs_b = embedder.embed_documents(texts_b)
    store.upsert(strategy="B_paragraph", chunks=chunks_strat_b, vectors=vecs_b)
    # Upsert C into same or distinct index tag
    store.upsert(strategy="C_meta_filter", chunks=chunks_strat_c, vectors=vecs_b)
    time_ingest_b = time.time() - t0
    print(f"    ✓ Strategy B & C indexed in {time_ingest_b:.2f}s")

    # 5. Run Benchmark Queries
    print("\n[5/5] Running Benchmark Evaluation on 20 Questions (15 dev / 5 test)...")
    questions_file = data_dir / "benchmark_questions.json"
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    benchmark_results = {}

    for strat_key, strat_db_tag in [
        ("Strategy_A_Fixed", "A_fixed"),
        ("Strategy_B_Paragraph", "B_paragraph"),
        ("Strategy_C_MetaFilter", "C_meta_filter")
    ]:
        print(f"\n  Evaluating {strat_key}...")
        traces = []
        t0 = time.time()
        for q in questions:
            q_vec = embedder.embed_query(q["question"])
            # Apply filter only for Strategy C if filter is defined
            q_filter = q.get("filter") if strat_key == "Strategy_C_MetaFilter" else None
            hits = store.search(strategy=strat_db_tag, query_vector=q_vec, top_k=3, filters=q_filter)
            eval_trace = evaluate_query(q, hits)
            traces.append(eval_trace)
        total_eval_time = time.time() - t0
        strat_summary = summarize(traces)
        strat_summary["query_time_seconds"] = round(total_eval_time, 3)
        strat_summary["traces"] = traces
        benchmark_results[strat_key] = strat_summary

        print(f"    -> Overall Hit@1: {strat_summary['overall']['hit@1_rate']*100:.1f}%, "
              f"Hit@3: {strat_summary['overall']['hit@3_rate']*100:.1f}% "
              f"(Dev Hit@3: {strat_summary['dev']['hit@3_rate']*100:.1f}%, "
              f"Test Hit@3: {strat_summary['test']['hit@3_rate']*100:.1f}%)")

    # Save benchmark outputs
    out_file = results_dir / "benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved detailed traces to {out_file}")

    # Generate experiment log CSV
    exp_log_file = root_dir / "artifacts" / "experiment_log.csv"
    with open(exp_log_file, "w", encoding="utf-8") as f:
        f.write("strategy,chunk_count,embed_time_s,hit1_overall,hit3_overall,hit3_dev,hit3_test\n")
        f.write(f"A_fixed,{len(chunks_strat_a)},{time_ingest_a:.2f},{benchmark_results['Strategy_A_Fixed']['overall']['hit@1_rate']},{benchmark_results['Strategy_A_Fixed']['overall']['hit@3_rate']},{benchmark_results['Strategy_A_Fixed']['dev']['hit@3_rate']},{benchmark_results['Strategy_A_Fixed']['test']['hit@3_rate']}\n")
        f.write(f"B_paragraph,{len(chunks_strat_b)},{time_ingest_b:.2f},{benchmark_results['Strategy_B_Paragraph']['overall']['hit@1_rate']},{benchmark_results['Strategy_B_Paragraph']['overall']['hit@3_rate']},{benchmark_results['Strategy_B_Paragraph']['dev']['hit@3_rate']},{benchmark_results['Strategy_B_Paragraph']['test']['hit@3_rate']}\n")
        f.write(f"C_meta_filter,{len(chunks_strat_c)},{time_ingest_b:.2f},{benchmark_results['Strategy_C_MetaFilter']['overall']['hit@1_rate']},{benchmark_results['Strategy_C_MetaFilter']['overall']['hit@3_rate']},{benchmark_results['Strategy_C_MetaFilter']['dev']['hit@3_rate']},{benchmark_results['Strategy_C_MetaFilter']['test']['hit@3_rate']}\n")
    print(f"✓ Saved experiment log to {exp_log_file}")
    print("\n[✓] Day 06 Pipeline Complete!")


if __name__ == "__main__":
    main()
