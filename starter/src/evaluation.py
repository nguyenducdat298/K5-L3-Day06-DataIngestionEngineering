"""Evaluate retrieved evidence and preserve per-query traces."""
from typing import Any


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check source/page and gold evidence; keep hit@1 and hit@3 trace."""
    gold_doc = question.get("gold_source_id")
    gold_page = question.get("gold_page")
    evidence_quote = question.get("evidence_quote", "").lower()

    hit_1 = False
    hit_3 = False
    found_at_rank = None

    for rank, h in enumerate(hits, start=1):
        # Match criterion: document matches AND (page matches OR evidence quote found in text)
        doc_matches = (h["document_id"] == gold_doc)
        page_matches = (h["page"] == gold_page)
        
        # Check text match for evidence quote
        clean_chunk_text = h["text"].lower()
        # Look for substantive overlap of evidence quote
        quote_words = [w for w in evidence_quote.split() if len(w) > 3]
        if quote_words:
            matched_words = sum(1 for w in quote_words if w in clean_chunk_text)
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

    return {
        "question_id": question.get("id"),
        "split": question.get("split"),
        "question": question.get("question"),
        "gold_source_id": gold_doc,
        "gold_page": gold_page,
        "filter": question.get("filter"),
        "hit@1": hit_1,
        "hit@3": hit_3,
        "found_at_rank": found_at_rank,
        "top_hits": [
            {
                "rank": idx + 1,
                "chunk_id": h["chunk_id"],
                "document_id": h["document_id"],
                "page": h["page"],
                "score": round(h["score"], 4),
                "text_snippet": h["text"][:120] + "..." if len(h["text"]) > 120 else h["text"]
            }
            for idx, h in enumerate(hits[:3])
        ]
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    summary = {}
    splits = set(t["split"] for t in traces)

    for s in splits:
        split_traces = [t for t in traces if t["split"] == s]
        total = len(split_traces)
        h1 = sum(1 for t in split_traces if t["hit@1"])
        h3 = sum(1 for t in split_traces if t["hit@3"])
        summary[s] = {
            "total_queries": total,
            "hit@1_count": h1,
            "hit@1_rate": round(h1 / total, 4) if total > 0 else 0.0,
            "hit@3_count": h3,
            "hit@3_rate": round(h3 / total, 4) if total > 0 else 0.0
        }

    total_all = len(traces)
    all_h1 = sum(1 for t in traces if t["hit@1"])
    all_h3 = sum(1 for t in traces if t["hit@3"])
    summary["overall"] = {
        "total_queries": total_all,
        "hit@1_count": all_h1,
        "hit@1_rate": round(all_h1 / total_all, 4) if total_all > 0 else 0.0,
        "hit@3_count": all_h3,
        "hit@3_rate": round(all_h3 / total_all, 4) if total_all > 0 else 0.0
    }
    return summary
