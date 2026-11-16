from __future__ import annotations

import math
from statistics import mean


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1.")


def precision_at_k(ranked_items: list[str], relevant_items: set[str], k: int) -> float:
    _validate_k(k)
    return sum(item in relevant_items for item in ranked_items[:k]) / k


def recall_at_k(ranked_items: list[str], relevant_items: set[str], k: int) -> float:
    _validate_k(k)
    if not relevant_items:
        return 0.0
    return len(set(ranked_items[:k]) & relevant_items) / len(relevant_items)


def reciprocal_rank_at_k(ranked_items: list[str], relevant_items: set[str], k: int) -> float:
    _validate_k(k)
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in relevant_items:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_items: list[str], relevant_items: set[str], k: int) -> float:
    _validate_k(k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked_items[:k], start=1)
        if item in relevant_items
    )
    ideal_hits = min(len(relevant_items), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    return dcg / ideal_dcg


def evaluate_retrieval(cases: list[dict], k: int = 5) -> dict:
    """Evaluate ranked page retrieval against a set of relevant page identifiers."""
    _validate_k(k)
    if not cases:
        raise ValueError("At least one evaluation case is required.")

    per_query = []
    for case in cases:
        retrieved = list(dict.fromkeys(str(item) for item in case.get("retrieved", [])))
        relevant = {str(item) for item in case.get("relevant", [])}
        per_query.append({
            "id": case.get("id", f"query-{len(per_query) + 1}"),
            "precision_at_k": precision_at_k(retrieved, relevant, k),
            "recall_at_k": recall_at_k(retrieved, relevant, k),
            "mrr_at_k": reciprocal_rank_at_k(retrieved, relevant, k),
            "ndcg_at_k": ndcg_at_k(retrieved, relevant, k),
        })

    metric_names = ("precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k")
    return {
        "k": k,
        "query_count": len(per_query),
        "metrics": {
            name: round(mean(item[name] for item in per_query), 6)
            for name in metric_names
        },
        "per_query": per_query,
    }


def build_retrieval_diagnostics(
    query: str,
    ranked_results: list[dict],
    relevant_pages: list[int | str],
) -> dict:
    """Return inspectable ranking evidence for one page-level retrieval query."""
    relevant = {str(page) for page in relevant_pages}
    ranked = [
        {
            "rank": rank,
            "page_number": str(item["page_number"]),
            "chunk_index": item["chunk_index"],
            "score": round(float(item["score"]), 6),
            "text_preview": str(item["text"])[:360],
        }
        for rank, item in enumerate(ranked_results, start=1)
    ]
    relevant_result = next(
        (item for item in ranked if item["page_number"] in relevant),
        None,
    )
    top_score = ranked[0]["score"] if ranked else None
    relevant_score = relevant_result["score"] if relevant_result else None
    return {
        "query": query,
        "relevant_pages": sorted(relevant),
        "ranked_results": ranked,
        "relevant_rank": relevant_result["rank"] if relevant_result else None,
        "top_score": top_score,
        "relevant_score": relevant_score,
        "top_to_relevant_score_gap": (
            round(top_score - relevant_score, 6)
            if top_score is not None and relevant_score is not None
            else None
        ),
    }
