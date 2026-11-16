import pytest

from app.evaluation import (
    evaluate_retrieval,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    rank_candidates_by_scores,
    rerank_top_n_candidates,
)


def test_retrieval_metrics_for_ranked_results():
    ranked = ["3", "1", "2"]
    relevant = {"1", "2"}
    assert precision_at_k(ranked, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, relevant, 3) == 1.0
    assert reciprocal_rank_at_k(ranked, relevant, 3) == 0.5
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(0.693426)


def test_retrieval_metrics_return_one_for_perfect_ranking():
    ranked = ["2", "1", "3"]
    relevant = {"1", "2"}
    assert precision_at_k(ranked, relevant, 2) == 1.0
    assert recall_at_k(ranked, relevant, 2) == 1.0
    assert reciprocal_rank_at_k(ranked, relevant, 2) == 1.0
    assert ndcg_at_k(ranked, relevant, 2) == 1.0


def test_retrieval_metrics_handle_no_relevant_results():
    assert precision_at_k(["1", "2"], set(), 2) == 0.0
    assert recall_at_k(["1", "2"], set(), 2) == 0.0
    assert reciprocal_rank_at_k(["1", "2"], set(), 2) == 0.0
    assert ndcg_at_k(["1", "2"], set(), 2) == 0.0


def test_evaluation_deduplicates_retrieved_pages():
    report = evaluate_retrieval(
        [{"id": "duplicate-page", "retrieved": ["1", "1", "2"], "relevant": ["2"]}],
        k=2,
    )
    assert report["metrics"]["precision_at_k"] == 0.5
    assert report["metrics"]["recall_at_k"] == 1.0
    assert report["metrics"]["mrr_at_k"] == 0.5


def test_evaluation_rejects_invalid_k_and_empty_cases():
    with pytest.raises(ValueError):
        precision_at_k(["1"], {"1"}, 0)
    with pytest.raises(ValueError):
        evaluate_retrieval([], 3)


def test_synthetic_benchmark_corpus_has_twelve_ground_truth_pages(tmp_path):
    import pymupdf as fitz

    from scripts.create_evaluation_corpus import PAGES, create_corpus

    pdf_path = create_corpus(tmp_path / "travel-policy-v2.pdf")
    document = fitz.open(pdf_path)
    assert document.page_count == 12
    assert len(PAGES) == 12
    assert "14 calendar days" in document[0].get_text()
    assert "INR 6,000 per night" in document[2].get_text()
    assert "INR 1,200 per employee" in document[3].get_text()
    assert "finance controller" in document[5].get_text()
    assert "12 months" in document[11].get_text()
    document.close()


def test_retrieval_diagnostics_explain_relevant_rank_and_score_gap():
    from app.evaluation import build_retrieval_diagnostics

    report = build_retrieval_diagnostics(
        query="Who approves a late claim?",
        ranked_results=[
            {"page_number": 9, "chunk_index": 0, "score": 0.82, "text": "Cancellation rules."},
            {"page_number": 6, "chunk_index": 0, "score": 0.77, "text": "Late claims need finance approval."},
            {"page_number": 3, "chunk_index": 0, "score": 0.61, "text": "Hotel limits."},
        ],
        relevant_pages=[6],
    )
    assert report["relevant_rank"] == 2
    assert report["top_score"] == 0.82
    assert report["relevant_score"] == 0.77
    assert report["top_to_relevant_score_gap"] == 0.05
    assert report["ranked_results"][1]["text_preview"] == "Late claims need finance approval."


def test_retrieval_diagnostics_report_missing_relevant_page():
    from app.evaluation import build_retrieval_diagnostics

    report = build_retrieval_diagnostics(
        query="Which page contains the policy?",
        ranked_results=[
            {"page_number": 2, "chunk_index": 0, "score": 0.7, "text": "Ground transport."},
        ],
        relevant_pages=[8],
    )
    assert report["relevant_rank"] is None
    assert report["relevant_score"] is None
    assert report["top_to_relevant_score_gap"] is None



def test_retrieval_diagnostics_include_hybrid_component_scores():
    from app.evaluation import build_retrieval_diagnostics

    diagnostics = build_retrieval_diagnostics(
        query="invoice deadline",
        ranked_results=[{
            "page_number": 5,
            "chunk_index": 0,
            "score": 0.7,
            "semantic_score": 0.8,
            "lexical_score": 2.5,
            "text": "Invoice submission deadline is ten days.",
        }],
        relevant_pages=[5],
    )
    assert diagnostics["ranked_results"][0]["semantic_score"] == 0.8
    assert diagnostics["ranked_results"][0]["lexical_score"] == 2.5


def test_reciprocal_rank_fusion_combines_rank_positions():
    from app.evaluation import reciprocal_rank_fusion

    semantic = [
        {"document_id": "d", "page_number": 1, "chunk_index": 0, "semantic_score": 0.9},
        {"document_id": "d", "page_number": 2, "chunk_index": 0, "semantic_score": 0.8},
        {"document_id": "d", "page_number": 3, "chunk_index": 0, "semantic_score": 0.7},
    ]
    lexical = [
        {"document_id": "d", "page_number": 3, "chunk_index": 0, "lexical_score": 4.0},
        {"document_id": "d", "page_number": 1, "chunk_index": 0, "lexical_score": 3.0},
        {"document_id": "d", "page_number": 2, "chunk_index": 0, "lexical_score": 2.0},
    ]
    fused = reciprocal_rank_fusion(semantic, lexical)
    assert [item["page_number"] for item in fused] == [1, 3, 2]
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)


def test_reciprocal_rank_fusion_rejects_invalid_rank_constant():
    from app.evaluation import reciprocal_rank_fusion
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], rank_constant=0)


def test_cross_encoder_scores_rerank_candidates_and_preserve_metadata():
    candidates = [
        {"page_number": 1, "semantic_score": 0.9, "text": "First"},
        {"page_number": 2, "semantic_score": 0.8, "text": "Second"},
    ]
    ranked = rank_candidates_by_scores(candidates, [-1.2, 2.4])
    assert [item["page_number"] for item in ranked] == [2, 1]
    assert ranked[0]["reranker_score"] == 2.4
    assert ranked[0]["text"] == "Second"


def test_cross_encoder_rejects_mismatched_or_non_finite_scores():
    with pytest.raises(ValueError):
        rank_candidates_by_scores([{"page_number": 1}], [])
    with pytest.raises(ValueError):
        rank_candidates_by_scores([{"page_number": 1}], [float("nan")])


def test_bounded_cross_encoder_reranks_only_semantic_top_n():
    candidates = [
        {"page_number": 1, "semantic_score": 0.9},
        {"page_number": 2, "semantic_score": 0.8},
        {"page_number": 3, "semantic_score": 0.7},
        {"page_number": 4, "semantic_score": 0.6},
    ]
    ranked = rerank_top_n_candidates(candidates, [0.1, 0.9, 0.5, 10.0], top_n=3)
    assert [item["page_number"] for item in ranked] == [2, 3, 1, 4]
    assert "reranker_score" not in ranked[-1]


def test_bounded_cross_encoder_rejects_invalid_top_n_or_score_count():
    with pytest.raises(ValueError):
        rerank_top_n_candidates([{"page_number": 1}], [0.5], top_n=0)
    with pytest.raises(ValueError):
        rerank_top_n_candidates([{"page_number": 1}], [], top_n=1)
