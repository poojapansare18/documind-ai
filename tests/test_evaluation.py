import pytest

from app.evaluation import (
    evaluate_retrieval,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
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
