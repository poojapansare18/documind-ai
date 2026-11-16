from __future__ import annotations

import json
from pathlib import Path

import pymupdf as fitz

from scripts.create_evaluation_corpus import create_diverse_corpus


ROOT = Path(__file__).resolve().parents[1]


def test_v3_benchmark_has_reviewable_labels_for_all_pages():
    benchmark = json.loads((ROOT / "evaluation" / "benchmark_v3.json").read_text(encoding="utf-8"))
    queries = benchmark["queries"]

    assert len(queries) == 48
    assert len({query["id"] for query in queries}) == len(queries)
    assert benchmark["corpus_profile"]["pages"] == 24
    assert benchmark["annotation_protocol"]["status"] == "draft_gold_labels_pending_independent_review"
    assert all(query["review_status"] == "pending_independent_review" for query in queries)
    assert all(query["evidence_excerpt"] and query["annotation_rationale"] for query in queries)

    page_counts = {}
    for query in queries:
        assert len(query["relevant_pages"]) == 1
        page = query["relevant_pages"][0]
        assert 1 <= page <= 24
        page_counts[page] = page_counts.get(page, 0) + 1

    assert set(page_counts) == set(range(1, 25))
    assert set(page_counts.values()) == {2}


def test_v3_corpus_generator_creates_24_pages_across_three_domains(tmp_path: Path):
    output = create_diverse_corpus(tmp_path / "policy-corpus-v3.pdf")
    pdf = fitz.open(output)
    try:
        assert len(pdf) == 24
        text = "\n".join(page.get_text() for page in pdf)
        assert "NORTHSTAR TRAVEL POLICY" in text
        assert "MERIDIAN PROCUREMENT STANDARD" in text
        assert "ORBIT INFORMATION SECURITY STANDARD" in text
        assert "three-way match" in text
        assert "Multi-factor authentication" in text
    finally:
        pdf.close()
