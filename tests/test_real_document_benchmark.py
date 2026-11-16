from __future__ import annotations

import json
from pathlib import Path

from scripts.create_real_document_corpus import SOURCE_PDF_PAGES


ROOT = Path(__file__).resolve().parents[1]


def test_real_document_benchmark_v4_page_mapping_and_queries() -> None:
    benchmark = json.loads((ROOT / "evaluation" / "benchmark_v4.json").read_text(encoding="utf-8"))
    mapping = benchmark["corpus_profile"]["output_page_mapping"]

    assert len(SOURCE_PDF_PAGES) == 16
    assert len(set(SOURCE_PDF_PAGES)) == len(SOURCE_PDF_PAGES)
    assert list(map(int, mapping.values())) == SOURCE_PDF_PAGES
    assert len(benchmark["queries"]) == 22
    assert all(len(query["relevant_pages"]) >= 1 for query in benchmark["queries"])
    assert all(
        1 <= page <= len(SOURCE_PDF_PAGES)
        for query in benchmark["queries"]
        for page in query["relevant_pages"]
    )
    assert all(query["evidence_excerpt"] for query in benchmark["queries"])
    assert all(query["annotation_rationale"] for query in benchmark["queries"])
    assert all(
        query["review_status"] == "pending_independent_review"
        for query in benchmark["queries"]
    )
