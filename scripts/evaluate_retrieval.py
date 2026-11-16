from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.evaluation import (
    build_retrieval_diagnostics,
    evaluate_retrieval,
    rank_candidates_by_scores,
    reciprocal_rank_fusion,
)
from scripts.create_evaluation_corpus import create_corpus


def run_benchmark(
    base_url: str,
    benchmark_path: Path,
    k: int | None = None,
    reranker_model_name: str | None = None,
) -> dict:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    top_k = int(benchmark.get("k", 5)) if k is None else k
    corpus_path = benchmark_path.parent / benchmark["corpus_path"]
    if not corpus_path.exists():
        create_corpus(corpus_path)

    headers = {}
    api_key = os.getenv("DOCUMIND_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key

    reranker = None
    if reranker_model_name:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        reranker = TextCrossEncoder(model_name=reranker_model_name)

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=180.0) as client:
        documents_response = client.get("/documents", headers=headers)
        documents_response.raise_for_status()
        documents = documents_response.json()
        document = next(
            (item for item in documents if item["filename"] == benchmark["corpus_filename"]),
            None,
        )
        if document is None:
            with corpus_path.open("rb") as pdf_file:
                upload_response = client.post(
                    "/documents/upload",
                    files={"file": (benchmark["corpus_filename"], pdf_file, "application/pdf")},
                    headers=headers,
                )
            upload_response.raise_for_status()
            document = upload_response.json()

        document_id = document["document_id"]
        index_response = client.post(f"/documents/{document_id}/index", headers=headers)
        index_response.raise_for_status()
        index_result = index_response.json()

        diagnostics = []
        semantic_cases = []
        lexical_cases = []
        rrf_cases = []
        cross_encoder_cases = []
        for query in benchmark["queries"]:
            response = client.post(
                "/documents/semantic-search",
                json={"query": query["query"], "limit": 20, "document_id": document_id},
                headers=headers,
            )
            response.raise_for_status()
            results = response.json()

            def page_order(items: list[dict]) -> list[str]:
                return list(dict.fromkeys(str(item["page_number"]) for item in items))

            semantic_results = sorted(results, key=lambda item: item["semantic_score"], reverse=True)
            lexical_results = sorted(results, key=lambda item: (item["lexical_score"], item["semantic_score"]), reverse=True)
            rrf_results = reciprocal_rank_fusion(semantic_results, lexical_results)
            cross_encoder_results = None
            if reranker is not None:
                reranker_scores = list(
                    reranker.rerank(query["query"], [item["text"] for item in semantic_results])
                )
                cross_encoder_results = rank_candidates_by_scores(semantic_results, reranker_scores)
            semantic_pages = page_order(semantic_results)
            lexical_pages = page_order(lexical_results)
            rrf_pages = page_order(rrf_results)
            relevant = [str(page) for page in query["relevant_pages"]]
            semantic_cases.append({
                "id": query["id"],
                "retrieved": semantic_pages[:top_k],
                "relevant": relevant,
            })
            lexical_cases.append({
                "id": query["id"],
                "retrieved": lexical_pages[:top_k],
                "relevant": relevant,
            })
            rrf_cases.append({
                "id": query["id"],
                "retrieved": rrf_pages[:top_k],
                "relevant": relevant,
            })
            if cross_encoder_results is not None:
                cross_encoder_pages = page_order(cross_encoder_results)
                cross_encoder_cases.append({
                    "id": query["id"],
                    "retrieved": cross_encoder_pages[:top_k],
                    "relevant": relevant,
                })
            diagnostic = build_retrieval_diagnostics(
                query=query["query"],
                ranked_results=results,
                relevant_pages=query["relevant_pages"],
            )
            relevant_set = set(relevant)
            semantic_ranked = semantic_pages
            lexical_ranked = lexical_pages
            diagnostic["semantic_relevant_rank"] = next(
                (i for i, page in enumerate(semantic_ranked, 1) if page in relevant_set), None
            )
            diagnostic["lexical_relevant_rank"] = next(
                (i for i, page in enumerate(lexical_ranked, 1) if page in relevant_set), None
            )
            diagnostic["semantic_top_pages"] = semantic_ranked[:top_k]
            diagnostic["lexical_top_pages"] = lexical_ranked[:top_k]
            diagnostic["rrf_relevant_rank"] = next((i for i, page in enumerate(rrf_pages, 1) if page in relevant_set), None)
            diagnostic["rrf_top_pages"] = rrf_pages[:top_k]
            diagnostic["rrf_ranked_candidates"] = [
                {"page_number": item["page_number"], "chunk_index": item["chunk_index"], "rrf_score": round(item["rrf_score"], 8)}
                for item in rrf_results
            ]
            if cross_encoder_results is not None:
                cross_encoder_pages = page_order(cross_encoder_results)
                diagnostic["cross_encoder_relevant_rank"] = next(
                    (i for i, page in enumerate(cross_encoder_pages, 1) if page in relevant_set), None
                )
                diagnostic["cross_encoder_top_pages"] = cross_encoder_pages[:top_k]
                diagnostic["cross_encoder_ranked_candidates"] = [
                    {
                        "page_number": item["page_number"],
                        "chunk_index": item["chunk_index"],
                        "reranker_score": item["reranker_score"],
                    }
                    for item in cross_encoder_results
                ]
            diagnostics.append(diagnostic)

    report = evaluate_retrieval(semantic_cases, top_k)
    report["ranking_comparison"] = {
        "semantic_only": report["metrics"],
        "lexical_only": evaluate_retrieval(lexical_cases, top_k)["metrics"],
        "rrf": evaluate_retrieval(rrf_cases, top_k)["metrics"],
    }
    if reranker is not None:
        report["ranking_comparison"]["cross_encoder"] = evaluate_retrieval(
            cross_encoder_cases, top_k
        )["metrics"]
        report["reranker_model"] = reranker_model_name
    for metric_row, diagnostic in zip(report["per_query"], diagnostics):
        metric_row.update(diagnostic)
    report["benchmark_name"] = benchmark["benchmark_name"]
    report["document"] = {
        "document_id": document_id,
        "filename": benchmark["corpus_filename"],
        "indexed_chunks": index_result["indexed_chunks"],
        "embedding_model": index_result["embedding_model"],
        "dimensions": index_result["dimensions"],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DocuMind retrieval ranking variants against a page-level ground-truth benchmark."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "evaluation" / "benchmark.json",
    )
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="Optional FastEmbed cross-encoder model; downloads its model on first use.",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = run_benchmark(
        args.base_url, args.benchmark.resolve(), args.k, args.reranker_model
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved evaluation report: {args.report}")


if __name__ == "__main__":
    main()
