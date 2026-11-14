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

from app.evaluation import evaluate_retrieval
from create_evaluation_corpus import create_corpus


def run_benchmark(base_url: str, benchmark_path: Path, k: int | None = None) -> dict:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    top_k = k or int(benchmark.get("k", 5))
    corpus_path = benchmark_path.parent / benchmark["corpus_path"]
    if not corpus_path.exists():
        create_corpus(corpus_path)

    headers = {}
    api_key = os.getenv("DOCUMIND_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key

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

        cases = []
        for query in benchmark["queries"]:
            response = client.post(
                "/documents/semantic-search",
                json={"query": query["query"], "limit": top_k, "document_id": document_id},
                headers=headers,
            )
            response.raise_for_status()
            results = response.json()
            ranked_pages = list(dict.fromkeys(str(item["page_number"]) for item in results))
            cases.append({
                "id": query["id"],
                "retrieved": ranked_pages,
                "relevant": [str(page) for page in query["relevant_pages"]],
            })

    report = evaluate_retrieval(cases, top_k)
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
        description="Evaluate DocuMind semantic retrieval against a page-level ground-truth benchmark."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "evaluation" / "benchmark.json",
    )
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = run_benchmark(args.base_url, args.benchmark.resolve(), args.k)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved evaluation report: {args.report}")


if __name__ == "__main__":
    main()
