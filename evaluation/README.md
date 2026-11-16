# DocuMind retrieval benchmark

This benchmark compares semantic-only, lexical-only, and reciprocal-rank-fusion retrieval at the **page level** using a synthetic, 12-page corporate travel policy and 24 manually authored questions. Each question has one known relevant page. The additional pages provide distractors, so a correct top result is no longer guaranteed by retrieving the entire three-page corpus.

## Run the benchmark

1. Install the project requirements.
2. Start DocuMind:

   ```powershell
   python -m uvicorn app.main:app --reload
   ```

3. Run the benchmark at k=3:

   ```powershell
   python scripts/evaluate_retrieval.py --k 3 --report evaluation/results.json
   ```

4. Run again at k=1 to measure first-result accuracy:

   ```powershell
   python scripts/evaluate_retrieval.py --k 1 --report evaluation/results-k1.json
   ```

The semantic score is cosine similarity from the configured FastEmbed model. The lexical score is BM25, calculated over indexed chunks. The API preserves semantic-only ranking as its default and returns semantic cosine similarity and BM25 lexical scores for analysis. The benchmark also evaluates reciprocal rank fusion (RRF), which combines rank positions rather than mixing raw score scales. RRF uses a rank constant of 60 and remains experimental.

The runner creates the synthetic PDF if it does not exist, uploads it if it is not already in the document library, indexes its chunks, and requests all indexed results for each query (up to the API limit of 20). It reports top-k metrics for semantic-only, lexical-only, and RRF rankings. An optional cross-encoder experiment can rerank the same candidate set. Per-query diagnostics include each candidate's component scores and the relevant page's rank under each ranking signal. The corpus uses a versioned filename so an older three-page benchmark document in a local database is not accidentally reused. The first indexing run downloads the local embedding model. If API-key authentication is enabled, set `DOCUMIND_API_KEY` in the environment or in the root `.env` file.

## Optional cross-encoder experiment

To compare a second-stage cross-encoder reranker with the existing ranking variants, run:

```powershell
python scripts/evaluate_retrieval.py --k 1 --reranker-model Xenova/ms-marco-MiniLM-L-6-v2 --report evaluation/reranker-k1.json
python scripts/evaluate_retrieval.py --k 3 --reranker-model Xenova/ms-marco-MiniLM-L-6-v2 --report evaluation/reranker-k3.json
python scripts/evaluate_retrieval.py --k 3 --reranker-model Xenova/ms-marco-MiniLM-L-6-v2 --reranker-top-n 3 --report evaluation/reranker-top3-k3.json
```

The optional stage uses FastEmbed's ONNX-based `TextCrossEncoder` to score each query against the candidate texts already retrieved by semantic search. It then ranks those candidates by cross-encoder score and adds a `cross_encoder` row to `ranking_comparison`, plus per-query ranks and candidate scores. Add `--reranker-top-n 3` to also evaluate a bounded variant: only the first three semantic candidates are reordered by cross-encoder score, while all remaining candidates retain their original semantic order. This is an offline safeguard experiment, not an API setting. It does not change the API or its default ranking. The first run downloads the model (approximately 0.08 GB); later runs use the local model cache. See the [FastEmbed reranker guide](https://qdrant.tech/documentation/fastembed/fastembed-rerankers/).


## Version 3: multi-domain evaluation pack

Version 3 adds a separate 24-page synthetic policy corpus covering three domains: corporate travel, procurement, and information security. It includes narrative pages, two-column layouts, and table-like policy blocks. The 48 questions contain two paraphrased queries per page, with one page-level relevant label, a supporting evidence excerpt, and a short annotation rationale.

The corpus is generated on demand and does not replace the v2 baseline:

```powershell
python scripts/evaluate_retrieval.py --benchmark evaluation/benchmark_v3.json --k 3 --report evaluation/v3-semantic-k3.json
python scripts/evaluate_retrieval.py --benchmark evaluation/benchmark_v3.json --k 1 --report evaluation/v3-semantic-k1.json
python scripts/evaluate_retrieval.py --benchmark evaluation/benchmark_v3.json --k 3 --reranker-model Xenova/ms-marco-MiniLM-L-6-v2 --reranker-top-n 3 --report evaluation/v3-reranker-top3-k3.json
```

The v3 labels are **draft gold labels, not independently reviewed labels**. Each query is paired with the page containing its supporting statement, and the JSON includes the evidence excerpt and rationale to make verification practical. Before reporting v3 as an independently labelled evaluation, two reviewers must independently judge the query-page pairs, record disagreements, and adjudicate them. The repository does not claim that this review has happened.

This corpus is still synthetic. It broadens domains and page layouts, but it does not substitute for permission-cleared real PDFs, OCR-heavy scans, complex real-world tables, multi-document retrieval, or answer-level factuality and citation evaluation. The current runner uploads the 24-page pack as one PDF and evaluates page-level retrieval within that document. Treat the results as a stronger internal engineering stress test, not an external benchmark or production guarantee.

## Metrics

- **Precision@k:** fraction of the top-k returned pages that are relevant.
- **Recall@k:** fraction of all relevant pages found in the top-k.
- **MRR@k:** reciprocal rank of the first relevant page; higher is better.
- **nDCG@k:** ranking quality that rewards relevant pages appearing earlier.

Results are averaged across all benchmark queries and also reported per query. With one relevant page per query, Precision@3 is at most 1/3 when that relevant page is found and three distinct pages are returned. Precision@1 directly measures whether the first returned page is relevant.

## Scope and limitations

This is a repeatable engineering baseline, not a broad evaluation of real-world PDFs. The corpus is synthetic, has one short page per topic, and uses manually authored queries. It evaluates page retrieval, not answer factuality, citation entailment, OCR accuracy, table extraction accuracy, or performance on diverse document layouts. Do not present these scores as an external industry benchmark. Add representative, permission-cleared documents and independently reviewed relevance labels before making production claims.


## Recorded comparison (24-query synthetic corpus)

The following results were produced by running the benchmark against the 12-page corpus with the local FastEmbed model. The API's production search order remains semantic-only.

| Ranking | Precision@1 | Recall@1 | MRR@1 | nDCG@1 |
|---|---:|---:|---:|---:|
| Semantic-only | 0.875000 | 0.875000 | 0.875000 | 0.875000 |
| Lexical-only (BM25) | 0.541667 | 0.541667 | 0.541667 | 0.541667 |
| RRF (rank constant 60) | 0.708333 | 0.708333 | 0.708333 | 0.708333 |

| Ranking | Precision@3 | Recall@3 | MRR@3 | nDCG@3 |
|---|---:|---:|---:|---:|
| Semantic-only | 0.333333 | 1.000000 | 0.930556 | 0.948411 |
| Lexical-only (BM25) | 0.250000 | 0.750000 | 0.631944 | 0.662200 |
| RRF (rank constant 60) | 0.319444 | 0.958333 | 0.819444 | 0.855155 |

**Interpretation:** semantic-only is the strongest of the three tested rankings at both cutoffs. RRF improves on BM25 alone but does not beat semantic-only: it lowers top-1 accuracy from 87.50% to 70.83%, and lowers MRR@3 from 0.930556 to 0.819444. For the `hotel-over-limit` query, semantic retrieval placed the relevant page first while RRF moved it to rank 2.

**Decision:** do not enable weighted fusion or RRF as the default retrieval strategy. Keep semantic-only ranking in the API. Retain BM25 and RRF only as offline comparison and diagnostic tools. These measurements are specific to this small synthetic corpus and do not establish that semantic-only will win on every real document collection.


## Recorded cross-encoder comparison (24-query synthetic corpus)

The optional reranker was evaluated with `Xenova/ms-marco-MiniLM-L-6-v2` against the same 12-page candidate set.

| Ranking | Precision@1 | Recall@1 | MRR@1 | nDCG@1 |
|---|---:|---:|---:|---:|
| Semantic-only | 0.875000 | 0.875000 | 0.875000 | 0.875000 |
| BM25 | 0.541667 | 0.541667 | 0.541667 | 0.541667 |
| RRF | 0.708333 | 0.708333 | 0.708333 | 0.708333 |
| Cross-encoder | **0.916667** | **0.916667** | **0.916667** | **0.916667** |

| Ranking | Precision@3 | Recall@3 | MRR@3 | nDCG@3 |
|---|---:|---:|---:|---:|
| Semantic-only | **0.333333** | **1.000000** | 0.930556 | **0.948411** |
| BM25 | 0.250000 | 0.750000 | 0.631944 | 0.662200 |
| RRF | 0.319444 | 0.958333 | 0.819444 | 0.855155 |
| Cross-encoder | 0.319444 | 0.958333 | **0.937500** | 0.942955 |

**Interpretation:** the cross-encoder improves top-1 accuracy from 21/24 to 22/24 queries. At k=3, it has a slightly higher MRR than semantic-only, but it fails to keep one relevant page in the top three. The `claim-required-evidence` query has relevant page 5: semantic-only ranks it third, while the cross-encoder moves it to sixth. Another query, `late-claim-authority`, has the relevant page at rank 2 under both methods.

**Decision:** this is a promising top-result improvement, but not a consistent top-k improvement. Do not replace semantic-only ranking in the API yet. Keep the cross-encoder as an optional offline experiment. Before production use, investigate the top-k regression, measure latency and memory, and validate against a larger, diverse, independently labelled document set. The benchmark remains synthetic and cannot establish general performance.


## Bounded cross-encoder follow-up

A post-run analysis of the supplied 24-query k=3 report simulated a bounded reranker that reorders only the semantic top three candidates and leaves the remaining semantic order unchanged. The results were:

| Ranking | Correct at rank 1 | Relevant page in top 3 | MRR@3 |
|---|---:|---:|---:|
| Semantic-only | 21/24 | 24/24 | 0.930556 |
| Full cross-encoder | 22/24 | 23/24 | 0.937500 |
| Cross-encoder limited to semantic top 3 | **22/24** | **24/24** | **0.951389** |

For `claim-required-evidence`, the relevant page was initially at semantic rank 3 but fell to cross-encoder rank 6. Limiting reordering to the semantic top three retains that page at rank 3. Across this supplied report, the bounded simulation retains the full reranker's top-1 hit count while restoring top-3 recall.

**Interpretation and limitation:** this is a retrospective simulation using the saved candidate scores, not a fresh model run. It is based on one small synthetic corpus and does not justify enabling reranking in the API. The `--reranker-top-n` option makes the bounded variant reproducible in future benchmark runs. Measure latency and evaluate on diverse, independently labelled PDFs before considering production use.
