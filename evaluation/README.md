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
