# DocuMind retrieval benchmark

This benchmark measures hybrid retrieval at the **page level** using a synthetic, 12-page corporate travel policy and 24 manually authored questions. Each question has one known relevant page. The additional pages provide distractors, so a correct top result is no longer guaranteed by retrieving the entire three-page corpus.

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

The semantic score is cosine similarity from the configured FastEmbed model. The lexical score is BM25, calculated over indexed chunks. The API combines 75% semantic score with 25% normalised BM25 score; the response exposes the combined score plus both component scores. This is a fixed baseline weighting, not a learned re-ranker.

The runner creates the synthetic PDF if it does not exist, uploads it if it is not already in the document library, indexes its chunks, and requests all indexed results for each query (up to the API limit of 20). It reports the top-k metrics for hybrid, semantic-only, and lexical-only rankings. Per-query diagnostics include each candidate's component scores and the relevant page's rank under all three ranking signals. This makes it possible to identify whether score fusion, rather than candidate retrieval, caused a ranking regression. The corpus uses a versioned filename so an older three-page benchmark document in a local database is not accidentally reused. The first indexing run downloads the local embedding model. If API-key authentication is enabled, set `DOCUMIND_API_KEY` in the environment or in the root `.env` file.

## Metrics

- **Precision@k:** fraction of the top-k returned pages that are relevant.
- **Recall@k:** fraction of all relevant pages found in the top-k.
- **MRR@k:** reciprocal rank of the first relevant page; higher is better.
- **nDCG@k:** ranking quality that rewards relevant pages appearing earlier.

Results are averaged across all benchmark queries and also reported per query. With one relevant page per query, Precision@3 is at most 1/3 when that relevant page is found and three distinct pages are returned. Precision@1 directly measures whether the first returned page is relevant.

## Scope and limitations

This is a repeatable engineering baseline, not a broad evaluation of real-world PDFs. The corpus is synthetic, has one short page per topic, and uses manually authored queries. It evaluates page retrieval, not answer factuality, citation entailment, OCR accuracy, table extraction accuracy, or performance on diverse document layouts. Do not present these scores as an external industry benchmark. Add representative, permission-cleared documents and independently reviewed relevance labels before making production claims.
