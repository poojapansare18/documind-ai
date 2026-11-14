# DocuMind retrieval benchmark

This benchmark measures semantic retrieval quality at the **page level** using a synthetic, three-page corporate travel policy. It has known relevant pages for five natural-language queries.

## Run the benchmark

1. Install the project requirements.
2. Start DocuMind:

   ```powershell
   python -m uvicorn app.main:app --reload
   ```

3. Run the benchmark:

   ```powershell
   python scripts/evaluate_retrieval.py --k 3 --report evaluation/results.json
   ```

The runner creates the synthetic PDF if it does not exist, uploads it if it is not already in the document library, indexes its chunks, runs the benchmark queries against semantic search, and writes a JSON report. The first indexing run downloads the local embedding model. If API-key authentication is enabled, set `DOCUMIND_API_KEY` in the environment or in the root `.env` file.

## Metrics

- **Precision@k:** fraction of the top-k returned pages that are relevant.
- **Recall@k:** fraction of all relevant pages found in the top-k.
- **MRR@k:** reciprocal rank of the first relevant page; higher is better.
- **nDCG@k:** ranking quality that rewards relevant pages appearing earlier.

Results are averaged across all benchmark queries and also reported per query.

## Scope and limitations

This is a repeatable baseline, not a broad evaluation of real-world PDFs. It evaluates page retrieval, not answer factuality, citation entailment, OCR accuracy, or table extraction accuracy. The corpus is synthetic and must not be presented as an external industry benchmark. Add representative, permission-cleared documents and manually reviewed relevance labels before using the metrics to make production claims.
