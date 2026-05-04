# Qrels-Based IR Metrics

RAG Sentinel now supports optional document-level qrels in retrieval datasets. This is additive to
the existing expectation-based QA evaluator.

## Expectation Metrics vs IR Metrics

Expectation checks answer deterministic QA questions such as:

- did the expected document appear as top 1?
- did an expected document appear anywhere in the returned set?
- did returned chunk text contain a required literal fragment?
- were forbidden documents or fragments absent?

Those checks continue to determine query pass/fail and the existing `metrics` section.

Qrels-based IR metrics answer ranking-quality questions using relevance labels. They are reported
separately under `ir_metrics` and do not change expectation pass/fail behavior.

## Qrels Shape

Add `qrels` to a query when relevance labels are available:

```json
{
  "query_id": "q_pdf_upload_limit",
  "query": "maximum PDF upload size",
  "alpha": 0.5,
  "qrels": [
    {
      "document_key": "doc_upload_limits",
      "relevance": 2
    },
    {
      "document_key": "doc_general_upload",
      "relevance": 1
    }
  ]
}
```

Rules:

- `qrels` are optional.
- `document_key` must match a dataset document key.
- `relevance` must be an integer greater than or equal to `0`.
- `relevance > 0` means relevant.
- `relevance = 0` is allowed for an explicitly judged non-relevant document, but is not required.

## Metrics

For queries with qrels, RAG Sentinel calculates document-level metrics at `k = 1, 5, 10`:

- `hit_rate_at_k`
- `recall_at_k`
- `precision_at_k`
- `mrr`
- `ndcg_at_k`

Duplicate retrieved document ids are deduplicated before document-level scoring so repeated chunks
from the same document do not inflate results.

Per-query artifacts include an `ir_evaluation` object with retrieved document ids, relevant
document keys and runtime ids, per-k hit/precision/recall/NDCG values, and reciprocal rank.

Run summaries include a separate `ir_metrics` section when at least one query has qrels. Existing
artifacts and datasets without qrels continue to load normally.

## Current Limitation

Qrels are document-level only. The schema accepts future-oriented fields such as `chunk_id`,
`chunk_index`, and `chunk_contains`, but they are not used for metrics yet. Chunk-level qrels should
wait until gRPC results expose stable chunk ids or stable chunk metadata.

The implementation is intentionally dependency-free and offline-testable. A `pytrec_eval`
integration can be added later if the project needs standardized TREC tooling or additional IR
metrics.
