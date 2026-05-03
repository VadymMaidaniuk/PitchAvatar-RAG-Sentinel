# Artifact Analysis Notes

## retrieval_smoke_v1 run qa-sentinel-retrieval_smoke_v1-1777818479321-cac34fcb

Artifact directory:

`artifacts/runs/qa-sentinel-retrieval_smoke_v1-1777818479321-cac34fcb/retrieval_smoke_v1`

Run status:

- `run_passed`: `false`
- `all_queries_passed`: `false`
- `cleanup_failed`: `false`
- cleanup result: both seeded documents were deleted by gRPC and verified absent

Metrics:

- `query_pass_rate`: `0.0`
- `top1_document_accuracy`: `0.5`
- `document_hit_rate_at_k`: `1.0`
- `top1_chunk_match_rate`: `1.0`
- `chunk_hit_rate_at_k`: `0.0`
- `forbidden_doc_violation_rate`: `0.5`
- `forbidden_chunk_violation_rate`: `1.0`

Failed query details:

- `q_smoke_release_window`
  - Query: `Harbor rollback approval code`
  - Top result was the expected release document.
  - Top chunk contained the required release fragments.
  - Failed `forbidden_docs_absent` because the billing document was also returned at rank 2.
  - Failed `forbidden_chunk_contains` because the rank 2 billing chunk contained `invoice dispute`.
  - Interpretation: this looks more like dataset calibration or query scoping strictness than a top-1
    retrieval failure. With `document_scope: all`, `top_k: 3`, and `threshold: 0.0`, the service can
    return a low-scored unrelated lower-ranked document in a tiny two-document corpus. The current
    forbidden expectation requires that not happen.

- `q_smoke_billing_dispute`
  - Query: `invoice dispute adjustment reason`
  - Returned zero results.
  - Failed `min_results`, `expected_top1`, and `expected_in_topk_chunk_contains`.
  - Interpretation: this should be checked before changing the dataset because the query terms are
    explicit in the billing document. Possible causes include the service text-search path for
    `alpha: 1.0`, service-side BM25 minimum-score behavior, analyzer/indexing behavior, or request
    scoping/configuration differences.

Recommended checks before changing the dataset:

- Confirm whether `alpha: 1.0` text search can retrieve the billing document in this environment
  with the same scoped `document_ids`.
- Compare the billing query with `alpha: 0.5` and `alpha: 0.0` in a controlled real run or diagnostic
  script, without loosening assertions silently.
- Check service-side BM25 minimum-score configuration and whether request `threshold: 0.0` is honored
  for pure text search.
- Inspect service logs for the billing query path and the scoped document ids used in the request.
- If lower-ranked unrelated documents are expected in tiny corpora, consider a follow-up dataset
  calibration such as stricter document scope, metadata filters, lower `top_k`, or removing forbidden
  checks from broad-scope smoke queries. Do not change these expectations without confirming the
  retrieval behavior above.
