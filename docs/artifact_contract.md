# Artifact Contract

RAG Sentinel artifacts are local JSON files written by dataset runs and read by static reports,
trends, and the Streamlit console. Report readers must tolerate older artifacts with missing
optional fields.

## Layout

Each dataset run writes files under:

```text
artifacts/runs/<run-id>/<dataset-id>/
  seed_manifest.json
  summary.json
  queries/
    <query-id>.json
```

`seed_manifest.json` records runtime document IDs created for the run. `summary.json` is the
aggregate contract. `queries/*.json` hold per-query request, response, and evaluation details.

## summary.json

Current summaries include:

- `dataset_id`: dataset identifier from the source dataset JSON.
- `run_id`: unique runtime run identifier generated from the Sentinel namespace and dataset ID.
- `run_passed`: final run status after query evaluation and cleanup policy are applied.
- `all_queries_passed`: retrieval expectation status only. Cleanup failures do not change this
  value.
- `cleanup_failed`: `true` when any seeded document cleanup could not be verified.
- `cleanup_warning`: human-readable cleanup policy message when cleanup failed, otherwise `null`.
- `seeded_documents`: object keyed by dataset document key. Values include `key`,
  `runtime_document_id`, `indexed_chunk_count`, and `metadata`.
- `query_results`: ordered list of per-query summaries. Each item includes `query_id`, `passed`,
  `latency_ms`, `evaluation`, `request`, `response`, `artifact_path`, and optional
  `ir_evaluation`.
- `cleanup_results`: structured cleanup attempts for seeded runtime documents.
- `metrics`: expectation-based aggregate retrieval and timing metrics.
- `ir_metrics`: qrels-based aggregate IR metrics when at least one query has qrels, otherwise
  `null`.
- `run_dir`: dataset artifact directory path as written by the executor.
- `run_error`: setup or execution error text when the run failed before normal completion,
  otherwise `null`.

## Query Artifacts

Each `queries/<query-id>.json` currently includes:

- `dataset_id`: source dataset ID.
- `query_id`: query case identifier.
- `latency_ms`: elapsed gRPC search call time in milliseconds.
- `request`: effective search request payload used by Sentinel.
- `response`: normalized response payload, usually `results` with `document_id`, `page_content`,
  `metadata`, and `score`.
- `evaluation`: deterministic expectation evaluation with `passed`, `checks`,
  `returned_document_ids`, and `result_count`.
- `ir_evaluation`: optional per-query qrels evaluation when the query defines qrels.

`summary.json` repeats the same request, response, and evaluation payloads in `query_results` so
older readers can work even when query detail files are missing.

## metrics vs ir_metrics

`metrics` are deterministic expectation-based QA metrics. They are calculated from explicit
dataset expectations and their check results. Examples include:

- `query_pass_rate`
- `top1_document_accuracy`
- `document_hit_rate_at_k`
- `top1_chunk_match_rate`
- `chunk_hit_rate_at_k`
- `forbidden_doc_violation_rate`
- `forbidden_chunk_violation_rate`
- `empty_query_pass_rate`

`ir_metrics` are qrels-based ranking metrics. They are separate from `metrics` and do not affect
query pass/fail behavior. Examples include:

- `queries_with_qrels`
- `queries_with_positive_qrels`
- `hit_rate_at_1`, `hit_rate_at_5`, `hit_rate_at_10`
- `recall_at_1`, `recall_at_5`, `recall_at_10`
- `precision_at_1`, `precision_at_5`, `precision_at_10`
- `mrr`
- `ndcg_at_1`, `ndcg_at_5`, `ndcg_at_10`

## Run Status

`all_queries_passed` means every retrieval expectation evaluation passed and no run error occurred.
It intentionally ignores cleanup policy.

`run_passed` is the final run result. It is `false` when queries fail. It is also `false` when
cleanup fails and `RAG_SENTINEL_FAIL_ON_CLEANUP_ERROR=true`.

When `RAG_SENTINEL_FAIL_ON_CLEANUP_ERROR=false`, query results can pass while `cleanup_failed=true`.
In that case `run_passed=true` and `cleanup_warning` explains that the cleanup failure was reported
as a warning.

## Cleanup Fields

`cleanup_failed` is derived from `cleanup_results`: any result with `cleanup_verified=false` makes
cleanup failed.

`cleanup_warning` is present only when cleanup failed. It states whether the cleanup policy failed
the overall run or preserved retrieval results as a warning.

Each cleanup result can include:

- `runtime_document_id`
- `grpc_delete_attempted`
- `grpc_delete_success`
- `grpc_message`
- `fallback_cleanup_used`
- `cleanup_verified`
- `cleanup_status`
- `cleanup_method`
- `cleanup_errors`
- `error`

Structured cleanup errors include method, error type, repr, and traceback text when available.

## Timing Metrics

Timing metrics live inside `metrics`:

- `total_run_ms`: elapsed dataset run time, including seed, search, and cleanup.
- `seed_total_ms`: total document seed time.
- `search_total_ms`: sum of query search latencies.
- `cleanup_total_ms`: total cleanup time.
- `p50_search_ms`: median search latency.
- `p95_search_ms`: nearest-rank p95 search latency.
- `max_search_ms`: maximum search latency.

Dry-run output is not a run artifact and does not write timing metrics.

## Per-Query IR Evaluation

When a query defines qrels, query artifacts and summary query results include `ir_evaluation` with:

- `has_qrels`
- `retrieved_document_ids`
- `relevant_document_keys`
- `relevant_document_ids`
- `relevant_count`
- `qrels`
- per-k `hit_at_k`, `precision_at_k`, `recall_at_k`, and `ndcg_at_k`
- `reciprocal_rank`

Current IR scoring is document-level. Future chunk-level qrel fields are accepted by the dataset
schema but are not scored yet.

## Compatibility

Artifact readers should remain backward compatible:

- Missing `metrics` means metric tables show `n/a`.
- Missing or `null` `ir_metrics` means IR sections are hidden or shown as unavailable.
- Missing query detail files should not break summary-based reporting.
- Missing status fields should display as `not reported`.
- New fields should be additive. Existing public dataset and artifact field names should not be
  renamed without a backward-compatible reader path.
