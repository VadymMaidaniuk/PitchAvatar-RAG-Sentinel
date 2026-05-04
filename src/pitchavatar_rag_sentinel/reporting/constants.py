from __future__ import annotations

SUMMARY_METRIC_NAMES = (
    "query_pass_rate",
    "top1_document_accuracy",
    "document_hit_rate_at_k",
    "top1_chunk_match_rate",
    "chunk_hit_rate_at_k",
    "forbidden_doc_violation_rate",
    "forbidden_chunk_violation_rate",
    "empty_query_pass_rate",
)

IR_METRIC_NAMES = (
    "queries_with_qrels",
    "queries_with_positive_qrels",
    "hit_rate_at_1",
    "hit_rate_at_5",
    "hit_rate_at_10",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "precision_at_1",
    "precision_at_5",
    "precision_at_10",
    "mrr",
    "ndcg_at_1",
    "ndcg_at_5",
    "ndcg_at_10",
)

TIMING_METRIC_NAMES = (
    "total_run_ms",
    "seed_total_ms",
    "search_total_ms",
    "cleanup_total_ms",
    "p50_search_ms",
    "p95_search_ms",
    "max_search_ms",
)

RUN_HISTORY_IR_METRIC_NAMES = (
    "hit_rate_at_1",
    "hit_rate_at_5",
    "recall_at_5",
    "precision_at_5",
    "mrr",
    "ndcg_at_5",
    "ndcg_at_10",
)

TREND_CHART_METRIC_NAMES = (
    "query_pass_rate",
    "hit_rate_at_1",
    "hit_rate_at_5",
    "recall_at_5",
    "precision_at_5",
    "mrr",
    "ndcg_at_5",
    "top1_document_accuracy",
    "chunk_hit_rate_at_k",
    "p95_search_ms",
)
