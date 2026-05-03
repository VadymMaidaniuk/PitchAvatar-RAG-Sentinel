from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any, TypeAlias

from pitchavatar_rag_sentinel.evaluators.retrieval import (
    CheckResult,
    RetrievalEvaluationResult,
)


MetricValue: TypeAlias = int | float | None
Metrics: TypeAlias = dict[str, MetricValue]
EvaluationLike: TypeAlias = RetrievalEvaluationResult | Mapping[str, Any]
CheckLike: TypeAlias = CheckResult | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RetrievalRunTimings:
    total_run_ms: float
    seed_total_ms: float
    search_latencies_ms: Sequence[float]
    cleanup_total_ms: float


@dataclass(frozen=True, slots=True)
class _CheckCounts:
    total: int
    passed: int

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float | None:
        return _rate(self.passed, self.total)

    @property
    def failure_rate(self) -> float | None:
        return _rate(self.failed, self.total)


def calculate_retrieval_metrics(
    evaluations: Iterable[EvaluationLike],
    *,
    timings: RetrievalRunTimings | None = None,
) -> Metrics:
    evaluation_list = list(evaluations)
    total_queries = len(evaluation_list)
    passed_queries = sum(1 for evaluation in evaluation_list if _evaluation_passed(evaluation))
    failed_queries = total_queries - passed_queries

    top1_document = _check_counts(evaluation_list, "expected_top1")
    document_hit_at_k = _check_counts(evaluation_list, "expected_in_topk")
    forbidden_doc = _check_counts(evaluation_list, "forbidden_docs_absent")
    empty_query = _check_counts(evaluation_list, "expect_empty")
    top1_chunk = _check_counts(evaluation_list, "expected_top1_chunk_contains")
    chunk_hit_at_k = _check_counts(evaluation_list, "expected_in_topk_chunk_contains")
    forbidden_chunk = _check_counts(evaluation_list, "forbidden_chunk_contains")

    metrics: Metrics = {
        "total_queries": total_queries,
        "passed_queries": passed_queries,
        "failed_queries": failed_queries,
        "query_pass_rate": _rate(passed_queries, total_queries),
        "top1_document_checks_total": top1_document.total,
        "top1_document_checks_passed": top1_document.passed,
        "top1_document_accuracy": top1_document.pass_rate,
        "document_hit_at_k_checks_total": document_hit_at_k.total,
        "document_hit_at_k_checks_passed": document_hit_at_k.passed,
        "document_hit_rate_at_k": document_hit_at_k.pass_rate,
        "forbidden_doc_checks_total": forbidden_doc.total,
        "forbidden_doc_violations": forbidden_doc.failed,
        "forbidden_doc_violation_rate": forbidden_doc.failure_rate,
        "empty_query_checks_total": empty_query.total,
        "empty_query_checks_passed": empty_query.passed,
        "empty_query_pass_rate": empty_query.pass_rate,
        "top1_chunk_checks_total": top1_chunk.total,
        "top1_chunk_checks_passed": top1_chunk.passed,
        "top1_chunk_match_rate": top1_chunk.pass_rate,
        "chunk_hit_at_k_checks_total": chunk_hit_at_k.total,
        "chunk_hit_at_k_checks_passed": chunk_hit_at_k.passed,
        "chunk_hit_rate_at_k": chunk_hit_at_k.pass_rate,
        "forbidden_chunk_checks_total": forbidden_chunk.total,
        "forbidden_chunk_violations": forbidden_chunk.failed,
        "forbidden_chunk_violation_rate": forbidden_chunk.failure_rate,
    }

    if timings is not None:
        metrics.update(_calculate_timing_metrics(timings))

    return metrics


def _check_counts(
    evaluations: Sequence[EvaluationLike],
    check_name: str,
) -> _CheckCounts:
    checks = [
        check
        for evaluation in evaluations
        for check in _evaluation_checks(evaluation)
        if _check_name(check) == check_name and _check_applicable(check)
    ]
    return _CheckCounts(
        total=len(checks),
        passed=sum(1 for check in checks if _check_passed(check)),
    )


def _calculate_timing_metrics(timings: RetrievalRunTimings) -> Metrics:
    search_latencies = [float(latency_ms) for latency_ms in timings.search_latencies_ms]
    return {
        "total_run_ms": _round_ms(timings.total_run_ms),
        "seed_total_ms": _round_ms(timings.seed_total_ms),
        "search_total_ms": _round_ms(sum(search_latencies)),
        "cleanup_total_ms": _round_ms(timings.cleanup_total_ms),
        "p50_search_ms": _median(search_latencies),
        "p95_search_ms": _nearest_rank_percentile(search_latencies, percentile=95),
        "max_search_ms": _round_ms(max(search_latencies)) if search_latencies else None,
    }


def _evaluation_passed(evaluation: EvaluationLike) -> bool:
    if isinstance(evaluation, Mapping):
        return bool(evaluation["passed"])
    return evaluation.passed


def _evaluation_checks(evaluation: EvaluationLike) -> Sequence[CheckLike]:
    if isinstance(evaluation, Mapping):
        return evaluation.get("checks", [])
    return evaluation.checks


def _check_name(check: CheckLike) -> str:
    if isinstance(check, Mapping):
        return str(check["name"])
    return check.name


def _check_passed(check: CheckLike) -> bool:
    if isinstance(check, Mapping):
        return bool(check["passed"])
    return check.passed


def _check_applicable(check: CheckLike) -> bool:
    if isinstance(check, Mapping):
        return bool(check.get("applicable", True))
    return check.applicable


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return _round_ms(sorted_values[middle])
    return _round_ms((sorted_values[middle - 1] + sorted_values[middle]) / 2)


def _nearest_rank_percentile(values: Sequence[float], *, percentile: int) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = ceil((percentile / 100) * len(sorted_values)) - 1
    return _round_ms(sorted_values[max(index, 0)])


def _round_ms(value: float) -> float:
    return round(float(value), 3)
