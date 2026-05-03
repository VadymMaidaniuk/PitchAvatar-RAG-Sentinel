from __future__ import annotations

import pytest

from pitchavatar_rag_sentinel.evaluators.retrieval import (
    CheckResult,
    RetrievalEvaluationResult,
)
from pitchavatar_rag_sentinel.reporting.metrics import calculate_retrieval_metrics

pytestmark = pytest.mark.offline


def make_evaluation(
    *,
    passed: bool,
    checks: list[CheckResult] | None = None,
) -> RetrievalEvaluationResult:
    return RetrievalEvaluationResult(
        passed=passed,
        checks=checks or [],
        returned_document_ids=[],
        result_count=0,
    )


def make_check(
    name: str,
    *,
    passed: bool,
    applicable: bool = True,
) -> CheckResult:
    return CheckResult(
        name=name,
        passed=passed,
        details="test check",
        applicable=applicable,
    )


def test_all_queries_pass_gives_query_pass_rate_one() -> None:
    metrics = calculate_retrieval_metrics(
        [
            make_evaluation(passed=True),
            make_evaluation(passed=True),
        ]
    )

    assert metrics["total_queries"] == 2
    assert metrics["passed_queries"] == 2
    assert metrics["failed_queries"] == 0
    assert metrics["query_pass_rate"] == 1.0


def test_mixed_pass_fail_gives_correct_query_counts() -> None:
    metrics = calculate_retrieval_metrics(
        [
            make_evaluation(passed=True),
            make_evaluation(passed=False),
            make_evaluation(passed=True),
        ]
    )

    assert metrics["total_queries"] == 3
    assert metrics["passed_queries"] == 2
    assert metrics["failed_queries"] == 1
    assert metrics["query_pass_rate"] == pytest.approx(2 / 3)


def test_top1_document_accuracy_is_calculated() -> None:
    metrics = calculate_retrieval_metrics(
        [
            make_evaluation(
                passed=True,
                checks=[make_check("expected_top1", passed=True)],
            ),
            make_evaluation(
                passed=False,
                checks=[make_check("expected_top1", passed=False)],
            ),
            make_evaluation(
                passed=True,
                checks=[make_check("expected_top1", passed=True)],
            ),
        ]
    )

    assert metrics["top1_document_checks_total"] == 3
    assert metrics["top1_document_checks_passed"] == 2
    assert metrics["top1_document_accuracy"] == pytest.approx(2 / 3)


def test_document_hit_rate_at_k_is_calculated() -> None:
    metrics = calculate_retrieval_metrics(
        [
            make_evaluation(
                passed=True,
                checks=[make_check("expected_in_topk", passed=True)],
            ),
            make_evaluation(
                passed=False,
                checks=[make_check("expected_in_topk", passed=False)],
            ),
        ]
    )

    assert metrics["document_hit_at_k_checks_total"] == 2
    assert metrics["document_hit_at_k_checks_passed"] == 1
    assert metrics["document_hit_rate_at_k"] == pytest.approx(0.5)


def test_top1_chunk_match_rate_is_calculated() -> None:
    metrics = calculate_retrieval_metrics(
        [
            make_evaluation(
                passed=True,
                checks=[make_check("expected_top1_chunk_contains", passed=True)],
            ),
            make_evaluation(
                passed=False,
                checks=[make_check("expected_top1_chunk_contains", passed=False)],
            ),
        ]
    )

    assert metrics["top1_chunk_checks_total"] == 2
    assert metrics["top1_chunk_checks_passed"] == 1
    assert metrics["top1_chunk_match_rate"] == pytest.approx(0.5)


def test_chunk_hit_rate_at_k_is_calculated() -> None:
    metrics = calculate_retrieval_metrics(
        [
            make_evaluation(
                passed=True,
                checks=[make_check("expected_in_topk_chunk_contains", passed=True)],
            ),
            make_evaluation(
                passed=False,
                checks=[make_check("expected_in_topk_chunk_contains", passed=False)],
            ),
            make_evaluation(
                passed=True,
                checks=[make_check("expected_in_topk_chunk_contains", passed=True)],
            ),
        ]
    )

    assert metrics["chunk_hit_at_k_checks_total"] == 3
    assert metrics["chunk_hit_at_k_checks_passed"] == 2
    assert metrics["chunk_hit_rate_at_k"] == pytest.approx(2 / 3)


def test_forbidden_doc_violation_rate_is_calculated() -> None:
    metrics = calculate_retrieval_metrics(
        [
            make_evaluation(
                passed=True,
                checks=[make_check("forbidden_docs_absent", passed=True)],
            ),
            make_evaluation(
                passed=False,
                checks=[make_check("forbidden_docs_absent", passed=False)],
            ),
            make_evaluation(
                passed=False,
                checks=[make_check("forbidden_docs_absent", passed=False)],
            ),
        ]
    )

    assert metrics["forbidden_doc_checks_total"] == 3
    assert metrics["forbidden_doc_violations"] == 2
    assert metrics["forbidden_doc_violation_rate"] == pytest.approx(2 / 3)


def test_forbidden_chunk_violation_rate_is_calculated() -> None:
    metrics = calculate_retrieval_metrics(
        [
            make_evaluation(
                passed=True,
                checks=[make_check("forbidden_chunk_contains", passed=True)],
            ),
            make_evaluation(
                passed=False,
                checks=[make_check("forbidden_chunk_contains", passed=False)],
            ),
        ]
    )

    assert metrics["forbidden_chunk_checks_total"] == 2
    assert metrics["forbidden_chunk_violations"] == 1
    assert metrics["forbidden_chunk_violation_rate"] == pytest.approx(0.5)


def test_metrics_with_zero_applicable_checks_return_null_rates() -> None:
    metrics = calculate_retrieval_metrics(
        [
            make_evaluation(
                passed=True,
                checks=[
                    make_check("expected_in_topk", passed=True, applicable=False),
                    make_check("forbidden_docs_absent", passed=True, applicable=False),
                ],
            )
        ]
    )

    assert metrics["document_hit_at_k_checks_total"] == 0
    assert metrics["document_hit_rate_at_k"] is None
    assert metrics["forbidden_doc_checks_total"] == 0
    assert metrics["forbidden_doc_violation_rate"] is None
    assert metrics["top1_chunk_match_rate"] is None
