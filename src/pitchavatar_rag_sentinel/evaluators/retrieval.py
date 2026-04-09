from __future__ import annotations

from dataclasses import asdict, dataclass

from pitchavatar_rag_sentinel.datasets.models import QueryCaseSpec


@dataclass(slots=True)
class CheckResult:
    name: str
    passed: bool
    details: str


@dataclass(slots=True)
class RetrievalEvaluationResult:
    passed: bool
    checks: list[CheckResult]
    returned_document_ids: list[str]
    result_count: int

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
            "returned_document_ids": self.returned_document_ids,
            "result_count": self.result_count,
        }


def evaluate_retrieval_query(
    query_case: QueryCaseSpec,
    returned_document_ids: list[str],
    key_to_runtime_id: dict[str, str],
) -> RetrievalEvaluationResult:
    checks: list[CheckResult] = []
    expectations = query_case.expectations
    result_count = len(returned_document_ids)

    if expectations.expect_empty:
        passed = result_count == 0
        checks.append(
            CheckResult(
                name="expect_empty",
                passed=passed,
                details=f"result_count={result_count}",
            )
        )
        return RetrievalEvaluationResult(
            passed=passed,
            checks=checks,
            returned_document_ids=returned_document_ids,
            result_count=result_count,
        )

    checks.append(
        CheckResult(
            name="min_results",
            passed=result_count >= expectations.min_results,
            details=f"result_count={result_count}, min_results={expectations.min_results}",
        )
    )

    if expectations.expected_top1:
        expected_top1_runtime = key_to_runtime_id[expectations.expected_top1]
        actual_top1 = returned_document_ids[0] if returned_document_ids else None
        checks.append(
            CheckResult(
                name="expected_top1",
                passed=actual_top1 == expected_top1_runtime,
                details=f"actual_top1={actual_top1}, expected_top1={expected_top1_runtime}",
            )
        )

    expected_runtime_ids = [key_to_runtime_id[key] for key in expectations.expected_in_topk]
    missing_expected = sorted(set(expected_runtime_ids) - set(returned_document_ids))
    checks.append(
        CheckResult(
            name="expected_in_topk",
            passed=not missing_expected,
            details=f"missing_expected={missing_expected}",
        )
    )

    forbidden_runtime_ids = [key_to_runtime_id[key] for key in expectations.forbidden_docs]
    unexpected_forbidden = sorted(set(forbidden_runtime_ids) & set(returned_document_ids))
    checks.append(
        CheckResult(
            name="forbidden_docs_absent",
            passed=not unexpected_forbidden,
            details=f"forbidden_present={unexpected_forbidden}",
        )
    )

    return RetrievalEvaluationResult(
        passed=all(check.passed for check in checks),
        checks=checks,
        returned_document_ids=returned_document_ids,
        result_count=result_count,
    )

