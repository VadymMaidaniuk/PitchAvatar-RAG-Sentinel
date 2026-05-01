from __future__ import annotations

from dataclasses import asdict, dataclass

from pitchavatar_rag_sentinel.datasets.models import QueryCaseSpec


@dataclass(slots=True)
class CheckResult:
    name: str
    passed: bool
    details: str
    level: str = "document"
    expected_fragments: list[str] | None = None
    matched_result_index: int | None = None
    matched_document_id: str | None = None
    failure_reason: str | None = None


@dataclass(slots=True)
class RetrievedChunk:
    document_id: str
    content: str | None


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
    returned_chunks: list[RetrievedChunk] | None = None,
) -> RetrievalEvaluationResult:
    checks: list[CheckResult] = []
    expectations = query_case.expectations
    result_count = len(returned_document_ids)
    chunks = returned_chunks or [
        RetrievedChunk(document_id=document_id, content=None)
        for document_id in returned_document_ids
    ]

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

    checks.extend(
        _evaluate_chunk_expectations(
            expected_top1_fragments=expectations.expected_top1_chunk_contains,
            expected_in_topk_fragments=expectations.expected_in_topk_chunk_contains,
            forbidden_fragments=expectations.forbidden_chunk_contains,
            returned_chunks=chunks,
        )
    )

    return RetrievalEvaluationResult(
        passed=all(check.passed for check in checks),
        checks=checks,
        returned_document_ids=returned_document_ids,
        result_count=result_count,
    )


def _evaluate_chunk_expectations(
    *,
    expected_top1_fragments: list[str],
    expected_in_topk_fragments: list[str],
    forbidden_fragments: list[str],
    returned_chunks: list[RetrievedChunk],
) -> list[CheckResult]:
    checks: list[CheckResult] = []

    if expected_top1_fragments:
        checks.append(
            _evaluate_expected_top1_chunk_contains(
                returned_chunks,
                expected_top1_fragments,
            )
        )

    if expected_in_topk_fragments:
        checks.append(
            _evaluate_expected_in_topk_chunk_contains(
                returned_chunks,
                expected_in_topk_fragments,
            )
        )

    if forbidden_fragments:
        checks.append(
            _evaluate_forbidden_chunk_contains(returned_chunks, forbidden_fragments)
        )

    return checks


def _evaluate_expected_top1_chunk_contains(
    returned_chunks: list[RetrievedChunk],
    expected_fragments: list[str],
) -> CheckResult:
    if not returned_chunks:
        return _chunk_check_result(
            name="expected_top1_chunk_contains",
            passed=False,
            expected_fragments=expected_fragments,
            failure_reason="no returned results to inspect",
        )

    top_chunk = returned_chunks[0]
    if _chunk_text_missing(top_chunk):
        return _chunk_check_result(
            name="expected_top1_chunk_contains",
            passed=False,
            expected_fragments=expected_fragments,
            matched_result_index=0,
            matched_document_id=top_chunk.document_id,
            failure_reason="top1 result does not expose chunk text/content",
        )

    missing_fragments = _missing_fragments(top_chunk.content or "", expected_fragments)
    passed = not missing_fragments
    return _chunk_check_result(
        name="expected_top1_chunk_contains",
        passed=passed,
        expected_fragments=expected_fragments,
        matched_result_index=0 if passed else None,
        matched_document_id=top_chunk.document_id if passed else None,
        failure_reason=(
            None
            if passed
            else f"top1 chunk is missing expected fragment(s): {missing_fragments}"
        ),
    )


def _evaluate_expected_in_topk_chunk_contains(
    returned_chunks: list[RetrievedChunk],
    expected_fragments: list[str],
) -> CheckResult:
    missing_text_indices = _missing_text_indices(returned_chunks)
    if missing_text_indices:
        return _chunk_check_result(
            name="expected_in_topk_chunk_contains",
            passed=False,
            expected_fragments=expected_fragments,
            failure_reason=(
                "returned result(s) do not expose chunk text/content: "
                f"{missing_text_indices}"
            ),
        )

    for index, chunk in enumerate(returned_chunks):
        if _contains_all_fragments(chunk.content or "", expected_fragments):
            return _chunk_check_result(
                name="expected_in_topk_chunk_contains",
                passed=True,
                expected_fragments=expected_fragments,
                matched_result_index=index,
                matched_document_id=chunk.document_id,
            )

    return _chunk_check_result(
        name="expected_in_topk_chunk_contains",
        passed=False,
        expected_fragments=expected_fragments,
        failure_reason="no returned chunk contains all expected fragment(s)",
    )


def _evaluate_forbidden_chunk_contains(
    returned_chunks: list[RetrievedChunk],
    forbidden_fragments: list[str],
) -> CheckResult:
    missing_text_indices = _missing_text_indices(returned_chunks)
    if missing_text_indices:
        return _chunk_check_result(
            name="forbidden_chunk_contains",
            passed=False,
            expected_fragments=forbidden_fragments,
            failure_reason=(
                "returned result(s) do not expose chunk text/content: "
                f"{missing_text_indices}"
            ),
        )

    for index, chunk in enumerate(returned_chunks):
        matched_forbidden = [
            fragment
            for fragment in forbidden_fragments
            if _normalized_contains(chunk.content or "", fragment)
        ]
        if matched_forbidden:
            return _chunk_check_result(
                name="forbidden_chunk_contains",
                passed=False,
                expected_fragments=forbidden_fragments,
                matched_result_index=index,
                matched_document_id=chunk.document_id,
                failure_reason=f"forbidden fragment(s) present: {matched_forbidden}",
            )

    return _chunk_check_result(
        name="forbidden_chunk_contains",
        passed=True,
        expected_fragments=forbidden_fragments,
    )


def _chunk_check_result(
    *,
    name: str,
    passed: bool,
    expected_fragments: list[str],
    matched_result_index: int | None = None,
    matched_document_id: str | None = None,
    failure_reason: str | None = None,
) -> CheckResult:
    details = {
        "expected_fragments": expected_fragments,
        "matched_result_index": matched_result_index,
        "matched_document_id": matched_document_id,
        "failure_reason": failure_reason,
    }
    return CheckResult(
        name=name,
        passed=passed,
        level="chunk",
        details=str(details),
        expected_fragments=expected_fragments,
        matched_result_index=matched_result_index,
        matched_document_id=matched_document_id,
        failure_reason=failure_reason,
    )


def _missing_text_indices(returned_chunks: list[RetrievedChunk]) -> list[int]:
    return [
        index
        for index, chunk in enumerate(returned_chunks)
        if _chunk_text_missing(chunk)
    ]


def _chunk_text_missing(chunk: RetrievedChunk) -> bool:
    return chunk.content is None or not chunk.content.strip()


def _contains_all_fragments(content: str, fragments: list[str]) -> bool:
    return all(_normalized_contains(content, fragment) for fragment in fragments)


def _missing_fragments(content: str, fragments: list[str]) -> list[str]:
    return [
        fragment
        for fragment in fragments
        if not _normalized_contains(content, fragment)
    ]


def _normalized_contains(content: str, fragment: str) -> bool:
    return _normalize_text(fragment) in _normalize_text(content)


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())
