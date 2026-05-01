from __future__ import annotations

from pathlib import Path

import pytest

from pitchavatar_rag_sentinel.datasets.loader import discover_datasets, load_dataset
from pitchavatar_rag_sentinel.datasets.models import QueryCaseSpec
from pitchavatar_rag_sentinel.evaluators.retrieval import (
    RetrievedChunk,
    evaluate_retrieval_query,
)

pytestmark = pytest.mark.offline


KEY_TO_RUNTIME_ID = {
    "doc_a": "runtime-doc-a",
    "doc_b": "runtime-doc-b",
    "doc_c": "runtime-doc-c",
}


def make_query(expectations: dict) -> QueryCaseSpec:
    return QueryCaseSpec.model_validate(
        {
            "query_id": "q_chunk",
            "query": "test query",
            "alpha": 0.5,
            "expectations": expectations,
        }
    )


def check_by_name(result, name: str):
    return next(check for check in result.checks if check.name == name)


def test_expected_top1_chunk_contains_passes() -> None:
    query = make_query(
        {
            "expected_top1": "doc_a",
            "expected_top1_chunk_contains": ["Quantum    retrieval", "ENTANGLEMENT"],
        }
    )

    result = evaluate_retrieval_query(
        query_case=query,
        returned_document_ids=["runtime-doc-a"],
        key_to_runtime_id=KEY_TO_RUNTIME_ID,
        returned_chunks=[
            RetrievedChunk(
                document_id="runtime-doc-a",
                content="The quantum retrieval chunk explains entanglement scope.",
            )
        ],
    )

    chunk_check = check_by_name(result, "expected_top1_chunk_contains")
    assert result.passed is True
    assert chunk_check.passed is True
    assert chunk_check.level == "chunk"
    assert chunk_check.matched_result_index == 0
    assert chunk_check.matched_document_id == "runtime-doc-a"
    artifact_check = next(
        check
        for check in result.to_dict()["checks"]
        if check["name"] == "expected_top1_chunk_contains"
    )
    assert artifact_check["level"] == "chunk"
    assert artifact_check["expected_fragments"] == ["Quantum    retrieval", "ENTANGLEMENT"]
    assert artifact_check["matched_result_index"] == 0
    assert artifact_check["matched_document_id"] == "runtime-doc-a"


def test_expected_top1_chunk_contains_fails_for_correct_document_wrong_chunk() -> None:
    query = make_query(
        {
            "expected_top1": "doc_a",
            "expected_top1_chunk_contains": ["target chunk phrase"],
        }
    )

    result = evaluate_retrieval_query(
        query_case=query,
        returned_document_ids=["runtime-doc-a"],
        key_to_runtime_id=KEY_TO_RUNTIME_ID,
        returned_chunks=[
            RetrievedChunk(
                document_id="runtime-doc-a",
                content="Correct document, but this is a different chunk.",
            )
        ],
    )

    document_check = check_by_name(result, "expected_top1")
    chunk_check = check_by_name(result, "expected_top1_chunk_contains")
    assert document_check.passed is True
    assert result.passed is False
    assert chunk_check.passed is False
    assert "missing expected fragment" in chunk_check.failure_reason


def test_expected_in_topk_chunk_contains_passes_for_lower_ranked_result() -> None:
    query = make_query(
        {
            "expected_in_topk_chunk_contains": ["needle phrase", "runbook"],
        }
    )

    result = evaluate_retrieval_query(
        query_case=query,
        returned_document_ids=["runtime-doc-a", "runtime-doc-b"],
        key_to_runtime_id=KEY_TO_RUNTIME_ID,
        returned_chunks=[
            RetrievedChunk(
                document_id="runtime-doc-a",
                content="First result is related but does not contain the target.",
            ),
            RetrievedChunk(
                document_id="runtime-doc-b",
                content="A lower ranked runbook chunk contains the needle phrase.",
            ),
        ],
    )

    chunk_check = check_by_name(result, "expected_in_topk_chunk_contains")
    assert result.passed is True
    assert chunk_check.passed is True
    assert chunk_check.matched_result_index == 1
    assert chunk_check.matched_document_id == "runtime-doc-b"


def test_forbidden_chunk_contains_fails_when_forbidden_text_appears() -> None:
    query = make_query(
        {
            "forbidden_chunk_contains": ["internal-only beta"],
        }
    )

    result = evaluate_retrieval_query(
        query_case=query,
        returned_document_ids=["runtime-doc-a"],
        key_to_runtime_id=KEY_TO_RUNTIME_ID,
        returned_chunks=[
            RetrievedChunk(
                document_id="runtime-doc-a",
                content="This chunk includes an internal-only beta reference.",
            )
        ],
    )

    chunk_check = check_by_name(result, "forbidden_chunk_contains")
    assert result.passed is False
    assert chunk_check.passed is False
    assert chunk_check.matched_result_index == 0
    assert "forbidden fragment" in chunk_check.failure_reason


def test_document_level_expectations_still_pass_as_before() -> None:
    query = make_query(
        {
            "expected_top1": "doc_a",
            "expected_in_topk": ["doc_b"],
            "forbidden_docs": ["doc_c"],
        }
    )

    result = evaluate_retrieval_query(
        query_case=query,
        returned_document_ids=["runtime-doc-a", "runtime-doc-b"],
        key_to_runtime_id=KEY_TO_RUNTIME_ID,
    )

    assert result.passed is True
    assert check_by_name(result, "expected_top1").passed is True
    assert check_by_name(result, "expected_in_topk").passed is True
    assert check_by_name(result, "forbidden_docs_absent").passed is True


def test_existing_datasets_without_chunk_fields_still_validate() -> None:
    datasets = [load_dataset(path) for path in discover_datasets(Path("datasets/retrieval"))]

    assert datasets
    for dataset in datasets:
        for query in dataset.queries:
            assert query.expectations.expected_top1_chunk_contains == []
            assert query.expectations.expected_in_topk_chunk_contains == []
            assert query.expectations.forbidden_chunk_contains == []
