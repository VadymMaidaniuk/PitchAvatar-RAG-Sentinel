from __future__ import annotations

import pytest

from pitchavatar_rag_sentinel.datasets.models import QueryCaseSpec
from pitchavatar_rag_sentinel.evaluators.ir_metrics import (
    calculate_query_ir_metrics,
    calculate_summary_ir_metrics,
)

pytestmark = pytest.mark.offline

KEY_TO_RUNTIME_ID = {
    "doc_a": "runtime-doc-a",
    "doc_b": "runtime-doc-b",
    "doc_c": "runtime-doc-c",
}


def make_qrels_query(qrels: list[dict]) -> QueryCaseSpec:
    return QueryCaseSpec.model_validate(
        {
            "query_id": "q_ir",
            "query": "test query",
            "alpha": 0.5,
            "qrels": qrels,
        }
    )


def test_query_without_qrels_returns_no_ir_metrics() -> None:
    query = make_qrels_query([])

    assert (
        calculate_query_ir_metrics(
            query_case=query,
            retrieved_document_ids=["runtime-doc-a"],
            key_to_runtime_id=KEY_TO_RUNTIME_ID,
        )
        is None
    )
    assert calculate_summary_ir_metrics([None]) is None


def test_hit_rate_at_k_calculation() -> None:
    query = make_qrels_query([{"document_key": "doc_a", "relevance": 2}])

    evaluation = calculate_query_ir_metrics(
        query_case=query,
        retrieved_document_ids=["runtime-doc-c", "runtime-doc-a"],
        key_to_runtime_id=KEY_TO_RUNTIME_ID,
    )
    summary = calculate_summary_ir_metrics([evaluation])

    assert evaluation is not None
    assert evaluation["hit_at_1"] is False
    assert evaluation["hit_at_5"] is True
    assert summary is not None
    assert summary["hit_rate_at_1"] == 0.0
    assert summary["hit_rate_at_5"] == 1.0


def test_recall_at_k_calculation() -> None:
    query = make_qrels_query(
        [
            {"document_key": "doc_a", "relevance": 2},
            {"document_key": "doc_b", "relevance": 1},
        ]
    )

    evaluation = calculate_query_ir_metrics(
        query_case=query,
        retrieved_document_ids=["runtime-doc-a", "runtime-doc-c"],
        key_to_runtime_id=KEY_TO_RUNTIME_ID,
    )

    assert evaluation is not None
    assert evaluation["recall_at_1"] == 0.5
    assert evaluation["recall_at_5"] == 0.5


def test_precision_at_k_calculation() -> None:
    query = make_qrels_query([{"document_key": "doc_a", "relevance": 1}])

    evaluation = calculate_query_ir_metrics(
        query_case=query,
        retrieved_document_ids=["runtime-doc-a", "runtime-doc-c"],
        key_to_runtime_id=KEY_TO_RUNTIME_ID,
    )

    assert evaluation is not None
    assert evaluation["precision_at_1"] == 1.0
    assert evaluation["precision_at_5"] == 0.2


def test_mrr_calculation() -> None:
    query = make_qrels_query([{"document_key": "doc_b", "relevance": 1}])

    evaluation = calculate_query_ir_metrics(
        query_case=query,
        retrieved_document_ids=["runtime-doc-c", "runtime-doc-b", "runtime-doc-a"],
        key_to_runtime_id=KEY_TO_RUNTIME_ID,
    )
    summary = calculate_summary_ir_metrics([evaluation])

    assert evaluation is not None
    assert evaluation["reciprocal_rank"] == 0.5
    assert summary is not None
    assert summary["mrr"] == 0.5


def test_ndcg_at_k_uses_graded_relevance() -> None:
    query = make_qrels_query(
        [
            {"document_key": "doc_a", "relevance": 2},
            {"document_key": "doc_b", "relevance": 1},
        ]
    )

    perfect = calculate_query_ir_metrics(
        query_case=query,
        retrieved_document_ids=["runtime-doc-a", "runtime-doc-b"],
        key_to_runtime_id=KEY_TO_RUNTIME_ID,
    )
    reversed_ranking = calculate_query_ir_metrics(
        query_case=query,
        retrieved_document_ids=["runtime-doc-b", "runtime-doc-a"],
        key_to_runtime_id=KEY_TO_RUNTIME_ID,
    )

    assert perfect is not None
    assert reversed_ranking is not None
    assert perfect["ndcg_at_5"] == pytest.approx(1.0)
    assert 0 < reversed_ranking["ndcg_at_5"] < 1


def test_duplicate_retrieved_docs_do_not_inflate_metrics() -> None:
    query = make_qrels_query([{"document_key": "doc_a", "relevance": 1}])

    evaluation = calculate_query_ir_metrics(
        query_case=query,
        retrieved_document_ids=[
            "runtime-doc-a",
            "runtime-doc-a",
            "runtime-doc-a",
        ],
        key_to_runtime_id=KEY_TO_RUNTIME_ID,
    )

    assert evaluation is not None
    assert evaluation["retrieved_document_ids"] == ["runtime-doc-a"]
    assert evaluation["recall_at_5"] == 1.0
    assert evaluation["precision_at_5"] == 0.2
    assert evaluation["ndcg_at_5"] == 1.0
