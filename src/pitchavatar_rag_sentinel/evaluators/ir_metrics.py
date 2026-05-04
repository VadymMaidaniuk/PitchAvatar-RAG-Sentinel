from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import log2
from typing import Any, TypeAlias

from pitchavatar_rag_sentinel.datasets.models import QueryCaseSpec, QueryQrelSpec

K_VALUES = (1, 5, 10)

IrMetricValue: TypeAlias = int | float | None
IrMetrics: TypeAlias = dict[str, IrMetricValue]
IrQueryEvaluation: TypeAlias = dict[str, Any]


def calculate_query_ir_metrics(
    *,
    query_case: QueryCaseSpec,
    retrieved_document_ids: Sequence[str],
    key_to_runtime_id: Mapping[str, str],
    k_values: Sequence[int] = K_VALUES,
) -> IrQueryEvaluation | None:
    if not query_case.qrels:
        return None

    relevance_by_key = _max_relevance_by_key(query_case.qrels)
    relevance_by_runtime_id = {
        key_to_runtime_id[document_key]: relevance
        for document_key, relevance in relevance_by_key.items()
    }
    relevant_document_keys = [
        document_key
        for document_key, relevance in relevance_by_key.items()
        if relevance > 0
    ]
    relevant_document_ids = [
        key_to_runtime_id[document_key] for document_key in relevant_document_keys
    ]
    relevant_document_id_set = set(relevant_document_ids)
    ranked_document_ids = _dedupe_preserving_order(retrieved_document_ids)

    evaluation: IrQueryEvaluation = {
        "has_qrels": True,
        "retrieved_document_ids": ranked_document_ids,
        "relevant_document_keys": relevant_document_keys,
        "relevant_document_ids": relevant_document_ids,
        "relevant_count": len(relevant_document_keys),
        "qrels": [
            {
                "document_key": document_key,
                "runtime_document_id": key_to_runtime_id[document_key],
                "relevance": relevance,
            }
            for document_key, relevance in relevance_by_key.items()
        ],
    }

    for k in k_values:
        top_k_document_ids = ranked_document_ids[:k]
        relevant_retrieved_count = _relevant_retrieved_count(
            top_k_document_ids,
            relevant_document_id_set,
        )
        evaluation[f"hit_at_{k}"] = relevant_retrieved_count > 0
        evaluation[f"precision_at_{k}"] = relevant_retrieved_count / k
        evaluation[f"recall_at_{k}"] = (
            relevant_retrieved_count / len(relevant_document_id_set)
            if relevant_document_id_set
            else None
        )
        evaluation[f"ndcg_at_{k}"] = _ndcg_at_k(
            ranked_document_ids,
            relevance_by_runtime_id,
            k=k,
        )

    evaluation["reciprocal_rank"] = _reciprocal_rank(
        ranked_document_ids,
        relevant_document_id_set,
    )
    return evaluation


def calculate_summary_ir_metrics(
    query_evaluations: Iterable[Mapping[str, Any] | None],
    *,
    k_values: Sequence[int] = K_VALUES,
) -> IrMetrics | None:
    evaluations = [
        evaluation
        for evaluation in query_evaluations
        if isinstance(evaluation, Mapping)
    ]
    if not evaluations:
        return None

    metrics: IrMetrics = {
        "queries_with_qrels": len(evaluations),
        "queries_with_positive_qrels": sum(
            1 for evaluation in evaluations if _positive_relevant_count(evaluation) > 0
        ),
    }
    for k in k_values:
        metrics[f"hit_rate_at_{k}"] = _mean(
            1.0 if bool(evaluation.get(f"hit_at_{k}")) else 0.0
            for evaluation in evaluations
        )
        metrics[f"recall_at_{k}"] = _mean(
            _optional_float(evaluation.get(f"recall_at_{k}"))
            for evaluation in evaluations
        )
        metrics[f"precision_at_{k}"] = _mean(
            _optional_float(evaluation.get(f"precision_at_{k}"))
            for evaluation in evaluations
        )
        metrics[f"ndcg_at_{k}"] = _mean(
            _optional_float(evaluation.get(f"ndcg_at_{k}"))
            for evaluation in evaluations
        )
    metrics["mrr"] = _mean(
        _optional_float(evaluation.get("reciprocal_rank"))
        for evaluation in evaluations
    )
    return metrics


def _max_relevance_by_key(qrels: Sequence[QueryQrelSpec]) -> dict[str, int]:
    relevance_by_key: dict[str, int] = {}
    for qrel in qrels:
        current_relevance = relevance_by_key.get(qrel.document_key)
        if current_relevance is None or qrel.relevance > current_relevance:
            relevance_by_key[qrel.document_key] = qrel.relevance
    return relevance_by_key


def _dedupe_preserving_order(document_ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ranked_document_ids: list[str] = []
    for document_id in document_ids:
        if document_id in seen:
            continue
        seen.add(document_id)
        ranked_document_ids.append(document_id)
    return ranked_document_ids


def _relevant_retrieved_count(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: set[str],
) -> int:
    return sum(1 for document_id in retrieved_document_ids if document_id in relevant_document_ids)


def _reciprocal_rank(
    ranked_document_ids: Sequence[str],
    relevant_document_ids: set[str],
) -> float | None:
    if not relevant_document_ids:
        return None
    for index, document_id in enumerate(ranked_document_ids, start=1):
        if document_id in relevant_document_ids:
            return 1 / index
    return 0.0


def _ndcg_at_k(
    ranked_document_ids: Sequence[str],
    relevance_by_runtime_id: Mapping[str, int],
    *,
    k: int,
) -> float | None:
    ideal_relevances = sorted(
        (relevance for relevance in relevance_by_runtime_id.values() if relevance > 0),
        reverse=True,
    )
    if not ideal_relevances:
        return None

    dcg = _dcg(
        relevance_by_runtime_id.get(document_id, 0)
        for document_id in ranked_document_ids[:k]
    )
    ideal_dcg = _dcg(ideal_relevances[:k])
    if ideal_dcg == 0:
        return None
    return dcg / ideal_dcg


def _dcg(relevances: Iterable[int]) -> float:
    return sum(
        ((2**relevance) - 1) / log2(index + 2)
        for index, relevance in enumerate(relevances)
        if relevance > 0
    )


def _positive_relevant_count(evaluation: Mapping[str, Any]) -> int:
    value = evaluation.get("relevant_count")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    relevant_document_ids = evaluation.get("relevant_document_ids")
    if isinstance(relevant_document_ids, list):
        return len(relevant_document_ids)
    return 0


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mean(values: Iterable[float | None]) -> float | None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)
