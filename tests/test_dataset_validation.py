from __future__ import annotations

import pytest

from pitchavatar_rag_sentinel.datasets.loader import load_dataset
from pitchavatar_rag_sentinel.datasets.models import RetrievalDataset

pytestmark = pytest.mark.offline


def test_duplicate_query_id_fails_dataset_validation() -> None:
    with pytest.raises(ValueError, match="query_id values must be unique"):
        RetrievalDataset.model_validate(
            {
                "dataset_id": "duplicate-query-id",
                "description": "duplicate query_id regression",
                "documents": [
                    {
                        "key": "doc_a",
                        "content": "Test document content.",
                        "metadata": {"user_id": "qa-test", "type": "txt"},
                    }
                ],
                "queries": [
                    {
                        "query_id": "q_duplicate",
                        "query": "first query",
                        "alpha": 0.5,
                        "expectations": {"expected_top1": "doc_a"},
                    },
                    {
                        "query_id": "q_duplicate",
                        "query": "second query",
                        "alpha": 0.5,
                        "expectations": {"expected_top1": "doc_a"},
                    },
                ],
            }
        )


def test_existing_dataset_without_qrels_still_validates() -> None:
    dataset = load_dataset("datasets/retrieval/retrieval_baseline_v1.json")

    assert dataset.queries
    assert all(query.qrels == [] for query in dataset.queries)


def test_qrels_unknown_document_key_fails_dataset_validation() -> None:
    with pytest.raises(ValueError, match="unknown document keys in qrels"):
        RetrievalDataset.model_validate(
            {
                "dataset_id": "bad-qrels-key",
                "description": "qrels validation regression",
                "documents": [
                    {
                        "key": "doc_a",
                        "content": "Test document content.",
                        "metadata": {"user_id": "qa-test", "type": "txt"},
                    }
                ],
                "queries": [
                    {
                        "query_id": "q_bad_qrel",
                        "query": "test query",
                        "alpha": 0.5,
                        "qrels": [
                            {
                                "document_key": "doc_missing",
                                "relevance": 1,
                            }
                        ],
                    }
                ],
            }
        )


def test_qrels_relevance_must_be_non_negative_integer() -> None:
    with pytest.raises(ValueError):
        RetrievalDataset.model_validate(
            {
                "dataset_id": "bad-qrels-relevance",
                "description": "qrels relevance validation regression",
                "documents": [
                    {
                        "key": "doc_a",
                        "content": "Test document content.",
                        "metadata": {"user_id": "qa-test", "type": "txt"},
                    }
                ],
                "queries": [
                    {
                        "query_id": "q_bad_relevance",
                        "query": "test query",
                        "alpha": 0.5,
                        "qrels": [
                            {
                                "document_key": "doc_a",
                                "relevance": -1,
                            }
                        ],
                    }
                ],
            }
        )
