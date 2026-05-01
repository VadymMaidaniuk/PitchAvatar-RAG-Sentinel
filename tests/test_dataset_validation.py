from __future__ import annotations

import pytest

from pitchavatar_rag_sentinel.datasets.models import RetrievalDataset


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
