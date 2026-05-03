from __future__ import annotations

from pathlib import Path

import pytest

from pitchavatar_rag_sentinel.datasets.loader import discover_datasets, load_dataset
from pitchavatar_rag_sentinel.executors.retrieval_flow import RetrievalFlowExecutor

pytestmark = [pytest.mark.integration, pytest.mark.grpc, pytest.mark.opensearch]


DATASET_ROOT = Path("datasets/retrieval")
STABLE_DATASET_CATEGORIES = {
    "smoke",
    "filters",
    "negative",
    "chunking",
    "multilingual",
    "regression",
}


def discover_stable_datasets(root: Path) -> list[Path]:
    return [
        path
        for path in discover_datasets(root)
        if path.parent == root or path.parent.name in STABLE_DATASET_CATEGORIES
    ]


@pytest.mark.retrieval
@pytest.mark.dataset
@pytest.mark.destructive
@pytest.mark.parametrize(
    "dataset_path",
    discover_stable_datasets(DATASET_ROOT),
    ids=lambda path: Path(path).stem,
)
def test_retrieval_dataset_flow(
    dataset_path: Path,
    retrieval_executor: RetrievalFlowExecutor,
) -> None:
    dataset = load_dataset(dataset_path)
    summary = retrieval_executor.run_dataset(dataset)

    failed_queries = [
        result.query_id for result in summary.query_results if not result.passed
    ]
    cleanup_not_verified = [
        result.runtime_document_id
        for result in summary.cleanup_results
        if not result.cleanup_verified
    ]

    assert not failed_queries, (
        f"Dataset {summary.dataset_id} has failed queries: {failed_queries}. "
        f"Artifacts: {summary.run_dir}"
    )
    assert not cleanup_not_verified, (
        f"Dataset {summary.dataset_id} did not clean up documents: {cleanup_not_verified}. "
        f"Artifacts: {summary.run_dir}"
    )
