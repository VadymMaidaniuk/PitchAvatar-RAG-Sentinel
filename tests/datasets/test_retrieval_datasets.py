from __future__ import annotations

from pathlib import Path

import pytest

from pitchavatar_rag_sentinel.datasets.loader import discover_datasets, load_dataset

pytestmark = pytest.mark.offline

DATASET_ROOT = Path("datasets/retrieval")
EXPECTED_CATEGORY_DIRS = {
    "smoke",
    "regression",
    "filters",
    "negative",
    "chunking",
    "multilingual",
    "precision",
    "diagnostics",
}
NEW_DATASET_IDS = {
    "retrieval_smoke_v1",
    "metadata_filtering_v1",
    "negative_queries_v1",
    "chunk_boundary_v1",
    "uk_en_mixed_v1",
    "retrieval_precision_v1",
    "alpha_matrix_v1",
}


def load_all_retrieval_datasets():
    return [(path, load_dataset(path)) for path in discover_datasets(DATASET_ROOT)]


def test_retrieval_dataset_category_layout_exists() -> None:
    existing_dirs = {
        path.name for path in DATASET_ROOT.iterdir() if path.is_dir()
    }

    assert EXPECTED_CATEGORY_DIRS <= existing_dirs


def test_all_retrieval_datasets_load_successfully() -> None:
    datasets = load_all_retrieval_datasets()

    assert datasets
    assert NEW_DATASET_IDS <= {dataset.dataset_id for _, dataset in datasets}


def test_retrieval_dataset_ids_are_unique() -> None:
    datasets = load_all_retrieval_datasets()
    dataset_ids = [dataset.dataset_id for _, dataset in datasets]

    assert len(dataset_ids) == len(set(dataset_ids))


def test_retrieval_query_ids_are_unique_within_each_dataset() -> None:
    for path, dataset in load_all_retrieval_datasets():
        query_ids = [query.query_id for query in dataset.queries]

        assert len(query_ids) == len(set(query_ids)), path


def test_root_baseline_datasets_still_load() -> None:
    assert load_dataset(DATASET_ROOT / "quantum_baseline.json").dataset_id == "quantum_baseline"
    assert (
        load_dataset(DATASET_ROOT / "retrieval_baseline_v1.json").dataset_id
        == "retrieval_baseline_v1"
    )


def test_new_datasets_include_chunk_level_expectations() -> None:
    datasets = {
        dataset.dataset_id: dataset
        for _, dataset in load_all_retrieval_datasets()
        if dataset.dataset_id in NEW_DATASET_IDS
    }

    assert datasets.keys() == NEW_DATASET_IDS
    assert any(
        query.expectations.expected_top1_chunk_contains
        or query.expectations.expected_in_topk_chunk_contains
        or query.expectations.forbidden_chunk_contains
        for dataset in datasets.values()
        for query in dataset.queries
    )
