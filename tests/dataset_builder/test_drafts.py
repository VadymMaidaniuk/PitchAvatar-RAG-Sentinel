from __future__ import annotations

import pytest

from pitchavatar_rag_sentinel.dataset_builder.drafts import (
    ExpectationDraft,
    QueryDraft,
    build_retrieval_dataset,
    make_document_key,
)
from pitchavatar_rag_sentinel.dataset_builder.parsers import (
    ParsedSection,
    ParsedSource,
    parse_source_text,
)
from pitchavatar_rag_sentinel.datasets.models import RetrievalDataset

pytestmark = pytest.mark.offline


def test_generated_dataset_validates_with_existing_schema() -> None:
    parsed = parse_source_text(
        "# Release Window\nRollback approval code RSM-17 is required.",
        source_file_name="runbook.md",
    )
    expected_key = make_document_key(parsed.source_file_name, parsed.sections[0])

    dataset = build_retrieval_dataset(
        dataset_id="runbook_draft",
        parsed_source=parsed,
        query_drafts=[
            QueryDraft(
                query_id="q_release_approval",
                query="rollback approval code",
                expectations=ExpectationDraft(
                    expected_top1=expected_key,
                    expected_in_topk=[expected_key],
                    expected_top1_chunk_contains=["RSM-17"],
                ),
            )
        ],
    )

    assert RetrievalDataset.model_validate(dataset.model_dump()).dataset_id == "runbook_draft"
    assert dataset.documents[0].key == expected_key
    assert dataset.queries[0].expectations.expected_top1 == expected_key


def test_generated_document_keys_are_unique() -> None:
    parsed = ParsedSource(
        source_file_name="duplicate.txt",
        extracted_text="Alpha Beta",
        sections=[
            ParsedSection(
                section_id="section_001_duplicate",
                title="Duplicate",
                text="Alpha",
                character_count=5,
            ),
            ParsedSection(
                section_id="section_001_duplicate",
                title="Duplicate",
                text="Beta",
                character_count=4,
            ),
        ],
    )

    dataset = build_retrieval_dataset(
        dataset_id="duplicate_sections",
        parsed_source=parsed,
        query_drafts=[],
    )

    document_keys = [document.key for document in dataset.documents]
    assert document_keys == ["duplicate_section_001_duplicate", "duplicate_section_001_duplicate_2"]
    assert len(document_keys) == len(set(document_keys))


def test_generated_query_ids_are_unique_when_missing() -> None:
    parsed = parse_source_text("Only source text.", source_file_name="source.txt")

    dataset = build_retrieval_dataset(
        dataset_id="generated_query_ids",
        parsed_source=parsed,
        query_drafts=[
            QueryDraft(query="same query"),
            QueryDraft(query="same query"),
        ],
    )

    query_ids = [query.query_id for query in dataset.queries]
    assert query_ids == ["q_001_same_query", "q_002_same_query"]
    assert len(query_ids) == len(set(query_ids))
