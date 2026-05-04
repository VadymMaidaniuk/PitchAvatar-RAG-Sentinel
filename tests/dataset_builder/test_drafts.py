from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from pitchavatar_rag_sentinel.dataset_builder.drafts import (
    ExpectationDraft,
    QueryDraft,
    build_retrieval_dataset,
    document_keys_for_source,
    make_document_key,
    make_source_document_key,
)
from pitchavatar_rag_sentinel.dataset_builder.parsers import (
    ParsedSection,
    ParsedSource,
    parse_source_file,
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


def test_pdf_dataset_validates_in_file_as_document_mode(
    pdf_source_factory: Callable[[Sequence[str], str], Path],
) -> None:
    pytest.importorskip("pypdf")
    parsed = parse_source_file(
        pdf_source_factory(
            ["Alpha rollout context.", "Beta approval code RSM-17."],
            "policy.pdf",
        )
    )
    expected_key = make_source_document_key(parsed.source_file_name)

    dataset = build_retrieval_dataset(
        dataset_id="pdf_file_mode",
        parsed_source=parsed,
        document_mode="file_as_document",
        query_drafts=[
            QueryDraft(
                query="approval code",
                expectations=ExpectationDraft(
                    expected_top1=expected_key,
                    expected_top1_chunk_contains=["RSM-17"],
                ),
            )
        ],
    )

    assert RetrievalDataset.model_validate(dataset.model_dump()).dataset_id == "pdf_file_mode"
    assert [document.key for document in dataset.documents] == [expected_key]
    assert dataset.documents[0].content == "Alpha rollout context. Beta approval code RSM-17."
    assert dataset.documents[0].metadata["document_mode"] == "file_as_document"


def test_pptx_dataset_validates_in_file_as_document_mode(
    pptx_source_factory: Callable[[str], Path],
) -> None:
    pytest.importorskip("pptx")
    parsed = parse_source_file(pptx_source_factory("deck.pptx"))
    expected_key = make_source_document_key(parsed.source_file_name)

    dataset = build_retrieval_dataset(
        dataset_id="pptx_file_mode",
        parsed_source=parsed,
        document_mode="file_as_document",
        query_drafts=[
            QueryDraft(
                query="rollback contact",
                expectations=ExpectationDraft(
                    expected_top1=expected_key,
                    expected_in_topk_chunk_contains=["rollback contact"],
                ),
            )
        ],
    )

    assert RetrievalDataset.model_validate(dataset.model_dump()).dataset_id == "pptx_file_mode"
    assert [document.key for document in dataset.documents] == [expected_key]
    assert "Launch Checklist" in dataset.documents[0].content
    assert "Risk Review" in dataset.documents[0].content


@pytest.mark.parametrize("source_kind", ["pdf", "pptx"])
def test_pdf_pptx_dataset_default_mode_is_file_as_document(
    source_kind: str,
    pdf_source_factory: Callable[[Sequence[str], str], Path],
    pptx_source_factory: Callable[[str], Path],
) -> None:
    if source_kind == "pdf":
        pytest.importorskip("pypdf")
        parsed = parse_source_file(
            pdf_source_factory(["Default PDF page one.", "Default PDF page two."], "default.pdf")
        )
    else:
        pytest.importorskip("pptx")
        parsed = parse_source_file(pptx_source_factory("default.pptx"))

    dataset = build_retrieval_dataset(
        dataset_id=f"{source_kind}_default_mode",
        parsed_source=parsed,
        query_drafts=[],
    )

    assert len(dataset.documents) == 1
    assert dataset.documents[0].key == make_source_document_key(parsed.source_file_name)
    assert dataset.documents[0].metadata["document_mode"] == "file_as_document"


@pytest.mark.parametrize("source_kind", ["pdf", "pptx"])
def test_pdf_pptx_dataset_validates_in_section_as_document_mode(
    source_kind: str,
    pdf_source_factory: Callable[[Sequence[str], str], Path],
    pptx_source_factory: Callable[[str], Path],
) -> None:
    if source_kind == "pdf":
        pytest.importorskip("pypdf")
        parsed = parse_source_file(
            pdf_source_factory(["Alpha page text.", "Beta page text."], "sections.pdf")
        )
    else:
        pytest.importorskip("pptx")
        parsed = parse_source_file(pptx_source_factory("sections.pptx"))

    expected_key = document_keys_for_source(parsed, document_mode="section_as_document")[0]
    dataset = build_retrieval_dataset(
        dataset_id=f"{source_kind}_section_mode",
        parsed_source=parsed,
        document_mode="section_as_document",
        query_drafts=[
            QueryDraft(
                query="section lookup",
                expectations=ExpectationDraft(expected_top1=expected_key),
            )
        ],
    )

    assert RetrievalDataset.model_validate(dataset.model_dump()).dataset_id == (
        f"{source_kind}_section_mode"
    )
    assert len(dataset.documents) == len(parsed.sections)
    assert dataset.documents[0].key == expected_key
    assert dataset.documents[0].metadata["document_mode"] == "section_as_document"
