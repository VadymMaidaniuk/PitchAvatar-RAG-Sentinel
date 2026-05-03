from __future__ import annotations

import pytest

from pitchavatar_rag_sentinel.dataset_builder.parsers import parse_source_text

pytestmark = pytest.mark.offline


def test_txt_parser_extracts_blank_line_sections() -> None:
    parsed = parse_source_text(
        "First paragraph about billing.\n\nSecond paragraph about release windows.",
        source_file_name="notes.txt",
    )

    assert parsed.source_file_name == "notes.txt"
    assert len(parsed.sections) == 2
    assert parsed.sections[0].section_id.startswith("section_001_")
    assert parsed.sections[0].text == "First paragraph about billing."
    assert parsed.sections[1].text == "Second paragraph about release windows."


def test_markdown_parser_splits_by_headings() -> None:
    parsed = parse_source_text(
        "# Release Window\nDeploy on Tuesday.\n\n## Rollback\nUse approval code RSM-17.",
        source_file_name="runbook.md",
    )

    assert [section.title for section in parsed.sections] == ["Release Window", "Rollback"]
    assert parsed.sections[0].text == "Release Window Deploy on Tuesday."
    assert parsed.sections[1].text == "Rollback Use approval code RSM-17."


def test_parser_normalizes_whitespace() -> None:
    parsed = parse_source_text(
        "Alpha\t\tbeta\n   gamma\r\n\r\nDelta    epsilon",
        source_file_name="source.txt",
    )

    assert [section.text for section in parsed.sections] == [
        "Alpha beta gamma",
        "Delta epsilon",
    ]
    assert parsed.extracted_text == "Alpha beta gamma Delta epsilon"


def test_parser_handles_empty_file_gracefully() -> None:
    parsed = parse_source_text(" \n\t\n ", source_file_name="empty.md")

    assert parsed.extracted_text == ""
    assert parsed.sections == []


def test_parser_strips_utf8_bom() -> None:
    parsed = parse_source_text("\ufeff# Alpha\nFirst section.", source_file_name="bom.md")

    assert parsed.sections[0].title == "Alpha"
    assert parsed.sections[0].text == "Alpha First section."
