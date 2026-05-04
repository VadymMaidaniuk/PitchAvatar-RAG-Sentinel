from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from pitchavatar_rag_sentinel.dataset_builder.parsers import (
    ParserDependencyError,
    parse_source_bytes,
    parse_source_file,
    parse_source_text,
)

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


def test_pdf_parser_extracts_text_from_text_layer_pdf(
    pdf_source_factory: Callable[[Sequence[str], str], Path],
) -> None:
    pytest.importorskip("pypdf")
    source_path = pdf_source_factory(
        [
            "First page explains rollout guardrails.",
            "Second page contains approval code RSM-17.",
        ],
        "policy.pdf",
    )

    parsed = parse_source_file(source_path)

    assert parsed.source_file_name == "policy.pdf"
    assert "First page explains rollout guardrails." in parsed.extracted_text
    assert "Second page contains approval code RSM-17." in parsed.extracted_text


def test_pdf_parser_creates_page_preview_sections(
    pdf_source_factory: Callable[[Sequence[str], str], Path],
) -> None:
    pytest.importorskip("pypdf")
    source_path = pdf_source_factory(["Alpha page text.", "Beta page text."], "pages.pdf")

    parsed = parse_source_file(source_path)

    assert [section.section_id for section in parsed.sections] == ["page_001", "page_002"]
    assert [section.title for section in parsed.sections] == ["Page 1", "Page 2"]
    assert parsed.sections[0].character_count == len("Alpha page text.")


def test_pptx_parser_extracts_text_from_synthetic_deck(
    pptx_source_factory: Callable[[str], Path],
) -> None:
    pytest.importorskip("pptx")
    parsed = parse_source_file(pptx_source_factory("deck.pptx"))

    assert "Launch Checklist" in parsed.extracted_text
    assert "Approve release window and rollback contact." in parsed.extracted_text
    assert "Escalate invoice anomalies before launch." in parsed.extracted_text
    assert "SLA-42" in parsed.extracted_text


def test_pptx_parser_creates_slide_preview_sections(
    pptx_source_factory: Callable[[str], Path],
) -> None:
    pytest.importorskip("pptx")
    parsed = parse_source_file(pptx_source_factory("sections.pptx"))

    assert [section.section_id for section in parsed.sections] == ["slide_001", "slide_002"]
    assert [section.title for section in parsed.sections] == [
        "Slide 1: Launch Checklist",
        "Slide 2: Risk Review",
    ]


def test_pptx_parser_extracts_slide_title_and_body_text(
    pptx_source_factory: Callable[[str], Path],
) -> None:
    pytest.importorskip("pptx")
    parsed = parse_source_file(pptx_source_factory("body.pptx"))

    assert parsed.sections[0].title == "Slide 1: Launch Checklist"
    assert "Launch Checklist" in parsed.sections[0].text
    assert "Approve release window and rollback contact." in parsed.sections[0].text
    assert "Risk Review" in parsed.sections[1].text
    assert "Escalate invoice anomalies before launch." in parsed.sections[1].text


def test_missing_pdf_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(module_name: str, *args: object, **kwargs: object) -> object:
        if module_name == "pypdf":
            raise ModuleNotFoundError("No module named 'pypdf'")
        return real_import_module(module_name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ParserDependencyError, match="Install parser extras"):
        parse_source_bytes(b"%PDF-1.4", source_file_name="missing.pdf")


def test_missing_pptx_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(module_name: str, *args: object, **kwargs: object) -> object:
        if module_name == "pptx":
            raise ModuleNotFoundError("No module named 'pptx'")
        return real_import_module(module_name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ParserDependencyError, match="Install parser extras"):
        parse_source_bytes(b"not a real pptx", source_file_name="missing.pptx")
