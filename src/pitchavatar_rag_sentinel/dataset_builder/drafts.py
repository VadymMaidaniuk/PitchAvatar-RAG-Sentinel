from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, Mapping, Sequence

from pitchavatar_rag_sentinel.dataset_builder.parsers import ParsedSection, ParsedSource
from pitchavatar_rag_sentinel.datasets.models import RetrievalDataset

DocumentMode = Literal["file_as_document", "section_as_document"]
_DOCUMENT_MODES = {"file_as_document", "section_as_document"}


@dataclass(frozen=True, slots=True)
class ExpectationDraft:
    expected_top1: str | None = None
    expected_in_topk: list[str] = field(default_factory=list)
    forbidden_docs: list[str] = field(default_factory=list)
    expected_top1_chunk_contains: list[str] = field(default_factory=list)
    expected_in_topk_chunk_contains: list[str] = field(default_factory=list)
    forbidden_chunk_contains: list[str] = field(default_factory=list)
    min_results: int = 1
    expect_empty: bool = False


@dataclass(frozen=True, slots=True)
class QueryDraft:
    query: str
    expectations: ExpectationDraft | Mapping[str, Any] = field(default_factory=ExpectationDraft)
    query_id: str | None = None
    alpha: Literal[0.0, 0.5, 1.0] = 0.5
    top_k: int = 10
    threshold: float = 0.3
    document_scope: Literal["all"] | list[str] = "all"
    filters: list[dict[str, Any]] = field(default_factory=list)


def build_retrieval_dataset(
    *,
    dataset_id: str,
    parsed_source: ParsedSource,
    query_drafts: Sequence[QueryDraft | Mapping[str, Any]],
    description: str | None = None,
    document_mode: DocumentMode | None = None,
) -> RetrievalDataset:
    resolved_document_mode = resolve_document_mode(parsed_source, document_mode=document_mode)
    document_payloads = _document_payloads(
        parsed_source,
        document_mode=resolved_document_mode,
    )
    query_payloads = _query_payloads(query_drafts)

    return RetrievalDataset.model_validate(
        {
            "dataset_id": dataset_id,
            "description": description
            or f"Draft retrieval dataset generated from {parsed_source.source_file_name}.",
            "documents": document_payloads,
            "queries": query_payloads,
        }
    )


def dataset_to_pretty_json(dataset: RetrievalDataset) -> str:
    return dataset.model_dump_json(indent=2)


def resolve_document_mode(
    parsed_source: ParsedSource,
    *,
    document_mode: DocumentMode | str | None = None,
) -> DocumentMode:
    if document_mode is not None:
        if document_mode not in _DOCUMENT_MODES:
            supported = ", ".join(sorted(_DOCUMENT_MODES))
            raise ValueError(f"unsupported document mode {document_mode!r}; supported modes: {supported}")
        return document_mode  # type: ignore[return-value]

    source_type = _source_type(parsed_source.source_file_name)
    if source_type in {"pdf", "pptx"}:
        return "file_as_document"
    return "section_as_document"


def document_keys_for_source(
    parsed_source: ParsedSource,
    *,
    document_mode: DocumentMode | str | None = None,
) -> list[str]:
    resolved_document_mode = resolve_document_mode(parsed_source, document_mode=document_mode)
    if resolved_document_mode == "file_as_document":
        return [make_source_document_key(parsed_source.source_file_name)]

    keys: list[str] = []
    used_keys: set[str] = set()
    for section in parsed_source.sections:
        key = _unique_key(make_document_key(parsed_source.source_file_name, section), used_keys)
        used_keys.add(key)
        keys.append(key)
    return keys


def make_source_document_key(source_file_name: str) -> str:
    return _slugify(re.sub(r"\.[^.]+$", "", source_file_name)) or "source"


def make_document_key(source_file_name: str, section: ParsedSection) -> str:
    source_slug = _slugify(re.sub(r"\.[^.]+$", "", source_file_name)) or "source"
    section_slug = _slugify(section.section_id) or "section"
    return f"{source_slug}_{section_slug}"


def _document_payloads(
    parsed_source: ParsedSource,
    *,
    document_mode: DocumentMode,
) -> list[dict[str, Any]]:
    if document_mode == "file_as_document":
        return [_source_document_payload(parsed_source)]

    return _section_document_payloads(parsed_source)


def _source_document_payload(parsed_source: ParsedSource) -> dict[str, Any]:
    source_type = _source_type(parsed_source.source_file_name)
    content = parsed_source.extracted_text or _normalize_whitespace(
        " ".join(section.text for section in parsed_source.sections)
    )
    return {
        "key": make_source_document_key(parsed_source.source_file_name),
        "content": content,
        "metadata": {
            "source_file_name": parsed_source.source_file_name,
            "source_section_count": str(len(parsed_source.sections)),
            "document_mode": "file_as_document",
            "type": source_type,
        },
        "min_expected_chunks": 1,
    }


def _section_document_payloads(parsed_source: ParsedSource) -> list[dict[str, Any]]:
    payloads = []
    used_keys: set[str] = set()
    source_type = _source_type(parsed_source.source_file_name)

    for section in parsed_source.sections:
        key = _unique_key(make_document_key(parsed_source.source_file_name, section), used_keys)
        used_keys.add(key)
        payloads.append(
            {
                "key": key,
                "content": section.text,
                "metadata": {
                    "source_file_name": parsed_source.source_file_name,
                    "source_section_id": section.section_id,
                    "source_section_title": section.title,
                    "document_mode": "section_as_document",
                    "type": source_type,
                },
                "min_expected_chunks": 1,
            }
        )

    return payloads


def _query_payloads(query_drafts: Sequence[QueryDraft | Mapping[str, Any]]) -> list[dict[str, Any]]:
    payloads = []
    for index, draft in enumerate(query_drafts, start=1):
        draft_payload = _draft_to_payload(draft)
        query_text = str(draft_payload.get("query", "")).strip()
        query_id = str(draft_payload.get("query_id") or "").strip()
        if not query_id:
            query_id = _generated_query_id(index, query_text)
        expectations = _expectations_payload(draft_payload.get("expectations", {}))
        payloads.append(
            {
                "query_id": query_id,
                "query": query_text,
                "alpha": draft_payload.get("alpha", 0.5),
                "top_k": draft_payload.get("top_k", 10),
                "threshold": draft_payload.get("threshold", 0.3),
                "document_scope": draft_payload.get("document_scope", "all"),
                "filters": draft_payload.get("filters", []),
                "expectations": expectations,
            }
        )
    return payloads


def _draft_to_payload(draft: QueryDraft | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(draft, Mapping):
        return dict(draft)
    if is_dataclass(draft):
        return asdict(draft)
    raise TypeError(f"unsupported query draft type: {type(draft)!r}")


def _expectations_payload(expectations: ExpectationDraft | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(expectations, Mapping):
        return dict(expectations)
    if is_dataclass(expectations):
        return asdict(expectations)
    raise TypeError(f"unsupported expectations draft type: {type(expectations)!r}")


def _generated_query_id(index: int, query: str) -> str:
    query_slug = _slugify(query)
    if query_slug:
        return f"q_{index:03d}_{query_slug}"
    return f"q_{index:03d}"


def _unique_key(base_key: str, used_keys: set[str]) -> str:
    if base_key not in used_keys:
        return base_key

    suffix = 2
    while f"{base_key}_{suffix}" in used_keys:
        suffix += 1
    return f"{base_key}_{suffix}"


def _source_type(source_file_name: str) -> str:
    return source_file_name.rsplit(".", maxsplit=1)[-1].lower()


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return re.sub(r"_+", "_", slug)[:64].strip("_")
