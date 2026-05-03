from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".txt"}
_DEFAULT_MAX_SECTION_CHARS = 1200
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True, slots=True)
class ParsedSection:
    section_id: str
    title: str
    text: str
    character_count: int


@dataclass(frozen=True, slots=True)
class ParsedSource:
    source_file_name: str
    extracted_text: str
    sections: list[ParsedSection]


def parse_source_file(path: str | Path) -> ParsedSource:
    source_path = Path(path)
    return parse_source_text(
        source_path.read_text(encoding="utf-8", errors="replace"),
        source_file_name=source_path.name,
    )


def parse_source_text(text: str, *, source_file_name: str) -> ParsedSource:
    extension = Path(source_file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"unsupported source type {extension!r}; supported types: {supported}")

    if extension == ".md":
        sections = _parse_markdown_sections(text)
    else:
        sections = _parse_txt_sections(text)

    return ParsedSource(
        source_file_name=source_file_name,
        extracted_text=_normalize_whitespace(" ".join(section.text for section in sections)),
        sections=sections,
    )


def _parse_markdown_sections(text: str) -> list[ParsedSection]:
    normalized_lines = _normalize_newlines(text).split("\n")
    raw_sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in normalized_lines:
        match = _HEADING_PATTERN.match(line)
        if match:
            if current_title is not None or _normalize_whitespace(" ".join(current_lines)):
                raw_sections.append((current_title or "Introduction", current_lines))
            current_title = _normalize_whitespace(match.group(2).strip("# "))
            current_lines = [current_title]
            continue
        current_lines.append(line)

    if current_title is not None or _normalize_whitespace(" ".join(current_lines)):
        raw_sections.append((current_title or "Introduction", current_lines))

    if not raw_sections:
        return []

    if len(raw_sections) == 1 and raw_sections[0][0] == "Introduction":
        return _parse_txt_sections(text)

    return _build_sections(
        (
            (title, normalized_text)
            for title, lines in raw_sections
            for normalized_text in [_normalize_whitespace(" ".join(lines))]
            if normalized_text
        )
    )


def _parse_txt_sections(text: str) -> list[ParsedSection]:
    normalized_text = _normalize_newlines(text)
    paragraphs = [
        _normalize_whitespace(part)
        for part in re.split(r"\n\s*\n+", normalized_text)
        if _normalize_whitespace(part)
    ]
    section_texts: list[str] = []
    for paragraph in paragraphs:
        section_texts.extend(_split_long_text(paragraph))

    return _build_sections(
        (None, section_text)
        for section_text in section_texts
        if section_text
    )


def _build_sections(section_items: Iterable[tuple[str | None, str]]) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    for index, item in enumerate(section_items, start=1):
        title, text = item
        normalized_title = title or _generated_title(index, text)
        sections.append(
            ParsedSection(
                section_id=_section_id(index, normalized_title),
                title=normalized_title,
                text=text,
                character_count=len(text),
            )
        )
    return sections


def _split_long_text(text: str, *, max_chars: int = _DEFAULT_MAX_SECTION_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sections = []
    remaining = text
    while len(remaining) > max_chars:
        split_at = _best_split_index(remaining, max_chars)
        sections.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        sections.append(remaining)
    return sections


def _best_split_index(text: str, max_chars: int) -> int:
    sentence_boundary = max(text.rfind(". ", 0, max_chars), text.rfind("? ", 0, max_chars))
    sentence_boundary = max(sentence_boundary, text.rfind("! ", 0, max_chars))
    if sentence_boundary >= max_chars // 2:
        return sentence_boundary + 1

    whitespace_boundary = text.rfind(" ", 0, max_chars)
    if whitespace_boundary >= max_chars // 2:
        return whitespace_boundary

    return max_chars


def _section_id(index: int, title: str) -> str:
    return f"section_{index:03d}_{_slugify(title) or 'untitled'}"


def _generated_title(index: int, text: str) -> str:
    words = text.split()
    if not words:
        return f"Section {index}"
    preview = " ".join(words[:8])
    if len(words) > 8:
        preview = f"{preview}..."
    return f"Section {index}: {preview}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return re.sub(r"_+", "_", slug)[:64].strip("_")


def _normalize_newlines(text: str) -> str:
    return text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
