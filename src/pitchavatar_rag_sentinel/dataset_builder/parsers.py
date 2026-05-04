from __future__ import annotations

import importlib
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".md", ".pdf", ".pptx", ".txt"}
TEXT_EXTENSIONS = {".md", ".txt"}
_DEFAULT_MAX_SECTION_CHARS = 1200
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_PARSER_INSTALL_HINT = (
    r'.venv\Scripts\python -m pip install -e ".[dev,report,ui,parsers]"'
)


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


class ParserDependencyError(RuntimeError):
    """Raised when an optional local parser adapter dependency is not installed."""


def parse_source_file(path: str | Path) -> ParsedSource:
    source_path = Path(path)
    extension = source_path.suffix.lower()
    if extension in TEXT_EXTENSIONS:
        return parse_source_text(
            source_path.read_text(encoding="utf-8", errors="replace"),
            source_file_name=source_path.name,
        )
    if extension == ".pdf":
        return _parse_pdf(source_path, source_file_name=source_path.name)
    if extension == ".pptx":
        return _parse_pptx(source_path, source_file_name=source_path.name)
    _raise_unsupported_extension(extension)


def parse_source_bytes(content: bytes, *, source_file_name: str) -> ParsedSource:
    extension = Path(source_file_name).suffix.lower()
    if extension in TEXT_EXTENSIONS:
        return parse_source_text(
            content.decode("utf-8", errors="replace"),
            source_file_name=source_file_name,
        )
    if extension == ".pdf":
        return _parse_pdf(BytesIO(content), source_file_name=source_file_name)
    if extension == ".pptx":
        return _parse_pptx(BytesIO(content), source_file_name=source_file_name)
    _raise_unsupported_extension(extension)


def parse_source_text(text: str, *, source_file_name: str) -> ParsedSource:
    extension = Path(source_file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        _raise_unsupported_extension(extension)
    if extension not in TEXT_EXTENSIONS:
        supported = ", ".join(sorted(TEXT_EXTENSIONS))
        raise ValueError(
            f"text parsing is only supported for {supported}; use parse_source_file or "
            f"parse_source_bytes for {extension!r}."
        )

    if extension == ".md":
        sections = _parse_markdown_sections(text)
    else:
        sections = _parse_txt_sections(text)

    return ParsedSource(
        source_file_name=source_file_name,
        extracted_text=_normalize_whitespace(" ".join(section.text for section in sections)),
        sections=sections,
    )


def _parse_pdf(source: str | Path | BytesIO, *, source_file_name: str) -> ParsedSource:
    pypdf = _require_dependency(
        "pypdf",
        feature_name="PDF parsing",
        package_name="pypdf",
    )

    try:
        reader = pypdf.PdfReader(source)
    except Exception as exc:
        raise ValueError(
            f"Could not read PDF {source_file_name!r}. Confirm the file is a valid, readable "
            "text-layer PDF."
        ) from exc

    if reader.is_encrypted:
        try:
            decrypt_result = reader.decrypt("")
        except Exception as exc:
            raise ValueError(
                f"PDF {source_file_name!r} is encrypted and cannot be parsed by the offline "
                "Dataset Builder. Provide an unencrypted text-layer PDF."
            ) from exc
        if not decrypt_result:
            raise ValueError(
                f"PDF {source_file_name!r} is encrypted and requires a password. Provide an "
                "unencrypted text-layer PDF."
            )

    sections: list[ParsedSection] = []
    try:
        pages = list(reader.pages)
    except Exception as exc:
        raise ValueError(
            f"Could not read pages from PDF {source_file_name!r}. Provide an unencrypted "
            "text-layer PDF."
        ) from exc

    for page_index, page in enumerate(pages, start=1):
        try:
            text = _normalize_whitespace(page.extract_text() or "")
        except Exception as exc:
            raise ValueError(
                f"Could not extract text from page {page_index} of PDF {source_file_name!r}. "
                "Only text-layer PDFs are supported; OCR is not available."
            ) from exc
        if not text:
            continue
        sections.append(
            ParsedSection(
                section_id=f"page_{page_index:03d}",
                title=f"Page {page_index}",
                text=text,
                character_count=len(text),
            )
        )

    return ParsedSource(
        source_file_name=source_file_name,
        extracted_text=_normalize_whitespace(" ".join(section.text for section in sections)),
        sections=sections,
    )


def _parse_pptx(source: str | Path | BytesIO, *, source_file_name: str) -> ParsedSource:
    pptx = _require_dependency(
        "pptx",
        feature_name="PPTX parsing",
        package_name="python-pptx",
    )

    try:
        presentation = pptx.Presentation(source)
    except Exception as exc:
        raise ValueError(
            f"Could not read PPTX {source_file_name!r}. Confirm the file is a valid .pptx file."
        ) from exc

    sections: list[ParsedSection] = []
    for slide_index, slide in enumerate(presentation.slides, start=1):
        title_text = _normalize_whitespace(_shape_text(getattr(slide.shapes, "title", None)))
        slide_text = _normalize_whitespace(" ".join(_slide_text_parts(slide)))
        section_title = f"Slide {slide_index}"
        if title_text:
            section_title = f"{section_title}: {title_text}"
        sections.append(
            ParsedSection(
                section_id=f"slide_{slide_index:03d}",
                title=section_title,
                text=slide_text,
                character_count=len(slide_text),
            )
        )

    return ParsedSource(
        source_file_name=source_file_name,
        extracted_text=_normalize_whitespace(" ".join(section.text for section in sections)),
        sections=sections,
    )


def _slide_text_parts(slide: Any) -> list[str]:
    parts: list[str] = []
    pending = deque(slide.shapes)
    while pending:
        shape = pending.popleft()
        if hasattr(shape, "shapes"):
            pending.extend(shape.shapes)
            continue
        if getattr(shape, "has_table", False):
            parts.extend(_table_text_parts(shape.table))
            continue
        shape_text = _shape_text(shape)
        if shape_text:
            parts.append(shape_text)
    return parts


def _shape_text(shape: Any) -> str:
    if shape is None or not getattr(shape, "has_text_frame", False):
        return ""
    return _normalize_whitespace(getattr(shape, "text", "") or "")


def _table_text_parts(table: Any) -> list[str]:
    parts: list[str] = []
    for row in table.rows:
        for cell in row.cells:
            cell_text = _normalize_whitespace(cell.text or "")
            if cell_text:
                parts.append(cell_text)
    return parts


def _require_dependency(module_name: str, *, feature_name: str, package_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ParserDependencyError(
            f"{feature_name} requires optional parser dependency {package_name!r}. "
            f"Install parser extras with: {_PARSER_INSTALL_HINT}"
        ) from exc


def _raise_unsupported_extension(extension: str) -> None:
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(f"unsupported source type {extension!r}; supported types: {supported}")


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
