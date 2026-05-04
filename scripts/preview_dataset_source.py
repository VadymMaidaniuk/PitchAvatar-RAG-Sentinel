from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pitchavatar_rag_sentinel.dataset_builder.parsers import ParserDependencyError, parse_source_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview local .txt, .md, .pdf, or .pptx source sections for a draft dataset."
    )
    parser.add_argument("source", type=Path, help="Path to a .txt, .md, .pdf, or .pptx source file.")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=120,
        help="Maximum characters to show for each section preview.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        parsed_source = parse_source_file(args.source)
    except (ParserDependencyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Source file: {parsed_source.source_file_name}")
    print(f"Source type: {_source_type(parsed_source.source_file_name)}")
    print(f"Total characters: {len(parsed_source.extracted_text)}")
    print(f"Sections: {len(parsed_source.sections)}")

    for section in parsed_source.sections:
        preview = _preview(section.text, args.preview_chars)
        print(
            f"- {section.section_id} | {section.title} | "
            f"{section.character_count} chars | {preview}"
        )

    return 0


def _preview(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)].rstrip()}..."


def _source_type(source_file_name: str) -> str:
    return Path(source_file_name).suffix.lower().lstrip(".") or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
