from __future__ import annotations

import argparse
from pathlib import Path

from pitchavatar_rag_sentinel.dataset_builder.parsers import parse_source_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview local .txt or .md source sections for a draft retrieval dataset."
    )
    parser.add_argument("source", type=Path, help="Path to a .txt or .md source file.")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=120,
        help="Maximum characters to show for each section preview.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parsed_source = parse_source_file(args.source)

    print(f"Source file: {parsed_source.source_file_name}")
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


if __name__ == "__main__":
    raise SystemExit(main())
